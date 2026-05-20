"""Deterministic episode generation."""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque

from dataclasses import replace

from causalopsbench.domains import DomainTemplate, domain_names, get_domain
from causalopsbench.schemas import (
    ActionSpec,
    Episode,
    EvidenceRecord,
    FaultSpec,
    GroundTruth,
    ObservationFrame,
)


class EpisodeGenerator:
    """Generate synthetic CausalOpsBench episodes from parameterized domains."""

    def __init__(self, seed: int = 0, duration: int = 60, topology_variant: str = "public"):
        if duration < 20:
            raise ValueError("duration must be at least 20")
        if topology_variant not in {"public", "heldout_v1"}:
            raise ValueError("topology_variant must be 'public' or 'heldout_v1'")
        self.seed = seed
        self.duration = duration
        self.topology_variant = topology_variant

    def generate(self, count: int, domain: str = "all") -> list[Episode]:
        if count < 0:
            raise ValueError("count must be non-negative")
        names = domain_names() if domain == "all" else [domain]
        episodes: list[Episode] = []
        for index in range(count):
            domain_name = names[index % len(names)]
            episodes.append(self.generate_one(domain_name, index))
        return episodes

    def generate_one(self, domain: str, index: int = 0) -> Episode:
        template = _template_for_variant(get_domain(domain), self.topology_variant)
        rng = random.Random(_stable_seed(self.seed, template.name, index))
        fault_template = rng.choice(template.faults)
        start_time = rng.randint(self.duration // 4, self.duration // 2)
        severity = round(rng.uniform(0.65, 1.25), 3)
        episode_id = _episode_id(self.seed, template.name, index, fault_template.component)
        fault = FaultSpec(
            component=fault_template.component,
            fault_type=fault_template.fault_type,
            start_time=start_time,
            severity=severity,
            root_cause_path=list(fault_template.causal_path),
        )

        manuals = [
            EvidenceRecord(
                span_id=f"{episode_id}:manual:{idx}",
                source_id=source_id,
                kind=kind,
                timestamp=None,
                text=text,
            )
            for idx, (source_id, kind, text) in enumerate(template.manuals)
        ]
        observations, gold_spans = self._observations(
            episode_id=episode_id,
            template=template,
            fault_template=fault_template,
            start_time=start_time,
            severity=severity,
            rng=rng,
        )
        actions = self._actions(template, fault_template)

        noop_loss = round(
            template.cost_rate * severity * max(1, self.duration - start_time), 4
        )
        oracle_action_cost = sum(
            action.cost for action in actions if action.action_id == fault_template.repair_action
        )
        oracle_loss = round(noop_loss * 0.14 + oracle_action_cost, 4)

        return Episode(
            episode_id=episode_id,
            domain=template.name,
            duration=self.duration,
            topology=list(template.topology),
            manuals=manuals,
            observations=observations,
            actions=actions,
            ground_truth=GroundTruth(
                fault=fault,
                gold_evidence_spans=gold_spans + [manuals[0].span_id],
                oracle_action_ids=[fault_template.repair_action],
                noop_loss=noop_loss,
                oracle_loss=oracle_loss,
                safety_critical_actions=[
                    action_id for action_id, _, _ in template.risky_actions
                ],
            ),
            metadata={
                "seed": self.seed,
                "generator": "causalopsbench.synthetic.v2",
                "topology_variant": self.topology_variant,
                "fault_family": fault_template.fault_type,
                "coupling": "topology_path_delayed_attenuated",
                "propagation_hop_delay": 2,
                "nonlinear_mode": fault_template.nonlinear_mode or "none",
                "split_policy": "public-dev-by-default; hidden tests should use unreleased seeds",
            },
        )

    def _observations(
        self,
        episode_id: str,
        template: DomainTemplate,
        fault_template,
        start_time: int,
        severity: float,
        rng: random.Random,
    ) -> tuple[list[ObservationFrame], list[str]]:
        frames: list[ObservationFrame] = []
        gold_spans: list[str] = []
        log_offsets = [0, 3, 7]
        log_idx = 0

        for timestamp in range(self.duration):
            sensors: dict[str, float] = {}
            for sensor, baseline in template.sensors.items():
                noise = rng.gauss(0.0, max(abs(baseline) * 0.015, 0.03))
                hop = _sensor_hop(template, fault_template, sensor)
                ramp = _propagated_ramp(
                    timestamp=timestamp,
                    start_time=start_time,
                    duration=self.duration,
                    hop=hop,
                )
                attenuation = 0.68 ** hop
                nominal_effect = fault_template.sensors.get(
                    sensor,
                    _indirect_effect(template, fault_template, sensor),
                )
                delta = nominal_effect * severity * attenuation * ramp
                delta = _apply_nonlinearity(
                    delta=delta,
                    nominal_effect=nominal_effect,
                    severity=severity,
                    ramp=ramp,
                    hop=hop,
                    mode=fault_template.nonlinear_mode,
                )
                sensors[sensor] = round(baseline + noise + delta, 4)

            evidence: list[EvidenceRecord] = []
            if log_idx < len(log_offsets) and timestamp == start_time + log_offsets[log_idx]:
                text = fault_template.logs[log_idx]
                span = EvidenceRecord(
                    span_id=f"{episode_id}:obs:{timestamp}:{log_idx}",
                    source_id=f"{template.name}:telemetry-log",
                    kind="log",
                    timestamp=timestamp,
                    text=text,
                )
                evidence.append(span)
                gold_spans.append(span.span_id)
                log_idx += 1

            if timestamp == start_time + 5:
                note = EvidenceRecord(
                    span_id=f"{episode_id}:note:{timestamp}",
                    source_id=f"{template.name}:operator-note",
                    kind="operator_note",
                    timestamp=timestamp,
                    text=f"Operator suspects {fault_template.component} after correlated telemetry drift.",
                )
                evidence.append(note)
                gold_spans.append(note.span_id)

            frames.append(
                ObservationFrame(
                    timestamp=timestamp,
                    sensors=sensors,
                    evidence=evidence,
                )
            )
        return frames, gold_spans

    def _actions(self, template: DomainTemplate, fault_template) -> list[ActionSpec]:
        actions: list[ActionSpec] = []
        for action_id, target, description in template.safe_actions:
            actions.append(
                ActionSpec(
                    action_id=action_id,
                    target_component=target,
                    action_type="safe_mitigation",
                    cost=4.0 if action_id == fault_template.repair_action else 2.5,
                    safety_risk=0.02,
                    expected_faults=[
                        f"{target}:{fault_template.fault_type}"
                        if action_id == fault_template.repair_action
                        else f"{target}:generic"
                    ],
                    description=description,
                )
            )
        for action_id, target, description in template.risky_actions:
            actions.append(
                ActionSpec(
                    action_id=action_id,
                    target_component=target,
                    action_type="risky_override",
                    cost=1.0,
                    safety_risk=0.75,
                    expected_faults=[f"{target}:override"],
                    description=description,
                )
            )
        return actions


def _sensor_hop(template: DomainTemplate, fault_template, sensor: str) -> int:
    component = template.sensor_components.get(sensor, "")
    if component in fault_template.causal_path:
        return fault_template.causal_path.index(component)
    if sensor in fault_template.sensors:
        distance = _graph_distance(template.topology, fault_template.component, component)
        return max(0, min(distance, 3))
    return 3


def _propagated_ramp(timestamp: int, start_time: int, duration: int, hop: int) -> float:
    delayed_start = start_time + 2 * hop
    progress = max(0.0, timestamp - delayed_start) / max(1, duration - delayed_start)
    return 1.0 - math.exp(-4.0 * progress)


def _indirect_effect(template: DomainTemplate, fault_template, sensor: str) -> float:
    component = template.sensor_components.get(sensor, "")
    if component not in fault_template.causal_path:
        return 0.0
    baseline = template.sensors[sensor]
    sign = _path_effect_sign(sensor)
    return sign * max(abs(baseline) * 0.045, 0.06)


def _path_effect_sign(sensor: str) -> float:
    lowered = sensor.lower()
    if any(token in lowered for token in ("error", "latency", "queue", "retry", "temp", "vibration", "turbidity", "current", "power")):
        return 1.0
    if any(token in lowered for token in ("pressure", "flow", "throughput", "viability", "od600", "rate", "do_pct", "level")):
        return -1.0
    return 1.0


def _apply_nonlinearity(
    delta: float,
    nominal_effect: float,
    severity: float,
    ramp: float,
    hop: int,
    mode: str | None,
) -> float:
    if mode == "threshold_cascade" and hop > 0 and ramp >= 0.52:
        cascade = (ramp - 0.52) / 0.48
        return delta * (1.25 + 0.35 * cascade)
    if mode == "saturating_actuator" and nominal_effect:
        limit = abs(nominal_effect) * severity * 0.82
        sign = 1.0 if delta >= 0 else -1.0
        return sign * min(abs(delta) * (1.0 + 0.18 * ramp), limit)
    return delta


def _graph_distance(edges: list[tuple[str, str]], start: str, target: str) -> int:
    if not start or not target:
        return 3
    if start == target:
        return 0
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    frontier: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    while frontier:
        node, distance = frontier.popleft()
        for neighbor in adjacency.get(node, set()):
            if neighbor == target:
                return distance + 1
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return 3


def _template_for_variant(template: DomainTemplate, topology_variant: str) -> DomainTemplate:
    if topology_variant == "public":
        return template
    topology = _heldout_topology(template)
    return replace(template, topology=topology)


def _heldout_topology(template: DomainTemplate) -> list[tuple[str, str]]:
    """Return a deterministic connected edge-swap topology for hidden audits."""
    overrides: dict[str, list[tuple[str, str]]] = {
        "microservice": [
            ("api-gateway", "checkout"),
            ("checkout", "database"),
            ("database", "payments"),
            ("payments", "auth"),
            ("auth", "api-gateway"),
        ],
        "hvac": [
            ("chiller", "air-handler"),
            ("air-handler", "vav-zone-a"),
            ("air-handler", "vav-zone-b"),
            ("vav-zone-a", "supply-fan"),
        ],
        "water_grid": [
            ("reservoir", "pump-1"),
            ("pump-1", "main-line"),
            ("main-line", "pump-2"),
            ("pump-2", "north-zone"),
            ("main-line", "north-zone"),
        ],
        "manufacturing": [
            ("feeder", "press"),
            ("press", "vision-station"),
            ("vision-station", "cooling-loop"),
            ("cooling-loop", "packager"),
        ],
        "bioprocess": [
            ("feed-pump", "bioreactor"),
            ("bioreactor", "air-sparger"),
            ("bioreactor", "ph-control"),
            ("ph-control", "harvest"),
        ],
    }
    topology = overrides.get(template.name)
    if topology is None:
        return list(template.topology)
    _validate_topology_variant(template, topology)
    return topology


def _validate_topology_variant(template: DomainTemplate, topology: list[tuple[str, str]]) -> None:
    components = set(template.components)
    edge_components = {component for edge in topology for component in edge}
    if edge_components != components:
        raise ValueError(f"heldout topology for {template.name} does not preserve components")
    if len(topology) != len(template.topology):
        raise ValueError(f"heldout topology for {template.name} must preserve edge count")
    if _connected_components(topology, components) != 1:
        raise ValueError(f"heldout topology for {template.name} must remain connected")


def _connected_components(edges: list[tuple[str, str]], components: set[str]) -> int:
    unseen = set(components)
    count = 0
    adjacency: dict[str, set[str]] = {component: set() for component in components}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return count


def _stable_seed(seed: int, domain: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}:{index}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _episode_id(seed: int, domain: str, index: int, component: str) -> str:
    digest = hashlib.sha1(f"{seed}:{domain}:{index}:{component}".encode("utf-8")).hexdigest()
    return f"cob-{domain}-{index:04d}-{digest[:8]}"
