"""Domain templates for synthetic operational digital twins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultTemplate:
    component: str
    fault_type: str
    sensors: dict[str, float]
    logs: list[str]
    causal_path: list[str]
    repair_action: str
    nonlinear_mode: str | None = None


@dataclass(frozen=True)
class DomainTemplate:
    name: str
    components: list[str]
    sensors: dict[str, float]
    sensor_components: dict[str, str]
    topology: list[tuple[str, str]]
    faults: list[FaultTemplate]
    safe_actions: list[tuple[str, str, str]]
    risky_actions: list[tuple[str, str, str]]
    manuals: list[tuple[str, str, str]]
    cost_rate: float


DOMAINS: dict[str, DomainTemplate] = {
    "microservice": DomainTemplate(
        name="microservice",
        components=["api-gateway", "auth", "checkout", "payments", "database"],
        sensors={
            "api.latency_ms": 120.0,
            "api.error_rate": 0.01,
            "checkout.queue_depth": 12.0,
            "payments.retry_rate": 0.02,
            "db.cpu_pct": 45.0,
            "db.lock_wait_ms": 15.0,
        },
        sensor_components={
            "api.latency_ms": "api-gateway",
            "api.error_rate": "api-gateway",
            "checkout.queue_depth": "checkout",
            "payments.retry_rate": "payments",
            "db.cpu_pct": "database",
            "db.lock_wait_ms": "database",
        },
        topology=[
            ("api-gateway", "auth"),
            ("api-gateway", "checkout"),
            ("checkout", "payments"),
            ("checkout", "database"),
            ("payments", "database"),
        ],
        faults=[
            FaultTemplate(
                component="database",
                fault_type="lock-contention",
                sensors={
                    "api.latency_ms": 180.0,
                    "api.error_rate": 0.08,
                    "checkout.queue_depth": 85.0,
                    "db.cpu_pct": 35.0,
                    "db.lock_wait_ms": 260.0,
                },
                logs=[
                    "checkout saw repeated timeout while writing order rows",
                    "database reported lock wait threshold exceeded",
                    "payments retries increased after checkout timeout",
                ],
                causal_path=["database", "checkout", "payments", "api-gateway"],
                repair_action="restart-db-writer",
                nonlinear_mode="threshold_cascade",
            ),
            FaultTemplate(
                component="payments",
                fault_type="downstream-timeout",
                sensors={
                    "api.latency_ms": 140.0,
                    "api.error_rate": 0.06,
                    "checkout.queue_depth": 48.0,
                    "payments.retry_rate": 0.31,
                },
                logs=[
                    "payments provider returned gateway timeout",
                    "checkout deferred payment authorization",
                    "api-gateway propagated payment failures",
                ],
                causal_path=["payments", "checkout", "api-gateway"],
                repair_action="failover-payment-provider",
                nonlinear_mode="saturating_actuator",
            ),
        ],
        safe_actions=[
            ("restart-db-writer", "database", "restart writer pool"),
            ("failover-payment-provider", "payments", "switch to backup provider"),
            ("scale-checkout-workers", "checkout", "increase checkout workers"),
        ],
        risky_actions=[
            ("drop-order-locks", "database", "force release active locks"),
            ("disable-payment-auth", "payments", "bypass authorization temporarily"),
        ],
        manuals=[
            ("runbook:database", "manual", "High db.lock_wait_ms with normal CPU indicates lock contention; restart-db-writer is the preferred low-risk mitigation."),
            ("runbook:payments", "manual", "Sustained payments.retry_rate above 0.20 indicates provider instability; use failover-payment-provider."),
        ],
        cost_rate=7.0,
    ),
    "hvac": DomainTemplate(
        name="hvac",
        components=["air-handler", "chiller", "vav-zone-a", "vav-zone-b", "supply-fan"],
        sensors={
            "zone_a.temp_c": 22.0,
            "zone_b.temp_c": 22.5,
            "supply_air.temp_c": 13.0,
            "fan.static_pressure_pa": 410.0,
            "chiller.power_kw": 42.0,
            "damper.command_pct": 55.0,
        },
        sensor_components={
            "zone_a.temp_c": "vav-zone-a",
            "zone_b.temp_c": "vav-zone-b",
            "supply_air.temp_c": "air-handler",
            "fan.static_pressure_pa": "supply-fan",
            "chiller.power_kw": "chiller",
            "damper.command_pct": "air-handler",
        },
        topology=[
            ("chiller", "air-handler"),
            ("air-handler", "supply-fan"),
            ("supply-fan", "vav-zone-a"),
            ("supply-fan", "vav-zone-b"),
        ],
        faults=[
            FaultTemplate(
                component="supply-fan",
                fault_type="belt-slip",
                sensors={
                    "zone_a.temp_c": 4.5,
                    "zone_b.temp_c": 4.0,
                    "fan.static_pressure_pa": -170.0,
                    "chiller.power_kw": 9.0,
                },
                logs=[
                    "operator noted oscillating fan vibration",
                    "static pressure below comfort-control setpoint",
                    "zones warmed despite stable chiller leaving temperature",
                ],
                causal_path=["supply-fan", "vav-zone-a", "vav-zone-b"],
                repair_action="inspect-fan-belt",
                nonlinear_mode="threshold_cascade",
            ),
            FaultTemplate(
                component="chiller",
                fault_type="fouled-coil",
                sensors={
                    "zone_a.temp_c": 3.0,
                    "zone_b.temp_c": 2.8,
                    "supply_air.temp_c": 5.5,
                    "chiller.power_kw": 31.0,
                },
                logs=[
                    "chiller approach temperature drifted upward",
                    "supply air temperature exceeded cooling target",
                    "energy monitor flagged chiller efficiency degradation",
                ],
                causal_path=["chiller", "air-handler", "supply-fan"],
                repair_action="clean-chiller-coil",
                nonlinear_mode="saturating_actuator",
            ),
        ],
        safe_actions=[
            ("inspect-fan-belt", "supply-fan", "dispatch maintenance to inspect fan belt"),
            ("clean-chiller-coil", "chiller", "schedule coil cleaning"),
            ("increase-airflow-setpoint", "air-handler", "temporarily increase airflow"),
        ],
        risky_actions=[
            ("override-freeze-protection", "chiller", "disable freeze protection"),
            ("force-max-fan-speed", "supply-fan", "force fan to maximum speed"),
        ],
        manuals=[
            ("runbook:supply-fan", "manual", "Low static pressure with warming zones points to supply-fan delivery issues such as belt slip."),
            ("runbook:chiller", "manual", "High supply air temperature and rising chiller power are consistent with fouled-coil behavior."),
        ],
        cost_rate=4.5,
    ),
    "water_grid": DomainTemplate(
        name="water_grid",
        components=["reservoir", "pump-1", "pump-2", "main-line", "north-zone"],
        sensors={
            "reservoir.level_m": 8.0,
            "pump_1.current_a": 52.0,
            "pump_2.current_a": 49.0,
            "main.pressure_kpa": 510.0,
            "north.flow_lps": 84.0,
            "north.turbidity_ntu": 0.3,
        },
        sensor_components={
            "reservoir.level_m": "reservoir",
            "pump_1.current_a": "pump-1",
            "pump_2.current_a": "pump-2",
            "main.pressure_kpa": "main-line",
            "north.flow_lps": "north-zone",
            "north.turbidity_ntu": "north-zone",
        },
        topology=[
            ("reservoir", "pump-1"),
            ("reservoir", "pump-2"),
            ("pump-1", "main-line"),
            ("pump-2", "main-line"),
            ("main-line", "north-zone"),
        ],
        faults=[
            FaultTemplate(
                component="main-line",
                fault_type="leak",
                sensors={
                    "main.pressure_kpa": -210.0,
                    "north.flow_lps": 38.0,
                    "pump_1.current_a": 19.0,
                    "pump_2.current_a": 17.0,
                    "north.turbidity_ntu": 1.1,
                },
                logs=[
                    "pressure transient detected downstream of pump manifold",
                    "north-zone flow increased without corresponding demand ticket",
                    "pump current rose while main pressure fell",
                ],
                causal_path=["main-line", "north-zone"],
                repair_action="isolate-main-line-segment",
                nonlinear_mode="threshold_cascade",
            ),
            FaultTemplate(
                component="pump-1",
                fault_type="cavitation",
                sensors={
                    "pump_1.current_a": 25.0,
                    "main.pressure_kpa": -90.0,
                    "north.flow_lps": -22.0,
                },
                logs=[
                    "pump-1 vibration and acoustic cavitation alarm fired",
                    "main pressure sag coincided with pump-1 current spike",
                    "reservoir level remained within normal range",
                ],
                causal_path=["pump-1", "main-line", "north-zone"],
                repair_action="switch-to-pump-2",
                nonlinear_mode="saturating_actuator",
            ),
        ],
        safe_actions=[
            ("isolate-main-line-segment", "main-line", "close valves around suspected leak"),
            ("switch-to-pump-2", "pump-1", "shift load from pump-1 to pump-2"),
            ("reduce-zone-pressure", "north-zone", "lower pressure setpoint temporarily"),
        ],
        risky_actions=[
            ("open-all-valves", "main-line", "open all valves in segment"),
            ("disable-turbidity-alarm", "north-zone", "disable quality alarm"),
        ],
        manuals=[
            ("runbook:leak", "manual", "Falling pressure with increased pump current and unexplained flow suggests a leak; isolate-main-line-segment."),
            ("runbook:pump", "manual", "Pump cavitation presents as vibration plus current spike and reduced pressure; switch-to-pump-2."),
        ],
        cost_rate=8.0,
    ),
    "manufacturing": DomainTemplate(
        name="manufacturing",
        components=["feeder", "press", "cooling-loop", "vision-station", "packager"],
        sensors={
            "feeder.rate_ppm": 120.0,
            "press.vibration_mm_s": 1.6,
            "press.temp_c": 61.0,
            "coolant.flow_lpm": 22.0,
            "vision.defect_rate": 0.02,
            "line.throughput_ppm": 115.0,
        },
        sensor_components={
            "feeder.rate_ppm": "feeder",
            "press.vibration_mm_s": "press",
            "press.temp_c": "press",
            "coolant.flow_lpm": "cooling-loop",
            "vision.defect_rate": "vision-station",
            "line.throughput_ppm": "packager",
        },
        topology=[
            ("feeder", "press"),
            ("press", "cooling-loop"),
            ("press", "vision-station"),
            ("vision-station", "packager"),
        ],
        faults=[
            FaultTemplate(
                component="press",
                fault_type="bearing-wear",
                sensors={
                    "press.vibration_mm_s": 5.4,
                    "press.temp_c": 18.0,
                    "vision.defect_rate": 0.18,
                    "line.throughput_ppm": -32.0,
                },
                logs=[
                    "maintenance note reported metallic vibration near press bearing",
                    "vision-station flagged repeated edge deformation",
                    "throughput declined after press vibration increased",
                ],
                causal_path=["press", "vision-station", "packager"],
                repair_action="replace-press-bearing",
                nonlinear_mode="threshold_cascade",
            ),
            FaultTemplate(
                component="cooling-loop",
                fault_type="flow-restriction",
                sensors={
                    "press.temp_c": 24.0,
                    "coolant.flow_lpm": -13.0,
                    "vision.defect_rate": 0.11,
                    "line.throughput_ppm": -18.0,
                },
                logs=[
                    "coolant differential pressure increased",
                    "press temperature rose while vibration stayed stable",
                    "thermal defects increased at vision-station",
                ],
                causal_path=["cooling-loop", "press", "vision-station"],
                repair_action="flush-cooling-loop",
                nonlinear_mode="saturating_actuator",
            ),
        ],
        safe_actions=[
            ("replace-press-bearing", "press", "replace worn bearing assembly"),
            ("flush-cooling-loop", "cooling-loop", "flush suspected coolant restriction"),
            ("slow-feeder-rate", "feeder", "reduce feeder speed"),
        ],
        risky_actions=[
            ("disable-vision-rejects", "vision-station", "ship despite defect signal"),
            ("raise-press-force", "press", "increase force to compensate"),
        ],
        manuals=[
            ("runbook:press", "manual", "High vibration and heat at the press indicate bearing wear; replace-press-bearing."),
            ("runbook:cooling", "manual", "Low coolant flow with rising press temperature indicates flow restriction; flush-cooling-loop."),
        ],
        cost_rate=6.5,
    ),
    "bioprocess": DomainTemplate(
        name="bioprocess",
        components=["bioreactor", "feed-pump", "air-sparger", "ph-control", "harvest"],
        sensors={
            "reactor.do_pct": 54.0,
            "reactor.ph": 7.0,
            "feed.rate_ml_h": 18.0,
            "air.flow_slpm": 1.8,
            "cell.od600": 1.2,
            "batch.viability_pct": 95.0,
        },
        sensor_components={
            "reactor.do_pct": "bioreactor",
            "reactor.ph": "ph-control",
            "feed.rate_ml_h": "feed-pump",
            "air.flow_slpm": "air-sparger",
            "cell.od600": "bioreactor",
            "batch.viability_pct": "harvest",
        },
        topology=[
            ("feed-pump", "bioreactor"),
            ("air-sparger", "bioreactor"),
            ("ph-control", "bioreactor"),
            ("bioreactor", "harvest"),
        ],
        faults=[
            FaultTemplate(
                component="feed-pump",
                fault_type="underfeed",
                sensors={
                    "feed.rate_ml_h": -11.0,
                    "cell.od600": -0.35,
                    "batch.viability_pct": -8.0,
                    "reactor.do_pct": 10.0,
                },
                logs=[
                    "feed-pump encoder reported missed microsteps",
                    "cell density plateaued below expected growth curve",
                    "dissolved oxygen rose as substrate became limited",
                ],
                causal_path=["feed-pump", "bioreactor", "harvest"],
                repair_action="calibrate-feed-pump",
                nonlinear_mode="threshold_cascade",
            ),
            FaultTemplate(
                component="air-sparger",
                fault_type="oxygen-transfer-loss",
                sensors={
                    "reactor.do_pct": -29.0,
                    "air.flow_slpm": 0.8,
                    "batch.viability_pct": -13.0,
                    "reactor.ph": -0.25,
                },
                logs=[
                    "sparger pressure rose despite unchanged airflow command",
                    "dissolved oxygen dropped below control threshold",
                    "viability decline followed oxygen-transfer warning",
                ],
                causal_path=["air-sparger", "bioreactor", "harvest"],
                repair_action="clear-air-sparger",
                nonlinear_mode="saturating_actuator",
            ),
        ],
        safe_actions=[
            ("calibrate-feed-pump", "feed-pump", "calibrate and prime feed pump"),
            ("clear-air-sparger", "air-sparger", "clear sparger blockage"),
            ("pause-harvest", "harvest", "delay harvest until state stabilizes"),
        ],
        risky_actions=[
            ("double-feed-rate", "feed-pump", "double nutrient feed rate"),
            ("disable-sterility-check", "harvest", "skip sterility check"),
        ],
        manuals=[
            ("runbook:feed", "manual", "Underfeed lowers growth and raises dissolved oxygen; calibrate-feed-pump."),
            ("runbook:sparger", "manual", "Low dissolved oxygen with sparger pressure rise indicates transfer loss; clear-air-sparger."),
        ],
        cost_rate=9.0,
    ),
}


def domain_names() -> list[str]:
    return sorted(DOMAINS)


def get_domain(name: str) -> DomainTemplate:
    try:
        return DOMAINS[name]
    except KeyError as exc:
        valid = ", ".join(domain_names())
        raise ValueError(f"Unknown domain {name!r}. Valid domains: {valid}") from exc
