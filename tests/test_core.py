import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from causalopsbench.baselines import (
    NoOpBaseline,
    OracleBaseline,
    RandomBaseline,
    ThresholdBaseline,
    TopologyRCABaseline,
)
from causalopsbench.cli import main as cli_main
from causalopsbench.evaluate import evaluate_baseline, evaluate_predictions
from causalopsbench.generator import EpisodeGenerator, _apply_nonlinearity, _graph_distance, _propagated_ramp
from causalopsbench.metrics import (
    evidence_f1,
    graph_distance,
    intervention_score,
    is_degenerate_intervention,
    score_episode,
)
from causalopsbench.schemas import dumps_json, episode_from_dict, write_jsonl


class CausalOpsBenchTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        left = EpisodeGenerator(seed=13, duration=40).generate(5)
        right = EpisodeGenerator(seed=13, duration=40).generate(5)
        self.assertEqual([ep.episode_id for ep in left], [ep.episode_id for ep in right])
        self.assertEqual(dumps_json(left[0]), dumps_json(right[0]))

    def test_json_roundtrip(self):
        episode = EpisodeGenerator(seed=2, duration=35).generate_one("microservice")
        restored = episode_from_dict(json.loads(dumps_json(episode)))
        self.assertEqual(episode.episode_id, restored.episode_id)
        self.assertEqual(episode.ground_truth.fault.label, restored.ground_truth.fault.label)

    def test_oracle_beats_noop(self):
        episode = EpisodeGenerator(seed=5, duration=45).generate_one("hvac")
        oracle_score = score_episode(episode, OracleBaseline().predict(episode))
        noop_score = score_episode(episode, NoOpBaseline().predict(episode))
        self.assertGreater(oracle_score.composite, noop_score.composite)
        self.assertGreater(oracle_score.intervention, noop_score.intervention)

    def test_threshold_runs_on_all_domains(self):
        episodes = EpisodeGenerator(seed=9, duration=50).generate(10, domain="all")
        scores, summary = evaluate_baseline(episodes, ThresholdBaseline())
        self.assertEqual(len(scores), 10)
        self.assertEqual(summary["count"], 10.0)
        self.assertGreaterEqual(summary["composite"], 0.0)
        self.assertLessEqual(summary["composite"], 1.0)

    def test_evaluate_baseline_accepts_one_shot_iterables(self):
        episodes = EpisodeGenerator(seed=9, duration=50).generate(4, domain="all")
        list_scores, list_summary = evaluate_baseline(episodes, ThresholdBaseline())
        iterable_scores, iterable_summary = evaluate_baseline(
            (episode for episode in episodes),
            ThresholdBaseline(),
        )

        self.assertEqual([score.episode_id for score in list_scores], [score.episode_id for score in iterable_scores])
        self.assertEqual(list_summary, iterable_summary)

    def test_oracle_worse_episodes_are_degenerate_and_excluded(self):
        episode = EpisodeGenerator(seed=5, duration=45).generate_one("hvac")
        modified_truth = replace(
            episode.ground_truth,
            oracle_loss=episode.ground_truth.noop_loss + 1.0,
        )
        modified = replace(episode, ground_truth=modified_truth)
        prediction = OracleBaseline().predict(modified)

        self.assertTrue(is_degenerate_intervention(modified))
        scores, summary = evaluate_predictions([modified], [prediction])

        self.assertEqual(scores, [])
        self.assertEqual(summary["n_total"], 1.0)
        self.assertEqual(summary["n_scored"], 0.0)
        self.assertEqual(summary["n_deg"], 1.0)

    def test_intervention_score_handles_non_improving_oracle_contrast(self):
        episode = EpisodeGenerator(seed=5, duration=45).generate_one("hvac")
        modified_truth = replace(
            episode.ground_truth,
            oracle_loss=episode.ground_truth.noop_loss + 1.0,
        )
        modified = replace(episode, ground_truth=modified_truth)

        score = intervention_score(
            modified,
            policy_loss=modified.ground_truth.noop_loss,
            safety_violations=0,
        )
        penalized = intervention_score(
            modified,
            policy_loss=modified.ground_truth.noop_loss,
            safety_violations=1,
        )

        self.assertEqual(score, 0.0)
        self.assertEqual(penalized, 0.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_causal_path_is_audit_only_for_scoring(self):
        episode = EpisodeGenerator(seed=5, duration=45).generate_one("hvac")
        prediction = OracleBaseline().predict(episode)
        changed_path = replace(prediction, causal_path=["unrelated", "audit", "path"])

        self.assertEqual(
            score_episode(episode, prediction),
            score_episode(episode, changed_path),
        )

    def test_topology_rca_runs_on_all_domains(self):
        episodes = EpisodeGenerator(seed=9, duration=50).generate(10, domain="all")
        scores, summary = evaluate_baseline(episodes, TopologyRCABaseline())
        self.assertEqual(len(scores), 10)
        self.assertEqual(summary["count"], 10.0)
        self.assertGreaterEqual(summary["composite"], 0.0)
        self.assertLessEqual(summary["composite"], 1.0)

    def test_generation_records_v2_coupling(self):
        episode = EpisodeGenerator(seed=3, duration=40).generate_one("water_grid")
        self.assertEqual(episode.metadata["generator"], "causalopsbench.synthetic.v2")
        self.assertEqual(episode.metadata["topology_variant"], "public")
        self.assertEqual(episode.metadata["coupling"], "topology_path_delayed_attenuated")
        self.assertIn(episode.metadata["nonlinear_mode"], {"threshold_cascade", "saturating_actuator"})

    def test_heldout_topology_variant_is_deterministic_and_connected(self):
        public = EpisodeGenerator(seed=3, duration=40).generate_one("microservice")
        left = EpisodeGenerator(
            seed=3,
            duration=40,
            topology_variant="heldout_v1",
        ).generate_one("microservice")
        right = EpisodeGenerator(
            seed=3,
            duration=40,
            topology_variant="heldout_v1",
        ).generate_one("microservice")
        self.assertEqual(left.topology, right.topology)
        self.assertNotEqual(public.topology, left.topology)
        self.assertEqual(left.metadata["topology_variant"], "heldout_v1")
        self.assertEqual(len(public.topology), len(left.topology))
        self.assertEqual(
            {component for edge in public.topology for component in edge},
            {component for edge in left.topology for component in edge},
        )
        components = {component for edge in left.topology for component in edge}
        for source in components:
            for target in components:
                self.assertLess(_graph_distance(left.topology, source, target), 3)
        self.assertIn(left.ground_truth.fault.component, components)
        self.assertTrue(set(left.ground_truth.oracle_action_ids))

    def test_generator_graph_distance_matches_scoring_helper_on_connected_topology(self):
        episode = EpisodeGenerator(seed=3, duration=40).generate_one("water_grid")
        components = sorted({component for edge in episode.topology for component in edge})
        for source in components:
            for target in components:
                self.assertEqual(
                    _graph_distance(episode.topology, source, target),
                    graph_distance(episode.topology, source, target),
                )

    def test_propagation_delay_and_nonlinearity_helpers(self):
        self.assertGreater(_propagated_ramp(timestamp=12, start_time=10, duration=60, hop=0), 0.0)
        self.assertEqual(_propagated_ramp(timestamp=12, start_time=10, duration=60, hop=2), 0.0)
        linear = 10.0
        cascaded = _apply_nonlinearity(
            delta=linear,
            nominal_effect=10.0,
            severity=1.0,
            ramp=0.8,
            hop=1,
            mode="threshold_cascade",
        )
        self.assertGreater(cascaded, linear)
        saturated = _apply_nonlinearity(
            delta=20.0,
            nominal_effect=10.0,
            severity=1.0,
            ramp=1.0,
            hop=0,
            mode="saturating_actuator",
        )
        self.assertLessEqual(saturated, 8.2)

    def test_non_oracle_baselines_do_not_read_hidden_fault_type(self):
        episode = EpisodeGenerator(seed=5, duration=45).generate_one("hvac")
        hidden_fault = replace(episode.ground_truth.fault, fault_type="secret-hidden-label")
        hidden_truth = replace(episode.ground_truth, fault=hidden_fault)
        modified = replace(episode, ground_truth=hidden_truth)
        for baseline in [RandomBaseline(seed=0), ThresholdBaseline(), TopologyRCABaseline()]:
            prediction = baseline.predict(modified)
            self.assertFalse(
                any("secret-hidden-label" in label for label in prediction.root_cause_topk),
                baseline.name,
            )

    def test_cli_accepts_topology_rca_baseline(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episodes = EpisodeGenerator(seed=1, duration=30).generate(3)
            episode_path = tmp_path / "episodes.jsonl"
            output_path = tmp_path / "scores.json"
            write_jsonl(episode_path, episodes)
            result = cli_main(
                [
                    "evaluate",
                    "--episodes",
                    str(episode_path),
                    "--baseline",
                    "topology_rca",
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output_path.exists())

    def test_cli_generate_accepts_heldout_topology_variant(self):
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "episodes.jsonl"
            result = cli_main(
                [
                    "generate",
                    "--count",
                    "2",
                    "--seed",
                    "1",
                    "--duration",
                    "30",
                    "--topology-variant",
                    "heldout_v1",
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(result, 0)
            restored = [
                episode_from_dict(json.loads(line))
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(restored)
            self.assertTrue(
                all(episode.metadata["topology_variant"] == "heldout_v1" for episode in restored)
            )

    def test_jsonl_writer_preserves_empty_and_trailing_newline_behavior(self):
        with TemporaryDirectory() as tmp:
            empty_path = Path(tmp) / "empty.jsonl"
            episode_path = Path(tmp) / "episodes.jsonl"
            write_jsonl(empty_path, [])
            self.assertEqual(empty_path.read_text(encoding="utf-8"), "")

            episode = EpisodeGenerator(seed=2, duration=35).generate_one("microservice")
            write_jsonl(episode_path, [episode])
            text = episode_path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(len(text.splitlines()), 1)

    def test_evidence_f1(self):
        self.assertEqual(evidence_f1(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(evidence_f1(["a"], ["a", "b"]), 2 / 3)
        self.assertEqual(evidence_f1([], ["a"]), 0.0)


if __name__ == "__main__":
    unittest.main()
