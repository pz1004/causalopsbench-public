import inspect
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
from unittest.mock import patch

from cob_ext.adapters import lbnl_fdd_adapter, rcaeval_re2_adapter
from cob_ext import model_client
from cob_ext.runners import run_external_agents
from cob_ext.runners import run_external_baselines
from cob_ext.reporting import make_external_tables
from cob_ext.scoring.bootstrap_ci import bootstrap_intervals
from cob_ext.scoring.score_external import evaluate_predictions, score_episode
from cob_ext.schemas import (
    ExternalEpisode,
    ExternalGroundTruth,
    ExternalPrediction,
    dumps_json,
    episode_from_dict,
    load_episodes_jsonl,
)


class ExternalValidationTests(unittest.TestCase):
    def test_external_episode_roundtrip_rejects_replay_metrics(self):
        episode = _simple_external_episode()
        restored = episode_from_dict(json.loads(dumps_json(episode)))
        self.assertEqual(restored.episode_id, episode.episode_id)
        invalid = replace(episode, supported_metrics=episode.supported_metrics + ["intervention"])
        with self.assertRaises(ValueError):
            invalid.validate()

    def test_score_external_refuses_native_replay_metrics(self):
        episode = replace(_simple_external_episode(), supported_metrics=["detection", "composite"])
        prediction = ExternalPrediction(
            episode_id=episode.episode_id,
            alarm_time=1,
            alarm_confidence=0.9,
            root_cause_topk=["svc-a:latency"],
            evidence_spans=["ev-1"],
        )
        with self.assertRaises(ValueError):
            score_episode(episode, prediction)

    def test_rcaeval_adapter_is_deterministic_and_emits_evidence(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            _write_rcaeval_fixture(raw)
            left, _ = rcaeval_re2_adapter.build_episodes(raw, ["SS"], max_evidence_spans=6, seed=7)
            right, _ = rcaeval_re2_adapter.build_episodes(raw, ["SS"], max_evidence_spans=6, seed=7)

        self.assertEqual([dumps_json(item) for item in left], [dumps_json(item) for item in right])
        self.assertEqual(len(left), 1)
        episode = left[0]
        self.assertEqual(episode.domain, "microservice")
        self.assertTrue(episode.ground_truth.gold_evidence_spans)
        self.assertTrue(all(span.startswith("re2-") for span in episode.ground_truth.gold_evidence_spans))

    def test_lbnl_adapter_marks_proxy_evidence(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            _write_lbnl_fixture(raw)
            episodes, _ = lbnl_fdd_adapter.build_episodes(
                raw,
                ["RTU"],
                window_minutes=2,
                stride_minutes=2,
                max_windows_per_csv=2,
                seed=7,
            )

        self.assertGreaterEqual(len(episodes), 1)
        faulted = [episode for episode in episodes if episode.ground_truth.is_faulted]
        self.assertTrue(faulted)
        evidence = faulted[0].observations["evidence"]
        self.assertTrue(evidence)
        self.assertTrue(all(item["proxy"] for item in evidence))
        self.assertEqual(faulted[0].metadata["evidence_label_type"], "proxy")

    def test_external_scoring_and_bootstrap(self):
        episode = _simple_external_episode()
        prediction = ExternalPrediction(
            episode_id=episode.episode_id,
            alarm_time=1,
            alarm_confidence=0.9,
            root_cause_topk=["svc-a:latency"],
            evidence_spans=["ev-1"],
            token_count=100,
            tool_calls=2,
            wall_time_s=0.1,
        )
        scores, summary = evaluate_predictions([episode], [prediction])
        self.assertEqual(len(scores), 1)
        self.assertGreater(summary["external_portability_score"], 0.9)
        intervals = bootstrap_intervals(scores, clusters=["fixture"], samples=20, seed=1)
        self.assertIn("external_portability_score", intervals)
        self.assertIn("ci_low", intervals["external_portability_score"])

    def test_external_rca_metrics_are_strict(self):
        episode = _simple_external_episode()
        wrong_fault = ExternalPrediction(
            episode_id=episode.episode_id,
            alarm_time=1,
            alarm_confidence=0.9,
            root_cause_topk=["svc-a:wrong_fault", "svc-b:latency"],
            evidence_spans=["ev-1"],
        )
        wrong_score = score_episode(episode, wrong_fault)
        self.assertEqual(wrong_score.root_cause_top1, 0.0)
        self.assertEqual(wrong_score.root_cause_topk, 0.0)

        exact_second = replace(wrong_fault, root_cause_topk=["svc-b:latency", "svc-a:latency"])
        exact_second_score = score_episode(episode, exact_second)
        self.assertEqual(exact_second_score.root_cause_top1, 0.0)
        self.assertEqual(exact_second_score.root_cause_topk, 1.0)

    def test_fault_free_episode_allows_empty_or_none_normal_rca(self):
        episode = replace(
            _simple_external_episode(),
            ground_truth=ExternalGroundTruth(
                fault_component="none",
                fault_type="normal",
                root_cause_label="none:normal",
                is_faulted=False,
                fault_start_time=None,
                gold_evidence_spans=[],
            ),
        )
        empty = ExternalPrediction(
            episode_id=episode.episode_id,
            alarm_time=None,
            alarm_confidence=0.1,
            root_cause_topk=[],
            evidence_spans=[],
        )
        none_normal = replace(empty, root_cause_topk=["none:normal"])
        self.assertEqual(score_episode(episode, empty).root_cause_top1, 1.0)
        self.assertEqual(score_episode(episode, none_normal).root_cause_topk, 1.0)

    def test_external_agent_uses_package_local_model_helpers_and_prompt_file(self):
        source = inspect.getsource(run_external_agents)
        self.assertNotIn("scripts.run_foundation_agent_experiments", source)
        self.assertEqual(model_client.parse_model_spec("ollama:gemma4:26b"), ("ollama", "gemma4:26b"))
        prompt = run_external_agents._system_prompt()
        self.assertIn("external public traces", prompt)
        prediction = run_external_agents._coerce_prediction(
            episode=_simple_external_episode(),
            data={
                "episode_id": "ext-1",
                "alarm_time": 1,
                "alarm_confidence": 0.95,
                "root_cause_topk": ["svc-a:latency"],
                "evidence_spans": ["ev-1", "not-visible"],
                "action_ids": ["not-visible"],
                "action_confidence": 0.4,
            },
            wall_time_s=0.1,
            tool_calls=4,
            raw_text='{"ok": true}',
            metadata={"prompt_eval_count": 10, "eval_count": 5},
            parse_status="direct_json",
        )
        self.assertTrue(prediction.parse_success)
        self.assertEqual(prediction.token_count, 15)
        self.assertEqual(prediction.evidence_spans, ["ev-1"])

    def test_external_agent_compacts_large_sensor_prompt_and_survives_timeout(self):
        episode = replace(
            _simple_external_episode(),
            observations={
                "sensors": [
                    {
                        "timestamp": index,
                        "values": {f"svc-a.metric-{metric}": float(index + metric) for metric in range(120)},
                    }
                    for index in range(100)
                ],
                "evidence": [
                    {
                        "span_id": "ev-1",
                        "source_id": "fixture",
                        "kind": "metric",
                        "timestamp": 1,
                        "component": "svc-a",
                        "text": "svc-a latency increased.",
                    }
                ],
            },
        )
        prompt = run_external_agents._episode_prompt(episode, max_steps=4)
        self.assertIn("sensor_summary", prompt)
        self.assertNotIn('"values"', prompt)
        self.assertLess(len(prompt), 40000)

        agent = run_external_agents.ExternalAgent(
            model_spec="ollama:gemma4:26b",
            policy="react-json",
            max_steps=4,
            temperature=0.0,
            timeout_s=0.01,
            ollama_host="http://localhost:11434",
            num_ctx=8192,
            keep_alive="5m",
            think=False,
        )
        with patch.object(agent, "_call_model", side_effect=TimeoutError("timed out")):
            prediction, raw = agent.predict(episode)
        self.assertFalse(prediction.parse_success)
        self.assertEqual(raw["json_parse_status"], "call_error")
        self.assertIn("TimeoutError", raw["call_error"])

    def test_external_baseline_runner_smoke(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_path = tmp_path / "episodes.jsonl"
            episode_path.write_text(dumps_json(_simple_external_episode()) + "\n", encoding="utf-8")
            output_dir = tmp_path / "out"
            result = run_external_baselines.main(
                [
                    "--episodes",
                    str(episode_path),
                    "--output-dir",
                    str(output_dir),
                    "--baselines",
                    "threshold",
                    "topology_heuristic",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "predictions" / "threshold.jsonl").exists())

    def test_external_table_writer_handles_baseline_and_agent_columns(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.csv"
            agents = tmp_path / "agents.csv"
            output_dir = tmp_path / "tables"
            baseline.write_text(
                "system,count,external_portability_score,detection,root_cause_top1,"
                "root_cause_topk,evidence_f1,parsing_success\n"
                "threshold,1,0.7,0.8,0.2,0.6,0.5,1.0\n",
                encoding="utf-8",
            )
            agents.write_text(
                "system,model_spec,count,external_portability_score,detection,root_cause_top1,"
                "root_cause_topk,evidence_f1,parsing_success\n"
                "Ollama-gemma4:e4b-ReAct,ollama:gemma4:e4b,1,0.8,0.9,0.3,0.7,0.6,1.0\n",
                encoding="utf-8",
            )

            result = make_external_tables.main(
                ["--summary-csv", str(baseline), str(agents), "--output-dir", str(output_dir)]
            )
            merged = (output_dir / "external_portability_summary.csv").read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertIn("model_spec", merged.splitlines()[0])
        self.assertIn("ollama:gemma4:e4b", merged)

    def test_adapter_cli_writes_manifest(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            _write_rcaeval_fixture(raw)
            output = Path(tmp) / "episodes.jsonl"
            manifest = Path(tmp) / "manifest.yaml"
            result = rcaeval_re2_adapter.main(
                [
                    "--raw_dir",
                    str(raw),
                    "--systems",
                    "SS",
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--max_evidence_spans",
                    "6",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(load_episodes_jsonl(output)), 1)
            self.assertIn("sha256", manifest.read_text(encoding="utf-8"))

    def test_rcaeval_adapter_handles_nested_zip_with_fault_path(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            archive = raw / "RE2-SS.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(
                    "SS/fault_cases/case-1/labels.csv",
                    "case_id,root_cause_service,root_cause_indicator,start_time\n"
                    "case-1,svc-a,latency,1\n",
                )
                handle.writestr(
                    "SS/fault_cases/case-1/metrics.csv",
                    "case_id,timestamp,service,metric,value\n"
                    "case-1,0,svc-a,latency,10\n"
                    "case-1,1,svc-a,latency,40\n",
                )
                handle.writestr(
                    "SS/fault_cases/case-1/logs.csv",
                    "case_id,timestamp,service,message\n"
                    "case-1,1,svc-a,error timeout\n",
                )
                handle.writestr(
                    "SS/fault_cases/case-1/traces.csv",
                    "case_id,timestamp,caller,callee,latency_ms,error\n"
                    "case-1,1,svc-a,svc-b,250,timeout\n",
                )

            episodes, blobs = rcaeval_re2_adapter.build_episodes(raw, ["SS"], max_evidence_spans=6, seed=7)

        self.assertEqual(len(blobs), 4)
        self.assertEqual(len(episodes), 1)
        self.assertTrue(episodes[0].observations["sensors"])
        self.assertTrue(episodes[0].ground_truth.gold_evidence_spans)

    def test_rcaeval_realistic_path_labels_and_wide_metrics(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            archive = raw / "RE2-SS.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("RE2-SS/payment_mem/1/inject_time.txt", "1705824978\n")
                handle.writestr(
                    "RE2-SS/payment_mem/1/simple_metrics.csv",
                    "time,payment_mem,front-end_error\n"
                    "1705824977,10,0\n"
                    "1705824978,40,1\n",
                )
                handle.writestr(
                    "RE2-SS/payment_mem/1/logs.csv",
                    "time,timestamp,container_name,message\n"
                    "08:04,1705824978000000000,payment,error timeout\n",
                )
                handle.writestr(
                    "RE2-SS/orders_disk/2/inject_time.txt",
                    "1705825000\n",
                )
                handle.writestr(
                    "RE2-SS/orders_disk/2/simple_metrics.csv",
                    "time,orders_disk,front-end_error\n"
                    "1705824999,5,0\n"
                    "1705825000,25,1\n",
                )

            episodes, _ = rcaeval_re2_adapter.build_episodes(raw, ["SS"], max_evidence_spans=6, seed=7)

        by_label = {episode.ground_truth.root_cause_label: episode for episode in episodes}
        self.assertIn("payment:mem", by_label)
        self.assertIn("orders:disk", by_label)
        self.assertEqual(len(episodes), 2)
        self.assertLessEqual(max(episode.duration for episode in episodes), 2)
        self.assertTrue(by_label["payment:mem"].observations["sensors"])


def _simple_external_episode() -> ExternalEpisode:
    return ExternalEpisode(
        episode_id="ext-1",
        source_dataset="fixture",
        domain="microservice",
        split="external",
        duration=4,
        topology=[("svc-a", "svc-b")],
        observations={
            "sensors": [
                {"timestamp": 0, "values": {"svc-a.latency": 10.0, "svc-b.latency": 10.0}},
                {"timestamp": 1, "values": {"svc-a.latency": 20.0, "svc-b.latency": 11.0}},
            ],
            "evidence": [
                {
                    "span_id": "ev-1",
                    "source_id": "fixture",
                    "kind": "metric",
                    "timestamp": 1,
                    "component": "svc-a",
                    "text": "svc-a latency increased.",
                }
            ],
        },
        candidate_actions=[],
        ground_truth=ExternalGroundTruth(
            fault_component="svc-a",
            fault_type="latency",
            root_cause_label="svc-a:latency",
            is_faulted=True,
            fault_start_time=1,
            gold_evidence_spans=["ev-1"],
        ),
        metadata={"candidate_root_causes": ["svc-a:latency", "svc-b:latency"]},
    )


def _write_rcaeval_fixture(raw: Path) -> None:
    (raw / "SS_labels.csv").write_text(
        "case_id,root_cause_service,root_cause_indicator,start_time\n"
        "case-1,svc-a,latency,1\n",
        encoding="utf-8",
    )
    (raw / "SS_metrics.csv").write_text(
        "case_id,timestamp,service,metric,value\n"
        "case-1,0,svc-a,latency,10\n"
        "case-1,1,svc-a,latency,40\n"
        "case-1,1,svc-b,latency,15\n",
        encoding="utf-8",
    )
    (raw / "SS_logs.csv").write_text(
        "case_id,timestamp,service,message\n"
        "case-1,1,svc-a,error timeout on checkout\n",
        encoding="utf-8",
    )
    (raw / "SS_traces.csv").write_text(
        "case_id,timestamp,caller,callee,latency_ms,error\n"
        "case-1,1,svc-a,svc-b,250,timeout\n",
        encoding="utf-8",
    )


def _write_lbnl_fixture(raw: Path) -> None:
    (raw / "RTU_fault_free.csv").write_text(
        "time,supply_temp,fan_power\n"
        "0,20,1\n"
        "1,20,1\n"
        "2,20,1\n",
        encoding="utf-8",
    )
    (raw / "RTU_fan_sev2.csv").write_text(
        "time,supply_temp,fan_power\n"
        "0,20,1\n"
        "1,28,4\n"
        "2,30,5\n"
        "3,31,5\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
