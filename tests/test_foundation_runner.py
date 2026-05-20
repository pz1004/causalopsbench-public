import unittest
from unittest.mock import patch

from cob_ext import model_client
from causalopsbench.generator import EpisodeGenerator
from scripts import run_foundation_agent_experiments as runner


class FoundationRunnerTests(unittest.TestCase):
    def test_parse_nested_ollama_model_spec(self):
        provider, model = runner._parse_model_spec("ollama:granite4.1:8b")
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "granite4.1:8b")

    def test_model_helper_wrappers_match_package_client(self):
        metadata = {"prompt_eval_count": "42", "eval_count": 11}
        self.assertEqual(
            runner._parse_model_spec("ollama:granite4.1:8b"),
            model_client.parse_model_spec("ollama:granite4.1:8b"),
        )
        self.assertEqual(runner._prob("1.5"), model_client.prob("1.5"))
        self.assertEqual(runner._str_list("svc-a:latency"), model_client.str_list("svc-a:latency"))
        self.assertEqual(runner._metadata_token_count(metadata), model_client.metadata_token_count(metadata))
        self.assertEqual(
            runner._collect_text({"content": [{"text": "hello"}]}),
            model_client.collect_text({"content": [{"text": "hello"}]}),
        )

    def test_ollama_requires_no_api_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            runner._validate_credentials(["ollama:qwen3.5:4b"])

    def test_call_ollama_extracts_response_and_metadata(self):
        fake_response = {
            "model": "qwen3.5:4b",
            "response": '{"episode_id":"ep","alarm_time":1}',
            "done": True,
            "prompt_eval_count": 42,
            "eval_count": 11,
            "total_duration": 123,
        }
        with patch.object(runner, "_post_json", return_value=fake_response) as post_json:
            text, metadata = runner._call_ollama(
                model="qwen3.5:4b",
                system="system",
                prompt="prompt",
                temperature=0.0,
                timeout_s=10.0,
                host="http://localhost:11434",
                num_ctx=8192,
                keep_alive="5m",
                think=False,
            )

        self.assertEqual(text, '{"episode_id":"ep","alarm_time":1}')
        self.assertEqual(metadata["prompt_eval_count"], 42)
        self.assertEqual(metadata["eval_count"], 11)
        self.assertEqual(metadata["requested_model"], "qwen3.5:4b")
        self.assertEqual(metadata["num_ctx"], 8192)
        self.assertEqual(metadata["temperature"], 0.0)
        self.assertFalse(metadata["think"])
        payload = post_json.call_args.args[1]
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["num_ctx"], 8192)

    def test_invalid_model_json_coerces_to_safe_empty_prediction(self):
        episode = EpisodeGenerator(seed=1, duration=30).generate_one("microservice")
        data, status = runner._extract_prediction_data_with_status("not json")
        prediction = runner._coerce_prediction(
            episode=episode,
            data=data,
            wall_time_s=0.1,
            tool_calls=4,
            raw_text="not json",
        )

        self.assertEqual(status, "no_json_object")
        self.assertIsNone(prediction.alarm_time)
        self.assertEqual(prediction.root_cause_topk, [])
        self.assertEqual(prediction.evidence_spans, [])
        self.assertEqual(prediction.action_ids, [])
        self.assertIn("Invalid or empty", prediction.postmortem)

    def test_view_ablation_removes_only_requested_public_fields(self):
        episode = EpisodeGenerator(seed=1, duration=30).generate_one("microservice")
        full = runner._public_episode_view(episode)
        no_evidence = runner._public_episode_view(episode, view_ablation="no-evidence")
        no_topology = runner._public_episode_view(episode, view_ablation="no-topology")
        no_manuals = runner._public_episode_view(episode, view_ablation="no-manuals")

        self.assertGreater(len(full["evidence"]), 0)
        self.assertGreater(len(full["manuals"]), 0)
        self.assertGreater(len(full["topology"]), 0)
        self.assertEqual(no_evidence["evidence"], [])
        self.assertGreater(len(no_evidence["manuals"]), 0)
        self.assertEqual(no_topology["topology"], [])
        self.assertGreater(len(no_topology["evidence"]), 0)
        self.assertEqual(no_manuals["manuals"], [])
        self.assertGreater(len(no_manuals["evidence"]), 0)


if __name__ == "__main__":
    unittest.main()
