"""Minimal model-client helpers for external validation agents."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any
from urllib import error, request


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    if ":" not in model_spec:
        raise ValueError("Model specs must use provider:model, e.g. ollama:qwen3.5:4b")
    provider, model = model_spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in {"openai", "anthropic", "ollama"}:
        raise ValueError("Supported providers: ollama, openai, anthropic")
    if not model:
        raise ValueError("Model name is empty")
    return provider, model


def validate_credentials(model_specs: list[str]) -> None:
    providers = {parse_model_spec(model_spec)[0] for model_spec in model_specs}
    missing = []
    if "openai" in providers and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if "anthropic" in providers and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise SystemExit("Missing required API credentials: " + ", ".join(missing))


def display_name(provider: str, model: str, policy: str) -> str:
    suffix = "ReAct" if policy == "react-json" else policy
    if provider == "ollama":
        return f"Ollama-{model}-{suffix}"
    if provider == "openai":
        return f"{model.upper()}-{suffix}"
    if provider == "anthropic":
        pretty = model.replace("claude-", "Claude-").replace("-", " ").title().replace(" ", "-")
        pretty = pretty.replace("Sonnet-4-6", "Sonnet-4.6")
        return f"{pretty}-{suffix}"
    return f"{provider}:{model}-{suffix}"


def call_ollama(
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
    host: str,
    num_ctx: int,
    keep_alive: str,
    think: bool,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": think,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    response = post_json(
        f"{host.rstrip('/')}/api/generate",
        payload,
        {"Content-Type": "application/json"},
        timeout_s,
    )
    metadata_keys = [
        "model",
        "created_at",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ]
    metadata = {key: response[key] for key in metadata_keys if key in response}
    metadata.update(
        {
            "requested_model": model,
            "ollama_host": host.rstrip("/"),
            "num_ctx": num_ctx,
            "keep_alive": keep_alive,
            "think": think,
            "temperature": temperature,
            "format": "json",
        }
    )
    return str(response.get("response") or ""), metadata


def call_openai(
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    response = post_json(
        "https://api.openai.com/v1/responses",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    return collect_text(response.get("output", []))


def call_anthropic(
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    payload = {
        "model": model,
        "max_tokens": 1200,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    return collect_text(response.get("content", []))


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def collect_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        pieces = []
        for key in ("text", "content", "output_text"):
            if isinstance(value.get(key), str):
                pieces.append(value[key])
        for item in value.values():
            if isinstance(item, (list, dict)):
                nested = collect_text(item)
                if nested:
                    pieces.append(nested)
        return "\n".join(pieces)
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := collect_text(item)))
    return ""


def extract_prediction_data_with_status(text: str) -> tuple[dict[str, Any], str]:
    if not text:
        return {}, "empty"
    try:
        parsed = json.loads(text)
        return (parsed, "direct_json") if isinstance(parsed, dict) else ({}, "non_object_json")
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}, "no_json_object"
        try:
            parsed = json.loads(match.group(0))
            return (parsed, "embedded_json") if isinstance(parsed, dict) else ({}, "embedded_non_object_json")
        except json.JSONDecodeError:
            return {}, "invalid_json"


def metadata_token_count(metadata: dict[str, Any]) -> int | None:
    prompt_count = _int_or_none(metadata.get("prompt_eval_count"))
    eval_count = _int_or_none(metadata.get("eval_count"))
    if prompt_count is None and eval_count is None:
        return None
    return (prompt_count or 0) + (eval_count or 0)


def prob(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
