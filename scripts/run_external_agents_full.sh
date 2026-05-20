#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPISODES_RE2="${EPISODES_RE2:-external_validation/data/processed/episodes/rcaeval_re2.jsonl}"
EPISODES_LBNL="${EPISODES_LBNL:-external_validation/data/processed/episodes/lbnl_rtu_sdahu.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-external_validation/outputs/agents}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-external_validation/outputs/baselines/summary.csv}"
TABLE_DIR="${TABLE_DIR:-external_validation/outputs/tables}"
FIGURE_DIR="${FIGURE_DIR:-external_validation/outputs/figures}"
LOG_DIR="${LOG_DIR:-external_validation/outputs/logs}"

POLICY="${POLICY:-react-json}"
MAX_STEPS="${MAX_STEPS:-4}"
TEMPERATURE="${TEMPERATURE:-0}"
TIMEOUT_S="${TIMEOUT_S:-300}"
NUM_CTX="${NUM_CTX:-8192}"
KEEP_ALIVE="${KEEP_ALIVE:-5m}"
THINK="${THINK:-false}"
PULL_MODELS="${PULL_MODELS:-true}"
MIN_PARSE_SUCCESS="${MIN_PARSE_SUCCESS:-0.01}"

# Space-separated model specs. Override, for example:
#   MODELS="ollama:gemma4:e4b" scripts/run_external_agents_full.sh
read -r -a MODEL_SPECS <<< "${MODELS:-ollama:gemma4:26b ollama:gemma4:e4b}"
if [[ "${#MODEL_SPECS[@]}" -eq 0 ]]; then
  echo "No models configured. Set MODELS to one or more provider:model specs." >&2
  exit 2
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

mkdir -p "$OUTPUT_DIR" "$TABLE_DIR" "$FIGURE_DIR" "$LOG_DIR"

for episode_file in "$EPISODES_RE2" "$EPISODES_LBNL"; do
  if [[ ! -f "$episode_file" ]]; then
    echo "Missing episode file: $episode_file" >&2
    echo "Run the external adapters first, or override EPISODES_RE2/EPISODES_LBNL." >&2
    exit 2
  fi
done

needs_ollama=false
for model_spec in "${MODEL_SPECS[@]}"; do
  if [[ "$model_spec" == ollama:* ]]; then
    needs_ollama=true
  fi
done

if [[ "$needs_ollama" == "true" ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama is not on PATH; install/start Ollama or choose non-Ollama MODEL specs." >&2
    exit 2
  fi

  if ! ollama list >/dev/null 2>&1; then
    echo "Cannot reach Ollama. Start the Ollama service, then rerun this script." >&2
    exit 2
  fi
fi

if [[ "$needs_ollama" == "true" && "$PULL_MODELS" == "true" ]]; then
  for model_spec in "${MODEL_SPECS[@]}"; do
    if [[ "$model_spec" == ollama:* ]]; then
      ollama pull "${model_spec#ollama:}"
    fi
  done
fi

log_file="$LOG_DIR/external_agents_$(date -u +%Y%m%dT%H%M%SZ).log"

PYTHONUNBUFFERED=1 python -m cob_ext.runners.run_external_agents \
  --episodes "$EPISODES_RE2" "$EPISODES_LBNL" \
  --output-dir "$OUTPUT_DIR" \
  --models "${MODEL_SPECS[@]}" \
  --policy "$POLICY" \
  --max-steps "$MAX_STEPS" \
  --temperature "$TEMPERATURE" \
  "${LIMIT_ARGS[@]}" \
  --num-ctx "$NUM_CTX" \
  --keep-alive "$KEEP_ALIVE" \
  --think "$THINK" \
  --timeout-s "$TIMEOUT_S" \
  2>&1 | tee "$log_file"

AGENT_SUMMARY="$OUTPUT_DIR/summary.csv"
if [[ ! -s "$AGENT_SUMMARY" ]]; then
  echo "Agent summary was not created: $AGENT_SUMMARY" >&2
  exit 1
fi

python - "$AGENT_SUMMARY" "$MIN_PARSE_SUCCESS" <<'PY'
import csv
import math
import sys

summary_path, raw_minimum = sys.argv[1], sys.argv[2]
minimum = float(raw_minimum)
with open(summary_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"No agent rows found in {summary_path}")

bad = []
for row in rows:
    try:
        value = float(row.get("parsing_success", "nan"))
    except ValueError:
        value = math.nan
    if math.isnan(value) or value < minimum:
        bad.append(f"{row.get('system', '<unknown>')}={value}")

if bad:
    joined = ", ".join(bad)
    raise SystemExit(
        f"Agent parsing_success below MIN_PARSE_SUCCESS={minimum}: {joined}. "
        "Inspect the raw JSONL/logs or rerun with a larger TIMEOUT_S. "
        "Set MIN_PARSE_SUCCESS=0 to disable this gate."
    )
PY

summary_inputs=("$AGENT_SUMMARY")
if [[ -s "$BASELINE_SUMMARY" ]]; then
  summary_inputs=("$BASELINE_SUMMARY" "$AGENT_SUMMARY")
else
  echo "Baseline summary not found at $BASELINE_SUMMARY; generating agent-only tables." >&2
fi

python -m cob_ext.reporting.make_external_tables \
  --summary-csv "${summary_inputs[@]}" \
  --output-dir "$TABLE_DIR"

python -m cob_ext.reporting.make_external_figures \
  --summary-csv "$TABLE_DIR/external_portability_summary.csv" \
  --output-dir "$FIGURE_DIR"

echo "External agent stage complete."
echo "Log: $log_file"
echo "Agent summary: $AGENT_SUMMARY"
echo "Tables: $TABLE_DIR"
echo "Figures: $FIGURE_DIR"
