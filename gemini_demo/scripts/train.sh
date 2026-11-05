#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-smoke}"
case "$PRESET" in
  smoke|one_epoch|qlora_smoke_8g|qlora_one_epoch_8g) ;;
  *) echo 'Usage: bash scripts/train.sh [qlora_smoke_8g|qlora_one_epoch_8g|smoke|one_epoch]' >&2; exit 2 ;;
esac
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python scripts/preflight.py --data "${DATA_DIR:-$ROOT/data}" --dataset "${DEMO_DATASET:-demo_train}"
if [[ -n "${DEMO_CONFIG_DIR:-}" ]]; then
  CONFIG_DIR="$DEMO_CONFIG_DIR"
elif [[ -w "$ROOT" ]]; then
  CONFIG_DIR="$ROOT/configs/generated"
else
  CONFIG_DIR="${GEMINI_DATA_OUT:-/gemini/output}/configs"
fi
mkdir -p "$CONFIG_DIR"
CONFIG_PATH="$CONFIG_DIR/${PRESET}_resolved.yaml"
if [[ -n "${DEMO_OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$DEMO_OUTPUT_DIR"
elif [[ -w "$ROOT" ]]; then
  OUTPUT_DIR="$ROOT/outputs/$PRESET"
else
  OUTPUT_DIR="${GEMINI_DATA_OUT:-/gemini/output}/$PRESET"
fi
python scripts/resolve_config.py --preset "$PRESET" --output-dir "$OUTPUT_DIR" --dest "$CONFIG_PATH"
echo 'Launching LLaMA-Factory ...'
llamafactory-cli train "$CONFIG_PATH"
