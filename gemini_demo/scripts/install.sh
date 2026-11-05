#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), "LLaMA-Factory main requires Python>=3.11"; print("Python:",sys.version)'
if [[ ! -d "$ROOT/.venv" ]]; then "$PYTHON_BIN" -m venv --system-site-packages "$ROOT/.venv"; fi
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip hatchling
if [[ -n "${LLAMAFACTORY_SOURCE:-}" ]]; then
  FACTORY_PATH="$(cd "$LLAMAFACTORY_SOURCE" && pwd)"
else
  FACTORY_PATH="$ROOT/LlamaFactory"
  if [[ ! -f "$FACTORY_PATH/pyproject.toml" ]]; then
    echo 'Cloning official LLaMA-Factory (GitHub network required)...'
    git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git "$FACTORY_PATH"
  fi
fi
python -m pip install -e "$FACTORY_PATH"
python -m pip install -U "qwen-vl-utils[decord]" huggingface_hub modelscope
python -c 'import torch,transformers;print("Torch:",torch.__version__,"CUDA:",torch.cuda.is_available(),"Transformers:",transformers.__version__)'
if [[ -d "$FACTORY_PATH/.git" ]]; then
  git -C "$FACTORY_PATH" rev-parse HEAD | tee "$ROOT/llamafactory_commit.txt"
fi
printf '\nInstalled. Activate with: source %s/.venv/bin/activate\n' "$ROOT"
