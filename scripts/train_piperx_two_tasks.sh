#!/usr/bin/env bash
# Fine-tune OpenWAM-Alpha for 5 epochs; default effective batch 256.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec bash scripts/train.sh --config-name=train_piperx_two_tasks "$@"
