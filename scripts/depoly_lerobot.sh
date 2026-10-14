#!/usr/bin/env bash

set -euo pipefail

.venv/bin/python scripts/deploy.py \
  --ckpt-dir outputs/piperx_two_tasks_alpha \
  --ckpt-name checkpoint_step_10470.safetensors \
  --prompt-embeddings outputs/piperx_two_tasks_alpha/prompt_embeddings.safetensors \
  --protocol openpi --inference-mode sync \
  --device cuda:0 --port 8848