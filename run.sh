#!/usr/bin/env bash
set -euo pipefail

# SmolVLA + LIBERO training pipeline (see TODO.md, docs/source/libero.mdx)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

HF_USER="${HF_USER:?Set HF_USER (in .env or env) to your Hugging Face username}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-4}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--wandb.enable=true)
fi

uv sync --locked --extra smolvla --extra libero

uv run lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="${HF_USER}/libero-test" \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task="${TASK_SUITE}" \
  --output_dir=./outputs/ \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=1000 \
  "${WANDB_ARGS[@]}"
