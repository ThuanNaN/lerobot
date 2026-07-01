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
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--wandb.enable=true)
fi

uv sync --locked --extra smolvla --extra libero

# https://huggingface.co/docs/lerobot/multi_gpu_training
TRAIN_CMD=(uv run lerobot-train)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  TRAIN_CMD=(uv run accelerate launch --multi_gpu --num_processes="${NUM_GPUS}" "$(uv run which lerobot-train)")
fi

"${TRAIN_CMD[@]}" \
  --policy.type=smolvla \
  --policy.repo_id="${HF_USER}/libero-vlai" \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task="${TASK_SUITE}" \
  --output_dir=./outputs/ \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=500 \
  "${WANDB_ARGS[@]}"

# Benchmark: full LIBERO protocol (4 suites x 10 episodes = 400 episodes)
CHECKPOINT_PATH="./outputs/checkpoints/last/pretrained_model"

uv run lerobot-eval \
  --policy.path="${CHECKPOINT_PATH}" \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --output_dir=./outputs/eval/
