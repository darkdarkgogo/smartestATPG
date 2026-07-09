#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/PodemQuest/vendor/deepgate_recgnn_extractor:${PYTHONPATH:-}"

BENCH_DIR="${1:-${ROOT_DIR}/PodemQuest/test}"
SAVE_PATH="${2:-${ROOT_DIR}/PodemQuest/checkpoints/podem_deepgate.pth}"
DEVICE="${3:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"
NUM_PATTERNS="${NUM_PATTERNS:-256}"
EXACT_PI_LIMIT="${EXACT_PI_LIMIT:-10}"
MAX_PAIRS="${MAX_PAIRS:-512}"
LR="${LR:-1e-3}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
PROB_WEIGHT="${PROB_WEIGHT:-0.2}"
FUNC_WEIGHT="${FUNC_WEIGHT:-1.0}"
SEED="${SEED:-7}"

mkdir -p "$(dirname "${SAVE_PATH}")"

python -m deepgate_recgnn_extractor.train \
  --bench_dir "${BENCH_DIR}" \
  --save_path "${SAVE_PATH}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_rounds "${NUM_ROUNDS}" \
  --num_patterns "${NUM_PATTERNS}" \
  --exact_pi_limit "${EXACT_PI_LIMIT}" \
  --max_pairs "${MAX_PAIRS}" \
  --lr "${LR}" \
  --train_ratio "${TRAIN_RATIO}" \
  --prob_weight "${PROB_WEIGHT}" \
  --func_weight "${FUNC_WEIGHT}" \
  --seed "${SEED}" \
  --device "${DEVICE}"
