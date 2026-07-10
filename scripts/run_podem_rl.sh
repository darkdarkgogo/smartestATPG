#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/vendor/deepgate_recgnn_extractor:${PYTHONPATH:-}"

INPUT_BENCH="${1:-${PROJECT_DIR}/test/03_c17.bench}"
OUTPUT_FILE="${2:-${PROJECT_DIR}/out_patterns_rl.txt}"
REPORT_FILE="${3:-${PROJECT_DIR}/out_report_rl.txt}"
DEEPGATE_CKPT="${4:-${PROJECT_DIR}/checkpoints/podem_deepgate.pth}"
RL_CKPT="${5:-${PROJECT_DIR}/checkpoints/podem_rl.pth}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"
mkdir -p "$(dirname "${REPORT_FILE}")"
mkdir -p "$(dirname "${DEEPGATE_CKPT}")"
mkdir -p "$(dirname "${RL_CKPT}")"

cd "${PROJECT_DIR}"

python -m PodemQuest.PodemQuest \
  -i "${INPUT_BENCH}" \
  -o "${OUTPUT_FILE}" \
  -r "${REPORT_FILE}" \
  -a rl \
  --deepgate_checkpoint "${DEEPGATE_CKPT}" \
  --rl_checkpoint "${RL_CKPT}"
