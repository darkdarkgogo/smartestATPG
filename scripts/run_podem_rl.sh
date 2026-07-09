#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/PodemQuest/src:${ROOT_DIR}/PodemQuest/vendor/deepgate_recgnn_extractor:${PYTHONPATH:-}"

INPUT_BENCH="${1:-${ROOT_DIR}/PodemQuest/test/c17.bench}"
OUTPUT_FILE="${2:-${ROOT_DIR}/PodemQuest/out_patterns_rl.txt}"
REPORT_FILE="${3:-${ROOT_DIR}/PodemQuest/out_report_rl.txt}"
DEEPGATE_CKPT="${4:-${ROOT_DIR}/PodemQuest/checkpoints/podem_deepgate.pth}"
RL_CKPT="${5:-${ROOT_DIR}/PodemQuest/checkpoints/podem_rl.pth}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"
mkdir -p "$(dirname "${REPORT_FILE}")"
mkdir -p "$(dirname "${DEEPGATE_CKPT}")"
mkdir -p "$(dirname "${RL_CKPT}")"

python -m PodemQuest.PodemQuest \
  -i "${INPUT_BENCH}" \
  -o "${OUTPUT_FILE}" \
  -r "${REPORT_FILE}" \
  -a rl \
  --deepgate_checkpoint "${DEEPGATE_CKPT}" \
  --rl_checkpoint "${RL_CKPT}"
