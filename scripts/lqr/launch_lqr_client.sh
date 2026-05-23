#!/usr/bin/env bash
set -euo pipefail

START="${START:-0}"
END="${END:-1}"
PORT="${PORT:-29056}"
OUT_DIR="${OUT_DIR:-outputs/libero_lqr/client}"
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket.}"

PYTHONPATH=. python evaluation/libero/client.py \
  --libero-benchmark libero_10 \
  --port "${PORT}" \
  --test-num 10 \
  --task-range "${START}" "${END}" \
  --out-dir "${OUT_DIR}" \
  --prompt "${PROMPT}"
