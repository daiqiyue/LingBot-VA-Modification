#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-29056}"
SAVE_ROOT="${SAVE_ROOT:-outputs/libero_lqr/server}"
LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"
SVD_DIR="${SVD_DIR:-outputs/lqr/svd_run}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full}"

PYTHONPATH=. python scripts/lqr/patch_infer_with_lqr.py \
  --config-name libero \
  --port "${PORT}" \
  --save_root "${SAVE_ROOT}" \
  --svd-dir "${SVD_DIR}" \
  --jac-dir-act "${JAC_DIR_ACT}" \
  --lqr-config "${LQR_CONFIG}"
