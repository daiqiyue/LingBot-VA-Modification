#!/usr/bin/env bash

# ===========================
# Lingbot LQR end-to-end runner
# ===========================
# Default usage:
#   bash scripts/lqr/run_lqr_pipeline.sh
#
# Common overrides:
#   NUM_EPISODES=5 TASK_RANGE_START=0 TASK_RANGE_END=2 bash scripts/lqr/run_lqr_pipeline.sh
#   PERTURB_SPEC=scripts/lqr/configs/perturb_spec_init_pos.yaml bash scripts/lqr/run_lqr_pipeline.sh
#   SKIP_EVAL=1 bash scripts/lqr/run_lqr_pipeline.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

if [[ "${ACTIVATE_CONDA:-1}" == "1" ]]; then
  source ~/.bashrc
  conda activate "${CONDA_ENV:-lingbot}"
fi

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

# -------- Data collection / pairing / SVD --------
CONFIG_NAME="${CONFIG_NAME:-libero}"
LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
TASK_ID="${TASK_ID:-0}"
NUM_EPISODES="${NUM_EPISODES:-10}"
TOPK_INFER_PER_TRAJ="${TOPK_INFER_PER_TRAJ:-10}"
SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
COLLECT_MODE="${COLLECT_MODE:-action}"
PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_init_pos.yaml}"
TARGET_VARIANTS="${TARGET_VARIANTS:-}"
PAIR_SEED="${PAIR_SEED:-0}"
K_TARGET="${K_TARGET:-32}"
P_OVER="${P_OVER:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
RIDGE="${RIDGE:-1e-3}"
JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"

OUT_BASE="${OUT_BASE:-outputs/lqr}"
PAIRS_DIR="${PAIRS_DIR:-${OUT_BASE}/pairs_${TS}}"
PAIRS_ALL_DIR="${PAIRS_ALL_DIR:-${OUT_BASE}/pairs_all_${TS}}"
SVD_DIR="${SVD_DIR:-${OUT_BASE}/svd_all_perturb_${TS}}"

# -------- Evaluation --------
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_OUT_BASE="${EVAL_OUT_BASE:-outputs/lqr_eval_all_perturb_${TS}}"
TASK_RANGE_START="${TASK_RANGE_START:-0}"
TASK_RANGE_END="${TASK_RANGE_END:-2}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
PORT="${PORT:-29056}"
PROMPT="${PROMPT:-}"
LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"

echo "===== Lingbot LQR Pipeline ====="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PERTURB_SPEC=${PERTURB_SPEC}"
echo "PAIRS_DIR=${PAIRS_DIR}"
echo "PAIRS_ALL_DIR=${PAIRS_ALL_DIR}"
echo "SVD_DIR=${SVD_DIR}"
echo "EVAL_OUT_BASE=${EVAL_OUT_BASE}"
echo "TS=${TS}"
echo "================================"

COLLECT_CMD=(
  python scripts/lqr/run_collect_inputs.py
  --config-name "${CONFIG_NAME}"
  --libero-benchmark "${LIBERO_BENCHMARK}"
  --task-id "${TASK_ID}"
  --num-episodes "${NUM_EPISODES}"
  --top-k-inference-per-traj "${TOPK_INFER_PER_TRAJ}"
  --selected-timesteps "${SELECTED_TIMESTEPS}"
  --mode "${COLLECT_MODE}"
  --perturb-spec "${PERTURB_SPEC}"
  --out-dir "${PAIRS_DIR}"
)
if [[ -n "${TARGET_VARIANTS}" ]]; then
  COLLECT_CMD+=(--target-variants "${TARGET_VARIANTS}")
fi
echo "[1/5] collect trajectory activations"
"${COLLECT_CMD[@]}"

echo "[2/5] build pool-based pairs"
python scripts/lqr/build_all_pairs.py \
  --collect-dir "${PAIRS_DIR}" \
  --out-dir "${PAIRS_ALL_DIR}" \
  --pair-seed "${PAIR_SEED}"

echo "[3/5] run SVD"
python scripts/lqr/run_partition_svd.py \
  --pairs-dir "${PAIRS_ALL_DIR}" \
  --out-dir "${SVD_DIR}" \
  --config-name "${CONFIG_NAME}" \
  --mode "${COLLECT_MODE}" \
  --selected-timesteps "${SELECTED_TIMESTEPS}" \
  --num-samples "${NUM_SAMPLES}" \
  --k-target "${K_TARGET}" \
  --p-over "${P_OVER}"

echo "[4/5] compute projected jacobians"
python scripts/lqr/run_compute_jacobians.py \
  --svd-dir "${SVD_DIR}" \
  --out-subdir "${JAC_SUBDIR}" \
  --ridge "${RIDGE}"

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[5/5] skipped eval (SKIP_EVAL=1)"
else
  EVAL_CMD=(
    python scripts/lqr/run_libero_lqr_eval.py
    --config-name "${CONFIG_NAME}"
    --libero-benchmark "${LIBERO_BENCHMARK}"
    --task-range "${TASK_RANGE_START}" "${TASK_RANGE_END}"
    --num-episodes "${EVAL_NUM_EPISODES}"
    --port "${PORT}"
    --svd-dir "${SVD_DIR}"
    --jac-dir-act "${JAC_SUBDIR}"
    --lqr-config "${LQR_CONFIG}"
    --perturb-spec "${PERTURB_SPEC}"
    --out-dir "${EVAL_OUT_BASE}"
  )
  if [[ -n "${PROMPT}" ]]; then
    EVAL_CMD+=(--prompt "${PROMPT}")
  fi
  echo "[5/5] run lqr eval"
  "${EVAL_CMD[@]}"
fi

echo ""
echo "Done."
echo "pairs_raw   : ${PAIRS_DIR}"
echo "pairs_all   : ${PAIRS_ALL_DIR}"
echo "svd_dir     : ${SVD_DIR}"
if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo "eval_out    : ${EVAL_OUT_BASE}"
fi
