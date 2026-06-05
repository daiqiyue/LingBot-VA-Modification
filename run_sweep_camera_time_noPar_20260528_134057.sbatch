#!/bin/bash
#SBATCH -J lqr_sweep_cam_tnp
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 20:00:00
#SBATCH --array=0-7%4
#SBATCH -o Lqr_sweep_camera_time_noPar-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
cd /storage/scratch1/9/qdai41/cosmos/LingBot-VA-Modification

export PYTHONPATH=.

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DEFAULT_MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

source scripts/lqr/slurm_port.sh

SVD_DIR="${SVD_DIR:-outputs/lqr_time_noPar/svd_cam_random_large_seed42_20260528_134057}"
JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot_time_noPar}"
PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"
TASK_IDS=(${TASK_IDS:-0 2 4})
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
INJECT_MODE="${INJECT_MODE:-action}"
RANDOM_CAMERA_BASE_SEED="${RANDOM_CAMERA_BASE_SEED:-99}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/lqr_sweep/time_noPar_camera_20260528_134057_tasks0_2_4}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:-${SWEEP_ROOT}/eval_seed${RANDOM_CAMERA_BASE_SEED}}"
SWEEP_RESULTS_MD="${SWEEP_RESULTS_MD:-${EVAL_SWEEP_ROOT}/sweep_success_rates.md}"
FORCE="${FORCE:-0}"

TOTAL=8

LAMBDAS=(5.0 15.0)
QS=(1.0 100.0)
RINITS=(5.0 20.0)
R_SCALE_TAU="${R_SCALE_TAU:-5.0}"

sanitize_tag_component() {
  local s="$1"
  s="${s//./p}"
  s="${s//-/_neg_}"
  printf '%s' "$s"
}

decode_combo() {
  local idx="$1"
  local -n _L="$2"
  local -n _Q="$3"
  local -n _RI="$4"

  if (( idx < 0 || idx >= TOTAL )); then
    echo "[error] combo index ${idx} out of range [0, $((TOTAL - 1))]" >&2
    return 1
  fi

  local nR=${#RINITS[@]}
  local nQ=${#QS[@]}
  local r_i=$((idx % nR))
  local tmp=$((idx / nR))
  local q_i=$((tmp % nQ))
  local l_i=$((tmp / nQ))
  _L="${LAMBDAS[$l_i]}"
  _Q="${QS[$q_i]}"
  _RI="${RINITS[$r_i]}"
}

combo_tag() {
  local idx="$1"
  local lambda_scale="$2"
  local q_scale="$3"
  local r_scale="$4"
  printf 'idx%s_lam%s_q%s_r%s' \
    "${idx}" \
    "$(sanitize_tag_component "${lambda_scale}")" \
    "$(sanitize_tag_component "${q_scale}")" \
    "$(sanitize_tag_component "${r_scale}")"
}

aggregate_sweep_results() {
  mkdir -p "${EVAL_SWEEP_ROOT}"
  (
    flock -w 300 9 || exit 1
    python scripts/lqr/aggregate_sweep_results.py \
      --out-root "${EVAL_SWEEP_ROOT}" \
      --output-md "${SWEEP_RESULTS_MD}"
  ) 9>"${EVAL_SWEEP_ROOT}/.sweep_aggregate.lock"
}

[[ -f "${SVD_DIR}/config.json" ]] || { echo "[error] missing ${SVD_DIR}/config.json" >&2; exit 2; }
[[ -f "${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt" ]] || { echo "[error] missing ${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt" >&2; exit 2; }

IDX="${SLURM_ARRAY_TASK_ID:-0}"
lambda_scale=""
q_scale=""
r_scale=""
decode_combo "${IDX}" lambda_scale q_scale r_scale

TAG="$(combo_tag "${IDX}" "${lambda_scale}" "${q_scale}" "${r_scale}")"
CFG_DIR="${EVAL_SWEEP_ROOT}/configs/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_JOB_ID:-local}"
CFG_PATH="${CFG_DIR}/lqr_${TAG}.yaml"
OUT_DIR="${EVAL_SWEEP_ROOT}/${TAG}"

if [[ "${FORCE}" != "1" && -f "${OUT_DIR}/summary.json" ]]; then
  echo "[sweep] skip ${TAG}: summary.json exists"
  aggregate_sweep_results
  exit 0
fi

mkdir -p "${CFG_DIR}" "${OUT_DIR}"
cat > "${CFG_PATH}" <<EOF
lambda_scale: ${lambda_scale}
q_scale: ${q_scale}
r_scale: ${r_scale}
r_scale_tau: ${R_SCALE_TAU}
r_scale_final: 1000000000.0
max_chunks: 50
qf_scale: 1.0
jac_dir_act: ${JAC_SUBDIR}
modality: action
inject_mode: ${INJECT_MODE}
EOF

echo "===== Camera-position time-noPar LQR sweep ====="
echo "  IDX=${IDX}/${TOTAL}"
echo "  TASK_IDS=${TASK_IDS[*]}"
echo "  SVD_DIR=${SVD_DIR}"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  lambda=${lambda_scale} q=${q_scale} r=${r_scale} tau=${R_SCALE_TAU} (fixed)"
echo "  OUT_DIR=${OUT_DIR}"
echo "  PORT=${PORT}"

python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-ids "${TASK_IDS[@]}" \
  --num-episodes "${EVAL_NUM_EPISODES}" \
  --startup-wait-sec "${EVAL_STARTUP_WAIT_SEC}" \
  --port "${PORT}" \
  --svd-dir "${SVD_DIR}" \
  --jac-dir-act "${JAC_SUBDIR}" \
  --lqr-config "${CFG_PATH}" \
  --inject-mode "${INJECT_MODE}" \
  --perturb-spec "${PERTURB_SPEC}" \
  --random-camera-base-seed "${RANDOM_CAMERA_BASE_SEED}" \
  --out-dir "${OUT_DIR}" \
  --resume

aggregate_sweep_results
