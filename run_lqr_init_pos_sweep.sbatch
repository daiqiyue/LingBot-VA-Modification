#!/bin/bash
#SBATCH -J lingbot_lqr_init_pos_sweep
#SBATCH -A gts-gchou3
#SBATCH -N 1
#SBATCH --gres=gpu:h200:1
#SBATCH -t 24:00:00
#SBATCH -o Lqr_InitPos_Sweep-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#SBATCH --array=0-77

# Init-position LQR pipeline + hyperparameter sweep (ctrlwam-aligned grid).
#
# Mirrors ctrlwam notebooks/lqr/e2e_scripts/run_task07_gripper_xyz_pipeline.sh step 5:
#   base grid  : LAMBDA x Q x RINIT x TAU = 4 x 2 x 3 x 3 = 72
#   extra Q    : {100,200,1000,2000,10000} at (lam=15, rinit=5, tau=3)
#   outlier    : (lam=20, q=1, rinit=10, tau=5)
# = 78 eval combos total.
#
# Each array task:
#   1. ensures collect/pair/SVD/jacobian artifacts exist (single builder via flock)
#   2. runs one LQR eval combo with eval_init_pos.yaml (base_seed=99 variants)
#
# Usage:
#   # full sweep (78 parallel array tasks, each builds-if-needed then evals one combo)
#   sbatch --export=ALL,TASK_ID=7,TASK_RANGE_START=7,TASK_RANGE_END=8 run_lqr_init_pos_sweep.sbatch
#
#   # build artifacts only (no eval)
#   sbatch --export=ALL,BUILD_ONLY=1,TASK_ID=7 run_lqr_init_pos_sweep.sbatch
#
#   # eval sweep only (artifacts must already exist under SVD_DIR)
#   sbatch --array=0-77 --export=ALL,SWEEP_ONLY=1,SVD_DIR=outputs/lqr_sweep/init_pos_task7/svd run_lqr_init_pos_sweep.sbatch
#
#   # rerun a combo even if summary.json exists
#   sbatch --array=12 --export=ALL,FORCE=1,... run_lqr_init_pos_sweep.sbatch
#
# After all array tasks finish:
#   python scripts/lqr/aggregate_sweep_results.py --out-root outputs/lqr_sweep/init_pos_task7/eval_sweep_seed99

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification

export PYTHONPATH=.

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DEFAULT_MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

# ---- Task / collect knobs (artifact building, seed=42 like ctrlwam upstream) ----
export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
export TASK_ID="${TASK_ID:-7}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_init_pos.yaml}"
export PAIR_INIT_BY_SIMILARITY="${PAIR_INIT_BY_SIMILARITY:-1}"
export PAIR_FEATURE="${PAIR_FEATURE:-proprio}"
export PAIR_MATCH_MODE="${PAIR_MATCH_MODE:-nn-greedy}"
export PAIR_MAX_ROWS="${PAIR_MAX_ROWS:--1}"
export PAIR_MAX_DISTANCE="${PAIR_MAX_DISTANCE:--1}"

export SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
export NUM_SAMPLES="${NUM_SAMPLES:--1}"
export K_TARGET="${K_TARGET:-64}"
export P_OVER="${P_OVER:-10}"
export PARTITIONS="${PARTITIONS:-}"
export JAC_METHOD="${JAC_METHOD:-vjp}"
export JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
export JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
export JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"
export INJECT_MODE="${INJECT_MODE:-auto}"

# ---- Eval sweep knobs (seed=99 rollouts, like ctrlwam SWEEP_BASE_SEED) ----
export TASK_RANGE_START="${TASK_RANGE_START:-7}"
export TASK_RANGE_END="${TASK_RANGE_END:-8}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
export SWEEP_BASE_SEED="${SWEEP_BASE_SEED:-99}"
export EVAL_PERTURB_SPEC="${EVAL_PERTURB_SPEC:-scripts/lqr/configs/eval_init_pos.yaml}"
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
source scripts/lqr/slurm_port.sh
export FORCE="${FORCE:-0}"
export BUILD_ONLY="${BUILD_ONLY:-0}"
export SWEEP_ONLY="${SWEEP_ONLY:-0}"
export PROMPT="${PROMPT:-}"

# ---- Output layout ----
SWEEP_TAG="${SWEEP_TAG:-init_pos_task${TASK_ID}}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/lqr_sweep/${SWEEP_TAG}}"
PAIRS_RAW_DIR="${PAIRS_RAW_DIR:-${SWEEP_ROOT}/pairs_init_pos_raw}"
PAIRED_DIR="${PAIRED_DIR:-${SWEEP_ROOT}/pairs_init_pos_paired}"
SVD_DIR="${SVD_DIR:-${SWEEP_ROOT}/svd}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:-${SWEEP_ROOT}/eval_sweep_seed${SWEEP_BASE_SEED}}"
SWEEP_RESULTS_MD="${SWEEP_RESULTS_MD:-${EVAL_SWEEP_ROOT}/sweep_success_rates.md}"
LOCK_DIR="${SWEEP_ROOT}/.locks"

BASE_GRID=72
EXTRA_Q_COUNT=5
TOTAL=$((BASE_GRID + EXTRA_Q_COUNT + 1))

# ctrlwam sweep_lqr_gripper_xyz.sh base grid
LAMBDAS=(0.5 5.0 10.0 15.0)
QS=(1.0 10.0)
RINITS=(5.0 10.0 20.0)
TAUS=(3.0 5.0 7.0)
EXTRA_QS=(100 200 1000 2000 10000)

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
  local -n _T="$5"

  if (( idx < 0 || idx >= TOTAL )); then
    echo "[error] combo index ${idx} out of range [0, $((TOTAL - 1))]" >&2
    return 1
  fi

  if (( idx < BASE_GRID )); then
    local nT=${#TAUS[@]}
    local nR=${#RINITS[@]}
    local nQ=${#QS[@]}

    local t_i=$((idx % nT))
    local tmp=$((idx / nT))
    local r_i=$((tmp % nR))
    tmp=$((tmp / nR))
    local q_i=$((tmp % nQ))
    local l_i=$((tmp / nQ))

    _L="${LAMBDAS[$l_i]}"
    _Q="${QS[$q_i]}"
    _RI="${RINITS[$r_i]}"
    _T="${TAUS[$t_i]}"
    return 0
  fi

  local extra_idx=$((idx - BASE_GRID))
  if (( extra_idx < EXTRA_Q_COUNT )); then
    _L="15.0"
    _Q="${EXTRA_QS[$extra_idx]}.0"
    _RI="5.0"
    _T="3.0"
    return 0
  fi

  _L="20.0"
  _Q="1.0"
  _RI="10.0"
  _T="5.0"
}

combo_tag() {
  local idx="$1"
  local lambda_scale="$2"
  local q_scale="$3"
  local r_scale="$4"
  local r_scale_tau="$5"
  local tag="idx${idx}_lam$(sanitize_tag_component "${lambda_scale}")_q$(sanitize_tag_component "${q_scale}")_r$(sanitize_tag_component "${r_scale}")_tau$(sanitize_tag_component "${r_scale_tau}")"
  printf '%s' "${tag}"
}

artifacts_ready() {
  [[ -f "${SVD_DIR}/svd_summary.pt" ]] \
    && [[ -f "${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt" ]]
}

ensure_artifacts() {
  mkdir -p "${LOCK_DIR}" "${SWEEP_ROOT}"

  if artifacts_ready; then
    echo "[build] artifacts already present under ${SVD_DIR}"
    return 0
  fi

  if [[ -n "${EXISTING_PAIRS_ALL_DIR:-}" ]]; then
    echo "[build] skip collect/pair; reuse EXISTING_PAIRS_ALL_DIR=${EXISTING_PAIRS_ALL_DIR}"
    export EXISTING_PAIRS_ALL_DIR
    export PAIRS_DIR="${EXISTING_PAIRS_ALL_DIR}"
    export PAIRED_DIR="${EXISTING_PAIRS_ALL_DIR}"
    export PAIRS_ALL_DIR="${EXISTING_PAIRS_ALL_DIR}"
    export SVD_DIR
    export SKIP_EVAL=1
    export ACTIVATE_CONDA=0
    bash scripts/lqr/run_lqr_pipeline.sh
    return 0
  fi

  echo "[build] running collect -> pair -> SVD -> jacobian under ${SWEEP_ROOT}"
  export OUT_BASE="${SWEEP_ROOT}"
  export PAIRS_DIR="${PAIRS_RAW_DIR}"
  export PAIRED_DIR="${PAIRED_DIR}"
  export PAIRS_ALL_DIR="${PAIRED_DIR}"
  export SVD_DIR
  export SKIP_EVAL=1
  export ACTIVATE_CONDA=0
  bash scripts/lqr/run_lqr_pipeline.sh
}

ensure_artifacts_with_lock() {
  mkdir -p "${LOCK_DIR}"
  (
    flock -w 7200 200 || exit 1
    ensure_artifacts
  ) 200>"${LOCK_DIR}/build.lock"

  local deadline=$((SECONDS + 7200))
  until artifacts_ready; do
    if (( SECONDS > deadline )); then
      echo "[error] timed out waiting for artifacts under ${SVD_DIR}" >&2
      exit 2
    fi
    echo "[build] waiting for artifacts under ${SVD_DIR} ..."
    sleep 30
  done
}

run_one_combo() {
  local idx="$1"
  local lambda_scale q_scale r_scale r_scale_tau
  decode_combo "${idx}" lambda_scale q_scale r_scale r_scale_tau

  local tag
  tag="$(combo_tag "${idx}" "${lambda_scale}" "${q_scale}" "${r_scale}" "${r_scale_tau}")"
  local cfg_dir="${EVAL_SWEEP_ROOT}/configs/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_JOB_ID:-local}"
  local cfg_path="${cfg_dir}/lqr_${tag}.yaml"
  local out_dir="${EVAL_SWEEP_ROOT}/${tag}"

  if [[ "${FORCE}" != "1" && -f "${out_dir}/summary.json" ]]; then
    echo "[sweep] SKIP combo ${idx}/${TOTAL} (${tag}) — summary.json exists"
    return 0
  fi

  mkdir -p "${cfg_dir}" "${out_dir}"
  cat > "${cfg_path}" <<EOF
lambda_scale: ${lambda_scale}
q_scale: ${q_scale}
r_scale: ${r_scale}
r_scale_tau: ${r_scale_tau}
r_scale_final: 1000000000.0
max_chunks: 50
qf_scale: 1.0
jac_dir_act: ${JAC_SUBDIR}
modality: action
inject_mode: ${INJECT_MODE}
EOF

  echo "===== LQR sweep combo ${idx}/${TOTAL} ====="
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "SVD_DIR=${SVD_DIR}"
  echo "EVAL_PERTURB_SPEC=${EVAL_PERTURB_SPEC}"
  echo "lambda_scale=${lambda_scale}"
  echo "q_scale=${q_scale}"
  echo "r_scale=${r_scale}"
  echo "r_scale_tau=${r_scale_tau}"
  echo "config=${cfg_path}"
  echo "out_dir=${out_dir}"
  echo "port=${PORT}"
  echo "==========================================="

  local eval_cmd=(
    python scripts/lqr/run_libero_lqr_eval.py
    --config-name "${CONFIG_NAME}"
    --libero-benchmark "${LIBERO_BENCHMARK}"
    --task-range "${TASK_RANGE_START}" "${TASK_RANGE_END}"
    --num-episodes "${EVAL_NUM_EPISODES}"
    --startup-wait-sec "${EVAL_STARTUP_WAIT_SEC}"
    --port "${PORT}"
    --svd-dir "${SVD_DIR}"
    --jac-dir-act "${JAC_SUBDIR}"
    --lqr-config "${cfg_path}"
    --inject-mode "${INJECT_MODE}"
    --perturb-spec "${EVAL_PERTURB_SPEC}"
    --out-dir "${out_dir}"
  )
  if [[ -n "${PROMPT}" ]]; then
    eval_cmd+=(--prompt "${PROMPT}")
  fi
  "${eval_cmd[@]}"
}

aggregate_sweep_results() {
  mkdir -p "${EVAL_SWEEP_ROOT}"
  local lock_file="${EVAL_SWEEP_ROOT}/.sweep_aggregate.lock"
  echo "[aggregate] writing success rates -> ${SWEEP_RESULTS_MD}"
  (
    flock -w 300 9 || exit 1
    python scripts/lqr/aggregate_sweep_results.py \
      --out-root "${EVAL_SWEEP_ROOT}" \
      --output-md "${SWEEP_RESULTS_MD}"
  ) 9>"${lock_file}"
}

echo "LingBot init-position LQR parameter sweep"
echo "  REPO=$(pwd)"
echo "  TASK_ID=${TASK_ID}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END})"
echo "  SWEEP_ROOT=${SWEEP_ROOT}"
echo "  SVD_DIR=${SVD_DIR}"
echo "  EVAL_SWEEP_ROOT=${EVAL_SWEEP_ROOT}"
echo "  TOTAL_COMBOS=${TOTAL} (72 base + 5 extra-q + 1 outlier)"
echo "  EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}"
echo "  BUILD_ONLY=${BUILD_ONLY} SWEEP_ONLY=${SWEEP_ONLY} FORCE=${FORCE}"
echo "  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>}"

if [[ "${SWEEP_ONLY}" != "1" ]]; then
  ensure_artifacts_with_lock
fi

if [[ "${BUILD_ONLY}" == "1" ]]; then
  echo "[done] BUILD_ONLY=1; artifacts ready under ${SVD_DIR}"
  exit 0
fi

if ! artifacts_ready; then
  echo "[error] missing artifacts under ${SVD_DIR}" >&2
  echo "        run with BUILD_ONLY=1 first, or set EXISTING_PAIRS_ALL_DIR / SVD_DIR" >&2
  exit 2
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_one_combo "${SLURM_ARRAY_TASK_ID}"
  aggregate_sweep_results
else
  for ((i = 0; i < TOTAL; i++)); do
    run_one_combo "${i}"
  done
  aggregate_sweep_results
fi
