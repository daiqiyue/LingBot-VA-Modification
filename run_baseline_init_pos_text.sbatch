#!/bin/bash
#SBATCH -J lingbot_policy_baseline_init_pos_text
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 24:00:00
#SBATCH -o Policy_Baseline_InitPos_Text-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# Pure policy rollout under init_position (gripper XYZ) perturbation with
# per-task prompts from task_text.txt (no LQR).
#
# Default: libero_10 tasks 1,2,3,7,9 — 20 episodes each.
#
# Examples:
#   sbatch run_baseline_init_pos_text.sbatch
#   sbatch --export=ALL,TASK_IDS="7" run_baseline_init_pos_text.sbatch

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification
export PYTHONPATH=.

export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"

export TASK_IDS="${TASK_IDS:-1 2 3 7 9}"
export NUM_EPISODES="${NUM_EPISODES:-20}"

export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_init_pos.yaml}"
export GRIPPER_XYZ_BASE_SEED="${GRIPPER_XYZ_BASE_SEED:-99}"

export TASK_TEXT_FILE="${TASK_TEXT_FILE:-task_text.txt}"
export RESUME="${RESUME:-0}"
export SAVE_ROOT="${SAVE_ROOT:-}"
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"

source scripts/lqr/slurm_port.sh
DEFAULT_MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"

TS="$(date +%Y%m%d_%H%M%S)"
TASK_TAG="$(echo "${TASK_IDS}" | tr ' ' '_')"
PERTURB_TAG="$(basename "${PERTURB_SPEC}" .yaml)"
PERTURB_TAG="${PERTURB_TAG#perturb_spec_}"
PERTURB_TAG="${PERTURB_TAG#eval_}"
export OUT_DIR="${OUT_DIR:-outputs/libero_baseline_${PERTURB_TAG}_text_seed${GRIPPER_XYZ_BASE_SEED}_tasks${TASK_TAG}_${TS}}"

get_task_prompt() {
  python - "$1" "${TASK_TEXT_FILE}" <<'PY'
import re
import sys

task_id = int(sys.argv[1])
text_path = sys.argv[2]
pattern = re.compile(rf"^Task\s+{task_id}:\s*(.+)\s*$")
for line in open(text_path, encoding="utf-8"):
    match = pattern.match(line.strip())
    if not match:
        continue
    prompt = match.group(1).strip()
    prompt = re.sub(r'^["\'\u201c\u201d]+|["\'\u201c\u201d]+$', "", prompt)
    print(prompt)
    break
else:
    raise SystemExit(f"no prompt found for task {task_id} in {text_path}")
PY
}

echo "===== Libero policy baseline (init_pos + task_text prompts, no LQR) ====="
echo "  REPO=$(pwd)"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  GRIPPER_XYZ_BASE_SEED=${GRIPPER_XYZ_BASE_SEED}"
echo "  TASK_TEXT_FILE=${TASK_TEXT_FILE}"
echo "  TASK_IDS=${TASK_IDS}"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  PORT=${PORT}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "==========================================================================="

FIRST_TASK=1
for TASK_ID in ${TASK_IDS}; do
  PROMPT="$(get_task_prompt "${TASK_ID}")"
  echo "----- task ${TASK_ID} prompt: ${PROMPT} -----"

  POLICY_EVAL_ARGS=(
    python scripts/lqr/run_libero_policy_eval.py
    --config-name "${CONFIG_NAME}"
    --libero-benchmark "${LIBERO_BENCHMARK}"
    --task-ids "${TASK_ID}"
    --num-episodes "${NUM_EPISODES}"
    --perturb-spec "${PERTURB_SPEC}"
    --gripper-xyz-base-seed "${GRIPPER_XYZ_BASE_SEED}"
    --prompt "${PROMPT}"
    --out-dir "${OUT_DIR}"
    --port "${PORT}"
    --master-port "${MASTER_PORT}"
    --startup-wait-sec "${EVAL_STARTUP_WAIT_SEC}"
    --save-root "${SAVE_ROOT}"
  )
  if [[ "${FIRST_TASK}" != "1" || "${RESUME}" == "1" ]]; then
    POLICY_EVAL_ARGS+=(--resume)
  fi

  "${POLICY_EVAL_ARGS[@]}"
  FIRST_TASK=0
done

echo "[done] summary: ${OUT_DIR}/summary.json"
