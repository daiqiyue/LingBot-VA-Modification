#!/bin/bash
#SBATCH -J lingbot_actadd
#SBATCH -A gts-gchou3
#SBATCH -N 1
#SBATCH --gres=gpu:h200:1
#SBATCH -t 20:00:00
#SBATCH -o Actadd_Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

conda activate lingbot
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/lingbot-va

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

# -------------------------------
# User-tunable knobs
# -------------------------------
PERTURB_SPEC="${PERTURB_SPEC:-scripts/activation_steering/configs/perturb_spec_init_pos.yaml}"
STEERING_CFG="${STEERING_CFG:-scripts/activation_steering/configs/steering_config.yaml}"
TOKEN_POLICY="${TOKEN_POLICY:-scripts/activation_steering/configs/token_policy.yaml}"

BENCHMARK="${BENCHMARK:-libero_10}"
PORT="${PORT:-29056}"
CONFIG_NAME="${CONFIG_NAME:-libero}"

# Default: task0 one-pass bank build.
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-1}"
TRACE_NUM_EPISODES="${TRACE_NUM_EPISODES:-1}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"

TS="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="outputs/actadd_onepass_${TS}"
TRACE_DIR="${WORK_DIR}/trace_bank"
STEERING_BANK="${WORK_DIR}/steering_bank.pt"
EVAL_DIR="${WORK_DIR}/steered_eval"

mkdir -p "${WORK_DIR}" "${TRACE_DIR}" "${EVAL_DIR}"

echo "[actadd] work dir: ${WORK_DIR}"
echo "[actadd] step1 one-pass build steering bank"
python scripts/activation_steering/run_libero_build_steering_bank_onepass.py \
  --config-name "${CONFIG_NAME}" \
  --libero-benchmark "${BENCHMARK}" \
  --task-range "${TASK_START}" "${TASK_END}" \
  --num-episodes "${TRACE_NUM_EPISODES}" \
  --port "${PORT}" \
  --trace-out-dir "${TRACE_DIR}" \
  --token-policy "${TOKEN_POLICY}" \
  --layers "15,19,22,25,27" \
  --modality both \
  --startup-wait-sec 240 \
  --perturb-spec "${PERTURB_SPEC}" \
  --phase-filter infer \
  --update-cache-filter 0 \
  --agg trimmed_mean \
  --normalize l2 \
  --out-path "${STEERING_BANK}"

echo "[actadd] step2 steered eval"
python scripts/activation_steering/run_libero_steered_eval.py \
  --config-name "${CONFIG_NAME}" \
  --libero-benchmark "${BENCHMARK}" \
  --task-range "${TASK_START}" "${TASK_END}" \
  --num-episodes "${EVAL_NUM_EPISODES}" \
  --port "${PORT}" \
  --steering-bank "${STEERING_BANK}" \
  --steering-config "${STEERING_CFG}" \
  --perturb-spec "${PERTURB_SPEC}" \
  --out-dir "${EVAL_DIR}"

echo "[actadd] done."
echo "[actadd] artifacts:"
echo "  steering bank:  ${STEERING_BANK}"
echo "  trace out dir:  ${TRACE_DIR}"
echo "  eval out dir:   ${EVAL_DIR}"
