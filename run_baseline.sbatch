#!/bin/bash
#SBATCH -J lingbot_libero_baseline
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 20:00:00
#SBATCH -o Libero_Baseline_Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

conda activate lingbot
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/lingbot-va
export PYTHONPATH=.

# ---- client settings (first perturb in scripts/lqr/configs/eval_camera.yaml) ----
# camera_z_pos_10: camera_rotate_deg=10, camera_axis=z
START=1
END=2
LIBERO_BENCHMARK=libero_10
SERVER_PORT=29056
TEST_NUM=10
OUT_DIR=outputs/libero/task1_camera_z_pos_10
AGENTVIEW_CAMERA_ROTATE_DEG=10
AGENTVIEW_CAMERA_ROTATE_AXIS=z

# ---- server / client orchestration ----
MASTER_PORT=29061
SAVE_ROOT=visualization/
STARTUP_WAIT_SEC=1200

LOG_DIR="logs/libero_baseline_${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/server.log"

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap stop_server EXIT

# ---- from evaluation/libero/launch_server.sh (background) ----
mkdir -p "${SAVE_ROOT}"
echo "[server] starting wan_va_server on port ${SERVER_PORT}"
python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port "${MASTER_PORT}" \
  wan_va/wan_va_server.py \
  --config-name libero \
  --port "${SERVER_PORT}" \
  --save_root "${SAVE_ROOT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[server] waiting for port ${SERVER_PORT} (max ${STARTUP_WAIT_SEC}s)..."
deadline=$((SECONDS + STARTUP_WAIT_SEC))
until python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${SERVER_PORT})); s.close()" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "[error] server not ready" >&2
    tail -n 50 "${SERVER_LOG}" >&2 || true
    exit 2
  fi
  sleep 2
done
echo "[server] ready"

# ---- from evaluation/libero/launch_client.sh (foreground) ----
echo "[client] evaluation/libero/client.py task-range ${START} ${END} -> ${OUT_DIR}"
echo "[client] perturbation=camera_z_pos_10 rotate_deg=${AGENTVIEW_CAMERA_ROTATE_DEG} axis=${AGENTVIEW_CAMERA_ROTATE_AXIS}"
python evaluation/libero/client.py \
  --libero-benchmark "${LIBERO_BENCHMARK}" \
  --port "${SERVER_PORT}" \
  --test-num "${TEST_NUM}" \
  --task-range "${START}" "${END}" \
  --out-dir "${OUT_DIR}" \
  --agentview-camera-rotate-deg "${AGENTVIEW_CAMERA_ROTATE_DEG}" \
  --agentview-camera-rotate-axis "${AGENTVIEW_CAMERA_ROTATE_AXIS}"

echo "[done]"
