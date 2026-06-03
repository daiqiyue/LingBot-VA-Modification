#!/usr/bin/env bash
# Parallel sbatch launcher for nominal (unperturbed) activation collection.
#
# Runs run_collect_inputs.py with the nominal-only perturb spec across
# NUM_TASKS × NUM_SHARDS GPU workers, then a dependent merge job stitches
# the per-shard manifests into one manifest.json per task.
#
# The output is meant to be consumed by submit_collect_activations.sh via
# its NOMINAL_RUN_DIR parameter, which injects the nominal records into
# the pairing step for any perturb spec.
#
# Usage:
#   TASK_IDS="0,1,2,3,4,5,6,7,8,9" NUM_EPISODES=20 NUM_SHARDS=4 \
#       bash scripts/lqr/submit_collect_nominal.sh
#
#   # resume merge only
#   EXISTING_RUN_DIR=outputs/lqr_nominal/collect_... MERGE_ONLY=1 \
#   TASK_IDS="0,1,2,3,4,5,6,7,8,9" NUM_SHARDS=4 NUM_EPISODES=20 \
#       bash scripts/lqr/submit_collect_nominal.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# --- collection knobs --------------------------------------------------------
CONFIG_NAME="${CONFIG_NAME:-libero}"
LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
TASK_IDS="${TASK_IDS:-${TASK_ID:-0}}"
NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BASE_SEED="${BASE_SEED:-0}"
TOPK_INFER_PER_TRAJ="${TOPK_INFER_PER_TRAJ:-10}"
SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
MODE="${MODE:-action}"
DISABLE_VIDEO="${DISABLE_VIDEO:-1}"

NOMINAL_SPEC="$REPO_ROOT/scripts/lqr/configs/perturb_spec_nominal.yaml"

# --- output / resume ---------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$REPO_ROOT/outputs/lqr_nominal}"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/collect_${TS}}"
MERGE_ONLY="${MERGE_ONLY:-0}"
EXISTING_RUN_DIR="${EXISTING_RUN_DIR:-}"

# --- slurm resources ---------------------------------------------------------
ACCOUNT="${ACCOUNT:-bhhv-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
WORKER_TIME="${WORKER_TIME:-01:00:00}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

CONDA_ENV_PATH="/projects/bhhv/jskifstad/LingBot-VA-Modification/.conda/envs/ling"
_NVIDIA_PFX="$CONDA_ENV_PATH/lib/python3.10/site-packages/nvidia"

# --- normalise task list -----------------------------------------------------
TASK_IDS_STR="$(echo "$TASK_IDS" | tr ',' ' ' | tr -s ' ')"
read -ra TASK_ID_ARRAY <<< "$TASK_IDS_STR"
NUM_TASKS="${#TASK_ID_ARRAY[@]}"
TOTAL_JOBS=$(( NUM_TASKS * NUM_SHARDS ))
LAST_JOB=$(( TOTAL_JOBS - 1 ))
EPS_PER_SHARD=$(( (NUM_EPISODES + NUM_SHARDS - 1) / NUM_SHARDS ))

[[ -n "$EXISTING_RUN_DIR" ]] && RUN_DIR="$EXISTING_RUN_DIR"
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== submit_collect_nominal ==="
echo "  TASK_IDS       : ${TASK_IDS_STR}  (${NUM_TASKS} task(s))"
echo "  NUM_EPISODES   : $NUM_EPISODES  x${NUM_SHARDS} shards = ${EPS_PER_SHARD} eps/shard"
echo "  TOTAL_JOBS     : $TOTAL_JOBS  (array 0-${LAST_JOB})"
echo "  BASE_SEED      : $BASE_SEED"
echo "  RUN_DIR        : $RUN_DIR"
echo "  MERGE_ONLY     : $MERGE_ONLY"
echo "  account        : $ACCOUNT  partition=$PARTITION_SLURM"
echo

# ---------- 1) collection array ----------------------------------------------
if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "[skip] MERGE_ONLY=1 — skipping collection array."
    COLLECT_JOB_ID=""
else
    COLLECT_JOB_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --job-name="lingnominal_collect_${TS}" \
        --array="0-${LAST_JOB}" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --time="$WORKER_TIME" \
        --output="$LOG_DIR/collect_%a_%A.out" \
        --error="$LOG_DIR/collect_%a_%A.err" \
        ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
        --wrap "
set -euo pipefail
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate '$CONDA_ENV_PATH'
export CUDA_HOME='$_NVIDIA_PFX/cuda_runtime'
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LD_LIBRARY_PATH='$_NVIDIA_PFX/cuda_runtime/lib:$_NVIDIA_PFX/cudnn/lib':\${LD_LIBRARY_PATH:-}
cd '$REPO_ROOT'
export PYTHONPATH='$REPO_ROOT'\${PYTHONPATH:+:\$PYTHONPATH}

TASK_ID_LIST=($TASK_IDS_STR)
TASK_IDX=\$(( SLURM_ARRAY_TASK_ID / $NUM_SHARDS ))
SHARD_IDX=\$(( SLURM_ARRAY_TASK_ID % $NUM_SHARDS ))
TASK_ID=\${TASK_ID_LIST[\$TASK_IDX]}

SHARD_SEED=\$(( $BASE_SEED + SHARD_IDX * $EPS_PER_SHARD ))

SHARD_EPS=$EPS_PER_SHARD
REMAINING=\$(( $NUM_EPISODES - SHARD_IDX * $EPS_PER_SHARD ))
(( REMAINING < SHARD_EPS )) && SHARD_EPS=\$REMAINING

SHARD_DIR='$RUN_DIR'/task_\${TASK_ID}/shard_\${SHARD_IDX}
mkdir -p \"\$SHARD_DIR\"

echo \"[array \$SLURM_ARRAY_TASK_ID] task=\$TASK_ID shard=\$SHARD_IDX seed=\$SHARD_SEED episodes=\$SHARD_EPS\"
nvidia-smi -L

COLLECT_ARGS=(
    python scripts/lqr/run_collect_inputs.py
    --config-name '$CONFIG_NAME'
    --libero-benchmark '$LIBERO_BENCHMARK'
    --task-id \"\$TASK_ID\"
    --num-episodes \"\$SHARD_EPS\"
    --top-k-inference-per-traj '$TOPK_INFER_PER_TRAJ'
    --selected-timesteps '$SELECTED_TIMESTEPS'
    --mode '$MODE'
    --perturb-spec '$NOMINAL_SPEC'
    --seed \"\$SHARD_SEED\"
    --out-dir \"\$SHARD_DIR\"
)
$([ "$DISABLE_VIDEO" = "1" ] && echo "COLLECT_ARGS+=(--disable-video)")
\"\${COLLECT_ARGS[@]}\"
echo \"[array \$SLURM_ARRAY_TASK_ID] done.\"
")

    echo "  collect array job : $COLLECT_JOB_ID  (array 0-${LAST_JOB})"
fi

# ---------- 2) merge manifests per task (no pairing) -------------------------
DEP_FLAG=""
[[ -n "$COLLECT_JOB_ID" ]] && DEP_FLAG="--dependency=afterok:${COLLECT_JOB_ID}"

MERGE_JOB_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    $DEP_FLAG \
    --job-name="lingnominal_merge_${TS}" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time="$MERGE_TIME" \
    --output="$LOG_DIR/merge_%j.out" \
    --error="$LOG_DIR/merge_%j.err" \
    --wrap "
set -euo pipefail
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate '$CONDA_ENV_PATH'
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
cd '$REPO_ROOT'
export PYTHONPATH='$REPO_ROOT'\${PYTHONPATH:+:\$PYTHONPATH}

python - <<'PYEOF'
import json
from pathlib import Path

run_dir  = Path('$RUN_DIR')
task_ids = [int(t) for t in '$TASK_IDS_STR'.split()]
n_shards = $NUM_SHARDS
num_eps  = $NUM_EPISODES

for task_id in task_ids:
    task_dir   = run_dir / f'task_{task_id}'
    merged_dir = task_dir / 'merged'
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f'\\n[merge] task {task_id} — combining {n_shards} shard(s)...', flush=True)
    all_records   = []
    base_manifest = None
    for s in range(n_shards):
        mpath = task_dir / f'shard_{s}' / 'manifest.json'
        if not mpath.exists():
            print(f'  WARNING: missing {mpath}, skipping', flush=True)
            continue
        m = json.loads(mpath.read_text(encoding='utf-8'))
        if base_manifest is None:
            base_manifest = {k: v for k, v in m.items() if k != 'records'}
        all_records.extend(m.get('records', []))
        print(f'  shard_{s}: {len(m.get(\"records\", []))} records', flush=True)

    if base_manifest is None:
        print(f'  ERROR: no shard manifests for task {task_id}', flush=True)
        continue

    base_manifest['records']      = all_records
    base_manifest['num_episodes'] = num_eps
    out = merged_dir / 'manifest.json'
    out.write_text(json.dumps(base_manifest, indent=2), encoding='utf-8')
    print(f'  wrote {len(all_records)} records -> {out}', flush=True)

print('\\n[merge] done.', flush=True)
PYEOF
")

echo "  merge job         : $MERGE_JOB_ID${COLLECT_JOB_ID:+ (afterok:$COLLECT_JOB_ID)}"
echo
echo "=== submitted ==="
echo "  run_dir : $RUN_DIR"
for tid in "${TASK_ID_ARRAY[@]}"; do
    printf "  task %-3s : %s/task_%s/merged/manifest.json\n" "$tid" "$RUN_DIR" "$tid"
done
echo
echo "Pass this run to submit_collect_activations.sh via:"
echo "  NOMINAL_RUN_DIR='$RUN_DIR' PERTURB_SPEC=... bash scripts/lqr/submit_collect_activations.sh"
echo
echo "Monitor:      squeue -u \$USER | grep lingnominal"
[[ -n "$COLLECT_JOB_ID" ]] && echo "Tail collect: tail -f $LOG_DIR/collect_0_${COLLECT_JOB_ID}.out"
echo "Tail merge:   tail -f $LOG_DIR/merge_${MERGE_JOB_ID}.out"
