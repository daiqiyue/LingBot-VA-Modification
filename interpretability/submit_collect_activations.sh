#!/usr/bin/env bash
# Parallel sbatch launcher for activation collection + pairing.
#
# Submits a single SLURM array of (NUM_TASKS × NUM_SHARDS) GPU workers.
# Array index i maps to:
#   task_idx  = i / NUM_SHARDS   -> which task ID
#   shard_idx = i % NUM_SHARDS   -> which episode slice
#
# A single dependent CPU merge job then runs per-task manifest merges and
# build_all_pairs.py, producing:
#   RUN_DIR/task_<ID>/merged/positive.pt
#   RUN_DIR/task_<ID>/merged/negative.pt
#
# Usage:
#   TASK_IDS="0,1,2" NUM_EPISODES=20 NUM_SHARDS=4 \
#   PERTURB_SPEC=scripts/lqr/configs/perturb_spec_camera.yaml \
#       bash scripts/lqr/submit_collect_activations.sh
#
#   # single task (backwards-compatible)
#   TASK_IDS=0 NUM_EPISODES=20 bash scripts/lqr/submit_collect_activations.sh
#
#   # re-run only the merge on an existing run
#   EXISTING_RUN_DIR=outputs/lqr_activations/collect_... MERGE_ONLY=1 \
#   TASK_IDS="0,1,2" NUM_SHARDS=4 NUM_EPISODES=20 \
#       bash scripts/lqr/submit_collect_activations.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# --- collection knobs --------------------------------------------------------
CONFIG_NAME="${CONFIG_NAME:-libero}"
LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
TASK_IDS="${TASK_IDS:-${TASK_ID:-0}}"          # comma or space separated
NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BASE_SEED="${BASE_SEED:-0}"
TOPK_INFER_PER_TRAJ="${TOPK_INFER_PER_TRAJ:-10}"
SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
MODE="${MODE:-action}"
PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"
PAIR_SEED="${PAIR_SEED:-0}"
DISABLE_VIDEO="${DISABLE_VIDEO:-1}"
NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR:-}"   # path to a submit_collect_nominal.sh run_dir
EXTRA_MERGE_DEPENDENCY="${EXTRA_MERGE_DEPENDENCY:-}" # optional SLURM job id(s), colon separated

# --- output / resume ---------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$REPO_ROOT/outputs/lqr_activations_init_pos}"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/collect_${TS}}"
MERGE_ONLY="${MERGE_ONLY:-0}"
EXISTING_RUN_DIR="${EXISTING_RUN_DIR:-}"

# --- slurm resources ---------------------------------------------------------
ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-${SLURM_ACCOUNT:-}}}"
PARTITION_SLURM="${PARTITION_SLURM:-${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}}"
QOS="${QOS:-${SLURM_JOB_QOS:-${SLURM_QOS:-}}}"
WORKER_TIME="${WORKER_TIME:-01:00:00}"
MERGE_TIME="${MERGE_TIME:-00:20:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"

SBATCH_ACCOUNT_ARGS=()
[[ -n "$ACCOUNT" ]] && SBATCH_ACCOUNT_ARGS+=(--account="$ACCOUNT")
SBATCH_PARTITION_ARGS=()
[[ -n "$PARTITION_SLURM" ]] && SBATCH_PARTITION_ARGS+=(--partition="$PARTITION_SLURM")
SBATCH_QOS_ARGS=()
[[ -n "$QOS" ]] && SBATCH_QOS_ARGS+=(--qos="$QOS")
SBATCH_GPU_ARGS=()
[[ -n "$GPU_GRES" ]] && SBATCH_GPU_ARGS+=(--gres="$GPU_GRES")

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/storage/scratch1/9/qdai41/.conda/envs/lingbot}"
_NVIDIA_PFX="$CONDA_ENV_PATH/lib/python3.10/site-packages/nvidia"

# --- normalise task list and derive dimensions --------------------------------
# accept "0,1,2" or "0 1 2" or "0, 1, 2"
TASK_IDS_STR="$(echo "$TASK_IDS" | tr ',' ' ' | tr -s ' ')"
read -ra TASK_ID_ARRAY <<< "$TASK_IDS_STR"
NUM_TASKS="${#TASK_ID_ARRAY[@]}"
TOTAL_JOBS=$(( NUM_TASKS * NUM_SHARDS ))
LAST_JOB=$(( TOTAL_JOBS - 1 ))
EPS_PER_SHARD=$(( (NUM_EPISODES + NUM_SHARDS - 1) / NUM_SHARDS ))

if [[ -n "$EXISTING_RUN_DIR" ]]; then
    RUN_DIR="$EXISTING_RUN_DIR"
fi
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== submit_collect_activations ==="
echo "  TASK_IDS       : ${TASK_IDS_STR}  (${NUM_TASKS} task(s))"
echo "  PERTURB_SPEC   : $PERTURB_SPEC"
echo "  NUM_EPISODES   : $NUM_EPISODES  x${NUM_SHARDS} shards = ${EPS_PER_SHARD} eps/shard"
echo "  TOTAL_JOBS     : $TOTAL_JOBS  (array 0-${LAST_JOB})"
echo "  BASE_SEED      : $BASE_SEED"
echo "  RUN_DIR        : $RUN_DIR"
echo "  NOMINAL_RUN_DIR: ${NOMINAL_RUN_DIR:-<none — positives must be in perturb spec>}"
echo "  MERGE_ONLY     : $MERGE_ONLY"
echo "  account        : $ACCOUNT  partition=$PARTITION_SLURM"
echo

# ---------- 1) collection array (all tasks × shards) -------------------------
if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "[skip] MERGE_ONLY=1 — skipping collection array."
    COLLECT_JOB_ID=""
else
    COLLECT_JOB_ID=$(sbatch --parsable \
        "${SBATCH_ACCOUNT_ARGS[@]}" \
        "${SBATCH_PARTITION_ARGS[@]}" \
        "${SBATCH_QOS_ARGS[@]}" \
        --job-name="lingact_collect_${TS}" \
        --array="0-${LAST_JOB}" \
        "${SBATCH_GPU_ARGS[@]}" \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --time="$WORKER_TIME" \
        --output="$LOG_DIR/collect_%a_%A.out" \
        --error="$LOG_DIR/collect_%a_%A.err" \
        ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
        --wrap "
set -eo pipefail
source ~/.bashrc
set -u
conda activate '$CONDA_ENV_PATH'
export CUDA_HOME='$_NVIDIA_PFX/cuda_runtime'
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LD_LIBRARY_PATH='$_NVIDIA_PFX/cuda_runtime/lib:$_NVIDIA_PFX/cudnn/lib':\${LD_LIBRARY_PATH:-}
cd '$REPO_ROOT'
export PYTHONPATH='$REPO_ROOT'${PYTHONPATH:+:\$PYTHONPATH}

TASK_ID_LIST=($TASK_IDS_STR)
TASK_IDX=\$(( SLURM_ARRAY_TASK_ID / $NUM_SHARDS ))
SHARD_IDX=\$(( SLURM_ARRAY_TASK_ID % $NUM_SHARDS ))
TASK_ID=\${TASK_ID_LIST[\$TASK_IDX]}

SHARD_SEED=\$(( $BASE_SEED + SHARD_IDX * $EPS_PER_SHARD ))

# clamp last shard to not overshoot NUM_EPISODES
SHARD_EPS=$EPS_PER_SHARD
REMAINING=\$(( $NUM_EPISODES - SHARD_IDX * $EPS_PER_SHARD ))
(( REMAINING < SHARD_EPS )) && SHARD_EPS=\$REMAINING

SHARD_DIR='$RUN_DIR'/task_\${TASK_ID}/shard_\${SHARD_IDX}
mkdir -p \"\$SHARD_DIR\"

echo \"[array \$SLURM_ARRAY_TASK_ID] task=\$TASK_ID shard=\$SHARD_IDX seed=\$SHARD_SEED episodes=\$SHARD_EPS\"
nvidia-smi -L

COLLECT_ARGS=(
    python interpretability/run_collect_inputs.py
    --config-name '$CONFIG_NAME'
    --libero-benchmark '$LIBERO_BENCHMARK'
    --task-id \"\$TASK_ID\"
    --num-episodes \"\$SHARD_EPS\"
    --top-k-inference-per-traj '$TOPK_INFER_PER_TRAJ'
    --selected-timesteps '$SELECTED_TIMESTEPS'
    --mode '$MODE'
    --perturb-spec '$PERTURB_SPEC'
    --seed \"\$SHARD_SEED\"
    --out-dir \"\$SHARD_DIR\"
)
$([ "$DISABLE_VIDEO" = "1" ] && echo "COLLECT_ARGS+=(--disable-video)")
\"\${COLLECT_ARGS[@]}\"
echo \"[array \$SLURM_ARRAY_TASK_ID] done.\"
")

    echo "  collect array job : $COLLECT_JOB_ID  (array 0-${LAST_JOB})"
fi

# ---------- 2) merge + build_all_pairs per task (CPU, depends on array) ------
DEP_IDS=()
[[ -n "$COLLECT_JOB_ID" ]] && DEP_IDS+=("$COLLECT_JOB_ID")
if [[ -n "$EXTRA_MERGE_DEPENDENCY" ]]; then
    IFS=':' read -ra EXTRA_DEP_IDS <<< "$EXTRA_MERGE_DEPENDENCY"
    for dep_id in "${EXTRA_DEP_IDS[@]}"; do
        [[ -n "$dep_id" ]] && DEP_IDS+=("$dep_id")
    done
fi
DEP_FLAG=""
if (( ${#DEP_IDS[@]} > 0 )); then
    DEP_JOINED="$(IFS=:; echo "${DEP_IDS[*]}")"
    DEP_FLAG="--dependency=afterany:${DEP_JOINED}"
fi

MERGE_JOB_ID=$(sbatch --parsable \
    "${SBATCH_ACCOUNT_ARGS[@]}" \
    "${SBATCH_PARTITION_ARGS[@]}" \
    "${SBATCH_QOS_ARGS[@]}" \
    "${SBATCH_GPU_ARGS[@]}" \
    $DEP_FLAG \
    --job-name="lingact_merge_${TS}" \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time="$MERGE_TIME" \
    --output="$LOG_DIR/merge_%j.out" \
    --error="$LOG_DIR/merge_%j.err" \
    --wrap "
set -eo pipefail
source ~/.bashrc
set -u
conda activate '$CONDA_ENV_PATH'
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
cd '$REPO_ROOT'
export PYTHONPATH='$REPO_ROOT'${PYTHONPATH:+:\$PYTHONPATH}

python - <<'PYEOF'
import json, subprocess, sys
from pathlib import Path

run_dir         = Path('$RUN_DIR')
nominal_run_dir = Path('$NOMINAL_RUN_DIR') if '$NOMINAL_RUN_DIR' else None
task_ids        = [int(t) for t in '$TASK_IDS_STR'.split()]
n_shards        = $NUM_SHARDS
num_eps         = $NUM_EPISODES
pair_seed       = $PAIR_SEED

paired  = []
skipped = []

for task_id in task_ids:
    task_dir   = run_dir / f'task_{task_id}'
    merged_dir = task_dir / 'merged'
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f'\\n[merge] task {task_id} — combining {n_shards} perturb shard(s)...', flush=True)
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
        print(f'  SKIP task {task_id}: no shard manifests found', flush=True)
        skipped.append((task_id, 'no shard manifests'))
        continue

    if nominal_run_dir is not None:
        nom_manifest_path = nominal_run_dir / f'task_{task_id}' / 'merged' / 'manifest.json'
        if not nom_manifest_path.exists():
            print(f'  SKIP task {task_id}: nominal manifest not found: {nom_manifest_path}', flush=True)
            skipped.append((task_id, 'nominal manifest missing'))
            continue
        nom = json.loads(nom_manifest_path.read_text(encoding='utf-8'))
        nom_records = nom.get('records', [])
        all_records = nom_records + all_records
        print(f'  nominal: {len(nom_records)} records injected from {nom_manifest_path}', flush=True)

    perturb_records = [r for r in all_records if r.get('variant_name') != 'nominal']
    n_fail = sum(1 for r in perturb_records if not r.get('trajectory_success', True))
    n_succ = sum(1 for r in perturb_records if     r.get('trajectory_success', True))
    nom_succ = sum(1 for r in all_records if r.get('variant_name') == 'nominal' and r.get('trajectory_success', False))
    print(f'  perturb: {n_succ} success / {n_fail} failure   nominal success: {nom_succ}', flush=True)

    if n_fail == 0:
        print(f'  SKIP task {task_id}: no perturb failures — collect more episodes', flush=True)
        skipped.append((task_id, f'0 perturb failures ({n_succ} successes)'))
        continue
    if nom_succ == 0:
        print(f'  SKIP task {task_id}: no nominal successes', flush=True)
        skipped.append((task_id, 'no nominal successes'))
        continue

    base_manifest['records']      = all_records
    base_manifest['num_episodes'] = num_eps
    out = merged_dir / 'manifest.json'
    out.write_text(json.dumps(base_manifest, indent=2), encoding='utf-8')
    print(f'  combined manifest: {len(all_records)} total records -> {out}', flush=True)

    print(f'  running build_all_pairs...', flush=True)
    subprocess.run([
        'python', 'scripts/lqr/build_all_pairs.py',
        '--collect-dir', str(merged_dir),
        '--out-dir',     str(merged_dir),
        '--pair-seed',   str(pair_seed),
    ], check=True)
    paired.append(task_id)
    print(f'  positive.pt / negative.pt written to {merged_dir}', flush=True)

print(f'\\n[merge] done — paired: {paired}', flush=True)
if skipped:
    print(f'[merge] skipped: {skipped}', flush=True)
PYEOF
")

echo "  merge job         : $MERGE_JOB_ID${COLLECT_JOB_ID:+ (afterany:$COLLECT_JOB_ID)}"
echo
echo "=== submitted ==="
echo "  run_dir : $RUN_DIR"
for tid in "${TASK_ID_ARRAY[@]}"; do
    printf "  task %-3s : %s/task_%s/merged/{positive,negative}.pt\n" "$tid" "$RUN_DIR" "$tid"
done
echo
echo "Monitor:      squeue -u \$USER | grep lingact"
[[ -n "$COLLECT_JOB_ID" ]] && echo "Tail collect: tail -f $LOG_DIR/collect_0_${COLLECT_JOB_ID}.out"
echo "Tail merge:   tail -f $LOG_DIR/merge_${MERGE_JOB_ID}.out"
