# LingBot LQR Pipeline

This directory runs the LingBot LIBERO LQR workflow:

1. collect positive/negative policy inputs
2. optionally pair unaligned success/failure buckets by similarity
3. run SVD and write contrastive vectors
4. fit projected Jacobians
5. evaluate with the LQR injector

The init-position and Gaussian-noise perturbation paths now mirror the ctrlwam LQR flows:

- `init_position`: ctrlwam task07 gripper-XYZ flow. The collector runs gripper-perturbed rollouts, writes successful rollout rows to `positive.npz`, failed rollout rows to `negative.npz`, then `pair_inputs_by_similarity.py` builds row-aligned pairs before SVD.
- `gaussian`: ctrlwam noise_extreme flow. The collector runs clean rollouts, then builds `negative.npz` by adding Gaussian pixel noise to both primary and wrist images with sigma 90 by default.

Rows are captured once per LingBot policy inference chunk, not once per environment action step. During SVD, LingBot hooks `transformer.blocks[*]` and stores `output[0].reshape(-1)` at the selected denoising steps for the selected modality (`action` by default). For every row-aligned pair it forms:

```python
delta = activation_positive - activation_negative
```

For each layer and selected timestep, SVD/PCA pools activation deltas **within each ctrlwam-style partition** (default: 3 layer groups sharing one V per timestep). The contrastive vector stored in `svd_summary.pt` is still per `(layer, timestep)`:

```python
V_k = top-k right singular vectors of deltas
c_mean[layer, t] = mean(delta) @ V_k
mu = norm(c_mean)
v = c_mean / mu
```

At injection time the same LingBot block output is projected into `V_k`; the LQR control is:

```python
alpha = lambda_scale * mu - v @ x_proj
u_tilde = K @ (alpha * v)
delta_activation = V_next @ u_tilde
```

The intra-layer update is added to the next block output. The cross-step update is computed at the final block of one selected denoising step and applied at block 0 of the next selected denoising step, matching ctrlwam's A-LQR structure.

## Quick Start

Run the init-position full pipeline:

```bash
sbatch run_lqr_init_pos_ctrlwam.sbatch
```

Run the Gaussian-noise full pipeline:

```bash
sbatch run_lqr_gaussian_noise_ctrlwam.sbatch
```

Run locally or inside an interactive allocation:

```bash
cd /storage/scratch1/9/qdai41/cosmos/LingBot-VA-Modification

PERTURB_SPEC=scripts/lqr/configs/perturb_spec_init_pos.yaml \
TASK_ID=7 NUM_EPISODES=50 PAIR_INIT_BY_SIMILARITY=1 \
bash scripts/lqr/run_lqr_pipeline.sh

PERTURB_SPEC=scripts/lqr/configs/perturb_spec_gaussian.yaml \
TASK_ID=0 NUM_EPISODES=10 PAIR_INIT_BY_SIMILARITY=0 \
bash scripts/lqr/run_lqr_pipeline.sh
```

Set `SKIP_EVAL=1` to stop after SVD and Jacobian artifacts are written.

## Main Driver

`scripts/lqr/run_lqr_pipeline.sh` is the end-to-end entry point. It creates timestamped outputs under `outputs/lqr` unless overridden.

Important outputs:

- `PAIRS_DIR`: raw collected `positive.npz` and `negative.npz`
- `PAIRED_DIR`: similarity-paired init-position pairs, only used when `PAIR_INIT_BY_SIMILARITY=1` or `auto` detects an init spec
- `SVD_DIR`: SVD basis files, `svd_summary.pt`, `projected_diffs.pt`, and contrastive vectors
- `${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt`: projected Jacobians
- `EVAL_OUT_BASE`: rollout videos and success JSON from LQR eval

## Adjustable Parameters

Core collection:

- `CONFIG_NAME`: LingBot config name. Default `libero`.
- `LIBERO_BENCHMARK`: LIBERO suite. Default `libero_10`.
- `TASK_ID`: task used for collecting SVD/Jacobian data.
- `NUM_EPISODES`: number of initial states to roll out during collection.
- `PERTURB_SPEC`: perturbation YAML. Use `scripts/lqr/configs/perturb_spec_init_pos.yaml` or `scripts/lqr/configs/perturb_spec_gaussian.yaml`.
- `OUT_BASE`: base output directory. Default `outputs/lqr`.
- `TS`: timestamp suffix. Set this to make repeatable output paths.

Init-position pairing:

- `PAIR_INIT_BY_SIMILARITY`: `auto`, `1`, or `0`. Use `1` for init-position. Gaussian should use `0`.
- `PAIR_FEATURE`: similarity feature for init pairing. Choices: `proprio`, `proprio_raw`, `proprio+wrist`. Default `proprio`.
- `PAIR_MATCH_MODE`: matching algorithm. Choices: `nn-greedy`, `nn-replace`, `optimal`. Default `nn-greedy`.
- `PAIR_MAX_ROWS`: cap paired rows after sorting by distance. `-1` keeps all.
- `PAIR_MAX_DISTANCE`: drop pairs above this feature-space distance. `-1` disables filtering.

SVD and contrastive vectors:

- `COLLECT_MODE`: activation mode for SVD hooks. Usually `action`.
- `SELECTED_TIMESTEPS`: comma-separated denoising timesteps, e.g. `0,10,20,30,40`.
- `NUM_SAMPLES`: rows used for SVD. `-1` uses all available paired rows.
- `K_TARGET`: projection rank per partition/timestep. Default `64`, matching the ctrlwam LQR scripts.
- `P_OVER`: PCA oversampling rank. Default `10`, matching the ctrlwam LQR scripts.
- `PARTITIONS`: ctrlwam-style layer groups for shared V, e.g. `0-9,10-19,20-29`. Empty = auto 3 partitions (for L=30: `0-9,10-19,20-29`; for L=28: `0-9,10-18,19-27`).

Jacobian fit (ctrlwam-aligned VJP):

- `run_compute_jacobians.py` runs autograd VJPs through LingBot transformer blocks on a real LIBERO observation (default: row 0 of the paired `negative.npz`), matching ctrlwam's `compute_jacobians_full.py`.
- `JAC_METHOD`: `vjp` (default) or legacy `ridge` (requires `projected_diffs.pt`).
- `JAC_OBS_INDEX`: which row of the inputs NPZ to linearize around (default `0`).
- `JAC_NUM_SHARDS`: parallel layer shards for multi-GPU Jacobian collection (default `1`).
- `JAC_SUBDIR`: Jacobian artifact subdirectory under `SVD_DIR`.
- `B_tilde` is skipped by default, consistent with Cosmos-Policy.

LQR steering:

- `lambda_scale`: setpoint multiplier. Default `15.0`, matching ctrlwam task07 gripper rollout defaults.
- `q_scale`: state cost. Default `1.0`.
- `r_scale`: initial control cost `R_init`. Default `5.0`.
- `r_scale_tau`: exponential time constant in policy chunks. Default `5.0`.
- `r_scale_final`: upper clamp for `R(c)`. Default `1e9`.
- `max_chunks`: number of chunk-specific Riccati gains to precompute. Chunks beyond this use the last gain.
- `qf_scale`: terminal cost scale. Default `1.0`.

The active control cost follows ctrlwam exactly:

```python
R(c) = min(r_scale_final, r_scale * exp(c / r_scale_tau))
```

The server resets `c=0` on `VA_Server._reset(...)` and increments it once per policy `_infer(...)` chunk.

Evaluation:

- `SKIP_EVAL`: `1` skips LQR rollout evaluation.
- `TASK_RANGE_START`, `TASK_RANGE_END`: evaluated task range `[start, end)`.
- `EVAL_NUM_EPISODES`: evaluation episodes per task. Default `20`.
- `PORT`: WebSocket port for evaluation.
- `EVAL_STARTUP_WAIT_SEC`: server startup wait timeout.
- `INJECT_MODE`: LQR injection mode. Default `auto`.
- `PROMPT`: optional prompt override for evaluation.
- `LQR_CONFIG`: LQR controller YAML.

Reuse existing artifacts:

- `EXISTING_COLLECT_DIR`: directory with raw `positive.npz` and `negative.npz`. If this is an init-position run, the driver can still pair it by similarity.
- `EXISTING_PAIRS_ALL_DIR`: already row-aligned pairs. This skips collection and similarity pairing.
- `SVD_DIR`: explicit output or existing SVD directory.

## Pure Perturbation Evaluation

`evaluation/libero/client.py` runs perturbation evaluation without LQR. The perturbation application now matches the two ctrlwam flows:

- Gaussian noise is applied to both `agentview` and wrist images by default.
- Gaussian RNG defaults to `seed=episode_idx`, matching ctrlwam noise_extreme collection.
- Gripper init perturbation uses the same `xyz_random_xlarge_3` preset and 10 post-init wait steps.

Examples:

```bash
python evaluation/libero/client.py \
  --task-range 0 2 \
  --test-num 20 \
  --agentview-noise-sigma 90 \
  --out-dir outputs/libero_noise_extreme

python evaluation/libero/client.py \
  --task-range 7 8 \
  --test-num 20 \
  --gripper-xyz-preset xyz_random_xlarge_3 \
  --gripper-xyz-base-seed 42 \
  --out-dir outputs/libero_init_pos
```

Use `--no-noise-apply-wrist` only if you intentionally want the older agentview-only behavior.

## Manual Step Runs

Collect init-position success/failure buckets:

```bash
python scripts/lqr/run_collect_pairs.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-id 7 \
  --num-episodes 50 \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --out-dir outputs/lqr/pairs_init_pos_raw
```

Pair those buckets:

```bash
python scripts/lqr/pair_inputs_by_similarity.py \
  --in-dir outputs/lqr/pairs_init_pos_raw \
  --out-dir outputs/lqr/pairs_init_pos_paired \
  --feature proprio \
  --match-mode nn-greedy
```

Collect Gaussian row-aligned pairs:

```bash
python scripts/lqr/run_collect_pairs.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-id 0 \
  --num-episodes 20 \
  --perturb-spec scripts/lqr/configs/perturb_spec_gaussian.yaml \
  --out-dir outputs/lqr/pairs_gaussian
```

Run SVD and Jacobians:

```bash
python scripts/lqr/run_partition_svd.py \
  --pairs-dir outputs/lqr/pairs_init_pos_paired \
  --out-dir outputs/lqr/svd_init_pos \
  --config-name libero \
  --mode action \
  --selected-timesteps 0,10,20,30,40 \
  --num-samples -1 \
  --k-target 64 \
  --p-over 10

python scripts/lqr/run_compute_jacobians.py \
  --svd-dir outputs/lqr/svd_init_pos \
  --out-subdir A_tilde_lingbot \
  --inputs-npz outputs/lqr/pairs_init_pos_paired/negative.npz \
  --obs-index 0 \
  --config-name libero
```
