# Lingbot LQR (Trajectory-TopK Activation Pairs)

This directory reimplements the offline LQR data pipeline around trajectory-level rollout outcomes:

- Collect activations directly during inference rollouts (`run_collect_inputs.py`)
- Keep first `K` inference chunks per trajectory (`--top-k-inference-per-traj`)
- Label each captured chunk with trajectory final outcome
- Filter pools:
  - positive: nominal + trajectory success
  - negative: perturbed + trajectory failure
- Build paired set by pool alignment (`build_all_pairs.py`)
- Run SVD / Jacobian / LQR steering as ctrlWAM-style downstream math

## 1) Collect trajectory records

```bash
python scripts/lqr/run_collect_inputs.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-id 0 \
  --num-episodes 10 \
  --top-k-inference-per-traj 10 \
  --selected-timesteps 0,10,20,30,40 \
  --mode action \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --out-dir outputs/lqr/pairs_raw
```

Output:
- `trajectory_records/*.pt`: one file per trajectory
- `manifest.json`: rollout metadata and record paths

## 2) Build pool-based pairs

```bash
python scripts/lqr/build_all_pairs.py \
  --collect-dir outputs/lqr/pairs_raw \
  --out-dir outputs/lqr/pairs_all \
  --pair-seed 0
```

Output:
- `positive.pt`
- `negative.pt`
- `manifest.json` with pool and filtering stats

## 3) Run SVD

```bash
python scripts/lqr/run_partition_svd.py \
  --pairs-dir outputs/lqr/pairs_all \
  --out-dir outputs/lqr/svd_all \
  --config-name libero \
  --mode action \
  --num-samples 200 \
  --k-target 32 \
  --selected-timesteps 0,10,20,30,40
```

If `positive.pt/negative.pt` exist, SVD consumes activation pairs directly.
If they do not exist, it falls back to observation pair mode (`positive.npz/negative.npz`).

## 4) Fit projected Jacobians

```bash
python scripts/lqr/run_compute_jacobians.py \
  --svd-dir outputs/lqr/svd_all \
  --out-subdir A_tilde_lingbot
```

## 5) LQR evaluation

```bash
python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 0 2 \
  --num-episodes 10 \
  --port 29056 \
  --svd-dir outputs/lqr/svd_all \
  --jac-dir-act A_tilde_lingbot \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --out-dir outputs/lqr_eval_all
```

## Notes

- Pair labels are trajectory-level, not chunk-level.
- No extra inference pass is required for pair construction: collection and activation tracing happen in one rollout pass.
- The online steering controller still uses ctrlWAM-style chained Riccati outputs (`K_intra`, `K_step`) and layer hooks.
