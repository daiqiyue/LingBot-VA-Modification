# LQR on Lingbot-VA

本目录实现了 Lingbot-VA 上的 Activation-LQR 流程，目标是：

- 离线工件（pairs/SVD/Jacobian）全部来自 Lingbot 自身模型表征空间
- 在线注入通过 runtime monkey-patch 完成
- 不修改 `wan_va/`、`evaluation/` 现有源码

## 方法概览

1. 采集 LIBERO 观测，构造正样本（原图）与负样本（加噪图）
2. 在 Lingbot 推理时抓取 transformer block 激活差分（pos-neg）
3. 对差分做低秩分解（SVD）得到 `V` 和对比方向统计
4. 在低秩空间拟合线性动力学 `A_tilde/B_tilde`
5. 在线推理用 LQR 控制律注入 activation steering

## 文件说明

- `run_collect_inputs.py`：采样并生成 `positive.npz` / `negative.npz`
- `run_partition_svd.py`：基于 Lingbot 推理提激活并做 SVD
- `run_compute_jacobians.py`：基于投影差分拟合 `A_tilde/B_tilde`
- `lqr_injector.py`：在线注入器（读取 SVD/Jacobian 工件并挂 hooks）
- `patch_infer_with_lqr.py`：server 入口（monkey-patch `VA_Server`）
- `run_libero_lqr_eval.py`：一体化启动 server + client 的评测脚本
- `configs/lqr_config.yaml`：LQR 默认参数

## 环境要求

- 在 `lingbot-va` 根目录运行
- 可用 Python 环境需包含：`torch`、`libero`、`mujoco` 及项目依赖
- 建议先设置：

```bash
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/lingbot-va
export PYTHONPATH=.
```

## 快速开始（多 perturbation）

### Step 1. 批量采集 nominal + 多个 perturbation

```bash
python scripts/lqr/run_collect_inputs.py \
  --libero-benchmark libero_10 \
  --task-id 0 \
  --num-samples 32 \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --mode both \
  --out-dir outputs/lqr/pairs
```

产物：

- `outputs/lqr/pairs/variants/<variant>.npz`（每个扰动一个 npz）
- `outputs/lqr/pairs/pairs/<variant>/positive.npz`
- `outputs/lqr/pairs/pairs/<variant>/negative.npz`
- `outputs/lqr/pairs/pairs/<variant>/manifest.json`
- `outputs/lqr/pairs/manifest.json`

说明：

- `pairs/<variant>/` 自动是 `nominal` 对 `variant` 的配对，适合直接喂给 SVD。
- 如果只想采一个 perturbation，可加 `--target-variants eef_y_pos_03`。

### Step 2. 选定一个 perturbation，生成 SVD 工件

下面示例针对 `eef_y_pos_03`：

```bash
python scripts/lqr/run_partition_svd.py \
  --pairs-dir outputs/lqr/pairs/pairs/eef_y_pos_03 \
  --out-dir outputs/lqr/svd_eef_y_pos_03 \
  --config-name libero \
  --mode action \
  --selected-timesteps 0,10,20,30,40 \
  --num-samples 16 \
  --k-target 32
```

产物：

- `outputs/lqr/svd_eef_y_pos_03/config.json`
- `outputs/lqr/svd_eef_y_pos_03/svd_summary.pt`
- `outputs/lqr/svd_eef_y_pos_03/V_part*_layers*-*_t*_k*.pt`
- `outputs/lqr/svd_eef_y_pos_03/projected_diffs.pt`

### Step 3. 生成 Jacobian/LQR 动力学工件

```bash
python scripts/lqr/run_compute_jacobians.py \
  --svd-dir outputs/lqr/svd_eef_y_pos_03 \
  --out-subdir A_tilde_lingbot
```

产物：

- `outputs/lqr/svd_eef_y_pos_03/A_tilde_lingbot/A_tilde__full.pt`

### Step 4. 启动 LQR server

```bash
PORT=29056 \
SVD_DIR=outputs/lqr/svd_eef_y_pos_03 \
JAC_DIR_ACT=A_tilde_lingbot \
PYTHONPATH=. \
bash scripts/lqr/launch_lqr_server.sh
```

### Step 5. 启动 client 验证

```bash
START=0 END=1 PORT=29056 PYTHONPATH=. bash scripts/lqr/launch_lqr_client.sh
```

## 一体化评测（可替代 Step 4 + 5）

```bash
python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 0 10 \
  --num-episodes 10 \
  --port 29056 \
  --svd-dir outputs/lqr/svd_eef_y_pos_03 \
  --jac-dir-act A_tilde_lingbot \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --out-dir outputs/lqr_eval
```

输出：

- `outputs/lqr_eval/metrics_nominal.json`
- `outputs/lqr_eval/metrics_perturbed.json`
- `outputs/lqr_eval/summary.json`

## 做目标 B：all perturbation 通用 LQR

如果你要先做“一个通用方向适配所有扰动”，流程是：

1) 先采集全部扰动（Step 1）
2) 把所有 `pairs/<variant>/` 合并成一个总 pair 集
3) 在总 pair 集上跑 SVD/Jacobian
4) 用这套工件评测各类扰动

### B1. 合并所有 perturbation pair

```bash
python scripts/lqr/build_all_pairs.py \
  --collect-dir outputs/lqr/pairs \
  --out-dir outputs/lqr/pairs_all
```

会生成：

- `outputs/lqr/pairs_all/positive.npz`
- `outputs/lqr/pairs_all/negative.npz`
- `outputs/lqr/pairs_all/manifest.json`

### B2. 用合并 pair 训练通用方向

```bash
python scripts/lqr/run_partition_svd.py \
  --pairs-dir outputs/lqr/pairs_all \
  --out-dir outputs/lqr/svd_all_perturb \
  --config-name libero \
  --mode action \
  --selected-timesteps 0,10,20,30,40 \
  --num-samples 32 \
  --k-target 32
```

### B3. 计算通用 Jacobian 工件

```bash
python scripts/lqr/run_compute_jacobians.py \
  --svd-dir outputs/lqr/svd_all_perturb \
  --out-subdir A_tilde_lingbot
```

### B4. 评测通用 LQR（覆盖全部扰动）

```bash
python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 0 10 \
  --num-episodes 10 \
  --port 29056 \
  --svd-dir outputs/lqr/svd_all_perturb \
  --jac-dir-act A_tilde_lingbot \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --out-dir outputs/lqr_eval_all_perturb
```

## 常见问题

- `ModuleNotFoundError: torch/libero`  
  当前环境缺少依赖，请切换到可运行 Lingbot 的 conda/env。

- `Missing activation key (...)`  
  说明 `--selected-timesteps` 与当前 mode 的实际推理步不匹配。先缩小为少量步（例如 `0,10,20`）再扩展。

- `k_target too large`  
  `k_target` 不能超过 `min(num_samples, feature_dim)`。先减小 `k_target` 或增加样本数。

- 如何支持多 perturbation 并分别训练 LQR  
  先用 `run_collect_inputs.py --mode both` 生成所有 `pairs/<variant>/`，然后对每个 `variant` 单独跑 Step 2/3/4。不要把不同扰动混到同一个 pair 目录里。

- server 启动后 client 连不上  
  检查 `PORT` 是否一致、server 是否已完成加载。
