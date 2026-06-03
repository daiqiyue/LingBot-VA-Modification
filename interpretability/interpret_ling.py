import sys
import json
import os
import glob
import gc

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from pathlib import Path

# Repo root is three directories above this notebook (notebooks/lqr/inputs/ -> ctrlwam/)
REPO_ROOT = Path(".").resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Root directories produced by submit_collect_nominal.sh and submit_collect_activations.sh
NOMINAL_RUN_DIR = Path("outputs/lqr_nominal/collect_20260528_140047")
PERTURB_RUN_DIR = Path("outputs/lqr_activations_init_pos/collect_20260528_153741")

OUTPUT_DIR       = "interpret_output_pos"
ACTIVATIONS_PATH = "activations_dict_gripper.npz"

TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

N_BLOCKS = 30  # WanTransformer3DModel num_layers


def _records_from_shards(run_dir: Path, task_id: int):
    """Aggregate records from all shard manifests for a task (no merge step needed)."""
    task_dir = run_dir / f"task_{task_id}"
    records, lang = [], ""
    for shard_manifest in sorted(task_dir.glob("shard_*/manifest.json")):
        m = json.loads(shard_manifest.read_text(encoding="utf-8"))
        records.extend(m.get("records", []))
        if not lang:
            lang = m.get("task_language", "")
    return records, lang


def get_task_records(task_id: int):
    """
    Positive = all nominal records (regardless of success).
    Negative = all perturbed records (regardless of success).
    Reads directly from shards — no merge step required.
    """
    nom_records, lang     = _records_from_shards(NOMINAL_RUN_DIR, task_id)
    perturb_records, plang = _records_from_shards(PERTURB_RUN_DIR, task_id)

    pos_paths = [Path(r["path"]) for r in nom_records]
    neg_paths = [Path(r["path"]) for r in perturb_records]
    return pos_paths, neg_paths, lang or plang


def collect_activations_from_pt(rec_path: Path) -> list:
    """
    Load a trajectory record saved by run_collect_inputs.py and extract
    per-block activations from each capture.

    Each capture stores activations keyed by (layer_idx, step_idx).
    We average over all available timesteps to produce one vector per block.

    Returns a list of dicts {block_idx: np.ndarray[D]}, one dict per capture.
    """
    rec = th.load(rec_path, weights_only=False)
    captures = rec.get("captures", [])
    rows = []
    for cap in captures:
        act = cap.get("activations", {})
        row = {}
        for layer_idx in range(N_BLOCKS):
            vecs = [v.cpu().numpy() if isinstance(v, th.Tensor) else np.asarray(v)
                    for (l, _t), v in act.items() if l == layer_idx]
            if vecs:
                row[layer_idx] = np.mean(vecs, axis=0)
        if row:
            rows.append(row)
    return rows


# Validate upfront — error immediately if a task is missing records
for t in TASKS:
    pos, neg, lang = get_task_records(t)
    if not pos:
        raise FileNotFoundError(f"Task {t}: no nominal records in {NOMINAL_RUN_DIR}")
    if not neg:
        raise FileNotFoundError(f"Task {t}: no perturb records in {PERTURB_RUN_DIR}")
    print(f"Task {t:2d}: {len(pos)} nominal (pos), {len(neg)} perturbed (neg)  —  {lang!r}")



# activations_by_task[t] = {"positive": [...rows...], "negative": [...rows...]}
filepath = Path(ACTIVATIONS_PATH)
if filepath.is_file():
    activations_by_task = np.load(filepath, allow_pickle=True)
    activations_by_task = {int(k): v.item() for k, v in activations_by_task.items()}
else:
    activations_by_task = {}

print(activations_by_task)

for t in TASKS:
    if t not in activations_by_task:
        pos_files, neg_files, lang = get_task_records(t)
        print(f"Task {t}: loading {len(pos_files)} pos + {len(neg_files)} neg records...")

        pos_rows = []
        for f in pos_files:
            pos_rows.extend(collect_activations_from_pt(f))
            print(f"  pos: {len(pos_rows)} captures loaded", end="\r")

        neg_rows = []
        for f in neg_files:
            neg_rows.extend(collect_activations_from_pt(f))
            print(f"  neg: {len(neg_rows)} captures loaded", end="\r")

        activations_by_task[t] = {"positive": pos_rows, "negative": neg_rows}
        print(f"Task {t}: done — {len(pos_rows)} positive, {len(neg_rows)} negative captures")

print("\nActivation collection complete")
str_keyed_dict = {str(k): v for k, v in activations_by_task.items()}

np.savez(ACTIVATIONS_PATH, **str_keyed_dict)


# Fit contrastive PCA per block per task.
# Uses gram matrix trick (n_samples x n_samples) since n_samples << n_features.
# Contrastive vector for pair i at block b = pos_rows[i][b] - neg_rows[i][b].

pca_by_task_block = {}
n_pairs_by_task = {}

for t in TASKS:
    pos_rows = activations_by_task[t]["positive"]
    neg_rows = activations_by_task[t]["negative"]
    n_pairs = min(len(pos_rows), len(neg_rows))
    n_pairs_by_task[t] = n_pairs

    if n_pairs < 3:
        raise ValueError(f"Task {t}: only {n_pairs} contrastive pair(s) — need at least 3 for PCA")

    pca_by_task_block[t] = {}
    for b in range(N_BLOCKS):
        X = np.stack([pos_rows[i][b] - neg_rows[i][b] for i in range(n_pairs)])  # [n_pairs, D]
        mean = X.mean(axis=0)
        X_c = X - mean
        G = X_c @ X_c.T
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        top_vals = eigenvalues[-3:][::-1].copy()
        top_vecs = eigenvectors[:, -3:][:, ::-1].copy()
        components = X_c.T @ top_vecs / np.sqrt(top_vals)
        pca_by_task_block[t][b] = {"mean": mean, "components": components}
        del X, X_c, G

    gc.collect()
    D = pca_by_task_block[t][0]["components"].shape[0]
    print(f"Task {t}: PCA fitted — {N_BLOCKS} blocks, {n_pairs} pairs, feature dim {D}")

print("PCA done")

# Project all positive and negative activations onto per-block PCA planes per task.

projs_by_task_block = {}

for t in TASKS:
    pos_rows = activations_by_task[t]["positive"]
    neg_rows = activations_by_task[t]["negative"]

    projs_by_task_block[t] = {}
    for b in range(N_BLOCKS):
        mean  = pca_by_task_block[t][b]["mean"]
        comps = pca_by_task_block[t][b]["components"]
        pos_projs = np.stack([(r[b] - mean) @ comps for r in pos_rows])
        neg_projs = np.stack([(r[b] - mean) @ comps for r in neg_rows])
        projs_by_task_block[t][b] = {"positive": pos_projs, "negative": neg_projs}

    print(f"Task {t}: projected {len(pos_rows)} pos, {len(neg_rows)} neg onto {N_BLOCKS} block planes")

print("Projections done")

from scipy.optimize import minimize


def fit_svm(pos_pts, neg_pts, C=10.0):
    """Soft-margin linear SVM via L-BFGS-B. Truncates to equal class sizes."""
    n = min(len(pos_pts), len(neg_pts))
    pos_pts, neg_pts = pos_pts[:n], neg_pts[:n]
    X = np.vstack([pos_pts, neg_pts])
    y = np.concatenate([np.ones(n), -np.ones(n)])
    d = X.shape[1]

    def obj_and_grad(params):
        w, b = params[:d], params[d]
        margins = y * (X @ w + b)
        mask = margins < 1
        loss = 0.5 * np.dot(w, w) + C * np.maximum(0, 1 - margins).sum()
        grad_w = w - C * (y[mask, None] * X[mask]).sum(axis=0)
        grad_b = float(-C * y[mask].sum())
        return loss, np.concatenate([grad_w, [grad_b]])

    res = minimize(obj_and_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    w_opt, b_opt = res.x[:d], res.x[d]
    margins_opt = y * (X @ w_opt + b_opt)
    hinge_loss = C * np.maximum(0, 1 - margins_opt).sum()
    return w_opt, b_opt, hinge_loss


def plot_hyperplane_2d(ax, w, bias, all_pts):
    xlim = all_pts[:, 0].min(), all_pts[:, 0].max()
    ylim = all_pts[:, 1].min(), all_pts[:, 1].max()
    xs = np.linspace(xlim[0], xlim[1], 300)
    if abs(w[1]) > 1e-12:
        ys = -(w[0] * xs + bias) / w[1]
    else:
        ax.axvline(-bias / w[0], color="k", ls="--", lw=0.8, alpha=0.6)
        return
    ax.plot(xs, ys, color="k", ls="--", lw=0.8, alpha=0.6)
    px, py = 0.05 * (xlim[1] - xlim[0]), 0.05 * (ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - px, xlim[1] + px)
    ax.set_ylim(ylim[0] - py, ylim[1] + py)


def plot_hyperplane_3d(ax, w, bias, all_pts):
    i = int(np.argmax(np.abs(w)))
    j, k = [d for d in range(3) if d != i]
    JJ, KK = np.meshgrid(
        np.linspace(all_pts[:, j].min(), all_pts[:, j].max(), 15),
        np.linspace(all_pts[:, k].min(), all_pts[:, k].max(), 15),
    )
    II = -(w[j] * JJ + w[k] * KK + bias) / w[i]
    II = np.clip(II, all_pts[:, i].min(), all_pts[:, i].max())
    coords = [None, None, None]
    coords[i], coords[j], coords[k] = II, JJ, KK
    ax.plot_surface(*coords, alpha=0.18, color="gray", linewidth=0, antialiased=False)


svm_by_task_block = {}
svm_losses_by_task_block = {}

processed = {}
for t in TASKS:
    # print(f"Running task {t}")
    svm_by_task_block[t] = {}
    svm_losses_by_task_block[t] = {}
    pos_pts = projs_by_task_block[t][0]["positive"]
    neg_pts = projs_by_task_block[t][0]["negative"]
    n = 2 * min(len(pos_pts), len(neg_pts))
    processed[t] = {}
    processed[t]["num_samples"] = n
    losses_per_task = []
    for b in range(N_BLOCKS):
        # print(f"Running block {b}")
        pos_pts = projs_by_task_block[t][b]["positive"]
        neg_pts = projs_by_task_block[t][b]["negative"]
        # print(len(pos_pts))

        if len(pos_pts) > 1 and len(neg_pts) > 1:
            w, bias, loss = fit_svm(pos_pts, neg_pts)
            svm_by_task_block[t][b] = {"w": w, "bias": bias}
            svm_losses_by_task_block[t][b] = loss
            losses_per_task.append(loss)
        else:
            svm_by_task_block[t][b] = None
    processed[t]["best_loss"] = min(losses_per_task)
    processed[t]["last_loss"] = losses_per_task[-1]
    processed[t]["best_avg_loss"] = (processed[t]["best_loss"] / 10.0) / n
    processed[t]["last_avg_loss"] = (losses_per_task[-1] / 10.0) / n
    print(f"  task {t} SVMs fitted", end="\r")

print(f"\nSVMs fitted for all {len(TASKS)} tasks × {N_BLOCKS} blocks")
# print(svm_losses_by_task_block)

import pandas as pd

df = pd.DataFrame.from_dict(svm_losses_by_task_block, orient="index")

# Optional: convert numpy floats
df = df.astype(float)

print("Raw data")
print(df.to_markdown())

df = pd.DataFrame.from_dict(processed, orient="index")

# Optional: convert numpy floats
df = df.astype(float)

print("Processed")
print(df.to_markdown())


# 2D scatter — one figure per task, one subplot per DiT block
N_COLS = 6
N_ROWS = (N_BLOCKS + N_COLS - 1) // N_COLS

COLORS = {"positive": "#2196F3", "negative": "#FF5722"}
LABELS = {"positive": "nominal", "negative": "perturbed"}

for t in TASKS:
    _, _, lang = get_task_records(t)
    n_pairs = n_pairs_by_task[t]
    n_pos = len(activations_by_task[t]["positive"])
    n_neg = len(activations_by_task[t]["negative"])

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3.5, N_ROWS * 3.2))
    axes = axes.flatten()

    for b in range(N_BLOCKS):
        ax = axes[b]
        pos_pts = projs_by_task_block[t][b]["positive"]
        neg_pts = projs_by_task_block[t][b]["negative"]
        n_bal = min(len(pos_pts), len(neg_pts))
        pos_pts, neg_pts = pos_pts[:n_bal], neg_pts[:n_bal]
        all_pts = np.vstack([pos_pts, neg_pts])

        ax.scatter(pos_pts[:, 0], pos_pts[:, 1], c=COLORS["positive"], alpha=0.35, s=6, label=LABELS["positive"], rasterized=True)
        ax.scatter(neg_pts[:, 0], neg_pts[:, 1], c=COLORS["negative"], alpha=0.35, s=6, label=LABELS["negative"], rasterized=True)

        if svm_by_task_block[t][b] is not None:
            plot_hyperplane_2d(ax, svm_by_task_block[t][b]["w"], svm_by_task_block[t][b]["bias"], all_pts)

        ax.set_title(f"Block {b}", fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC1", fontsize=6, labelpad=1)
        ax.set_ylabel("PC2", fontsize=6, labelpad=1)

    for b in range(N_BLOCKS, len(axes)):
        axes[b].set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", fontsize=11, markerscale=4,
               title="Condition", title_fontsize=10)

    fig.suptitle(
        f"Task {t} — DiT block activations projected onto contrastive PCA planes\n"
        f"{lang!r}\n"
        f"(PCA fit on {n_pairs} pairs; {n_pos} positive, {n_neg} negative projected)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(f"{OUTPUT_DIR}/cosmos_contrastive_pca_2d_task{t:02d}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path}")


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

for t in TASKS:
    _, _, lang = get_task_records(t)
    n_pairs = n_pairs_by_task[t]
    n_pos = len(activations_by_task[t]["positive"])
    n_neg = len(activations_by_task[t]["negative"])

    fig = plt.figure(figsize=(N_COLS * 3.5, N_ROWS * 3.2))

    for b in range(N_BLOCKS):
        ax = fig.add_subplot(N_ROWS, N_COLS, b + 1, projection="3d")
        pos_pts = projs_by_task_block[t][b]["positive"]
        neg_pts = projs_by_task_block[t][b]["negative"]
        n_bal = min(len(pos_pts), len(neg_pts))
        pos_pts, neg_pts = pos_pts[:n_bal], neg_pts[:n_bal]
        all_pts = np.vstack([pos_pts, neg_pts])

        if svm_by_task_block[t][b] is not None:
            plot_hyperplane_3d(ax, svm_by_task_block[t][b]["w"], svm_by_task_block[t][b]["bias"], all_pts)

        ax.scatter(pos_pts[:, 0], pos_pts[:, 1], pos_pts[:, 2],
                   c=COLORS["positive"], alpha=0.35, s=4, label=LABELS["positive"], rasterized=True)
        ax.scatter(neg_pts[:, 0], neg_pts[:, 1], neg_pts[:, 2],
                   c=COLORS["negative"], alpha=0.35, s=4, label=LABELS["negative"], rasterized=True)

        ax.set_title(f"Block {b}", fontsize=8, pad=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("PC1", fontsize=5, labelpad=-8)
        ax.set_ylabel("PC2", fontsize=5, labelpad=-8)
        ax.set_zlabel("PC3", fontsize=5, labelpad=-8)

    handles, labels_ = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", fontsize=11, markerscale=4,
               title="Condition", title_fontsize=10)

    fig.suptitle(
        f"Task {t} — DiT block activations, top 3 contrastive PCs (3D)\n"
        f"{lang!r}\n"
        f"(PCA fit on {n_pairs} pairs; {n_pos} positive, {n_neg} negative projected)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = f"{OUTPUT_DIR}/cosmos_contrastive_pca_3d_task{t:02d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path}")


import itertools
import pandas as pd

PCS = [1, 2]  # which PCs to plot (1-indexed); e.g. [3, 1] → x=PC3, y=PC1

assert len(PCS) == 2, "PCS must have exactly 2 entries"

COLORS  = {"positive": "#2196F3", "negative": "#FF5722"}
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

n_pcs_needed = max(max(PCS), 3)  # always compute at least 3 for the 3D SVM table
pc_cols = [p - 1 for p in PCS]  # 0-indexed column selection into sorted-desc components

N_COLS = 6
N_ROWS = (N_BLOCKS + N_COLS - 1) // N_COLS

# avg hinge loss per sample, keyed by (t1, t2) then block index
pair_losses    = {}
pair_losses_3d = {}

for COMBINED_TASKS in itertools.combinations(TASKS, 2):
    COMBINED_TASKS = list(COMBINED_TASKS)
    pair_key = tuple(COMBINED_TASKS)

    missing = set(COMBINED_TASKS) - set(activations_by_task)
    if missing:
        raise KeyError(f"Tasks {sorted(missing)} not in activations_by_task.")

    TASK_MARKER = {t: MARKERS[i % len(MARKERS)] for i, t in enumerate(COMBINED_TASKS)}

    # ── combined PCA per block ────────────────────────────────────────────────
    combined_pca = {}
    for b in range(N_BLOCKS):
        vecs = []
        for t in COMBINED_TASKS:
            pos_rows = activations_by_task[t]["positive"]
            neg_rows = activations_by_task[t]["negative"]
            n_pairs = min(len(pos_rows), len(neg_rows))
            vecs.extend(pos_rows[i][b] - neg_rows[i][b] for i in range(n_pairs))

        X = np.stack(vecs)
        mean = X.mean(axis=0)
        X_c = X - mean
        G = X_c @ X_c.T
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        n_keep = min(n_pcs_needed, len(eigenvalues))
        top_vals = eigenvalues[-n_keep:][::-1].copy()
        top_vecs = eigenvectors[:, -n_keep:][:, ::-1].copy()
        all_components = X_c.T @ top_vecs / np.sqrt(top_vals)
        combined_pca[b] = {
            "mean": mean,
            "components":    all_components[:, pc_cols],  # [D, 2] for plotting
            "components_3d": all_components[:, :3],        # [D, 3] for 3D SVM
        }

    # ── project all rows ──────────────────────────────────────────────────────
    combined_projs    = {}
    combined_projs_3d = {}
    for t in COMBINED_TASKS:
        combined_projs[t]    = {}
        combined_projs_3d[t] = {}
        for label in ("positive", "negative"):
            rows = activations_by_task[t][label]
            combined_projs[t][label] = np.stack(
                [np.stack([(r[b] - combined_pca[b]["mean"]) @ combined_pca[b]["components"]
                           for b in range(N_BLOCKS)])
                 for r in rows]
            )  # [N_rows, N_BLOCKS, 2]
            combined_projs_3d[t][label] = np.stack(
                [np.stack([(r[b] - combined_pca[b]["mean"]) @ combined_pca[b]["components_3d"]
                           for b in range(N_BLOCKS)])
                 for r in rows]
            )  # [N_rows, N_BLOCKS, 3]

    # ── SVM + loss collection ─────────────────────────────────────────────────
    combined_svm = {}
    block_losses    = {}
    block_losses_3d = {}
    for b in range(N_BLOCKS):
        all_pos    = np.vstack([combined_projs[t]["positive"][:, b, :]    for t in COMBINED_TASKS])
        all_neg    = np.vstack([combined_projs[t]["negative"][:, b, :]    for t in COMBINED_TASKS])
        all_pos_3d = np.vstack([combined_projs_3d[t]["positive"][:, b, :] for t in COMBINED_TASKS])
        all_neg_3d = np.vstack([combined_projs_3d[t]["negative"][:, b, :] for t in COMBINED_TASKS])
        n_total = len(all_pos) + len(all_neg)
        if len(all_pos) > 1 and len(all_neg) > 1:
            w, bias, hinge = fit_svm(all_pos, all_neg)
            combined_svm[b] = {"w": w, "bias": bias}
            block_losses[b] = hinge / n_total / 10

            _, _, hinge_3d = fit_svm(all_pos_3d, all_neg_3d)
            block_losses_3d[b] = hinge_3d / n_total / 10
        else:
            combined_svm[b] = None
            block_losses[b]    = float("nan")
            block_losses_3d[b] = float("nan")

    pair_losses[pair_key]    = block_losses
    pair_losses_3d[pair_key] = block_losses_3d

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3.5, N_ROWS * 3.2))
    axes = axes.flatten()

    for b in range(N_BLOCKS):
        ax = axes[b]
        all_pts_list = []
        for t in COMBINED_TASKS:
            for label in ("positive", "negative"):
                pts = combined_projs[t][label][:, b, :]
                all_pts_list.append(pts)
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=COLORS[label], marker=TASK_MARKER[t],
                           alpha=0.35, s=6, label=f"task {t} {LABELS[label]}", rasterized=True)
        all_pts = np.vstack(all_pts_list)
        if combined_svm[b] is not None:
            plot_hyperplane_2d(ax, combined_svm[b]["w"], combined_svm[b]["bias"], all_pts)
        ax.set_title(f"Block {b}", fontsize=8, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"PC{PCS[0]}", fontsize=6, labelpad=1)
        ax.set_ylabel(f"PC{PCS[1]}", fontsize=6, labelpad=1)

    for b in range(N_BLOCKS, len(axes)):
        axes[b].set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", fontsize=9, markerscale=2,
               title=f"tasks {COMBINED_TASKS}", title_fontsize=9, ncol=len(COMBINED_TASKS))

    n_pairs_total = sum(min(len(activations_by_task[t]["positive"]),
                            len(activations_by_task[t]["negative"])) for t in COMBINED_TASKS)
    fig.suptitle(
        f"Combined tasks {COMBINED_TASKS} — DiT block activations  (PC{PCS[0]} vs PC{PCS[1]})\n"
        f"(PCA fit on {n_pairs_total} pooled contrastive pairs; color=condition  shape=task)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(f"{OUTPUT_DIR}/cosmos_combined_tasks_{'_'.join(str(t) for t in COMBINED_TASKS)}_pc{PCS[0]}v{PCS[1]}_2d.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.show()
    print(f"Saved → {out_path}")

# ── hinge loss summary tables ─────────────────────────────────────────────────
def _loss_summary(losses_dict, label):
    df = pd.DataFrame.from_dict(losses_dict, orient="index").astype(float)
    df.index = [str(p) for p in df.index]
    df.columns = [f"b{b}" for b in df.columns]
    summary = pd.DataFrame({
        "best block":            df.idxmin(axis=1),
        "best avg loss":         df.min(axis=1),
        f"b{N_BLOCKS-1} avg loss": df[f"b{N_BLOCKS-1}"],
    })
    print(f"\n### Avg hinge loss per sample — all task pairs  [{label}]\n")
    print(df.to_markdown(floatfmt=".4f"))
    print(f"\n### Summary: best block and last block  [{label}]\n")
    print(summary.to_markdown(floatfmt=".4f"))

_loss_summary(pair_losses,    "2D SVM — top-2 PCs")
_loss_summary(pair_losses_3d, "3D SVM — top-3 PCs")


