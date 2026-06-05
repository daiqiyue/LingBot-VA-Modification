"""
Camera-ready condensed figures for interpret_cosmos.ipynb results.

Produces two sets of figures matching the make_fig.py style:
  1. Per-task   : one figure, rows=tasks,      cols=First/Best/Last DiT block
  2. Per-pair   : one PDF per task pair,        cols=First/Best/Last DiT block
                  (colour=condition, shape=task)

SVM hinge loss is shown as a text annotation in every subfigure.
"""

import itertools
import json
import os
import glob
import gc

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_SCALE = 1.5
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16 * FONT_SCALE

ROW_LABEL_FONTSIZE  = 18 * FONT_SCALE
COL_TITLE_FONTSIZE  = 20 * FONT_SCALE
ANNOTATION_FONTSIZE = 16 * FONT_SCALE
LEGEND_FONTSIZE     = 16 * FONT_SCALE

COLORS  = {"positive": "#2196F3", "negative": "#FF5722"}
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]

# ── Paths / tasks ─────────────────────────────────────────────────────────────
def _parse_tasks_env(value, default):
    if not value.strip():
        return default
    return [int(x) for x in value.replace(",", " ").split()]


# DATA_DIR         = "policy_inputs_new_cam"
DATA_DIR         = os.environ.get("DATA_DIR", "policy_inputs_noise")
# ACTIVATIONS_PATH = "activations_dict.npz"
# OUTPUT_DIR       = "gripper_large_figs"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "actual_gripper_final_figs")
ACTIVATIONS_PATH = os.environ.get("ACTIVATIONS_PATH", "activations_dict_gripper.npz")

TASKS        = _parse_tasks_env(os.environ.get("TASKS", ""), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
CAMERA_TASKS = _parse_tasks_env(os.environ.get("CAMERA_TASKS", ""), TASKS)
CAMERA_PAIRS = list(itertools.combinations(TASKS, 2))

# ── Data helpers ──────────────────────────────────────────────────────────────
def get_task_dirs(task_id):
    return sorted(glob.glob(os.path.join(DATA_DIR, f"libero_10__task{task_id:02d}_*")))

def get_task_language(task_id):
    for d in get_task_dirs(task_id):
        manifest = os.path.join(d, "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                return json.load(f)["policy_prompt"]
    raise FileNotFoundError(f"No manifest.json found for task {task_id}")

# ── Load activations ──────────────────────────────────────────────────────────
raw = np.load(ACTIVATIONS_PATH, allow_pickle=True)
# activations_dict.npz also contains per-run keys like "libero_10__task00__..." — skip those
_int_keys = {k for k in raw.keys() if k.lstrip("-").isdigit()}
available = {int(k) for k in _int_keys}
missing = set(TASKS) - available
if missing:
    raise FileNotFoundError(f"Tasks {sorted(missing)} not in {ACTIVATIONS_PATH}.")

activations_by_task = {int(k): raw[k].item() for k in _int_keys if int(k) in set(TASKS)}

_sample  = next(iter(activations_by_task.values()))
_rows    = _sample["positive"] or _sample["negative"]
N_BLOCKS = len(_rows[0])
print(f"Loaded {len(TASKS)} tasks, {N_BLOCKS} blocks from {ACTIVATIONS_PATH!r}")

# ── SVM ───────────────────────────────────────────────────────────────────────
_loss_buf = []

def fit_svm(pos_pts, neg_pts, C=10.0):
    X = np.vstack([pos_pts, neg_pts])
    y = np.concatenate([np.ones(len(pos_pts)), -np.ones(len(neg_pts))])
    d = X.shape[1]
    # print(f"DDDDDD: {d}")
    _loss_buf.clear()

    def obj_and_grad(params):
        w, b = params[:d], params[d]
        margins = y * (X @ w + b)
        mask = margins < 1
        loss = 0.5 * np.dot(w, w) + C * np.maximum(0, 1 - margins).sum()
        grad_w = w - C * (y[mask, None] * X[mask]).sum(axis=0)
        grad_b = float(-C * y[mask].sum())
        _loss_buf.append(loss - 0.5 * np.dot(w, w))
        return loss, np.concatenate([grad_w, [grad_b]])

    res = minimize(obj_and_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return res.x[:d], res.x[d], _loss_buf[-1]


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
    px = 0.05 * (xlim[1] - xlim[0])
    py = 0.05 * (ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - px, xlim[1] + px)
    ax.set_ylim(ylim[0] - py, ylim[1] + py)

# ── Per-task contrastive PCA + projections ────────────────────────────────────
print("Fitting per-task PCA …")
pca_by_task_block   = {}
projs_by_task_block = {}
n_pairs_by_task     = {}

for t in TASKS:
    pos_rows = activations_by_task[t]["positive"]
    neg_rows = activations_by_task[t]["negative"]
    n_pairs  = min(len(pos_rows), len(neg_rows))
    n_pairs_by_task[t] = n_pairs
    if n_pairs < 3:
        raise ValueError(f"Task {t}: only {n_pairs} pairs — need ≥ 3")

    pca_by_task_block[t] = {}
    for b in range(N_BLOCKS):
        X    = np.stack([pos_rows[i][b] - neg_rows[i][b] for i in range(n_pairs)])
        mean = X.mean(axis=0)
        X_c  = X - mean
        G    = X_c @ X_c.T
        evals, evecs = np.linalg.eigh(G)
        top_vals = evals[-3:][::-1].copy()
        top_vecs = evecs[:, -3:][:, ::-1].copy()
        comps = X_c.T @ top_vecs / np.sqrt(top_vals)
        pca_by_task_block[t][b] = {"mean": mean, "components": comps}
        del X, X_c, G
    gc.collect()

    projs_by_task_block[t] = {}
    for b in range(N_BLOCKS):
        mean  = pca_by_task_block[t][b]["mean"]
        comps = pca_by_task_block[t][b]["components"]
        projs_by_task_block[t][b] = {
            "positive": np.stack([(r[b] - mean) @ comps for r in pos_rows]),
            "negative": np.stack([(r[b] - mean) @ comps for r in neg_rows]),
        }
    print(f"  task {t} done", end="\r")
print("Per-task PCA done        ")

# ── Per-task SVMs ─────────────────────────────────────────────────────────────
print("Fitting per-task SVMs …")
svm_by_task_block        = {}
svm_losses_by_task_block = {}

for t in TASKS:
    svm_by_task_block[t]        = {}
    svm_losses_by_task_block[t] = {}
    for b in range(N_BLOCKS):
        pos_pts = projs_by_task_block[t][b]["positive"]
        neg_pts = projs_by_task_block[t][b]["negative"]
        if len(pos_pts) > 1 and len(neg_pts) > 1:
            w, bias, loss = fit_svm(pos_pts, neg_pts)
            svm_by_task_block[t][b]        = {"w": w, "bias": bias}
            svm_losses_by_task_block[t][b] = loss
        else:
            svm_by_task_block[t][b]        = None
            svm_losses_by_task_block[t][b] = float("nan")
    print(f"  task {t} SVMs done", end="\r")
print("Per-task SVMs done        ")

# ── Shared helpers ────────────────────────────────────────────────────────────
def _annotate_loss(ax, b, avg_loss):
    ax.text(
        0.97, 0,
        f"Block {b}\navg loss: {avg_loss:.4f}",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONTSIZE, ha="right", va="top",
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )


def _style_ax(ax, row_idx, col_idx, role_label, row_label):
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax.grid(True, color="gray", alpha=0.4, linewidth=0.5, linestyle=":")
    if row_idx == 0:
        ax.set_title(role_label, fontsize=COL_TITLE_FONTSIZE, fontweight="bold", pad=5)
    if col_idx == 0:
        ax.annotate(
            row_label,
            xy=(-0.02, 0.5), xycoords="axes fraction",
            fontsize=ROW_LABEL_FONTSIZE,
            rotation=90, ha="center", va="center",
            annotation_clip=False,
        )


def _col_specs(loss_by_block):
    first_b = 0
    last_b  = N_BLOCKS - 1
    interior = [b for b in range(N_BLOCKS)
                if b not in (first_b, last_b) and not np.isnan(loss_by_block[b])]
    best_b = min(interior, key=lambda b: loss_by_block[b]) if interior else first_b
    roles: dict[int, list[str]] = {}
    for b, role in [(first_b, "First"), (best_b, "Best"), (last_b, "Last")]:
        roles.setdefault(b, []).append(role)
    specs = [(b, " / ".join(roles[b])) for b in sorted(roles)]
    while len(specs) < 3:
        specs.append(specs[-1])
    return specs

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-task camera-ready figure
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating per-task camera-ready figure …")

import matplotlib.gridspec as gridspec

n_per_col  = 5  # tasks per column group
col_groups = [CAMERA_TASKS[:n_per_col], CAMERA_TASKS[n_per_col:]]

# 7 columns: [0-2] left group | [3] narrow spacer | [4-6] right group
fig_cam = plt.figure(figsize=(6 * 6.2, n_per_col * 5.5))
gs_cam  = gridspec.GridSpec(
    n_per_col, 7,
    figure=fig_cam,
    width_ratios=[1, 1, 1, 0.01, 1, 1, 1],
    hspace=0.08, wspace=0.08,
)

for grp_idx, grp_tasks in enumerate(col_groups):
    col_offset = grp_idx * 4  # 0 (left) or 4 (right), skipping spacer at col 3
    for row_idx, t in enumerate(grp_tasks):
        n_pos   = len(activations_by_task[t]["positive"])
        n_neg   = len(activations_by_task[t]["negative"])
        n_total = n_pos + n_neg

        specs = _col_specs(svm_losses_by_task_block[t])

        for col_idx, (b, role_label) in enumerate(specs):
            ax      = fig_cam.add_subplot(gs_cam[row_idx, col_offset + col_idx])
            pos_pts = projs_by_task_block[t][b]["positive"]
            neg_pts = projs_by_task_block[t][b]["negative"]
            n_plot  = min(len(pos_pts), len(neg_pts))
            all_pts = np.vstack([pos_pts[:n_plot], neg_pts[:n_plot]])

            for label, color in COLORS.items():
                pts = projs_by_task_block[t][b][label][:n_plot]
                ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.7, s=12,
                           label=label, rasterized=True)

            if svm_by_task_block[t][b] is not None:
                plot_hyperplane_2d(ax, svm_by_task_block[t][b]["w"],
                                   svm_by_task_block[t][b]["bias"], all_pts)

            avg_loss = svm_losses_by_task_block[t][b] / (10.0 * n_total)
            _annotate_loss(ax, b, avg_loss)
            _style_ax(ax, row_idx, col_idx, role_label, f"Task {t}")

fig_cam.axes[0].legend(
    loc="upper left", fontsize=LEGEND_FONTSIZE, markerscale=3,
    title="Condition", title_fontsize=LEGEND_FONTSIZE, framealpha=0.85,
)
gs_cam.tight_layout(fig_cam, rect=[0.02, 0.0, 0.98, 1])

out_path = Path(f"{OUTPUT_DIR}/ling_<task>_all_tasks_horizontal.pdf")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 1b.  Per-task camera-ready figure — 3D
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating per-task 3D camera-ready figure …")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 7 columns: [0-2] left group | [3] narrow spacer | [4-6] right group
fig_cam3d = plt.figure(figsize=(6 * 7.2, n_per_col * 6.2))
gs_3d     = gridspec.GridSpec(
    n_per_col, 7,
    figure=fig_cam3d,
    # width_ratios=[1, 1, 1, 0.03, 1, 1, 1],
    width_ratios=[0.9, 0.9, 0.9, 0.03, 0.9, 0.9, 0.9],
    hspace=0.15, wspace=0.05,
)

for grp_idx, grp_tasks in enumerate(col_groups):
    col_offset = grp_idx * 4
    for row_idx, t in enumerate(grp_tasks):
        n_pos   = len(activations_by_task[t]["positive"])
        n_neg   = len(activations_by_task[t]["negative"])
        n_total = n_pos + n_neg

        specs = _col_specs(svm_losses_by_task_block[t])

        for col_idx, (b, role_label) in enumerate(specs):
            ax = fig_cam3d.add_subplot(gs_3d[row_idx, col_offset + col_idx], projection="3d")
            pos_pts = projs_by_task_block[t][b]["positive"]
            neg_pts = projs_by_task_block[t][b]["negative"]
            n_plot  = min(len(pos_pts), len(neg_pts))
            all_pts = np.vstack([pos_pts[:n_plot], neg_pts[:n_plot]])

            for label, color in COLORS.items():
                pts = projs_by_task_block[t][b][label][:n_plot]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=color, alpha=0.85, s=10, label=label, rasterized=True)

            if svm_by_task_block[t][b] is not None:
                plot_hyperplane_3d(ax, svm_by_task_block[t][b]["w"],
                                   svm_by_task_block[t][b]["bias"], all_pts)

            avg_loss = svm_losses_by_task_block[t][b] / (10.0 * n_total)
            ax.text2D(
                0.97, 0.04,
                f"Block {b}\navg loss: {avg_loss:.4f}",
                transform=ax.transAxes,
                fontsize=ANNOTATION_FONTSIZE, ha="right", va="top",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xlabel("PC1", fontsize=LEGEND_FONTSIZE, labelpad=-3)
            ax.set_ylabel("PC2", fontsize=LEGEND_FONTSIZE, labelpad=-3)
            ax.set_zlabel("PC3", fontsize=LEGEND_FONTSIZE, labelpad=-3)

            if row_idx == 0:
                ax.set_title(role_label, fontsize=COL_TITLE_FONTSIZE, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.text2D(
                    -0.06, 0.5, f"Task {t}",
                    transform=ax.transAxes,
                    fontsize=ROW_LABEL_FONTSIZE,
                    rotation=90, ha="center", va="center",
                )

fig_cam3d.axes[0].legend(
    loc="upper left", fontsize=LEGEND_FONTSIZE, markerscale=3,
    title="Condition", title_fontsize=LEGEND_FONTSIZE, framealpha=0.55,
)
fig_cam3d.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.13,
                           hspace=0.15, wspace=0.05)

out_path_3d = Path(f"{OUTPUT_DIR}/ling_<task>_all_tasks_horizontal_3d.pdf")
out_path_3d.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path_3d, dpi=150, bbox_inches="tight", pad_inches=0.65)
print(f"Saved → {out_path_3d}")
plt.close()

# # ─────────────────────────────────────────────────────────────────────────────
# # 2.  Per-pair camera-ready figures
# # ─────────────────────────────────────────────────────────────────────────────
# print(f"\nGenerating {len(CAMERA_PAIRS)} per-pair camera-ready figures …")

# for t1, t2 in CAMERA_PAIRS:
#     pair_tasks  = [t1, t2]
#     task_marker = {t: MARKERS[i] for i, t in enumerate(pair_tasks)}

#     # combined contrastive PCA (pool pairs from both tasks per block)
#     combined_pca = {}
#     for b in range(N_BLOCKS):
#         vecs = []
#         for t in pair_tasks:
#             pos_rows = activations_by_task[t]["positive"]
#             neg_rows = activations_by_task[t]["negative"]
#             n_pairs  = min(len(pos_rows), len(neg_rows))
#             vecs.extend(pos_rows[i][b] - neg_rows[i][b] for i in range(n_pairs))
#         X    = np.stack(vecs)
#         mean = X.mean(axis=0)
#         X_c  = X - mean
#         G    = X_c @ X_c.T
#         evals, evecs = np.linalg.eigh(G)
#         top_vals = evals[-3:][::-1].copy()
#         top_vecs = evecs[:, -3:][:, ::-1].copy()
#         comps = X_c.T @ top_vecs / np.sqrt(top_vals)
#         combined_pca[b] = {"mean": mean, "components": comps}

#     # project all observations
#     combined_projs = {t: {} for t in pair_tasks}
#     for t in pair_tasks:
#         for b in range(N_BLOCKS):
#             mean  = combined_pca[b]["mean"]
#             comps = combined_pca[b]["components"]
#             combined_projs[t][b] = {
#                 "positive": np.stack([(r[b] - mean) @ comps for r in activations_by_task[t]["positive"]]),
#                 "negative": np.stack([(r[b] - mean) @ comps for r in activations_by_task[t]["negative"]]),
#             }

#     # fit combined SVM per block
#     combined_svm  = {}
#     combined_loss = {}
#     for b in range(N_BLOCKS):
#         all_pos = np.vstack([combined_projs[t][b]["positive"] for t in pair_tasks])
#         all_neg = np.vstack([combined_projs[t][b]["negative"] for t in pair_tasks])
#         n_total = len(all_pos) + len(all_neg)
#         if len(all_pos) > 1 and len(all_neg) > 1:
#             w, bias, loss = fit_svm(all_pos, all_neg)
#             combined_svm[b]  = {"w": w, "bias": bias}
#             combined_loss[b] = loss / (10.0 * n_total)
#         else:
#             combined_svm[b]  = None
#             combined_loss[b] = float("nan")

#     specs = _col_specs(combined_loss)

#     fig, axes = plt.subplots(1, 3, figsize=(3 * 6.2, 5.5), squeeze=False)

#     for col_idx, (b, role_label) in enumerate(specs):
#         ax = axes[0, col_idx]
#         all_pts_list = []
#         for t in pair_tasks:
#             for label, color in COLORS.items():
#                 pts = combined_projs[t][b][label]
#                 ax.scatter(pts[:, 0], pts[:, 1],
#                            c=color, marker=task_marker[t], alpha=0.7, s=12,
#                            label=f"task {t} {label}", rasterized=True)
#                 all_pts_list.append(pts)
#         all_pts = np.vstack(all_pts_list)

#         if combined_svm[b] is not None:
#             plot_hyperplane_2d(ax, combined_svm[b]["w"], combined_svm[b]["bias"], all_pts)

#         _annotate_loss(ax, b, combined_loss[b])
#         _style_ax(ax, 0, col_idx, role_label, f"Tasks {t1}&{t2}")

#     axes[0, 0].legend(
#         loc="upper left", fontsize=LEGEND_FONTSIZE, markerscale=3,
#         title="Cond. / Task", title_fontsize=LEGEND_FONTSIZE, framealpha=0.85,
#     )
#     plt.tight_layout(rect=[0.04, 0.0, 0.98, 1])

#     out_path = Path(f"{OUTPUT_DIR}/cosmos_pair_{t1:02d}_{t2:02d}_camera_ready.pdf")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     print(f"  Saved 2D → {out_path}")
#     plt.close()

#     # 3D figure for this pair
#     fig3d = plt.figure(figsize=(3 * 7.2, 6.2))
#     for col_idx, (b, role_label) in enumerate(specs):
#         ax = fig3d.add_subplot(1, 3, col_idx + 1, projection="3d")
#         all_pts_list = []
#         for t in pair_tasks:
#             for label, color in COLORS.items():
#                 pts = combined_projs[t][b][label]
#                 ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
#                            c=color, marker=task_marker[t], alpha=0.85, s=10,
#                            label=f"task {t} {label}", rasterized=True)
#                 all_pts_list.append(pts)
#         all_pts = np.vstack(all_pts_list)

#         if combined_svm[b] is not None:
#             plot_hyperplane_3d(ax, combined_svm[b]["w"], combined_svm[b]["bias"], all_pts)

#         ax.text2D(
#             0.97, 0.04,
#             f"Block {b}\navg loss: {combined_loss[b]:.4f}",
#             transform=ax.transAxes,
#             fontsize=ANNOTATION_FONTSIZE, ha="right", va="top",
#             clip_on=False,
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
#         )
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.set_xlabel("PC1", fontsize=LEGEND_FONTSIZE, labelpad=-3)
#         ax.set_ylabel("PC2", fontsize=LEGEND_FONTSIZE, labelpad=-3)
#         ax.set_zlabel("PC3", fontsize=LEGEND_FONTSIZE, labelpad=-3)
#         ax.set_title(role_label, fontsize=COL_TITLE_FONTSIZE, fontweight="bold", pad=8)
#         if col_idx == 0:
#             ax.text2D(
#                 -0.06, 0.5, f"Tasks {t1}&{t2}",
#                 transform=ax.transAxes,
#                 fontsize=ROW_LABEL_FONTSIZE,
#                 rotation=90, ha="center", va="center",
#             )

#     fig3d.axes[0].legend(
#         loc="upper left", fontsize=LEGEND_FONTSIZE, markerscale=3,
#         title="Cond. / Task", title_fontsize=LEGEND_FONTSIZE, framealpha=0.55,
#     )
#     fig3d.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.05,
#                           hspace=0.1, wspace=0.05)

#     out_path_3d = Path(f"{OUTPUT_DIR}/cosmos_pair_{t1:02d}_{t2:02d}_camera_ready_3d.pdf")
#     out_path_3d.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path_3d, dpi=150, bbox_inches="tight", pad_inches=0.65)
#     print(f"  Saved 3D → {out_path_3d}")
#     plt.close()

# print("\nAll camera-ready figures saved.")
