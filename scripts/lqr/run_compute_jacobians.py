import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _fit_linear_map(xs: List[torch.Tensor], ys: List[torch.Tensor], ridge: float) -> torch.Tensor:
    # Solve Y = A X in least-squares with ridge: A = Y X^T (X X^T + λI)^-1
    X = torch.stack(xs, dim=1).float()  # [k, N]
    Y = torch.stack(ys, dim=1).float()  # [k, N]
    k = X.shape[0]
    XXt = X @ X.transpose(0, 1)
    reg = ridge * torch.eye(k, dtype=XXt.dtype)
    A = (Y @ X.transpose(0, 1)) @ torch.linalg.inv(XXt + reg)
    return A.float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Lingbot-native projected Jacobian maps for LQR.")
    parser.add_argument("--svd-dir", type=Path, required=True, help="SVD output directory from run_partition_svd.py")
    parser.add_argument("--out-subdir", type=str, default="A_tilde_lingbot")
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    _ = torch.load(args.svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
    proj_raw = torch.load(args.svd_dir / "projected_diffs.pt", map_location="cpu", weights_only=False)
    config_json = __import__("json").loads((args.svd_dir / "config.json").read_text(encoding="utf-8"))

    projected_diffs: Dict[Tuple[int, int, int], torch.Tensor] = proj_raw["projected_diffs"]
    selected_timesteps = list(config_json["selected_timesteps"])
    n_layers = int(config_json["L"])

    # Infer sample count from keys.
    sample_ids = sorted({k[0] for k in projected_diffs.keys()})
    A_tilde = {}
    B_tilde = {}

    for t in selected_timesteps:
        for l_in in range(n_layers - 1):
            xs = []
            ys = []
            for s in sample_ids:
                k_x = (s, l_in, t)
                k_y = (s, l_in + 1, t)
                if k_x not in projected_diffs or k_y not in projected_diffs:
                    continue
                xs.append(projected_diffs[k_x])
                ys.append(projected_diffs[k_y])
            if len(xs) < 2:
                continue
            A_tilde[(t, l_in)] = _fit_linear_map(xs, ys, ridge=args.ridge)

    for idx in range(len(selected_timesteps) - 1):
        t = selected_timesteps[idx]
        t_next = selected_timesteps[idx + 1]
        xs = []
        ys = []
        for s in sample_ids:
            k_x = (s, n_layers - 1, t)
            k_y = (s, 0, t_next)
            if k_x not in projected_diffs or k_y not in projected_diffs:
                continue
            xs.append(projected_diffs[k_x])
            ys.append(projected_diffs[k_y])
        if len(xs) < 2:
            continue
        B_tilde[(t,)] = _fit_linear_map(xs, ys, ridge=args.ridge)

    out_dir = args.svd_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "A_tilde__full.pt"
    torch.save(
        {
            "A_tilde": A_tilde,
            "B_tilde": B_tilde,
            "prompt": config_json.get("prompt"),
            "mode": config_json.get("mode"),
            "selected_timesteps": selected_timesteps,
            "ridge": float(args.ridge),
            "num_samples": len(sample_ids),
        },
        out_path,
    )
    print(f"[jac] wrote {out_path}")
    print(f"[jac] A_tilde entries: {len(A_tilde)}")
    print(f"[jac] B_tilde entries: {len(B_tilde)}")


if __name__ == "__main__":
    main()
