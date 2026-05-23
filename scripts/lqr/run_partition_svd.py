import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class _CallCtx:
    action_mode: bool
    update_cache: int
    step_idx: int


class LingbotActivationTracer:
    def __init__(self, layers: List[int], selected_timesteps: List[int], mode: str) -> None:
        self.layers = set(layers)
        self.selected_timesteps = set(selected_timesteps)
        self.mode = mode
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current: Optional[_CallCtx] = None
        self._hook_handles = []
        self.captured: Dict[Tuple[int, int], torch.Tensor] = {}

    def reset_chunk(self) -> None:
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current = None
        self.captured = {}

    def begin_call(self, action_mode: bool, update_cache: int) -> None:
        if action_mode:
            step_idx = self.action_step_idx
            self.action_step_idx += 1
        else:
            step_idx = self.video_step_idx
            self.video_step_idx += 1
        self.current = _CallCtx(action_mode=bool(action_mode), update_cache=int(update_cache), step_idx=step_idx)

    def end_call(self) -> None:
        self.current = None

    def _mode_allow(self, action_mode: bool) -> bool:
        if self.mode == "both":
            return True
        if self.mode == "action":
            return action_mode
        return not action_mode

    def register_hooks(self, transformer: torch.nn.Module) -> None:
        blocks = transformer.blocks
        for idx, block in enumerate(blocks):
            if idx not in self.layers:
                continue
            self._hook_handles.append(block.register_forward_hook(self._hook_fn(idx)))

    def _hook_fn(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.current is None:
                return output
            if not self._mode_allow(self.current.action_mode):
                return output
            if self.current.step_idx not in self.selected_timesteps:
                return output
            self.captured[(layer_idx, self.current.step_idx)] = output[0].detach().reshape(-1).float().cpu()
            return output

        return hook

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def _default_master_port() -> str:
    return str(12355 + (os.getpid() % 1000))


def _ensure_dist_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _default_master_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _build_server(config_name: str):
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    _ensure_dist_env()
    if not dist.is_initialized():
        init_distributed(world_size=1, local_rank=0, rank=0)

    cfg = copy.deepcopy(VA_CONFIGS[config_name])
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    return VA_Server(cfg)


def _obs_from_npz(npz_obj, idx: int, cam_keys: List[str]) -> Dict:
    cam0 = npz_obj["primary_images"][idx]
    cam1 = npz_obj["wrist_images"][idx]
    return {
        "obs": [
            {
                cam_keys[0]: np.ascontiguousarray(cam0),
                cam_keys[1]: np.ascontiguousarray(cam1),
            }
        ]
    }


def _parse_int_csv(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lingbot-native partition SVD for LQR.")
    parser.add_argument("--pairs-dir", type=Path, required=True, help="Directory containing positive.npz and negative.npz.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["video", "action", "both"], default="action")
    parser.add_argument("--layers", type=str, default="", help="Comma separated layer ids; empty means all layers.")
    parser.add_argument("--selected-timesteps", type=str, default="0,10,20,30,40")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--k-target", type=int, default=32)
    parser.add_argument("--p-over", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    pos_path = args.pairs_dir / "positive.npz"
    neg_path = args.pairs_dir / "negative.npz"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(f"Expected positive/negative npz in {args.pairs_dir}")
    pos = np.load(pos_path)
    neg = np.load(neg_path)

    manifest_path = args.pairs_dir / "manifest.json"
    prompt = args.prompt
    if prompt is None and manifest_path.exists():
        pair_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt = pair_manifest.get("task_language")
    if not prompt:
        raise ValueError("Prompt is required. Pass --prompt or provide pairs manifest with task_language.")

    server = _build_server(args.config_name)
    transformer = server.transformer
    n_layers = len(transformer.blocks)
    layers = _parse_int_csv(args.layers) if args.layers.strip() else list(range(n_layers))
    selected_timesteps = _parse_int_csv(args.selected_timesteps)
    tracer = LingbotActivationTracer(layers=layers, selected_timesteps=selected_timesteps, mode=args.mode)
    tracer.register_hooks(transformer)

    original_forward = transformer.forward

    def patched_forward(*f_args, **f_kwargs):
        action_mode = bool(f_kwargs.get("action_mode", False))
        update_cache = int(f_kwargs.get("update_cache", 0))
        tracer.begin_call(action_mode=action_mode, update_cache=update_cache)
        try:
            return original_forward(*f_args, **f_kwargs)
        finally:
            tracer.end_call()

    transformer.forward = patched_forward

    cam_keys = list(server.job_config.obs_cam_keys)
    n_total = int(min(args.num_samples, pos["primary_images"].shape[0], neg["primary_images"].shape[0]))
    diffs: Dict[Tuple[int, int], List[torch.Tensor]] = {(l, t): [] for l in layers for t in selected_timesteps}

    try:
        for i in range(n_total):
            obs_pos = _obs_from_npz(pos, i, cam_keys)
            obs_neg = _obs_from_npz(neg, i, cam_keys)

            server._reset(prompt=prompt)
            tracer.reset_chunk()
            server._infer(obs_pos, frame_st_id=0)
            cap_pos = dict(tracer.captured)

            server._reset(prompt=prompt)
            tracer.reset_chunk()
            server._infer(obs_neg, frame_st_id=0)
            cap_neg = dict(tracer.captured)

            for key in diffs:
                if key not in cap_pos or key not in cap_neg:
                    raise RuntimeError(
                        f"Missing activation key {key} at sample {i}. "
                        "Adjust --mode/--selected-timesteps to valid inference steps."
                    )
                diffs[key].append((cap_pos[key] - cap_neg[key]).float().cpu())
    finally:
        tracer.close()
        transformer.forward = original_forward

    partitions = [(l, l) for l in range(n_layers)]
    layer_to_part = list(range(n_layers))
    t_to_idx = {t: i for i, t in enumerate(selected_timesteps)}
    c_means = torch.zeros(n_layers, len(selected_timesteps), args.k_target, dtype=torch.float32)
    projected_diffs: Dict[Tuple[int, int, int], torch.Tensor] = {}

    for (layer, t), delta_list in diffs.items():
        X = torch.stack(delta_list, dim=0)  # [N, D]
        n, d = X.shape
        if args.k_target > min(n, d):
            raise ValueError(
                f"k_target={args.k_target} too large for key (layer={layer}, t={t}) with "
                f"N={n}, D={d}. Reduce k_target or increase num-samples."
            )
        mean = X.mean(dim=0)
        Xc = X - mean
        q = min(args.k_target + args.p_over, min(n, d))
        _, _, V = torch.pca_lowrank(Xc, q=q, center=False)
        Vk = V[:, : args.k_target].contiguous().float()

        p_idx = layer_to_part[layer]
        v_path = args.out_dir / f"V_part{p_idx}_layers{layer}-{layer}_t{t}_k{args.k_target}.pt"
        torch.save({"V": Vk}, v_path)

        c_means[layer, t_to_idx[t]] = (mean @ Vk).float()
        for s_idx in range(n):
            projected_diffs[(s_idx, layer, t)] = (X[s_idx] @ Vk).float().cpu()

    if args.mode == "action":
        sampling_steps = int(server.job_config.action_num_inference_steps)
    elif args.mode == "video":
        sampling_steps = int(server.job_config.num_inference_steps)
    else:
        sampling_steps = int(max(server.job_config.num_inference_steps, server.job_config.action_num_inference_steps))
    cfg = {
        "config_name": args.config_name,
        "prompt": prompt,
        "mode": args.mode,
        "selected_timesteps": selected_timesteps,
        "sampling_steps": sampling_steps,
        "L": n_layers,
        "k_target": int(args.k_target),
        "partitions": partitions,
        "num_samples": n_total,
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    torch.save(
        {
            "c_means": c_means,
            "layer_to_part": layer_to_part,
            "selected_timesteps": selected_timesteps,
            "mode": args.mode,
            "num_samples": n_total,
            "k_target": int(args.k_target),
        },
        args.out_dir / "svd_summary.pt",
    )
    torch.save(
        {
            "projected_diffs": projected_diffs,
            "mode": args.mode,
            "prompt": prompt,
            "selected_timesteps": selected_timesteps,
            "k_target": int(args.k_target),
        },
        args.out_dir / "projected_diffs.pt",
    )
    print(f"[svd] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
