import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from scripts.activation_steering.common import alpha_schedule, maybe_load_yaml, parse_int_list


def _ensure_lingbot_logging() -> None:
    # patch script imports wan_va_server as a module, so wan_va_server.__main__
    # does not run init_logger() automatically.
    from wan_va.utils.logging import init_logger, logger as root_logger

    if not root_logger.handlers:
        init_logger()
    root_logger.setLevel(logging.INFO)


@dataclass
class CallContext:
    action_mode: bool
    update_cache: int
    step_idx: int
    total_steps: int
    chunk_id: int


class SteeringInjector:
    def __init__(
        self,
        steering_bank_path: str,
        layers: List[int],
        alpha: float,
        alpha_schedule_name: str,
        modality: str,
        apply_on: str,
        video_steps: int,
        action_steps: int,
    ) -> None:
        raw = torch.load(steering_bank_path, map_location="cpu", weights_only=False)
        if "vectors" in raw:
            raw = raw["vectors"]
        self.bank = raw
        self.layers = set(layers)
        self.alpha = float(alpha)
        self.alpha_schedule_name = alpha_schedule_name
        self.modality = modality
        self.apply_on = apply_on
        self.video_steps = max(video_steps, 1)
        self.action_steps = max(action_steps, 1)
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.chunk_id = 0
        self.current: Optional[CallContext] = None
        self._hook_handles = []

    def _allow_modality(self, action_mode: bool) -> bool:
        if self.modality == "both":
            return True
        if self.modality == "video":
            return not action_mode
        if self.modality == "action":
            return action_mode
        return False

    def _allow_cache_mode(self, update_cache: int) -> bool:
        if self.apply_on == "include_cache_write":
            return True
        return update_cache == 0

    def _bank_key(self, action_mode: bool, layer_idx: int) -> Tuple[Optional[str], Optional[str]]:
        mode_key = "action" if action_mode else "video"
        if mode_key not in self.bank:
            return None, None
        layer_key = str(layer_idx)
        if layer_key not in self.bank[mode_key]:
            return mode_key, None
        return mode_key, layer_key

    def _vector_for(self, action_mode: bool, layer_idx: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        mode_key, layer_key = self._bank_key(action_mode, layer_idx)
        if mode_key is None or layer_key is None:
            return None
        vec = self.bank[mode_key][layer_key]
        if not torch.is_tensor(vec):
            vec = torch.tensor(vec)
        return vec.to(device=device, dtype=dtype)

    def on_chunk_start(self, chunk_id: int) -> None:
        self.chunk_id = chunk_id
        self.video_step_idx = 0
        self.action_step_idx = 0

    def begin_call(self, action_mode: bool, update_cache: int) -> None:
        if action_mode:
            step_idx = self.action_step_idx
            total_steps = self.action_steps
            self.action_step_idx += 1
        else:
            step_idx = self.video_step_idx
            total_steps = self.video_steps
            self.video_step_idx += 1
        self.current = CallContext(
            action_mode=action_mode,
            update_cache=update_cache,
            step_idx=step_idx,
            total_steps=total_steps,
            chunk_id=self.chunk_id,
        )

    def end_call(self) -> None:
        self.current = None

    def register_hooks(self, transformer_model: torch.nn.Module) -> None:
        if not hasattr(transformer_model, "blocks"):
            raise RuntimeError("Expected transformer model with `blocks` attribute")
        for idx, block in enumerate(transformer_model.blocks):
            if idx not in self.layers:
                continue
            handle = block.register_forward_hook(self._hook_fn(idx))
            self._hook_handles.append(handle)

    def _hook_fn(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.current is None:
                return output
            if not self._allow_modality(self.current.action_mode):
                return output
            if not self._allow_cache_mode(self.current.update_cache):
                return output
            vec = self._vector_for(self.current.action_mode, layer_idx, output.device, output.dtype)
            if vec is None:
                return output
            cur_alpha = alpha_schedule(
                mode=self.alpha_schedule_name,
                base_alpha=self.alpha,
                step_idx=self.current.step_idx,
                total_steps=self.current.total_steps,
            )
            if cur_alpha == 0.0:
                return output
            if vec.ndim == 1:
                steer = vec.view(1, 1, -1)
            elif vec.ndim == 2:
                steer = vec.unsqueeze(0)
            else:
                raise ValueError(f"Steering vector must be 1D or 2D, got {vec.shape}")
            return output + cur_alpha * steer

        return hook

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LingBot server with runtime activation steering patch.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)

    parser.add_argument("--steering-bank", type=str, required=True)
    parser.add_argument("--layers", type=str, default="22")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-schedule", type=str, choices=["flat", "linear_decay", "cosine_decay"], default="linear_decay")
    parser.add_argument("--modality", type=str, choices=["video", "action", "both"], default="both")
    parser.add_argument(
        "--apply-on",
        type=str,
        choices=["transient_only", "include_cache_write"],
        default="transient_only",
    )
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--steering-config", type=str, default=None, help="Optional YAML/JSON to override steering args.")
    return parser.parse_args()


def _load_injector(args: argparse.Namespace) -> SteeringInjector:
    merged = {
        "layers": parse_int_list(args.layers),
        "alpha": args.alpha,
        "alpha_schedule": args.alpha_schedule,
        "modality": args.modality,
        "apply_on": args.apply_on,
        "video_steps": args.video_steps,
        "action_steps": args.action_steps,
    }
    if args.steering_config:
        cfg = maybe_load_yaml(args.steering_config)
        merged.update(cfg)
    return SteeringInjector(
        steering_bank_path=args.steering_bank,
        layers=list(merged["layers"]),
        alpha=float(merged["alpha"]),
        alpha_schedule_name=str(merged["alpha_schedule"]),
        modality=str(merged["modality"]),
        apply_on=str(merged["apply_on"]),
        video_steps=int(merged["video_steps"]),
        action_steps=int(merged["action_steps"]),
    )


def main() -> None:
    args = _parse_args()
    injector = _load_injector(args)
    os.environ.setdefault("PYTHONPATH", ".")
    _ensure_lingbot_logging()

    import wan_va.wan_va_server as base_server

    original_init = base_server.VA_Server.__init__
    original_infer_chunk = base_server.VA_Server._infer

    def patched_init(self, job_config):
        original_init(self, job_config)
        transformer = self.transformer
        injector.register_hooks(transformer)
        original_forward = transformer.forward

        def patched_forward(*f_args, **f_kwargs):
            action_mode = bool(f_kwargs.get("action_mode", False))
            update_cache = int(f_kwargs.get("update_cache", 0))
            injector.begin_call(action_mode=action_mode, update_cache=update_cache)
            try:
                return original_forward(*f_args, **f_kwargs)
            finally:
                injector.end_call()

        transformer.forward = patched_forward

    def patched_infer(self, obs, frame_st_id=0):
        injector.on_chunk_start(chunk_id=int(frame_st_id))
        return original_infer_chunk(self, obs, frame_st_id=frame_st_id)

    base_server.VA_Server.__init__ = patched_init
    base_server.VA_Server._infer = patched_infer

    try:
        base_server.run(args)
    finally:
        injector.close()


if __name__ == "__main__":
    main()
