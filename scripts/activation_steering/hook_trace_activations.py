import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from scripts.activation_steering.common import (
    append_jsonl,
    ensure_dir,
    maybe_load_yaml,
    parse_int_list,
)


def _ensure_lingbot_logging() -> None:
    # hook script imports wan_va_server as a module, so wan_va_server.__main__
    # does not run init_logger() automatically.
    from wan_va.utils.logging import init_logger, logger as root_logger

    if not root_logger.handlers:
        init_logger()
    root_logger.setLevel(logging.INFO)


@dataclass
class TraceCallContext:
    call_idx: int
    action_mode: bool
    update_cache: int
    phase: str
    chunk_id: int


class ActivationTracer:
    def __init__(
        self,
        out_dir: str,
        layers: List[int],
        modality: str,
        token_policy: Dict,
        save_format: str = "pt",
    ) -> None:
        self.layers = set(layers)
        self.modality = modality
        self.token_policy = token_policy
        self.save_format = save_format
        self.out_dir = ensure_dir(out_dir)
        self.trace_dir = ensure_dir(os.path.join(self.out_dir, "traces"))
        self.manifest_path = os.path.join(self.out_dir, "traces_manifest.jsonl")
        self.phase = "infer"
        self.chunk_id = 0
        self.call_idx = 0
        self.current: Optional[TraceCallContext] = None
        self.current_layers: Dict[int, torch.Tensor] = {}
        self.frame_chunk_size = int(token_policy.get("frame_chunk_size", 4))
        self.action_per_frame = int(token_policy.get("action_per_frame", 4))
        self._hook_handles = []

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def on_chunk_start(self, chunk_id: int) -> None:
        self.chunk_id = chunk_id

    def _allow_modality(self, action_mode: bool) -> bool:
        if self.modality == "both":
            return True
        if self.modality == "video":
            return not action_mode
        if self.modality == "action":
            return action_mode
        return False

    def begin_call(self, action_mode: bool, update_cache: int) -> None:
        self.current = TraceCallContext(
            call_idx=self.call_idx,
            action_mode=action_mode,
            update_cache=update_cache,
            phase=self.phase,
            chunk_id=self.chunk_id,
        )
        self.current_layers = {}

    def end_call(self) -> None:
        if self.current is None:
            return
        mode_name = "action" if self.current.action_mode else "video"
        file_base = f"call_{self.current.call_idx:08d}"
        out_path = os.path.join(self.trace_dir, f"{file_base}.pt")
        payload = {
            "meta": {
                "call_idx": self.current.call_idx,
                "mode": mode_name,
                "update_cache": self.current.update_cache,
                "phase": self.current.phase,
                "chunk_id": self.current.chunk_id,
            },
            "layers": {str(k): v.cpu() for k, v in self.current_layers.items()},
        }
        torch.save(payload, out_path)
        append_jsonl(
            self.manifest_path,
            {
                "call_idx": self.current.call_idx,
                "mode": mode_name,
                "update_cache": self.current.update_cache,
                "phase": self.current.phase,
                "chunk_id": self.current.chunk_id,
                "path": out_path,
            },
        )
        self.call_idx += 1
        self.current = None
        self.current_layers = {}

    def _pool_tokens(self, hidden_states: torch.Tensor, action_mode: bool) -> torch.Tensor:
        # hidden_states shape: [B, L, C]
        x = hidden_states.float().mean(dim=0)  # [L, C] average over batch/CFG branches
        L = x.shape[0]
        if self.token_policy.get("pooling", "mean") == "all":
            return x.mean(dim=0)

        if action_mode:
            focus = self.token_policy.get("action_frame_focus", "first_group")
            n_per_frame = self.action_per_frame
            n_frames = max(1, L // max(1, n_per_frame))
            if focus == "all":
                return x.mean(dim=0)
            use_frames = 1 if focus == "first_group" else min(2, n_frames)
            cut = min(L, use_frames * n_per_frame)
            return x[:cut].mean(dim=0)

        focus = self.token_policy.get("video_frame_focus", "first2")
        if focus == "all":
            return x.mean(dim=0)
        n_frames = max(1, self.frame_chunk_size)
        n_per_frame = max(1, L // n_frames)
        use_frames = 1 if focus == "first1" else min(2, n_frames)
        cut = min(L, use_frames * n_per_frame)
        return x[:cut].mean(dim=0)

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
            pooled = self._pool_tokens(output, action_mode=self.current.action_mode)
            self.current_layers[layer_idx] = pooled.detach()
            return output

        return hook

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run server and trace transformer activations.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="15,19,22,25,27")
    parser.add_argument("--modality", type=str, choices=["video", "action", "both"], default="both")
    parser.add_argument("--token-policy", type=str, default=None, help="YAML/JSON token policy path.")
    parser.add_argument("--save-format", type=str, choices=["pt"], default="pt")
    parser.add_argument("--run-tag", type=str, default="trace")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PYTHONPATH", ".")
    _ensure_lingbot_logging()
    token_policy = {
        "video_frame_focus": "first2",
        "action_frame_focus": "first_group",
        "pooling": "mean",
        "frame_chunk_size": 4,
        "action_per_frame": 4,
    }
    if args.token_policy:
        token_policy.update(maybe_load_yaml(args.token_policy))

    trace_root = ensure_dir(os.path.join(args.out_dir, args.run_tag))
    tracer = ActivationTracer(
        out_dir=trace_root,
        layers=parse_int_list(args.layers),
        modality=args.modality,
        token_policy=token_policy,
        save_format=args.save_format,
    )

    import wan_va.wan_va_server as base_server

    original_init = base_server.VA_Server.__init__
    original_infer = base_server.VA_Server._infer
    original_kv = base_server.VA_Server._compute_kv_cache

    def patched_init(self, job_config):
        original_init(self, job_config)
        token_policy["frame_chunk_size"] = int(getattr(job_config, "frame_chunk_size", token_policy["frame_chunk_size"]))
        token_policy["action_per_frame"] = int(getattr(job_config, "action_per_frame", token_policy["action_per_frame"]))
        tracer.register_hooks(self.transformer)
        original_forward = self.transformer.forward

        def patched_forward(*f_args, **f_kwargs):
            action_mode = bool(f_kwargs.get("action_mode", False))
            update_cache = int(f_kwargs.get("update_cache", 0))
            tracer.begin_call(action_mode=action_mode, update_cache=update_cache)
            try:
                return original_forward(*f_args, **f_kwargs)
            finally:
                tracer.end_call()

        self.transformer.forward = patched_forward

    def patched_infer(self, obs, frame_st_id=0):
        tracer.set_phase("infer")
        tracer.on_chunk_start(int(frame_st_id))
        return original_infer(self, obs, frame_st_id=frame_st_id)

    def patched_kv(self, obs):
        tracer.set_phase("kv_cache")
        tracer.on_chunk_start(int(getattr(self, "frame_st_id", 0)))
        return original_kv(self, obs)

    base_server.VA_Server.__init__ = patched_init
    base_server.VA_Server._infer = patched_infer
    base_server.VA_Server._compute_kv_cache = patched_kv

    try:
        base_server.run(args)
    finally:
        tracer.close()


if __name__ == "__main__":
    main()
