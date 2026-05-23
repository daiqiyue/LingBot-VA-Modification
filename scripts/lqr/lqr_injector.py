import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from scripts.lqr.lqr_core_ctrlwam import VCache, chained_riccati


@dataclass
class CallContext:
    action_mode: bool
    update_cache: int
    step_idx: int
    total_steps: int
    chunk_id: int


class LQRInjector:
    """Lingbot runtime adapter for CtrlWAM Activation-LQR."""

    def __init__(
        self,
        svd_dir: str,
        jac_dir_act: str,
        lambda_scale: float,
        q_scale: float,
        r_scale: float,
        qf_scale: float,
        modality: str,
        apply_on: str,
        video_steps: int,
        action_steps: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.svd_dir = Path(svd_dir).resolve()
        self.jac_dir_act = jac_dir_act
        self.lambda_scale = float(lambda_scale)
        self.q_scale = float(q_scale)
        self.r_scale = float(r_scale)
        self.qf_scale = float(qf_scale)
        self.modality = modality
        self.apply_on = apply_on
        self.video_steps = max(int(video_steps), 1)
        self.action_steps = max(int(action_steps), 1)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.chunk_id = 0
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current: Optional[CallContext] = None
        self._in_hook = False
        self._hook_handles = []
        self._u_step_pending = None

        cfg = json.loads((self.svd_dir / "config.json").read_text(encoding="utf-8"))
        self.sel_t = list(cfg["selected_timesteps"])
        self.T_diff = len(self.sel_t)
        self.L = int(cfg["L"])
        self.k_target = int(cfg["k_target"])
        self.partitions = [tuple(p) for p in cfg["partitions"]]
        self.layer_to_part = []

        summary = torch.load(self.svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
        self.layer_to_part = list(summary["layer_to_part"])
        c_means = summary["c_means"].float()
        self.mu = c_means.norm(dim=-1)
        self.v = c_means / self.mu.unsqueeze(-1).clamp(min=1e-12)

        jac_path = self.svd_dir / self.jac_dir_act / "A_tilde__full.pt"
        raw = torch.load(jac_path, map_location="cpu", weights_only=False)
        A_dict = raw.get("A_tilde", {})
        B_dict = raw.get("B_tilde", {})
        sel_idx_of = {t: i for i, t in enumerate(self.sel_t)}
        A_tilde = torch.zeros(self.T_diff, self.L - 1, self.k_target, self.k_target, dtype=torch.float32)
        for (t, l_in), Atl in A_dict.items():
            if t in sel_idx_of:
                A_tilde[sel_idx_of[t], l_in] = Atl.float()
        B_tilde = torch.zeros(max(self.T_diff - 1, 0), self.k_target, self.k_target, dtype=torch.float32)
        for key, Bt in B_dict.items():
            t = key[0]
            if t in sel_idx_of and sel_idx_of[t] < self.T_diff - 1:
                B_tilde[sel_idx_of[t]] = Bt.float()
        K_intra, K_step = chained_riccati(
            A_tilde=A_tilde,
            B_tilde=B_tilde,
            q_scale=self.q_scale,
            r_scale=self.r_scale,
            qf_scale=self.qf_scale,
            device=self.device,
        )
        self.K_intra = K_intra.to(device=self.device, dtype=torch.float32)
        self.K_step = K_step.to(device=self.device, dtype=torch.float32)
        self.v = self.v.to(device=self.device, dtype=torch.float32)
        self.mu = self.mu.to(device=self.device, dtype=torch.float32)

        self.vcache = VCache(
            self.svd_dir,
            self.partitions,
            self.layer_to_part,
            self.sel_t,
            self.k_target,
            device=self.device,
            dtype=torch.bfloat16,
        )

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

    def on_chunk_start(self, chunk_id: int) -> None:
        self.chunk_id = int(chunk_id)
        self.video_step_idx = 0
        self.action_step_idx = 0
        self._u_step_pending = None

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
            action_mode=bool(action_mode),
            update_cache=int(update_cache),
            step_idx=step_idx,
            total_steps=total_steps,
            chunk_id=self.chunk_id,
        )

    def end_call(self) -> None:
        self.current = None

    def _selected(self) -> Tuple[Optional[int], Optional[int]]:
        if self.current is None:
            return None, None
        step = self.current.step_idx
        if step < 0:
            return None, None
        try:
            sel = self.sel_t.index(step)
        except ValueError:
            return None, None
        return step, sel

    def register_hooks(self, transformer_model: torch.nn.Module) -> None:
        if not hasattr(transformer_model, "blocks"):
            raise RuntimeError("Expected transformer model with `blocks` attribute")
        blocks = transformer_model.blocks
        n_blocks = len(blocks)
        if n_blocks != self.L:
            raise RuntimeError(f"LQR expects {self.L} blocks, but transformer has {n_blocks}")
        self.vcache.preload([(p, t) for p in range(len(self.partitions)) for t in self.sel_t])

        self._hook_handles.append(blocks[0].register_forward_hook(self._cross_step_apply_hook()))
        for l_in in range(self.L - 1):
            self._hook_handles.append(blocks[l_in + 1].register_forward_hook(self._intra_hook(l_in)))
        self._hook_handles.append(blocks[self.L - 1].register_forward_hook(self._cross_step_compute_hook()))

    def _intra_hook(self, l_in: int):
        def hook(_module, _inputs, output):
            if self._in_hook:
                return output
            if self.current is None:
                return output
            if not self._allow_modality(self.current.action_mode):
                return output
            if not self._allow_cache_mode(self.current.update_cache):
                return output
            step, sel = self._selected()
            if step is None:
                return output
            out = output
            z_full = out[0].detach().reshape(-1)
            V_in = self.vcache.for_layer(l_in, step)
            self._in_hook = True
            try:
                x_proj = (z_full.to(V_in.dtype) @ V_in).float()
                v_fp = self.v[l_in, sel]
                mu_fp = self.mu[l_in, sel]
                K_fp = self.K_intra[sel, l_in]
                alpha = self.lambda_scale * mu_fp - v_fp @ x_proj
                u_tilde = K_fp @ (alpha * v_fp)
            finally:
                self._in_hook = False
            V_out = self.vcache.for_layer(l_in + 1, step)
            self._in_hook = True
            try:
                u_full = V_out @ u_tilde.to(V_out.dtype)
            finally:
                self._in_hook = False
            u_add = u_full.to(out.dtype).reshape_as(out[0]).unsqueeze(0)
            return out + u_add

        return hook

    def _cross_step_compute_hook(self):
        def hook(_module, _inputs, output):
            if self._in_hook:
                return output
            if self.current is None:
                return output
            if not self._allow_modality(self.current.action_mode):
                return output
            if not self._allow_cache_mode(self.current.update_cache):
                return output
            step, sel = self._selected()
            if step is None or sel >= self.T_diff - 1:
                return output
            z_full = output[0].detach().reshape(-1)
            V_in = self.vcache.for_layer(self.L - 1, step)
            self._in_hook = True
            try:
                x_proj = (z_full.to(V_in.dtype) @ V_in).float()
                v_fp = self.v[self.L - 1, sel]
                mu_fp = self.mu[self.L - 1, sel]
                K_fp = self.K_step[sel]
                alpha = self.lambda_scale * mu_fp - v_fp @ x_proj
                u_tilde = K_fp @ (alpha * v_fp)
                self._u_step_pending = {"src_sel": sel, "u_tilde": u_tilde.detach()}
            finally:
                self._in_hook = False
            return output

        return hook

    def _cross_step_apply_hook(self):
        def hook(_module, _inputs, output):
            if self._in_hook:
                return output
            if self.current is None:
                return output
            if not self._allow_modality(self.current.action_mode):
                return output
            if not self._allow_cache_mode(self.current.update_cache):
                return output
            step, sel = self._selected()
            pending = self._u_step_pending
            if step is None or pending is None or sel == 0 or pending["src_sel"] != sel - 1:
                return output
            V_dest = self.vcache.for_layer(0, step)
            self._in_hook = True
            try:
                u_full = V_dest @ pending["u_tilde"].to(V_dest.dtype)
            finally:
                self._in_hook = False
            u_add = u_full.to(output.dtype).reshape_as(output[0]).unsqueeze(0)
            self._u_step_pending = None
            return output + u_add

        return hook

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
