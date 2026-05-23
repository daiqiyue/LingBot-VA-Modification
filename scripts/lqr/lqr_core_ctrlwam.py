"""CtrlWAM Activation-LQR core primitives.

This file preserves the original algorithmic components used by the CtrlWAM
LQR runtime: VCache, chained_riccati, SteeringRuntime, install_lqr_hooks.
"""

import time
from collections import OrderedDict
from pathlib import Path

import torch


class VCache:
    def __init__(
        self,
        svd_dir,
        partitions,
        layer_to_part,
        sel_t,
        k_target,
        device,
        dtype=torch.bfloat16,
        max_gpu_tiles=None,
    ):
        self.svd_dir = Path(svd_dir)
        self.partitions = partitions
        self.layer_to_part = layer_to_part
        self.sel_t = list(sel_t)
        self.k_target = k_target
        self.device = device
        self.dtype = dtype
        self.max_gpu = max_gpu_tiles or (len(partitions) * len(sel_t))

        self._cpu = {}
        self._gpu = OrderedDict()
        self.stats = {"cpu_loads": 0, "gpu_swaps": 0, "gpu_hits": 0, "cpu_to_gpu_s": 0.0}

    def _vfile(self, p_idx, t_id):
        a, b = self.partitions[p_idx]
        pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
        matches = sorted(self.svd_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no V file matching {pattern} in {self.svd_dir}")
        preferred = [m for m in matches if m.name.endswith(f"_k{self.k_target}.pt")]
        return (preferred or matches)[0]

    def _load_cpu(self, p_idx, t_id):
        key = (p_idx, t_id)
        V = self._cpu.get(key)
        if V is not None:
            return V
        fp = self._vfile(p_idx, t_id)
        print(
            f"  disk-load {fp.name} ({fp.stat().st_size / 1e9:.2f} GB) -> CPU {self.dtype} ...",
            flush=True,
        )
        t0 = time.time()
        raw = torch.load(fp, map_location="cpu", weights_only=False)["V"]
        V = raw.to(dtype=self.dtype).contiguous()
        del raw
        self._cpu[key] = V
        self.stats["cpu_loads"] += 1
        print(
            f"    CPU resident (p={p_idx}, t={t_id}): {tuple(V.shape)}  "
            f"~{V.element_size() * V.numel() / 1e9:.2f} GB  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        return V

    def for_layer(self, layer_idx, t_id):
        p_idx = self.layer_to_part[layer_idx]
        key = (p_idx, t_id)
        V = self._gpu.get(key)
        if V is not None:
            self._gpu.move_to_end(key)
            self.stats["gpu_hits"] += 1
            return V
        V_cpu = self._load_cpu(p_idx, t_id)
        if self.device.type == "cpu":
            self._gpu[key] = V_cpu
            return V_cpu
        while len(self._gpu) >= self.max_gpu:
            _, ev = self._gpu.popitem(last=False)
            del ev
            torch.cuda.empty_cache()
            self.stats["gpu_swaps"] += 1
        t0 = time.time()
        V = V_cpu.to(device=self.device, non_blocking=False).contiguous()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stats["cpu_to_gpu_s"] += time.time() - t0
        self._gpu[key] = V
        return V

    def preload(self, partition_t_pairs):
        for p_idx, t_id in partition_t_pairs:
            self._load_cpu(p_idx, t_id)


def chained_riccati(A_tilde, B_tilde, q_scale, r_scale, qf_scale, device):
    T_diff, L_minus1, r, _ = A_tilde.shape
    L = L_minus1 + 1
    K_total = T_diff * L - 1 if T_diff > 0 else 0

    _dtype = torch.float64
    I_r = torch.eye(r, dtype=_dtype, device=device)
    Q_chain = (q_scale * I_r).expand(K_total, r, r).contiguous()
    R_chain = (r_scale * I_r).expand(K_total, r, r).contiguous()
    S_T = (qf_scale * I_r).contiguous()

    A_dev = A_tilde.to(device=device, dtype=_dtype)
    B_dev = B_tilde.to(device=device, dtype=_dtype)
    A_chain = torch.zeros(K_total, r, r, dtype=_dtype, device=device)
    for t in range(T_diff):
        for l in range(L - 1):
            A_chain[t * L + l] = A_dev[t, l]
        if t < T_diff - 1 and B_dev.numel():
            A_chain[t * L + (L - 1)] = B_dev[t]

    Tn = A_chain.shape[0]
    S = torch.zeros(Tn + 1, r, r, dtype=_dtype, device=device)
    K = torch.zeros(Tn, r, r, dtype=_dtype, device=device)
    S[Tn] = S_T
    for k in reversed(range(Tn)):
        Ak = A_chain[k]
        P = S[k + 1] + R_chain[k]
        F = S[k + 1] @ Ak
        G = Q_chain[k] + Ak.transpose(-2, -1) @ S[k + 1] @ Ak
        Kk = torch.linalg.solve(P, F)
        K[k] = Kk
        Snew = G - F.transpose(-2, -1) @ Kk
        S[k] = 0.5 * (Snew + Snew.transpose(-2, -1))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    K_chain = K.float().cpu()

    K_intra = torch.zeros(T_diff, L - 1, r, r, dtype=torch.float32)
    K_step = torch.zeros(max(T_diff - 1, 0), r, r, dtype=torch.float32)
    for t in range(T_diff):
        for l in range(L - 1):
            K_intra[t, l] = K_chain[t * L + l]
        if t < T_diff - 1:
            K_step[t] = K_chain[t * L + (L - 1)]
    return K_intra, K_step


class SteeringRuntime:
    def __init__(
        self,
        *,
        L,
        T_diff,
        sel_t,
        denoise_t_start,
        denoise_t_end,
        T_p_denoise,
        H_p,
        W_p,
        D,
        sampling_steps,
        lambda_scale,
        vcache,
        lqr,
    ):
        self.L = L
        self.T_diff = T_diff
        self.sel_t = list(sel_t)
        self.sel_idx_of = {t: i for i, t in enumerate(self.sel_t)}
        self.denoise_t_start = denoise_t_start
        self.denoise_t_end = denoise_t_end
        self.T_p_denoise = T_p_denoise
        self.H_p = H_p
        self.W_p = W_p
        self.D = D
        self.sampling_steps = sampling_steps

        self.lambda_scale = float(lambda_scale)
        self.vcache = vcache
        self.lqr = lqr

        self.pass_idx = -1
        self.u_step_pending = None
        self.in_ad = False
        self.u_norm_log = []

    def reset_chunk(self):
        self.pass_idx = -1
        self.u_step_pending = None

    def is_selected_step(self, pass_idx):
        if pass_idx < 0:
            return None, None
        step = pass_idx
        sel = self.sel_idx_of.get(step)
        if sel is None:
            return None, None
        return step, sel


def install_lqr_hooks(model, rt):
    L = rt.L
    handles = []

    def _pass_tick(_block, _args):
        if not rt.in_ad:
            rt.pass_idx += 1

    def make_intra_hook(l_in):
        def hook(block, args, output):
            if rt.in_ad:
                return None
            step, sel = rt.is_selected_step(rt.pass_idx)
            if step is None:
                return None
            z_full = output[0, rt.denoise_t_start : rt.denoise_t_end, :, :, :].detach().reshape(-1)
            z_dt = output.dtype
            V_in = rt.vcache.for_layer(l_in, step)
            rt.in_ad = True
            try:
                x_proj = (z_full.to(V_in.dtype) @ V_in).float()
                v_fp = rt.lqr["v"][l_in, sel]
                mu_fp = rt.lqr["mu"][l_in, sel]
                K_fp = rt.lqr["K_intra"][sel, l_in]
                alpha = rt.lambda_scale * mu_fp - v_fp @ x_proj
                u_tilde = K_fp @ (alpha * v_fp)
                rt.u_norm_log.append((sel, l_in, float(u_tilde.norm()), float(alpha)))
            finally:
                rt.in_ad = False
            del V_in
            V_out = rt.vcache.for_layer(l_in + 1, step)
            rt.in_ad = True
            try:
                u_full = V_out @ u_tilde.to(V_out.dtype)
            finally:
                rt.in_ad = False
            u_add = u_full.to(z_dt).reshape(rt.T_p_denoise, rt.H_p, rt.W_p, rt.D)
            output[0, rt.denoise_t_start : rt.denoise_t_end, :, :, :] = (
                output[0, rt.denoise_t_start : rt.denoise_t_end, :, :, :] + u_add
            )
            return output

        return hook

    def cross_step_compute(block, args, output):
        if rt.in_ad:
            return None
        step, sel = rt.is_selected_step(rt.pass_idx)
        if step is None or sel >= rt.T_diff - 1:
            return None
        z_full = output[0, rt.denoise_t_start : rt.denoise_t_end, :, :, :].detach().reshape(-1)
        V_in = rt.vcache.for_layer(L - 1, step)
        rt.in_ad = True
        try:
            x_proj = (z_full.to(V_in.dtype) @ V_in).float()
            v_fp = rt.lqr["v"][L - 1, sel]
            mu_fp = rt.lqr["mu"][L - 1, sel]
            K_fp = rt.lqr["K_step"][sel]
            alpha = rt.lambda_scale * mu_fp - v_fp @ x_proj
            u_tilde = K_fp @ (alpha * v_fp)
            rt.u_norm_log.append((sel, -1, float(u_tilde.norm()), float(alpha)))
            rt.u_step_pending = {"src_sel": sel, "u_tilde": u_tilde.detach()}
        finally:
            rt.in_ad = False
        return None

    def cross_step_apply(block, args, output):
        if rt.in_ad:
            return None
        step, sel = rt.is_selected_step(rt.pass_idx)
        pending = rt.u_step_pending
        if step is None or pending is None or sel == 0 or pending["src_sel"] != sel - 1:
            return None
        u_tilde = pending["u_tilde"]
        V_dest = rt.vcache.for_layer(0, step)
        rt.in_ad = True
        try:
            u_full = V_dest @ u_tilde.to(V_dest.dtype)
        finally:
            rt.in_ad = False
        u_add = u_full.to(output.dtype).reshape(rt.T_p_denoise, rt.H_p, rt.W_p, rt.D)
        output[0, rt.denoise_t_start : rt.denoise_t_end, :, :, :] = (
            output[0, rt.denoise_t_start : rt.denoise_t_end, :, :, :] + u_add
        )
        rt.u_step_pending = None
        return output

    handles.append(model.net.blocks[0].register_forward_pre_hook(_pass_tick))
    handles.append(model.net.blocks[0].register_forward_hook(cross_step_apply))
    for l_in in range(L - 1):
        handles.append(model.net.blocks[l_in + 1].register_forward_hook(make_intra_hook(l_in)))
    handles.append(model.net.blocks[L - 1].register_forward_hook(cross_step_compute))
    return handles
