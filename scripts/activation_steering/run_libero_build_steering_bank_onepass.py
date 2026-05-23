import argparse
import os
import signal
import socket
import subprocess
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from scripts.activation_steering.common import ensure_dir, maybe_load_yaml, read_jsonl, write_json


def _aggregate(stacked: torch.Tensor, method: str) -> torch.Tensor:
    if stacked.shape[0] == 0:
        raise ValueError("No samples provided for aggregation")
    if method == "mean":
        return stacked.mean(dim=0)
    if method == "trimmed_mean":
        if stacked.shape[0] < 5:
            return stacked.mean(dim=0)
        lo = int(0.1 * stacked.shape[0])
        hi = max(lo + 1, int(0.9 * stacked.shape[0]))
        vals, _ = torch.sort(stacked, dim=0)
        return vals[lo:hi].mean(dim=0)
    if method == "median_of_means":
        k = min(5, max(1, stacked.shape[0]))
        chunks = torch.chunk(stacked, k, dim=0)
        means = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
        return means.median(dim=0).values
    raise ValueError(f"Unknown agg method: {method}")


def _l2_normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(p=2) + 1e-8)


def _load_variants(spec_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = maybe_load_yaml(spec_path)
    nominal = None
    perturbed: List[Dict[str, Any]] = []
    for v in spec.get("variants", []):
        name = str(v.get("name", ""))
        if name == "nominal":
            nominal = v
        else:
            perturbed.append(v)
    if nominal is None:
        nominal = {"name": "nominal"}
    if not perturbed:
        raise ValueError("No non-nominal variant found in perturb spec.")
    return nominal, perturbed


def _build_trace_server_cmd(args: argparse.Namespace, run_tag: str) -> List[str]:
    cmd = [
        "python",
        "scripts/activation_steering/hook_trace_activations.py",
        "--config-name",
        args.config_name,
        "--port",
        str(args.port),
        "--out-dir",
        args.trace_out_dir,
        "--layers",
        args.layers,
        "--modality",
        args.modality,
        "--run-tag",
        run_tag,
    ]
    if args.token_policy:
        cmd += ["--token-policy", args.token_policy]
    return cmd


def _build_client_cmd(args: argparse.Namespace, out_dir: str, variant: Optional[Dict[str, Any]]) -> List[str]:
    cmd = [
        "python",
        "evaluation/libero/client.py",
        "--libero-benchmark",
        args.libero_benchmark,
        "--port",
        str(args.port),
        "--test-num",
        str(args.num_episodes),
        "--task-range",
        str(args.task_range[0]),
        str(args.task_range[1]),
        "--out-dir",
        out_dir,
    ]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if variant:
        if variant.get("eef_delta") is not None:
            dx, dy, dz = variant["eef_delta"]
            cmd += ["--eef-delta", str(dx), str(dy), str(dz)]
        if variant.get("eef_preposition_steps") is not None:
            cmd += ["--eef-preposition-steps", str(int(variant["eef_preposition_steps"]))]
        if variant.get("eef_step_size") is not None:
            cmd += ["--eef-step-size", str(float(variant["eef_step_size"]))]
        if variant.get("eef_tolerance") is not None:
            cmd += ["--eef-tolerance", str(float(variant["eef_tolerance"]))]
        if variant.get("camera_rotate_deg") is not None:
            cmd += ["--agentview-camera-rotate-deg", str(float(variant["camera_rotate_deg"]))]
            cmd += ["--agentview-camera-rotate-axis", str(variant.get("camera_axis", "z"))]
    return cmd


def _wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_trace_pass(args: argparse.Namespace, run_tag: str, variant: Optional[Dict[str, Any]], out_dir: str) -> None:
    server_cmd = _build_trace_server_cmd(args, run_tag)
    print(f"[onepass] starting trace server ({run_tag}): {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd)
    try:
        if not _wait_for_port("127.0.0.1", args.port, args.startup_wait_sec):
            raise RuntimeError(f"Trace server for {run_tag} did not listen on port {args.port}")
        client_cmd = _build_client_cmd(args, out_dir=out_dir, variant=variant)
        print(f"[onepass] running client ({run_tag}): {' '.join(client_cmd)}")
        subprocess.run(client_cmd, check=True)
    finally:
        _stop_process(server_proc)
        time.sleep(2)


def _collect_samples(
    manifest_path: str,
    phase_filter: Optional[str],
    update_cache_filter: Optional[int],
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    rows = read_jsonl(manifest_path)
    out: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if phase_filter is not None and r.get("phase") != phase_filter:
            continue
        if update_cache_filter is not None and int(r.get("update_cache", -1)) != int(update_cache_filter):
            continue
        payload = torch.load(r["path"], map_location="cpu", weights_only=False)
        mode = str(payload.get("meta", {}).get("mode", r.get("mode", "video")))
        for layer, vec in payload.get("layers", {}).items():
            if not torch.is_tensor(vec):
                vec = torch.tensor(vec)
            vec = vec.float()
            if vec.ndim > 1:
                vec = vec.mean(dim=0)
            out[mode][str(layer)].append(vec)
    return out


def _build_bank_from_samples(
    pos_samples: Dict[str, Dict[str, List[torch.Tensor]]],
    neg_samples: Dict[str, Dict[str, List[torch.Tensor]]],
    agg: str,
    normalize: str,
) -> Dict[str, Any]:
    vectors: Dict[str, Dict[str, torch.Tensor]] = {"video": {}, "action": {}}
    stats: Dict[str, Any] = {"sample_count": {}, "missing_keys": []}
    for mode in ("video", "action"):
        layers = sorted(set(pos_samples.get(mode, {}).keys()) | set(neg_samples.get(mode, {}).keys()))
        for layer in layers:
            pos_list = pos_samples.get(mode, {}).get(layer, [])
            neg_list = neg_samples.get(mode, {}).get(layer, [])
            if not pos_list or not neg_list:
                stats["missing_keys"].append(
                    {
                        "mode": mode,
                        "layer": layer,
                        "pos_count": len(pos_list),
                        "neg_count": len(neg_list),
                    }
                )
                continue
            pos_v = _aggregate(torch.stack(pos_list, dim=0), agg)
            neg_v = _aggregate(torch.stack(neg_list, dim=0), agg)
            delta = pos_v - neg_v
            if normalize == "l2":
                delta = _l2_normalize(delta)
            vectors[mode][layer] = delta
            stats["sample_count"][f"{mode}:{layer}"] = {"pos": len(pos_list), "neg": len(neg_list)}
    return {"vectors": vectors, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass LIBERO steering bank construction.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-range", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--port", type=int, default=29056)

    parser.add_argument("--trace-out-dir", type=str, required=True)
    parser.add_argument("--token-policy", type=str, default=None)
    parser.add_argument("--layers", type=str, default="15,19,22,25,27")
    parser.add_argument("--modality", choices=["video", "action", "both"], default="both")
    parser.add_argument("--startup-wait-sec", type=int, default=240)

    parser.add_argument("--perturb-spec", type=str, required=True)
    parser.add_argument("--phase-filter", type=str, default="infer")
    parser.add_argument("--update-cache-filter", type=int, default=0)
    parser.add_argument("--agg", choices=["mean", "trimmed_mean", "median_of_means"], default="trimmed_mean")
    parser.add_argument("--normalize", choices=["l2", "none"], default="l2")
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.trace_out_dir)
    ensure_dir(os.path.dirname(args.out_path) or ".")

    nominal_variant, neg_variants = _load_variants(args.perturb_spec)
    neg_names = [str(v.get("name", f"variant_{i}")) for i, v in enumerate(neg_variants)]
    print(f"[onepass] nominal variant: {nominal_variant.get('name', 'nominal')}")
    print(f"[onepass] perturb variants ({len(neg_variants)}): {neg_names}")

    _run_trace_pass(
        args=args,
        run_tag="nominal_trace",
        variant=nominal_variant,
        out_dir=os.path.join(args.trace_out_dir, "trace_rollout_nominal"),
    )
    neg_manifest_paths: List[str] = []
    for i, neg_variant in enumerate(neg_variants):
        run_tag = f"perturbed_trace_{i:02d}_{str(neg_variant.get('name', 'variant')).replace(' ', '_')}"
        _run_trace_pass(
            args=args,
            run_tag=run_tag,
            variant=neg_variant,
            out_dir=os.path.join(args.trace_out_dir, f"trace_rollout_{run_tag}"),
        )
        neg_manifest_paths.append(os.path.join(args.trace_out_dir, run_tag, "traces_manifest.jsonl"))

    pos_manifest = os.path.join(args.trace_out_dir, "nominal_trace", "traces_manifest.jsonl")
    if not os.path.exists(pos_manifest):
        raise FileNotFoundError(f"Missing nominal manifest: {pos_manifest}")
    for p in neg_manifest_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing perturbed manifest: {p}")

    pos_samples = _collect_samples(
        pos_manifest,
        phase_filter=args.phase_filter,
        update_cache_filter=args.update_cache_filter,
    )
    neg_samples: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    for neg_manifest in neg_manifest_paths:
        cur = _collect_samples(
            neg_manifest,
            phase_filter=args.phase_filter,
            update_cache_filter=args.update_cache_filter,
        )
        for mode, mode_layers in cur.items():
            for layer, layer_vecs in mode_layers.items():
                neg_samples[mode][layer].extend(layer_vecs)

    bank_payload = _build_bank_from_samples(
        pos_samples=pos_samples,
        neg_samples=neg_samples,
        agg=args.agg,
        normalize=args.normalize,
    )

    nonempty = sum(len(v) for v in bank_payload["vectors"].values())
    if nonempty == 0:
        raise RuntimeError("No steering vectors were produced. Check phase/update_cache filters and traces.")

    torch.save(bank_payload, args.out_path)
    write_json(
        args.out_path + ".meta.json",
        {
            "trace_out_dir": args.trace_out_dir,
            "nominal_variant": nominal_variant,
            "perturb_variants": neg_variants,
            "task_range": args.task_range,
            "num_episodes": args.num_episodes,
            "phase_filter": args.phase_filter,
            "update_cache_filter": args.update_cache_filter,
            "stats": bank_payload["stats"],
            "layers": {k: sorted(list(v.keys())) for k, v in bank_payload["vectors"].items()},
        },
    )
    print(f"[onepass] saved steering bank: {args.out_path}")


if __name__ == "__main__":
    main()
