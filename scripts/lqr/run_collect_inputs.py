import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from scripts.lqr.common import maybe_load_yaml


def _extract_obs(obs):
    # Match Lingbot LIBERO client preprocessing.
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return agentview, eye_in_hand


def _construct_env(env_args):
    for _ in range(5):
        try:
            return OffScreenRenderEnv(**env_args)
        except Exception:
            continue
    raise RuntimeError("Failed to construct OffScreenRenderEnv after retries")


def _normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"cannot normalize near-zero vector: {v.tolist()}")
    return v / n


def _axis_angle_to_quat(axis, angle_rad):
    axis = _normalize(axis)
    h = angle_rad / 2.0
    return np.array(
        [np.cos(h), axis[0] * np.sin(h), axis[1] * np.sin(h), axis[2] * np.sin(h)],
        dtype=np.float64,
    )


def _quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _rotate_vector(vec, axis, angle_rad):
    vec = np.asarray(vec, dtype=np.float64)
    axis = _normalize(axis)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def _rotmat_to_quat_wxyz(rotmat):
    m = np.asarray(rotmat, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _find_body_pos_by_keywords(env_in, include_keywords):
    model = env_in.sim.model
    data = env_in.sim.data
    include_keywords = tuple(k.lower() for k in include_keywords)
    candidates = []
    for body_id in range(model.nbody):
        body_name = model.body_id2name(body_id)
        if not body_name:
            continue
        body_name_l = body_name.lower()
        if all(k in body_name_l for k in include_keywords):
            candidates.append((len(body_name), body_id, body_name))
    if not candidates:
        return None, None
    _, body_id, body_name = sorted(candidates, key=lambda x: (x[0], x[2]))[0]
    return np.asarray(data.body_xpos[body_id], dtype=np.float64).copy(), body_name


def _lookat_quat_wxyz(cam_pos, target_pos, world_up=np.array([0.0, 0.0, 1.0], dtype=np.float64)):
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    forward = _normalize(target_pos - cam_pos)
    cam_z = -forward
    cam_x = np.cross(world_up, cam_z)
    if np.linalg.norm(cam_x) < 1e-6:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cam_x = np.cross(alt_up, cam_z)
    cam_x = _normalize(cam_x)
    cam_y = _normalize(np.cross(cam_z, cam_x))
    rotmat = np.column_stack([cam_x, cam_y, cam_z])
    return _rotmat_to_quat_wxyz(rotmat)


def _infer_agentview_orbit_center_and_target(env_in):
    robot_base_pos, _ = _find_body_pos_by_keywords(env_in, ["robot0", "base"])
    if robot_base_pos is None:
        robot_base_pos, _ = _find_body_pos_by_keywords(env_in, ["robot0"])
    table_pos, _ = _find_body_pos_by_keywords(env_in, ["table"])
    if robot_base_pos is not None and table_pos is not None:
        center = 0.5 * (robot_base_pos + table_pos)
        target = center.copy()
        target[2] = max(robot_base_pos[2], table_pos[2]) + 0.08
        return center, target
    if table_pos is not None:
        center = table_pos.copy()
        target = table_pos.copy()
        target[2] += 0.10
        return center, target
    if robot_base_pos is not None:
        center = robot_base_pos.copy()
        target = robot_base_pos.copy()
        target[2] += 0.18
        return center, target
    return None, None


def _apply_agentview_camera_rotation(env_in, rotate_deg: Optional[float], rotate_axis: str = "z"):
    if rotate_deg is None or rotate_deg == 0:
        return
    axis_by_name = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if rotate_axis not in axis_by_name:
        raise ValueError(f"camera axis must be x/y/z, got {rotate_axis}")
    axis = axis_by_name[rotate_axis]
    cam_id = env_in.sim.model.camera_name2id("agentview")
    old_cam_pos = np.asarray(env_in.sim.model.cam_pos[cam_id], dtype=np.float64).copy()
    old_quat = np.asarray(env_in.sim.model.cam_quat[cam_id], dtype=np.float64)
    center, target = _infer_agentview_orbit_center_and_target(env_in)
    if center is None or target is None:
        dq = _axis_angle_to_quat(axis, np.deg2rad(rotate_deg))
        new_q = _quat_multiply(dq, old_quat)
        new_q = new_q / np.linalg.norm(new_q)
        env_in.sim.model.cam_quat[cam_id] = new_q
        env_in.sim.forward()
        return
    rel_vec = old_cam_pos - center
    new_cam_pos = center + _rotate_vector(rel_vec, axis, np.deg2rad(rotate_deg))
    new_quat = _lookat_quat_wxyz(new_cam_pos, target)
    env_in.sim.model.cam_pos[cam_id] = new_cam_pos
    env_in.sim.model.cam_quat[cam_id] = new_quat
    env_in.sim.forward()


def _apply_eef_delta_preposition(
    env_in,
    raw_obs,
    eef_delta=None,
    max_steps=80,
    step_size=0.01,
    tolerance=0.01,
):
    if eef_delta is None:
        return raw_obs
    start_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
    target_pos = start_pos + np.asarray(eef_delta, dtype=np.float64)
    obs = raw_obs
    for _ in range(int(max_steps)):
        cur_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        error = target_pos - cur_pos
        dist = float(np.linalg.norm(error))
        if dist <= float(tolerance):
            break
        delta = error
        if dist > float(step_size):
            delta = error / dist * float(step_size)
        action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, _, done, _ = env_in.step(action)
        if done:
            break
    return obs


def _apply_noise(rng: np.random.Generator, img: np.ndarray, sigma: float) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=float(sigma), size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _load_variants(spec_path: Optional[Path], noise_sigma: Optional[float]) -> List[Dict[str, Any]]:
    if spec_path is None:
        if noise_sigma is None:
            return [{"name": "nominal"}]
        return [
            {"name": "nominal"},
            {"name": f"image_noise_sigma_{int(noise_sigma)}", "image_noise_sigma": float(noise_sigma)},
        ]
    spec = maybe_load_yaml(str(spec_path))
    variants = spec.get("variants", [])
    if not variants:
        raise ValueError("perturb spec must contain non-empty `variants` list")
    out = []
    for i, v in enumerate(variants):
        if "name" not in v:
            v["name"] = f"variant_{i}"
        out.append(v)
    return out


def _select_variants(
    variants: List[Dict[str, Any]],
    mode: str,
    target_variants: Optional[str],
) -> List[Dict[str, Any]]:
    keep_names = None
    if target_variants:
        keep_names = {x.strip() for x in target_variants.split(",") if x.strip()}
    selected = []
    for v in variants:
        name = v["name"]
        if keep_names is not None and name not in keep_names:
            continue
        if mode == "nominal" and name != "nominal":
            continue
        if mode == "perturbed" and name == "nominal":
            continue
        selected.append(v)
    if mode != "nominal" and not any(v["name"] == "nominal" for v in selected):
        # Always include nominal for pair generation.
        nominal = [v for v in variants if v["name"] == "nominal"]
        if nominal:
            selected = nominal + selected
        else:
            selected = [{"name": "nominal"}] + selected
    if not selected:
        raise ValueError("No variants selected after filtering.")
    return selected


def _collect_variant_samples(
    env_args: Dict[str, Any],
    init_states: np.ndarray,
    sample_init_ids: List[int],
    warmup_steps: int,
    variant: Dict[str, Any],
    seed: int,
):
    rng = np.random.default_rng(seed=seed)
    env = _construct_env(env_args)
    primary = []
    wrist = []
    proprios = []
    try:
        for init_idx in sample_init_ids:
            env.reset()
            raw_obs = env.set_init_state(init_states[init_idx])
            for _ in range(warmup_steps):
                raw_obs, _, _, _ = env.step([0.0] * 7)

            _apply_agentview_camera_rotation(
                env,
                rotate_deg=variant.get("camera_rotate_deg"),
                rotate_axis=str(variant.get("camera_axis", "z")),
            )
            if variant.get("camera_rotate_deg") not in (None, 0):
                raw_obs, _, _, _ = env.step([0.0] * 7)

            raw_obs = _apply_eef_delta_preposition(
                env,
                raw_obs,
                eef_delta=variant.get("eef_delta"),
                max_steps=variant.get("eef_preposition_steps", 80),
                step_size=variant.get("eef_step_size", 0.01),
                tolerance=variant.get("eef_tolerance", 0.01),
            )

            agent, eye = _extract_obs(raw_obs)
            sigma = variant.get("image_noise_sigma", None)
            if sigma is not None and float(sigma) > 0:
                agent = _apply_noise(rng, agent, float(sigma))
                eye = _apply_noise(rng, eye, float(sigma))
            primary.append(agent)
            wrist.append(eye)
            proprios.append(np.asarray(raw_obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)), dtype=np.float32))
    finally:
        env.close()
    return {
        "primary_images": np.stack(primary, axis=0),
        "wrist_images": np.stack(wrist, axis=0),
        "proprios": np.stack(proprios, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Lingbot-native perturbation variants and pair files for LQR.")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--perturb-spec", type=Path, default=None, help="YAML/JSON with `variants` list.")
    parser.add_argument("--mode", choices=["nominal", "perturbed", "both"], default="both")
    parser.add_argument("--target-variants", type=str, default=None, help="Optional comma-separated names from perturb spec.")
    parser.add_argument("--noise-sigma", type=float, default=None, help="Fallback when no perturb-spec is provided.")
    parser.add_argument("--export-pairs", action="store_true", default=True)
    parser.add_argument("--no-export-pairs", dest="export_pairs", action="store_false")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants_dir = args.out_dir / "variants"
    pairs_dir = args.out_dir / "pairs"
    variants_dir.mkdir(parents=True, exist_ok=True)
    if args.export_pairs:
        pairs_dir.mkdir(parents=True, exist_ok=True)

    bench_cls = benchmark.get_benchmark_dict()[args.libero_benchmark]
    bench = bench_cls()
    task = bench.get_task(args.task_id)
    task_lang = task.language
    init_states = bench.get_task_init_states(args.task_id)
    sample_init_ids = [i % init_states.shape[0] for i in range(int(args.num_samples))]

    env_args = {
        "bddl_file_name": bench.get_task_bddl_file_path(args.task_id),
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
    }

    variants = _load_variants(args.perturb_spec, args.noise_sigma)
    selected_variants = _select_variants(variants, mode=args.mode, target_variants=args.target_variants)

    variant_files = {}
    variant_arrays = {}
    for idx, variant in enumerate(selected_variants):
        name = str(variant["name"])
        data = _collect_variant_samples(
            env_args=env_args,
            init_states=init_states,
            sample_init_ids=sample_init_ids,
            warmup_steps=int(args.warmup_steps),
            variant=variant,
            seed=int(args.seed + idx * 1000),
        )
        out_npz = variants_dir / f"{name}.npz"
        np.savez(out_npz, **data)
        variant_arrays[name] = data
        variant_files[name] = str(out_npz.resolve())
        print(f"[collect] variant={name} -> {out_npz}")

    pair_files = {}
    if args.export_pairs and "nominal" in variant_arrays:
        nominal = variant_arrays["nominal"]
        for name, data in variant_arrays.items():
            if name == "nominal":
                continue
            pdir = pairs_dir / name
            pdir.mkdir(parents=True, exist_ok=True)
            pos_fp = pdir / "positive.npz"
            neg_fp = pdir / "negative.npz"
            np.savez(pos_fp, **nominal)
            np.savez(neg_fp, **data)
            pair_manifest = {
                "pair_name": name,
                "positive_variant": "nominal",
                "negative_variant": name,
                "prompt": task_lang,
                "task_language": task_lang,
                "num_samples": int(args.num_samples),
                "files": {
                    "positive": str(pos_fp.resolve()),
                    "negative": str(neg_fp.resolve()),
                },
            }
            (pdir / "manifest.json").write_text(json.dumps(pair_manifest, indent=2), encoding="utf-8")
            pair_files[name] = {
                "positive": str(pos_fp.resolve()),
                "negative": str(neg_fp.resolve()),
                "manifest": str((pdir / "manifest.json").resolve()),
            }
            print(f"[pairs] nominal vs {name} -> {pdir}")

    manifest = {
        "libero_benchmark": args.libero_benchmark,
        "task_id": int(args.task_id),
        "task_language": task_lang,
        "num_samples": int(args.num_samples),
        "camera_height": int(args.camera_height),
        "camera_width": int(args.camera_width),
        "warmup_steps": int(args.warmup_steps),
        "seed": int(args.seed),
        "mode": args.mode,
        "target_variants": args.target_variants,
        "variants": selected_variants,
        "variant_files": variant_files,
        "pair_files": pair_files,
    }
    manifest_fp = args.out_dir / "manifest.json"
    manifest_fp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[collect] wrote {manifest_fp}")


if __name__ == "__main__":
    main()
