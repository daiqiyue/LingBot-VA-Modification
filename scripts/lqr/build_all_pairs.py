import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    obj = np.load(path)
    return {k: obj[k] for k in obj.files}


def _validate_pair_shapes(pos: Dict[str, np.ndarray], neg: Dict[str, np.ndarray], pair_name: str) -> int:
    required = ("primary_images", "wrist_images", "proprios")
    for key in required:
        if key not in pos or key not in neg:
            raise ValueError(f"[{pair_name}] missing key {key} in positive/negative npz")
        if pos[key].shape[0] != neg[key].shape[0]:
            raise ValueError(
                f"[{pair_name}] sample count mismatch for {key}: "
                f"positive={pos[key].shape[0]} negative={neg[key].shape[0]}"
            )
    n = int(pos["primary_images"].shape[0])
    return n


def _load_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple perturbation pair dirs into one all-perturbation pair set.")
    parser.add_argument(
        "--collect-dir",
        type=Path,
        required=True,
        help="Collection root from run_collect_inputs.py (contains pairs/<variant>/...).",
    )
    parser.add_argument(
        "--pair-variants",
        type=str,
        default=None,
        help="Optional comma-separated variant names. Default: all subdirs in pairs/.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output merged pair directory.")
    args = parser.parse_args()

    pairs_root = args.collect_dir / "pairs"
    if not pairs_root.exists():
        raise FileNotFoundError(f"pairs directory not found: {pairs_root}")

    if args.pair_variants:
        variants = [v.strip() for v in args.pair_variants.split(",") if v.strip()]
    else:
        variants = sorted([p.name for p in pairs_root.iterdir() if p.is_dir()])

    if not variants:
        raise ValueError("No pair variants selected.")

    pos_primary: List[np.ndarray] = []
    pos_wrist: List[np.ndarray] = []
    pos_prop: List[np.ndarray] = []
    neg_primary: List[np.ndarray] = []
    neg_wrist: List[np.ndarray] = []
    neg_prop: List[np.ndarray] = []
    per_variant_counts = {}
    per_variant_prompts = {}
    per_variant_task_languages = {}

    for name in variants:
        pdir = pairs_root / name
        pos = _load_npz(pdir / "positive.npz")
        neg = _load_npz(pdir / "negative.npz")
        n = _validate_pair_shapes(pos, neg, name)
        per_variant_counts[name] = n

        pos_primary.append(pos["primary_images"])
        pos_wrist.append(pos["wrist_images"])
        pos_prop.append(pos["proprios"])
        neg_primary.append(neg["primary_images"])
        neg_wrist.append(neg["wrist_images"])
        neg_prop.append(neg["proprios"])

        pair_manifest = _load_json_if_exists(pdir / "manifest.json")
        if pair_manifest:
            prompt = pair_manifest.get("prompt")
            task_language = pair_manifest.get("task_language")
            if prompt:
                per_variant_prompts[name] = prompt
            if task_language:
                per_variant_task_languages[name] = task_language

    merged_positive = {
        "primary_images": np.concatenate(pos_primary, axis=0),
        "wrist_images": np.concatenate(pos_wrist, axis=0),
        "proprios": np.concatenate(pos_prop, axis=0),
    }
    merged_negative = {
        "primary_images": np.concatenate(neg_primary, axis=0),
        "wrist_images": np.concatenate(neg_wrist, axis=0),
        "proprios": np.concatenate(neg_prop, axis=0),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pos_fp = args.out_dir / "positive.npz"
    neg_fp = args.out_dir / "negative.npz"
    np.savez(pos_fp, **merged_positive)
    np.savez(neg_fp, **merged_negative)

    collect_manifest = _load_json_if_exists(args.collect_dir / "manifest.json")
    merged_task_language = None
    merged_prompt = None
    if collect_manifest:
        merged_task_language = collect_manifest.get("task_language")
        merged_prompt = collect_manifest.get("task_language")
    if not merged_task_language and per_variant_task_languages:
        uniq = sorted(set(per_variant_task_languages.values()))
        if len(uniq) == 1:
            merged_task_language = uniq[0]
    if not merged_prompt and per_variant_prompts:
        uniq = sorted(set(per_variant_prompts.values()))
        if len(uniq) == 1:
            merged_prompt = uniq[0]
    if merged_prompt is None and merged_task_language is not None:
        merged_prompt = merged_task_language

    manifest = {
        "mode": "all_perturbation_merged_pairs",
        "collect_dir": str(args.collect_dir.resolve()),
        "source_pair_variants": variants,
        "per_variant_counts": per_variant_counts,
        "per_variant_prompts": per_variant_prompts,
        "per_variant_task_languages": per_variant_task_languages,
        "task_language": merged_task_language,
        "prompt": merged_prompt,
        "num_pairs_total": int(merged_positive["primary_images"].shape[0]),
        "files": {
            "positive": str(pos_fp.resolve()),
            "negative": str(neg_fp.resolve()),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[merge] variants: {variants}")
    print(f"[merge] total pairs: {manifest['num_pairs_total']}")
    print(f"[merge] wrote {pos_fp}")
    print(f"[merge] wrote {neg_fp}")
    print(f"[merge] wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
