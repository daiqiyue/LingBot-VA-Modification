import json
import os
import time
import hashlib
from typing import Any, Dict, Iterable, List


def maybe_load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"YAML file requested but pyyaml is unavailable: {path}"
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must decode to a dict")
    return data


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_int_list(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def stable_run_id(parts: List[str]) -> str:
    raw = "|".join(parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"run_{digest}"


def now_ts() -> int:
    return int(time.time())


def alpha_schedule(
    mode: str,
    base_alpha: float,
    step_idx: int,
    total_steps: int,
    active_ratio: float = 0.6,
) -> float:
    if total_steps <= 0:
        return base_alpha
    x = step_idx / max(1, total_steps - 1)
    if x > active_ratio:
        return 0.0
    if mode == "flat":
        return base_alpha
    if mode == "linear_decay":
        return base_alpha * (1.0 - x / max(active_ratio, 1e-6))
    if mode == "cosine_decay":
        import math

        y = x / max(active_ratio, 1e-6)
        return base_alpha * 0.5 * (1.0 + math.cos(math.pi * y))
    raise ValueError(f"Unsupported alpha schedule: {mode}")
