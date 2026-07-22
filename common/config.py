from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml # type: ignore


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            return value
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_config(path: str | os.PathLike) -> SimpleNamespace:
    cfg_path = Path(path).resolve()

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = cfg_path.parents[1]
    raw["project_root"] = str(project_root)
    raw["config_path"] = str(cfg_path)

    def resolve_path(p):
        if p is None:
            return None
        p = str(p)
        if os.path.isabs(p) or ":" in p[:3]:
            return p
        return str(project_root / p)

    raw["model_yaml"] = resolve_path(raw.get("model_yaml"))
    raw["output_dir"] = resolve_path(raw.get("output_dir", "outputs/baseline"))
    raw["resume_weights"] = resolve_path(raw.get("resume_weights"))

    data_dir = raw.get("data_dir", "data/dummy")
    if not os.path.exists(data_dir) and os.path.exists("data/dummy"):
        print(f"[config] Config data_dir '{data_dir}' not found; using local fallback 'data/dummy'")
        data_dir = "data/dummy"

    raw["train_images"] = os.path.join(data_dir, "images", "train")
    raw["train_labels"] = os.path.join(data_dir, "labels", "train")
    raw["val_images"] = os.path.join(data_dir, "images", "val")
    raw["val_labels"] = os.path.join(data_dir, "labels", "val")
    raw["test_images"] = os.path.join(data_dir, "images", "test")
    raw["test_labels"] = os.path.join(data_dir, "labels", "test")

    names = raw.get("names", {})
    if isinstance(names, dict):
        raw["names"] = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        raw["names"] = {i: str(v) for i, v in enumerate(names)}

    return _to_namespace(raw)


def ensure_output_dirs(cfg: SimpleNamespace) -> None:
    out = Path(cfg.output_dir)
    for sub in ["checkpoints", "logs", "plots", "predictions"]:
        (out / sub).mkdir(parents=True, exist_ok=True)