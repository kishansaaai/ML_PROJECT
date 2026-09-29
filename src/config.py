from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_root"] = str(path.parent)
    return cfg


def project_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_root"]) / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for_run() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
