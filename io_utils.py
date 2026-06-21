"""Small path and JSON helpers for the thesis scripts."""

import math
import os
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch


def ensure_dir(path) -> None:
    os.makedirs(path, exist_ok=True)


def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def suffix_from_saved_models_name(model_dir):
    name = os.path.basename(os.path.normpath(str(model_dir)))
    if name.startswith("Saved_models_"):
        return name.replace("Saved_models_", "", 1)
    return name


def infer_history_dir(model_dir, history_root, history_prefix: str = "History_"):
    suffix = suffix_from_saved_models_name(model_dir)
    return os.path.join(str(history_root), history_prefix + suffix)


def infer_results_dir(model_dir, results_root, results_prefix: str = "Results_"):
    suffix = suffix_from_saved_models_name(model_dir)
    return os.path.join(str(results_root), results_prefix + suffix)


def checkpoint_path(model_dir, fold, checkpoint_name: str = "best_auprc.pt"):
    return os.path.join(str(model_dir), f"GAT_CV_10_{fold}", checkpoint_name)


def available_folds(model_dir, n_folds: int = 10, checkpoint_name: str = "best_auprc.pt") -> List[int]:
    return [
        fold
        for fold in range(n_folds)
        if os.path.isfile(checkpoint_path(model_dir, fold, checkpoint_name))
    ]


def load_checkpoint(path, map_location="cpu"):
    return torch.load(str(path), map_location=map_location, weights_only=False)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch values to JSON-safe Python values."""
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return to_jsonable(obj.item())
        return to_jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return to_jsonable(obj.item())
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return to_jsonable(obj.item())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
