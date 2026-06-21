"""
XAI.py (Custom metrics only, clean single-output pipeline)

This script:
1) Loads a trained CNN-GAT (PyG) model and a chosen CV fold.
2) Generates:
   - Temporal explanations: Integrated Gradients on raw node signals (graph_data.x).
   - Spatial explanations: GNNExplainer on CNN node-features (model.cnn(x)) and edges.
3) Quantifies explanation quality using custom metrics only (no Quantus dependency):
   - Continuity  -> local Lipschitz estimate
   - Correctness -> parameter randomisation sanity correlation
   - Coherence   -> global temporal coherence (mean pairwise Jaccard overlap of top-k IG time-blocks across channels)
                  (+ optional Top-K intersection vs channel-level ground truth if provided)
   - FaithfulnessCorrelation -> perturbation-based correlation metric
   - TopKDeletion_drop -> necessity via deleting top-k IG time-blocks (per channel)

Notes:
- Single-output binary model is used directly (no 2-class wrappers).
- MyUtils.calculateIG and MyUtils.calculateGNNexpl are used for explanations.
"""

import os
import json
import copy
import gc
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import Models_senn as Model
import MyUtils_senn_test as MyUtils
from torch_geometric.data import Data
from tqdm import tqdm


# -------------------------
# Utilities
# -------------------------
def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalise_by_absmax(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale attribution by max absolute value (if nonzero)."""
    a = np.asarray(a, dtype=float)
    m = float(np.max(np.abs(a))) if a.size else 0.0
    if m < eps:
        return a.copy()
    return a / m


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch objects to JSON-safe Python types."""
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return _to_jsonable(obj.item())
        return _to_jsonable(obj.detach().cpu().numpy())

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _to_jsonable(obj.item())
        return [_to_jsonable(v) for v in obj.tolist()]

    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj

    return str(obj)


def _extract_numeric_values(obj: Any) -> List[float]:
    """Flatten nested containers to finite numeric values."""
    out: List[float] = []

    def rec(x: Any):
        if torch.is_tensor(x):
            if x.ndim == 0:
                rec(x.item())
            else:
                rec(x.detach().cpu().numpy())
            return
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)
            return
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                rec(x.item())
            else:
                for v in x.ravel():
                    rec(v.item() if hasattr(v, "item") else v)
            return
        if isinstance(x, (str, bytes)):
            return
        if isinstance(x, (np.bool_, bool)):
            out.append(float(bool(x)))
            return
        if isinstance(x, (np.integer, int, np.floating, float)):
            v = float(x)
            if np.isfinite(v):
                out.append(v)
            return

    rec(obj)
    return out


def _reduce_to_channel_importance(a: np.ndarray, positive_only: bool = True) -> np.ndarray:
    """
    Reduce attribution tensor to channel-level importances.
    - IG: shape (12, T)
    - GNN node mask: shape (12, F)
    Returns shape (12,).
    """
    a = _to_numpy(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D attribution (12, *) but got shape {a.shape}")
    if positive_only:
        a = np.maximum(a, 0.0)
    return a.sum(axis=1)


def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)

    vals = x[order]
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str = "spearman") -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")

    if kind == "spearman":
        ra = _rankdata_1d(a)
        rb = _rankdata_1d(b)
        return float(np.corrcoef(ra, rb)[0, 1])

    return float(np.corrcoef(a, b)[0, 1])

def _corr_nan_to_zero(a: np.ndarray, b: np.ndarray, kind: str = "spearman") -> float:
    """Like _safe_corr but returns 0.0 when correlation is undefined (e.g., constant arrays)."""
    v = _safe_corr(a, b, kind=kind)
    return 0.0 if (v != v) else float(v)




def _compute_feature_std(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-feature std over the batch dimension (axis=0)."""
    x = np.asarray(x, dtype=float)
    std = x.std(axis=0, ddof=0)
    return np.maximum(std, eps)


def _sample_relative_noise(std: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with per-feature std = alpha * std."""
    return rng.normal(loc=0.0, scale=alpha * std, size=std.shape)



def randomise_module_parameters_inplace(module: torch.nn.Module) -> None:
    """Reinitialise module parameters in-place using reset_parameters where available."""
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
            try:
                m.reset_parameters()
            except Exception:
                pass


def _extract_model_output(output: Any, key: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if not isinstance(output, dict):
        raise TypeError(f"Expected tensor or dict output, got {type(output)!r}")

    aliases = {
        "prob": ["prob", "probability", "probs", "probabilities"],
        "logit": ["logit", "logits"],
        "explanation": ["explanation", "focus_map", "F_map"],
    }.get(key, [key])

    for name in aliases:
        if name in output:
            return output[name]
    raise KeyError(f"Could not find '{key}' in model output. Available keys: {sorted(output.keys())}")


def _build_model(model_kind: str, ckpt: Dict[str, Any], return_explanations: bool = False) -> torch.nn.Module:
    model_kind = str(model_kind).strip().lower()
    if model_kind == "base":
        print("building base model...")
        model = Model.EEG_GAT_Model()
    elif model_kind == "senn":
        print("building senn rawx model...")
        model = Model.SENN_raw(
            global_min=ckpt.get("global_min", None),
            return_node_scores=False,
            return_fmap=return_explanations,
        )
    elif model_kind == "senn_fixed":
        print("building sen fixed concepts model...")
        model = Model.SENN_fixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=return_explanations,
        )
    else:
        raise ValueError("model_kind must be 'base' or 'senn'")
    model.load_state_dict(ckpt["model_state_dict"])
    return model

# -------------------------
# Prediction helpers (single-output model)
# -------------------------

@torch.no_grad()
def _predict_p1_full(
    model: torch.nn.Module,
    x_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Predict P(class=1) for raw inputs x_batch of shape (B,12,T) or (12,T).
    Supports both legacy tensor outputs and the new dictionary outputs.
    """
    x = torch.tensor(np.asarray(x_batch), dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)

    model.eval()
    out = []
    for i in range(x.shape[0]):
        batch = torch.zeros(x.shape[1], dtype=torch.long, device=device)
        model_out = model(x[i], edge_index, batch)
        p1 = _extract_model_output(model_out, "logit").view(-1)[0]
        out.append(float(p1.item()))
    return np.asarray(out, dtype=float)



@torch.no_grad()
def _predict_p1_gnn(
    gnn: torch.nn.Module,
    x_nf_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Predict P(class=1) from node-feature batches for the GNN head only.
    x_nf_batch shape (B,12,F) or (12,F).
    """
    x = torch.tensor(np.asarray(x_nf_batch), dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)

    gnn.eval()
    out = []
    for i in range(x.shape[0]):
        batch = torch.zeros(x.shape[1], dtype=torch.long, device=device)
        logit = gnn(x[i], edge_index, batch).view(-1)[0]
        p1 = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
        # out.append(float(p1.item()))
        out.append(float(logit.item()))
    return np.asarray(out, dtype=float)


def _target_scores_binary_from_p1(p1: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """
    For true labels y in {0,1}, return target score:
    - y=1 -> p1
    - y=0 -> 1-p1
    """
    y = np.asarray(y_batch, dtype=int).reshape(-1)
    p1 = np.asarray(p1, dtype=float).reshape(-1)
    if p1.size != y.size:
        raise ValueError(f"Shape mismatch in target-score selection: p1={p1.shape}, y={y.shape}")
    return np.where(y == 1, p1, 1.0 - p1)


# -------------------------
# Explainer wrappers (custom, no Quantus)
# -------------------------
def ig_explainer_raw(
    model: torch.nn.Module,
    inputs: Any,
    targets: Any,
    abs: bool = False,
    normalise: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Feature attributions on raw input via MyUtils.calculateIG.
    Returns shape (B, 12, T).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    edge_index: torch.Tensor = kwargs["edge_index"]
    device: torch.device = kwargs["device"]
    thr = kwargs.get("thr", 0.5)

    x_np = _to_numpy(inputs)
    if x_np.ndim == 2:
        x_np = x_np[None, ...]  # (1,12,T)
    # model-type explanations: no targets needed for single-output probability
    t_np = None

    a_list = []
    for i in range(x_np.shape[0]):
        g = Data(
            x=torch.tensor(x_np[i], dtype=torch.float32),
            edge_index=edge_index,
        ).to(device)
        exp = MyUtils.calculateIG(model, g, thr=thr,target_key="logit")

        a = exp.node_mask.detach().cpu().numpy()  # (12,T)

        if abs:
            a = np.abs(a)
        if normalise:
            a = _normalise_by_absmax(a)

        a_list.append(a)

    return np.stack(a_list, axis=0)


def ig_explainer_channel(model, inputs, targets, abs: bool = False, normalise: bool = False, **kwargs) -> np.ndarray:
    """
    IG reduced to channel-level importances, returning (B, 12).
    Positive-only aggregation follows your visual convention.
    """
    a = ig_explainer_raw(model, inputs, targets, abs=False, normalise=False, **kwargs)
    a_ch = np.stack([_reduce_to_channel_importance(ai, positive_only=True) for ai in a], axis=0)

    if abs:
        a_ch = np.abs(a_ch)
    if normalise:
        a_ch = _normalise_by_absmax(a_ch)

    return a_ch


def gnn_node_explainer(model, inputs, targets, abs: bool = False, normalise: bool = False, **kwargs) -> np.ndarray:
    """
    GNNExplainer node mask in node-feature space.
    Inputs expected shape (B, 12, F). Returns shape (B, 12, F).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    edge_index: torch.Tensor = kwargs["edge_index"]
    device: torch.device = kwargs["device"]
    thr = kwargs.get("thr", 0.5)
    epochs = int(kwargs.get("epochs", GNNEXPL_EPOCHS_QUANT))

    x_np = _to_numpy(inputs)
    if x_np.ndim == 2:
        x_np = x_np[None, ...]
    # model-type explanations: no targets needed for single-output probability
    t_np = None

    out = []
    for i in range(x_np.shape[0]):
        g = Data(x=torch.tensor(x_np[i], dtype=torch.float32), edge_index=edge_index).to(device)
        exp = MyUtils.calculateGNNexpl(model, g, thr=thr, epochs=epochs)

        a = exp.node_mask.detach().cpu().numpy()
        if abs:
            a = np.abs(a)
        if normalise:
            a = _normalise_by_absmax(a)
        out.append(a)

    return np.stack(out, axis=0)


def gnn_node_explainer_channel(model, inputs, targets, abs: bool = False, normalise: bool = False, **kwargs) -> np.ndarray:
    """GNN node mask reduced to channel importances, output (B,12)."""
    a = gnn_node_explainer(model, inputs, targets, abs=False, normalise=False, **kwargs)
    a_ch = np.stack([_reduce_to_channel_importance(ai, positive_only=True) for ai in a], axis=0)

    if abs:
        a_ch = np.abs(a_ch)
    if normalise:
        a_ch = _normalise_by_absmax(a_ch)

    return a_ch


def gnn_edge_explainer(model, inputs, targets, abs: bool = False, normalise: bool = False, **kwargs) -> np.ndarray:
    """
    GNNExplainer edge mask in node-feature space.
    Inputs expected shape (B, 12, F). Returns shape (B, E).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gnn_or_model = kwargs.get("model_for_gnn", model)
    edge_index: torch.Tensor = kwargs["edge_index"]
    device: torch.device = kwargs["device"]
    thr = kwargs.get("thr", 0.5)
    epochs = int(kwargs.get("epochs", GNNEXPL_EPOCHS_QUANT))

    x_np = _to_numpy(inputs)
    if x_np.ndim == 2:
        x_np = x_np[None, ...]
    # model-type explanations: no targets needed for single-output probability
    t_np = None

    out = []
    for i in range(x_np.shape[0]):
        g = Data(x=torch.tensor(x_np[i], dtype=torch.float32), edge_index=edge_index).to(device)
        # use full model that exposes .gnn in MyUtils.calculateGNNexpl
        exp = MyUtils.calculateGNNexpl(model, g, thr=thr, epochs=epochs)

        a = exp.edge_mask.detach().cpu().numpy()
        if abs:
            a = np.abs(a)
        if normalise:
            a = _normalise_by_absmax(a)
        out.append(a)

    return np.stack(out, axis=0)



def focusmap_explainer_raw(
    model: torch.nn.Module,
    inputs: Any,
    targets: Any,
    abs: bool = False,
    normalise: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Extract the SENN focus map from the model output dictionary.
    Returns shape (B, 12, T).
    """
    edge_index: torch.Tensor = kwargs["edge_index"]
    device: torch.device = kwargs["device"]

    x_np = _to_numpy(inputs)
    if x_np.ndim == 2:
        x_np = x_np[None, ...]

    a_list = []
    model.eval()
    with torch.no_grad():
        for i in range(x_np.shape[0]):
            x_t = torch.tensor(x_np[i], dtype=torch.float32, device=device)
            batch = torch.zeros(x_t.shape[0], dtype=torch.long, device=device)
            out = model(x_t, edge_index, batch)
            a = _extract_model_output(out, "explanation").detach().cpu().numpy()
            # a = _extract_model_output(out, "theta_x").detach().cpu().numpy() # we can also define explanation by pure relevance scores, althought that is only logical for 
            # fixed stadnarized concepts instead of identity conceptizer. (i.e. sample amplitude at t_i is not interpretable as concepts, therefore focus map)
            # If it were e.g. psd power in e.g. beta-band  it can be described as more feature like and the value of h(x) is less needed
            # Theta only as explanation mostly valid for prototype based concepts, i.e. we tested that concept h_a(x) corresponds to seizure pattern a; then explanation can be taken as theta_a(x) only
            # But for concepts where numerical values are used, we need focus map, since the value of h(x) also is explicitly important in prediciton

            if abs:
                a = np.abs(a)
            if normalise:
                a = _normalise_by_absmax(a)

            a_list.append(a)

    return np.stack(a_list, axis=0)


def focusmap_explainer_channel(model, inputs, targets, abs: bool = False, normalise: bool = False, **kwargs) -> np.ndarray:
    a = focusmap_explainer_raw(model, inputs, targets, abs=False, normalise=False, **kwargs)
    a_ch = np.stack([_reduce_to_channel_importance(ai, positive_only=True) for ai in a], axis=0)
    if abs:
        a_ch = np.abs(a_ch)
    if normalise:
        a_ch = _normalise_by_absmax(a_ch)
    return a_ch


def focusmap_local_lipschitz_estimate(
    *,
    x_raw: np.ndarray,
    y_batch: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    nr_samples: int = 20,
    perturb_std: float = 0.03,
    perturb_alpha: Optional[float] = None,
) -> List[float]:
    x_raw = np.asarray(x_raw, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    rng = np.random.default_rng(0)
    std_raw = _compute_feature_std(x_raw)

    scores: List[float] = []
    for i in range(x_raw.shape[0]):
        x0 = x_raw[i:i + 1]
        y0 = y_batch[i:i + 1]

        a0 = focusmap_explainer_raw(
            model=model,
            inputs=x0,
            targets=y0,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
        )[0]
        x0_flat = x0.reshape(-1)

        ratios = []
        for _ in range(int(nr_samples)):
            if perturb_alpha is not None:
                noise = _sample_relative_noise(std_raw, float(perturb_alpha), rng)[None, ...]
            else:
                noise = rng.normal(loc=0.0, scale=float(perturb_std), size=x0.shape)

            x1 = x0 + noise
            a1 = focusmap_explainer_raw(
                model=model,
                inputs=x1,
                targets=y0,
                abs=False,
                normalise=False,
                edge_index=edge_index,
                device=device,
            )[0]

            num = float(np.linalg.norm((a1 - a0).ravel(), ord=2))
            den = float(np.linalg.norm((x1.reshape(-1) - x0_flat), ord=2) + 1e-12)
            ratios.append(num / den)

        scores.append(float(np.max(ratios)) if len(ratios) else float("nan"))
    return scores


def focusmap_parameter_randomisation_sanity(
    *,
    x_raw: np.ndarray,
    y_batch: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    corr_kind: str = "spearman",
    n_random_models: int = 3,
) -> List[float]:
    x_raw = np.asarray(x_raw, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)

    a_ref = focusmap_explainer_raw(
        model=model,
        inputs=x_raw,
        targets=y_batch,
        abs=False,
        normalise=False,
        edge_index=edge_index,
        device=device,
    )

    state = copy.deepcopy(model.state_dict())
    per_round = []

    for _ in range(int(max(1, n_random_models))):
        randomise_module_parameters_inplace(model)
        model.eval()

        a_rand = focusmap_explainer_raw(
            model=model,
            inputs=x_raw,
            targets=y_batch,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
        )

        corr = [_corr_nan_to_zero(a_ref[i], a_rand[i], kind=corr_kind) for i in range(x_raw.shape[0])]
        per_round.append(corr)

    model.load_state_dict(state)
    model.eval()

    per_round = np.asarray(per_round, dtype=float)
    out = np.nanmean(per_round, axis=0)
    return [float(v) for v in out]

# -------------------------
# Custom metrics
# -------------------------
def feature_local_lipschitz_estimate(
    *,
    x_raw: np.ndarray,
    y_batch: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    nr_samples: int = 20,
    perturb_std: float = 0.03,
    perturb_alpha: Optional[float] = None,
    thr: float = 0.5,
) -> List[float]:
    """Custom local Lipschitz estimate for IG explanations."""

    x_raw = np.asarray(x_raw, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    rng = np.random.default_rng(0)
    std_raw = _compute_feature_std(x_raw)

    normalise_attr = False

    scores: List[float] = []
    for i in range(x_raw.shape[0]):
        x0 = x_raw[i:i + 1]
        y0 = y_batch[i:i + 1]

        a0 = ig_explainer_raw(
            model=model,
            inputs=x0,
            targets=y0,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
            thr=thr,
        )[0]

        if normalise_attr:
                a0 = _normalise_by_absmax(a0)
        x0_flat = x0.reshape(-1)

        ratios = []
        for _ in range(int(nr_samples)):
            if perturb_alpha is not None:
                noise = _sample_relative_noise(std_raw, float(perturb_alpha), rng)[None, ...]
            else:
                noise = rng.normal(loc=0.0, scale=float(perturb_std), size=x0.shape)
            
            x1 = x0 + noise
            a1 = ig_explainer_raw(
                model=model,
                inputs=x1,
                targets=y0,
                abs=False,
                normalise=False,
                edge_index=edge_index,
                device=device,
                thr=thr,
            )[0]

            if normalise_attr:
                a1 = _normalise_by_absmax(a1)

            num = float(np.linalg.norm((a1 - a0).ravel(), ord=2))
            den = float(np.linalg.norm((x1.reshape(-1) - x0_flat), ord=2) + 1e-12)
            ratios.append(num / den)

        scores.append(float(np.max(ratios)) if len(ratios) else float("nan"))
    return scores


def feature_parameter_randomisation_sanity(
    *,
    x_raw: np.ndarray,
    y_batch: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    corr_kind: str = "spearman",
    n_random_models: int = 3,
    thr: float = 0.5,
) -> List[float]:
    """
    Custom MPRT-style sanity check for feature attributions (IG).
    Lower correlation between original and randomized-model explanations is better.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)

    a_ref = ig_explainer_raw(
        model=model,
        inputs=x_raw,
        targets=y_batch,
        abs=False,
        normalise=False,
        edge_index=edge_index,
        device=device,
        thr=thr,
    )

    state = copy.deepcopy(model.state_dict())
    per_round = []

    for _ in range(int(max(1, n_random_models))):
        randomise_module_parameters_inplace(model)
        model.eval()

        a_rand = ig_explainer_raw(
            model=model,
            inputs=x_raw,
            targets=y_batch,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
            thr=thr,
        )

        corr = [_corr_nan_to_zero(a_ref[i], a_rand[i], kind=corr_kind) for i in range(x_raw.shape[0])]
        per_round.append(corr)

    #reset model
    model.load_state_dict(state)
    model.eval()

    per_round = np.asarray(per_round, dtype=float)  # (R,B)
    out = np.nanmean(per_round, axis=0)
    return [float(v) for v in out]


def edge_local_lipschitz_estimate(
    *,
    x_nf: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    nr_samples: int = 5,
    perturb_std: float = 0.05,
    perturb_alpha: Optional[float] = None,
    epochs: int = 80,
    thr: float = 0.5,
) -> List[float]:
    """Custom local Lipschitz estimate for GNNExplainer edge masks."""

    normalise_attr = False

    x_nf = np.asarray(x_nf, dtype=float)
    rng = np.random.default_rng(0)
    std_nf = _compute_feature_std(x_nf)

    scores: List[float] = []
    for i in range(x_nf.shape[0]):
        x0 = x_nf[i:i + 1]
        a0 = gnn_edge_explainer(
            model=model,
            inputs=x0,
            targets=np.array([1]),
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
            epochs=epochs,
            thr=thr,
        )[0]

        if normalise_attr:
                a0 = _normalise_by_absmax(a0)


        x0_flat = x0.reshape(-1)

        ratios = []
        for _ in range(int(nr_samples)):
            
            if perturb_alpha is not None:
                noise = _sample_relative_noise(std_nf, float(perturb_alpha), rng)[None, ...]
            else:
                noise = rng.normal(loc=0.0, scale=float(perturb_std), size=x0.shape)
            x1 = x0 + noise
            x1 = x0 + noise
            a1 = gnn_edge_explainer(
                model=model,
                inputs=x1,
                targets=np.array([1]),
                abs=False,
                normalise=False,
                edge_index=edge_index,
                device=device,
                epochs=epochs,
                thr=thr,
            )[0]

            if normalise_attr:
                a1 = _normalise_by_absmax(a1)

            num = float(np.linalg.norm((a1 - a0).ravel(), ord=2))
            den = float(np.linalg.norm((x1.reshape(-1) - x0_flat), ord=2) + 1e-12)
            ratios.append(num / den)

        scores.append(float(np.max(ratios)) if len(ratios) else float("nan"))

    return scores


def edge_parameter_randomisation_sanity(
    *,
    x_nf: np.ndarray,
    gnn: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    epochs: int = 80,
    corr_kind: str = "spearman",
    n_random_models: int = 3,
) -> List[float]:
    """Custom MPRT-style sanity check for GNN edge attributions."""
    x_nf = np.asarray(x_nf, dtype=float)

    # Build lightweight model shim exposing .cnn and .gnn for MyUtils.calculateGNNexpl
    class _ModelShim(torch.nn.Module):
        def __init__(self, gnn_module):
            super().__init__()
            self.gnn = gnn_module
            self.cnn = torch.nn.Identity()

    model_shim = _ModelShim(gnn).to(device)
    model_shim.eval()

    a_ref = gnn_edge_explainer(
        model=model_shim,
        inputs=x_nf,
        targets=np.ones((x_nf.shape[0],), dtype=int),
        abs=False,
        normalise=False,
        edge_index=edge_index,
        device=device,
        epochs=epochs,
    )

    state = copy.deepcopy(gnn.state_dict())
    rounds = []

    for _ in range(int(max(1, n_random_models))):
        randomise_module_parameters_inplace(gnn)
        gnn.eval()

        a_rand = gnn_edge_explainer(
            model=model_shim,
            inputs=x_nf,
            targets=np.ones((x_nf.shape[0],), dtype=int),
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
            epochs=epochs,
        )

        corr = [_corr_nan_to_zero(a_ref[i], a_rand[i], kind=corr_kind) for i in range(x_nf.shape[0])]
        rounds.append(corr)

    gnn.load_state_dict(state)
    gnn.eval()

    rounds = np.asarray(rounds, dtype=float)
    out = np.nanmean(rounds, axis=0)
    return [float(v) for v in out]


def _edge_to_input_importance(a_edge: np.ndarray, edge_index: torch.Tensor, x_shape: Tuple[int, int]) -> np.ndarray:
    """Map edge attributions (E,) to node-feature importances (N,F)."""
    n_nodes, n_feat = int(x_shape[0]), int(x_shape[1])
    imp = np.zeros((n_nodes, n_feat), dtype=float)
    ei = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    a_edge = np.asarray(a_edge, dtype=float).reshape(-1)

    e_count = min(a_edge.size, ei.shape[1])
    for e in range(e_count):
        u = int(ei[0, e]); v = int(ei[1, e]); w = float(a_edge[e])
        if 0 <= u < n_nodes:
            imp[u, :] += 0.5 * w
        if 0 <= v < n_nodes:
            imp[v, :] += 0.5 * w
    return imp


def _logit(p: float, eps: float = 1e-6,inv: bool = False) -> float:
    if inv:
        # inverse: logit -> probability (sigmoid)
        return float(1.0 / (1.0 + np.exp(-p)))
    else:
        # forward: probability -> logit
        p = float(np.clip(p, eps, 1.0 - eps))
        return float(np.log(p / (1.0 - p)))


def faithfulness_correlation_metric_feature(
    *,
    predict_p1_fn: Callable[[np.ndarray], np.ndarray],
    x_batch: np.ndarray,
    y_batch: np.ndarray,  # kept for API compatibility; not used for model-type faithfulness
    a_batch: np.ndarray,
    nr_runs: int = 40,
    subset_frac: float = 0.10,
    baseline_value: float = 0.0,
    corr_kind: str = "pearson",
    positive_only: bool = False,
    use_logit_drop: bool = True,
    # EEG-specific: mask contiguous time blocks (per channel) instead of individual samples
    block_len: Optional[int] = None,
    min_blocks: int = 1,
) -> List[float]:
    """
    Faithfulness Correlation (model-type, single-output), EEG block masking:

    For each sample i:
      - sample random *contiguous time blocks* (per channel) until ~|S| features are covered
      - replace those values with baseline
      - compute score drop: f(x) - f(x_pert), where f is p1 (or logit(p1))
      - compute attribution mass over the masked subset

    Returns one correlation per sample in [-1, 1].

    Notes:
      - This avoids the "remove one timepoint" issue in EEG where neighboring samples are highly correlated.
      - Works for x shaped (B, C, T). For non-2D per-sample shapes, falls back to flat random masking.
    """
    x_batch = np.asarray(x_batch, dtype=float)
    a_batch = np.asarray(a_batch, dtype=float)

    p1_orig = predict_p1_fn(x_batch)

    out: List[float] = []
    n_runs = int(max(3, nr_runs))
    frac = float(np.clip(subset_frac, 1e-6, 1.0))

    rng = np.random.default_rng(0)

    for i in range(x_batch.shape[0]):
        x_i = x_batch[i].copy()
        a_i = a_batch[i].copy()

        fx = float(p1_orig[i])
        fx_s = fx if use_logit_drop else _logit(fx,inv=True) 

        if positive_only:
            a_i = np.maximum(a_i, 0.0)

        # EEG-like shape: (C, T)
        if x_i.ndim == 2 and a_i.ndim == 2:
            C, T = int(x_i.shape[0]), int(x_i.shape[1])
            n_feat = C * T
            if n_feat <= 1:
                out.append(float("nan"))
                continue

            k = int(np.ceil(frac * n_feat))
            k = max(1, min(k, n_feat - 1))

            # Default block length: ~2.5% of T, clipped.
            if block_len is None:
                L = max(4, int(round(0.025 * T)))
                L = min(L, max(4, T // 2))
            else:
                L = int(max(1, min(int(block_len), T)))

            attr_sums: List[float] = []
            score_drops: List[float] = []

            for _ in range(n_runs):
                mask = np.zeros((C, T), dtype=bool)
                masked = 0
                blocks_used = 0

                while (masked < k) or (blocks_used < int(max(1, min_blocks))):
                    c = int(rng.integers(0, C))
                    if T <= 1:
                        t0, t1 = 0, 1
                    else:
                        t0 = int(rng.integers(0, max(1, T - L + 1)))
                        t1 = min(T, t0 + L)

                    before = int(mask[c, t0:t1].sum())
                    mask[c, t0:t1] = True
                    after = int(mask[c, t0:t1].sum())
                    masked += (after - before)
                    blocks_used += 1

                    if blocks_used > (C * 20):
                        break

                x_pert = x_i.copy()
                x_pert[mask] = baseline_value

                p1_pert = float(predict_p1_fn(x_pert[None, ...])[0])
                fxp =  p1_pert if use_logit_drop else  _logit(p1_pert,inv=True)

                attr_sums.append(float(a_i[mask].sum()))
                score_drops.append(float(fx_s - fxp))

            out.append(_safe_corr(np.asarray(attr_sums), np.asarray(score_drops), kind=corr_kind))

        else:
            # Fallback: flat random masking
            a_flat = a_i.reshape(-1)
            n_feat = int(a_flat.size)
            if n_feat <= 1:
                out.append(float("nan"))
                continue

            k = int(np.ceil(frac * n_feat))
            k = max(1, min(k, n_feat - 1))

            attr_sums: List[float] = []
            score_drops: List[float] = []

            for _ in range(n_runs):
                idx = rng.choice(n_feat, size=k, replace=False)
                x_flat = x_i.reshape(-1).copy()
                x_flat[idx] = baseline_value
                x_pert = x_flat.reshape(x_i.shape)

                p1_pert = float(predict_p1_fn(x_pert[None, ...])[0])
                fxp =  p1_pert if use_logit_drop else  _logit(p1_pert,inv=True)

                attr_sums.append(float(np.sum(a_flat[idx])))
                score_drops.append(float(fx_s - fxp))

            out.append(_safe_corr(np.asarray(attr_sums), np.asarray(score_drops), kind=corr_kind))

    return [float(v) for v in out]

# -------------------------
# Top-K block utilities (EEG/CNN friendly)
# -------------------------
def _default_block_len(T: int) -> int:
    """Default contiguous block length for time masking.

    Mirrors the faithfulness block default: ~2.5% of T, clipped to [4, T//2].
    """
    T = int(T)
    if T <= 1:
        return 1
    L = max(4, int(round(0.025 * T)))
    L = min(L, max(4, T // 2))
    return int(max(1, min(L, T)))


def _topk_time_blocks_mask_1d(
    attr_1d: np.ndarray,
    *,
    k_blocks: int,
    block_len: int,
    disallow_overlap: bool = True,
) -> np.ndarray:
    """Select top-k contiguous blocks on a 1D attribution curve.

    Returns:
        mask: bool array shape (T,) marking the selected samples.
    """
    a = np.asarray(attr_1d, dtype=float).reshape(-1)
    T = int(a.size)
    if T == 0:
        return np.zeros((0,), dtype=bool)

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    L = int(max(1, min(int(block_len), T)))
    K = int(max(0, k_blocks))
    if K == 0:
        return np.zeros((T,), dtype=bool)

    # If the block is as long as the signal, mask everything.
    if L >= T:
        return np.ones((T,), dtype=bool)

    # Window sums via cumulative sum.
    cs = np.concatenate([[0.0], np.cumsum(a, dtype=float)])
    # scores[s] = sum_{t=s}^{s+L-1} a[t]
    scores = cs[L:] - cs[:-L]  # length T-L+1

    used = np.zeros((T,), dtype=bool)
    mask = np.zeros((T,), dtype=bool)

    # Greedy selection of highest-scoring non-overlapping blocks.
    for _ in range(K):
        best_s = None
        best_v = -np.inf

        for s in range(scores.size):
            s0 = int(s)
            s1 = int(s0 + L)
            if disallow_overlap and used[s0:s1].any():
                continue
            v = float(scores[s0])
            if v > best_v:
                best_v = v
                best_s = s0

        if best_s is None:
            break

        s0 = int(best_s)
        s1 = int(s0 + L)
        mask[s0:s1] = True
        used[s0:s1] = True

    return mask

def _topk_time_blocks_mask_global(
    attr_2d: np.ndarray,
    *,
    k_blocks_total: int,
    block_len: int,
    positive_only: bool = False,
    abs_val: bool = False,
    disallow_overlap: bool = True,
    only_positive_blocks: bool = True,
) -> np.ndarray:
    """
    Select top-k contiguous time blocks globally across all channels.

    Args:
        attr_2d: attribution array (C, T)
        k_blocks_total: total number of blocks to delete across all channels
        block_len: block length L in samples
        positive_only: if True, use ReLU(attr) for ranking blocks
        abs_val: if True, rank by |attr| (applied after positive_only)
        disallow_overlap: if True, do not allow overlap within the same channel
        only_positive_blocks: if True, only select blocks with score > 0

    Returns:
        mask: bool array (C, T)
    """
    a = np.asarray(attr_2d, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected (C,T) attribution but got shape {a.shape}")

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if positive_only:
        a = np.maximum(a, 0.0)
    if abs_val:
        a = np.abs(a)

    C, T = int(a.shape[0]), int(a.shape[1])
    L = int(max(1, min(int(block_len), T)))
    K = int(max(0, int(k_blocks_total)))

    if K == 0 or T == 0:
        return np.zeros((C, T), dtype=bool)

    out = np.zeros((C, T), dtype=bool)
    used = np.zeros((C, T), dtype=bool)

    candidates = []
    # candidate = (score, channel, start, end)
    for c in range(C):
        ac = a[c].reshape(-1)

        if L >= T:
            score = float(np.sum(ac))
            candidates.append((score, c, 0, T))
            continue

        cs = np.concatenate([[0.0], np.cumsum(ac, dtype=float)])
        scores = cs[L:] - cs[:-L]  # length T-L+1

        for s in range(scores.size):
            s0 = int(s)
            s1 = int(s0 + L)
            score = float(scores[s0])
            candidates.append((score, c, s0, s1))

    # Highest-scoring blocks first
    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = 0
    for score, c, s0, s1 in candidates:
        if only_positive_blocks and score <= 0.0:
            break

        if disallow_overlap and used[c, s0:s1].any():
            continue

        out[c, s0:s1] = True
        used[c, s0:s1] = True
        selected += 1

        if selected >= K:
            break

    return out

def _topk_time_blocks_mask_per_channel(
    attr_2d: np.ndarray,
    *,
    k_blocks_per_channel: int,
    block_len: int,
    positive_only: bool = False,
    abs_val: bool = False,
    disallow_overlap: bool = True,
) -> np.ndarray:
    """Top-k contiguous time-block selection per channel.

    Args:
        attr_2d: attribution array (C, T)
        k_blocks_per_channel: number of blocks to delete *per channel*
        block_len: block length L in samples
        positive_only: if True, use ReLU(attr) for ranking blocks
        abs_val: if True, rank by |attr| (applied after positive_only)
    Returns:
        mask: bool array (C, T)
    """
    a = np.asarray(attr_2d, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected (C,T) attribution but got shape {a.shape}")

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if positive_only:
        a = np.maximum(a, 0.0)
    if abs_val:
        a = np.abs(a)

    C, T = int(a.shape[0]), int(a.shape[1])
    L = _default_block_len(T) if block_len is None else int(block_len)
    L = int(max(1, min(L, T)))

    out = np.zeros((C, T), dtype=bool)
    for c in range(C):
        out[c] = _topk_time_blocks_mask_1d(
            a[c],
            k_blocks=int(k_blocks_per_channel),
            block_len=L,
            disallow_overlap=disallow_overlap,
        )
    return out


def feature_topk_block_deletion_drop(
    *,
    predict_p1_fn: Callable[[np.ndarray], np.ndarray],
    x_batch: np.ndarray,
    y_batch: np.ndarray,  # kept for API compatibility; not used
    a_batch: np.ndarray,
    k_blocks_per_channel: Optional[int] = None,
    k_frac: float = 0.10,
    baseline_value: float = 0.0,
    block_len: Optional[int] = None,
    positive_only: bool = False,
    abs_val: bool = False,
    disallow_overlap: bool = True,
    use_logit_drop: bool = True,
) -> List[float]:
    """Feature necessity via *top-k block deletion* for IG explanations.

    Behaviour (EEG/CNN friendly):
      - For each sample i and each channel c:
          pick the K most relevant contiguous blocks (length=L) according to IG
          (optionally ReLU/abs for ranking),
          mask them by setting the input samples to baseline_value.
      - Measure output drop: f(x) - f(x_masked), where f is p1 or logit(p1).

    K is interpreted as "blocks per channel". If k_blocks_per_channel is None,
    it is derived from k_frac by masking roughly k_frac*T samples per channel:
        K ≈ ceil((k_frac*T)/L)

    Returns:
        One drop value per sample (higher => more "necessary" explanation).
    """
    x_batch = np.asarray(x_batch, dtype=float)
    a_batch = np.asarray(a_batch, dtype=float)

    p1_orig = predict_p1_fn(x_batch)

    drops: List[float] = []

    for i in range(x_batch.shape[0]):
        x_i = x_batch[i].copy()
        a_i = a_batch[i].copy()

        if x_i.ndim != 2 or a_i.ndim != 2:
            raise ValueError(
                f"Top-k block deletion expects x and a shaped (C,T); got x={x_i.shape}, a={a_i.shape}"
            )

        C, T = int(x_i.shape[0]), int(x_i.shape[1])
        L = _default_block_len(T) if block_len is None else int(block_len)
        L = int(max(1, min(L, T)))

        if k_blocks_per_channel is None:
            # mask ~k_frac*T samples per channel -> K blocks
            k_pts = int(np.ceil(float(k_frac) * T))
            K = int(max(1, int(np.ceil(k_pts / max(1, L)))))
        else:
            K = int(max(1, int(k_blocks_per_channel)))

        mask = _topk_time_blocks_mask_per_channel(
            a_i,
            k_blocks_per_channel=K,
            block_len=L,
            positive_only=positive_only,
            abs_val=abs_val,
            disallow_overlap=disallow_overlap,
        )

        x_pert = x_i.copy()
        x_pert[mask] = float(baseline_value)

        p1_pert = float(predict_p1_fn(x_pert[None, ...])[0])

        fx = float(p1_orig[i])
        if use_logit_drop:
            drop = fx - p1_pert
        else:
            drop = _logit(fx,inv=True) - _logit(p1_pert,inv=True)
            

        drops.append(float(drop))

    return drops

def feature_topk_block_deletion_drop_global(
    *,
    predict_p1_fn: Callable[[np.ndarray], np.ndarray],
    x_batch: np.ndarray,
    y_batch: np.ndarray,  # kept for API compatibility; not used
    a_batch: np.ndarray,
    k_blocks_total: Optional[int] = None,
    k_frac: float = 0.10,
    baseline_value: float = 0.0,
    block_len: Optional[int] = None,
    positive_only: bool = False,
    abs_val: bool = False,
    disallow_overlap: bool = True,
    only_positive_blocks: bool = True,
    use_logit_drop: bool = True,
) -> List[float]:
    """
    Feature necessity via global top-k block deletion across all channels.

    Behaviour:
      - For each sample i:
          pick the K most relevant contiguous blocks (length=L) globally
          across all channels according to attribution scores
          (optionally ReLU/abs for ranking),
          mask them by setting input samples to baseline_value.
      - Measure output drop: f(x) - f(x_masked)

    If k_blocks_total is None, it is derived from k_frac as roughly
    k_frac of all samples across the full (C,T) map:
        K ≈ ceil((k_frac * C * T) / L)
    """
    x_batch = np.asarray(x_batch, dtype=float)
    a_batch = np.asarray(a_batch, dtype=float)

    p1_orig = predict_p1_fn(x_batch)
    drops: List[float] = []

    for i in range(x_batch.shape[0]):
        x_i = x_batch[i].copy()
        a_i = a_batch[i].copy()

        if x_i.ndim != 2 or a_i.ndim != 2:
            raise ValueError(
                f"Top-k block deletion expects x and a shaped (C,T); got x={x_i.shape}, a={a_i.shape}"
            )

        C, T = int(x_i.shape[0]), int(x_i.shape[1])
        L = _default_block_len(T) if block_len is None else int(block_len)
        L = int(max(1, min(L, T)))

        if k_blocks_total is None:
            k_pts_total = int(np.ceil(float(k_frac) * C * T))
            K = int(max(1, int(np.ceil(k_pts_total / max(1, L)))))
        else:
            K = int(max(1, int(k_blocks_total)))

        mask = _topk_time_blocks_mask_global(
            a_i,
            k_blocks_total=K,
            block_len=L,
            positive_only=positive_only,
            abs_val=abs_val,
            disallow_overlap=disallow_overlap,
            only_positive_blocks=only_positive_blocks,
        )

        x_pert = x_i.copy()
        x_pert[mask] = float(baseline_value)

        p1_pert = float(predict_p1_fn(x_pert[None, ...])[0])

        fx = float(p1_orig[i])
        if use_logit_drop:
            drop = fx - p1_pert
        else:
            drop = _logit(fx, inv=True) - _logit(p1_pert, inv=True)

        drops.append(float(drop))

    return drops
def faithfulness_correlation_metric_edge(
    *,
    gnn: torch.nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,  # kept for API compatibility; not used
    a_edge_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    nr_runs: int = 40,
    subset_frac: float = 0.10,
    corr_kind: str = "pearson",
    positive_only: bool = True,
    use_logit_drop: bool = True,
    # NEW:
    
) -> List[float]:
    """
    Faithfulness Correlation for edge explanations.

    selection="random":
        Bhatt et al. faithfulness correlation.
        Random subsets of edges are deleted.

    
    """
    

    x_batch = np.asarray(x_batch, dtype=float)
    a_edge_batch = np.asarray(a_edge_batch, dtype=float)

    # Original predictions
    p1_orig = _predict_p1_gnn(gnn, x_batch, edge_index=edge_index, device=device)

    ei = edge_index.detach().cpu().numpy()
    E = int(ei.shape[1])

    rng = np.random.default_rng(0)
    out: List[float] = []

    for i in range(x_batch.shape[0]):
        x_i = x_batch[i]
        a_i = a_edge_batch[i].reshape(-1)

        if positive_only:
            a_i = np.maximum(a_i, 0.0)

        if E <= 1:
            out.append(float("nan"))
            continue

        k = int(np.ceil(subset_frac * E))
        k = max(1, min(k, E - 1))

        fx = float(p1_orig[i])
        fx_s =fx if use_logit_drop else _logit(fx,inv=True) 

        attr_sums = []
        score_drops = []

        for _ in range(max(3, nr_runs)):
           
            idx = rng.choice(E, size=k, replace=False)
            

            keep = np.ones(E, dtype=bool)
            keep[idx] = False
            ei_pert = ei[:, keep]

            edge_index_pert = torch.tensor(
                ei_pert, dtype=torch.long, device=device
            )

            p1_pert = float(
                _predict_p1_gnn(
                    gnn,
                    x_i[None, ...],
                    edge_index=edge_index_pert,
                    device=device,
                )[0]
            )

            fxp = p1_pert if use_logit_drop else _logit(p1_pert,inv=True)

            attr_sums.append(float(a_i[idx].sum()))
            score_drops.append(float(fx_s - fxp))

        out.append(
            _safe_corr(
                np.asarray(attr_sums),
                np.asarray(score_drops),
                kind=corr_kind,
            )
        )

    return [float(v) for v in out]

def edge_topk_deletion_drop(
    *,
    gnn: torch.nn.Module,
    x_batch: np.ndarray,
    a_edge_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.10,
    positive_only: bool = False,
    use_logit_drop: bool = True,
) -> List[float]:
    """
    Edge necessity metric:
    Delete top-k most relevant edges and measure output drop.
    """
    x_batch = np.asarray(x_batch, dtype=float)
    a_edge_batch = np.asarray(a_edge_batch, dtype=float)

    p1_orig = _predict_p1_gnn(gnn, x_batch, edge_index=edge_index, device=device)

    ei = edge_index.detach().cpu().numpy()
    E = ei.shape[1]

    drops = []

    for i in range(x_batch.shape[0]):
        a_i = a_edge_batch[i].reshape(-1)
        if positive_only:
            a_i = np.maximum(a_i, 0.0)

        k = max(1, int(np.ceil(k_frac * E)))
        idx = np.argsort(-a_i)[:k]

        keep = np.ones(E, dtype=bool)
        keep[idx] = False
        ei_pert = ei[:, keep]

        edge_index_pert = torch.tensor(
            ei_pert, dtype=torch.long, device=device
        )

        p1p = float(
            _predict_p1_gnn(
                gnn,
                x_batch[i][None, ...],
                edge_index=edge_index_pert,
                device=device,
            )[0]
        )

        fx = float(p1_orig[i])
        if use_logit_drop:
            drop = fx - p1p
        else:
            drop = _logit(fx,inv=True) - _logit(p1p,inv=True)

        drops.append(float(drop))

    return drops



def _align_temporal_mask_to_T(mask_1d: np.ndarray, T: int, fs: int = 32, thr: float = 0.5) -> np.ndarray:
    """Align a temporal ground-truth mask to length T.

    Supported:
      - mask length == T  -> returned as-is (thresholded to bool)
      - mask length == T//fs (e.g. 12 at 1 Hz for a 12s window with fs=32) -> upsample by repeating each second fs times
      - mask length * fs == T -> same as above

    Returns:
      bool array of shape (T,)
    """
    m = np.asarray(mask_1d).reshape(-1)
    if m.size == 0:
        return np.zeros((T,), dtype=bool)

    # binarise
    m = (m.astype(float) >= float(thr))

    if int(m.size) == int(T):
        return m.astype(bool)

    fs = int(max(1, fs))
    if int(m.size) * fs == int(T):
        return np.repeat(m.astype(bool), fs)

    if int(T) % fs == 0 and int(m.size) == int(T // fs):
        return np.repeat(m.astype(bool), fs)

    raise ValueError(f"Cannot align mask of length {m.size} to T={T} with fs={fs}.")



def relevance_mass_accuracy_temporal(
    *,
    a_batch: np.ndarray,
    s_batch: np.ndarray,
    abs_val: bool = True,
    positive_only: bool = False,
    normalise: bool = True,
    eps: float = 1e-12,
    fs: int = 32,
    s_threshold: float = 0.5,
) -> List[float]:
    """Relevance Mass Accuracy (temporal).

    This implements the core idea of Relevance Mass Accuracy used in explanation
    localisation papers: compute the fraction of total relevance that falls inside
    a ground-truth region (here: seizure timepoints inside a window).

    For EEG IG attributions A with shape (B,C,T) we:
      1) Convert to non-negative relevance (abs or ReLU).
      2) Aggregate over channels -> r(t) = sum_c A(c,t).
      3) Score per sample = sum_t r(t) * s(t) / (sum_t r(t) + eps),
         where s(t) is the temporal ground-truth mask aligned to length T.

    If s(t) is all zeros, the numerator is 0, so the score is 0.

    Args:
      a_batch: (B,C,T) or (B,T) attribution batch.
      s_batch: (B,T) temporal GT masks, or (B,T//fs) 1Hz masks.
      normalise: if True, normalise relevance per sample so sum_t r(t)=1 before masking
                 (this does not change the final ratio but is numerically stable).
    """
    a = np.asarray(a_batch, dtype=float)
    s = np.asarray(s_batch)

    if a.ndim == 2:
        a = a[:, None, :]  # (B,1,T)
    if a.ndim != 3:
        raise ValueError(f"Expected a_batch shaped (B,C,T) or (B,T), got {a.shape}")

    B, C, T = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])

    # Align s to (B,T)
    if s.ndim == 1:
        s = np.broadcast_to(s[None, :], (B, s.size))
    if s.ndim != 2 or s.shape[0] != B:
        raise ValueError(f"Expected s_batch shaped (B,*) compatible with B={B}, got {s.shape}")

    s_aligned = np.stack([_align_temporal_mask_to_T(s[i], T, fs=fs, thr=s_threshold) for i in range(B)], axis=0).astype(float)

    # non-negative relevance
    if abs_val:
        a = np.abs(a)
    if positive_only:
        a = np.maximum(a, 0.0)

    # aggregate over channels
    r = a.sum(axis=1)  # (B,T)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    if normalise:
        denom = r.sum(axis=1, keepdims=True) + float(eps)
        r = r / denom

    num = (r * s_aligned).sum(axis=1)
    den = r.sum(axis=1) + float(eps)  # equals ~1 if normalise=True
    scores = num / den
    return [float(v) for v in scores]


def relevance_mass_accuracy_temporal_global(
    *,
    a_batch: np.ndarray,
    s_batch: np.ndarray,
    abs_val: bool = False,
    positive_only: bool = True,
    eps: float = 1e-12,
    fs: int = 32,
    s_threshold: float = 0.5,
    normalize_per_window = False,
    balance_mode: str = "by_window_count",   # "pooled", "by_window_count", "seizure_only"
) -> float:
    """Global Relevance Mass Accuracy (temporal).

    Aggregates relevance mass across the *entire dataset*:

        RMA_global = (sum_{i,t} r_i(t) * s_i(t)) / (sum_{i,t} r_i(t) + eps)

    where r_i(t) is the channel-summed, non-negative relevance at time t for sample i.

    This is useful when you evaluate a mixed dataset of seizure and non-seizure windows:
    windows with s(t)=0 everywhere (non-seizure) contribute only to the denominator,
    so the global score is no longer trivially 0/1 per split.
    """
    a = np.asarray(a_batch, dtype=float)
    s = np.asarray(s_batch)

    if a.ndim == 2:
        a = a[:, None, :]
    if a.ndim != 3:
        raise ValueError(f"Expected a_batch shaped (B,C,T) or (B,T), got {a.shape}")

    B, _, T = a.shape

    if s.ndim == 1:
        s = np.broadcast_to(s[None, :], (B, s.size))
    if s.ndim != 2 or s.shape[0] != B:
        raise ValueError(f"Expected s_batch shaped (B,*) compatible with B={B}, got {s.shape}")

    s_aligned = np.stack(
        [_align_temporal_mask_to_T(s[i], T, fs=fs, thr=s_threshold) for i in range(B)],
        axis=0,
    ).astype(float)

    if abs_val:
        a = np.abs(a)
    if positive_only:
        a = np.maximum(a, 0.0)

    r = a.sum(axis=1)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    # if relevance_transform == "log":
    #     r = np.log10(1.0 + r)

    num_i = (r * s_aligned).sum(axis=1)
    den_i = r.sum(axis=1) + eps

    has_seiz = s_aligned.sum(axis=1) > 0
    n_seiz = max(int(has_seiz.sum()), 1)
    n_nonseiz = max(int((~has_seiz).sum()), 1)

    if balance_mode == "pooled":
        return float(num_i.sum() / (den_i.sum() + eps))

    elif balance_mode == "by_window_count":
        # weight each window inversely by class count
        weights = np.where(has_seiz, 1.0 / n_seiz, 1.0 / n_nonseiz)
        return float((weights * num_i).sum() / ((weights * den_i).sum() + eps))

    elif balance_mode == "seizure_only":
        if not np.any(has_seiz):
            return 0.0
        return float(num_i[has_seiz].sum() / (den_i[has_seiz].sum() + eps))

    else:
        raise ValueError(f"Unknown balance_mode={balance_mode!r}")



def edge_relevance_mass_accuracy_temporal_proxy(
    *,
    edge_mask_batch: np.ndarray,
    x_raw_batch: np.ndarray,
    edge_index: torch.Tensor,
    s_batch: np.ndarray,
    interaction: str = "abs_product",
    eps: float = 1e-12,
    fs: int = 32,
    s_threshold: float = 0.5,
) -> List[float]:
    """Temporal relevance-mass proxy for *edge-only* explanations.

    Problem:
      - Your GT mask is temporal (within-window seizure annotation).
      - GNNExplainer edge masks are static per window (E,) with no time axis.

    Proxy idea (works without extra annotations):
      1) Convert the static edge importance weights w_e into a time series by weighting
         a simple per-edge interaction signal computed from raw EEG at each timepoint:
            inter_e(t) = |x_u(t) * x_v(t)|  (default)  OR  |x_u(t) - x_v(t)|
         where (u,v) is the edge endpoints.
      2) Aggregate across edges:
            r(t) = sum_e w_e * inter_e(t)
      3) Compute temporal relevance mass accuracy of r(t) w.r.t. the seizure mask s(t).

    This yields a per-window score in [0,1] that you can interpret as:
      "Do the edges highlighted by GNNExplainer correspond to interactions that are strong
       during seizure-annotated timepoints?"

    Note:
      - This is *not* the same as having time-resolved edge explanations.
      - It is an explicit approximation to bridge static edge importance and temporal GT.
    """
    w_batch = np.asarray(edge_mask_batch, dtype=float)
    x_batch = np.asarray(x_raw_batch, dtype=float)
    s = np.asarray(s_batch)

    if w_batch.ndim != 2:
        raise ValueError(f"Expected edge_mask_batch shaped (B,E), got {w_batch.shape}")
    if x_batch.ndim != 3:
        raise ValueError(f"Expected x_raw_batch shaped (B,C,T), got {x_batch.shape}")

    B, C, T = int(x_batch.shape[0]), int(x_batch.shape[1]), int(x_batch.shape[2])

    # Align s to (B,T)
    if s.ndim == 1:
        s = np.broadcast_to(s[None, :], (B, s.size))
    if s.ndim != 2 or s.shape[0] != B:
        raise ValueError(f"Expected s_batch shaped (B,*) compatible with B={B}, got {s.shape}")
    s_aligned = np.stack([_align_temporal_mask_to_T(s[i], T, fs=fs, thr=s_threshold) for i in range(B)], axis=0).astype(float)

    ei = edge_index.detach().cpu().numpy()
    u = ei[0].astype(int)
    v = ei[1].astype(int)
    E = int(ei.shape[1])

    interaction = str(interaction).lower().strip()

    out: List[float] = []
    for i in range(B):
        w = w_batch[i].reshape(-1)
        if w.size != E:
            # safe truncate/pad
            w = w[:E] if w.size > E else np.pad(w, (0, E - w.size), constant_values=0.0)

        # ensure non-negative (GNNExplainer edge masks are usually >=0, but be safe)
        w = np.maximum(w, 0.0)

        x = x_batch[i]  # (C,T)

        # gather endpoint signals: (E,T)
        xu = x[u]  # advanced indexing
        xv = x[v]

        if interaction == "abs_diff":
            inter = np.abs(xu - xv)
        else:
            # default: abs_product
            inter = np.abs(xu * xv)

        # weighted sum over edges -> (T,)
        r = (w[:, None] * inter).sum(axis=0)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

        denom = float(r.sum() + eps)
        score = float((r * s_aligned[i]).sum() / denom) if denom > 0 else 0.0
        out.append(score)

    return out

# -------------------------
# Runner (custom metrics only)
# -------------------------

def run_custom_metrics_co12(
    model: torch.nn.Module,
    graphs: Sequence[Data],
    thr: float,
    results_dir: str,
    device: torch.device,
    eeg_fs: int = 32,
    edge_rma_interaction: str = "abs_product",
    model_kind: str = "base",
) -> Dict[str, Any]:
    """
    Custom metric runner.

    model_kind='base':
        - IG on raw input, explained on graph logits.
        - GNNExplainer on CNN node-features, explained on graph logits.

    model_kind='senn':
        - focus map from SENN_raw output dictionary.
        - same feature-level metrics as IG, evaluated on the focus map.
    """
    if len(graphs) == 0:
        raise ValueError("graphs is empty")

    model_kind = str(model_kind).strip().lower()
    model.eval()
    edge_index = graphs[0].edge_index.to(device)

    x_raw = np.stack([g.x.detach().cpu().numpy() for g in graphs], axis=0)
    y_true = np.array([int(g.y.detach().cpu().view(-1)[0].item()) for g in graphs], dtype=int)

    s_time = None
    if all(hasattr(g, "y_mask") for g in graphs):
        try:
            s_time = np.stack([_to_numpy(getattr(g, "y_mask")).squeeze() for g in graphs], axis=0)
        except Exception:
            s_time = None
    else:
        y0 = _to_numpy(graphs[0].y).squeeze()
        if np.asarray(y0).ndim == 1:
            try:
                s_time = np.stack([_to_numpy(g.y).squeeze() for g in graphs], axis=0)
            except Exception:
                s_time = None

    custom_lip_nr_ig = int(os.getenv("CUSTOM_LIP_NR_IG", str(LIPSCHITZ_NR_SAMPLES_IG)))
    custom_lip_std_ig = float(os.getenv("CUSTOM_LIP_STD_IG", "0.05"))
    custom_lip_alpha_ig = 0.05
    custom_lip_nr_edge = int(os.getenv("CUSTOM_LIP_NR_EDGE", str(LIPSCHITZ_NR_SAMPLES_GNN)))
    custom_lip_std_edge = float(os.getenv("CUSTOM_LIP_STD_EDGE", "0.05"))
    custom_lip_alpha_edge = 0.05
    custom_mprt_rounds = int(os.getenv("CUSTOM_MPRT_ROUNDS", "3"))

    faith_nr_runs = int(os.getenv("FAITH_CORR_NR_RUNS", "40"))
    faith_subset_frac = float(os.getenv("FAITH_CORR_SUBSET_FRAC", "0.10"))

    faith_block_len = os.getenv("FAITH_BLOCK_LEN", "")
    faith_block_len = int(faith_block_len) if faith_block_len.strip() != "" else None
    faith_min_blocks = int(os.getenv("FAITH_MIN_BLOCKS", "4"))

    ig_del_block_len_env = os.getenv("IG_DEL_BLOCK_LEN", "")
    ig_del_block_len = int(ig_del_block_len_env) if ig_del_block_len_env.strip() != "" else faith_block_len
    ig_del_k_env = os.getenv("IG_DEL_TOPK_PER_CHANNEL", "")
    ig_del_topk_per_channel = int(ig_del_k_env) if ig_del_k_env.strip() != "" else None
    ig_del_frac = float(os.getenv("IG_DEL_FRAC", str(faith_subset_frac)))

    edge_epochs = int(os.getenv("GNNEXPL_EPOCHS_QUANT", str(GNNEXPL_EPOCHS_QUANT)))
    rma_s_threshold = float(os.getenv("RMA_S_THRESHOLD", "0.5"))

    results: Dict[str, Any] = {
        "meta": {
            "framework": "custom_only_single_output",
            "model_kind": model_kind,
            "thr": float(thr),
            "n_samples": int(len(graphs)),
            "eeg_fs": int(eeg_fs),
            "has_temporal_gt": bool(s_time is not None),
            "rma_s_threshold": float(rma_s_threshold),
            "edge_rma_interaction": str(edge_rma_interaction),
            "lipschitz_nr_samples_ig": int(custom_lip_nr_ig),
            "lipschitz_std_ig": float(custom_lip_std_ig),
            "lipschitz_alpha_ig": None if custom_lip_alpha_ig is None else float(custom_lip_alpha_ig),
            "lipschitz_nr_samples_edge": int(custom_lip_nr_edge),
            "lipschitz_std_edge": float(custom_lip_std_edge),
            "lipschitz_alpha_edge": None if custom_lip_alpha_edge is None else float(custom_lip_alpha_edge),
            "gnnexpl_epochs_quant": int(edge_epochs),
            "faith_corr_nr_runs": int(faith_nr_runs),
            "faith_corr_subset_frac": float(faith_subset_frac),
            "faith_block_len": None if faith_block_len is None else int(faith_block_len),
            "faith_min_blocks": int(faith_min_blocks),
            "mprt_rounds": int(custom_mprt_rounds),
        },
        "y_true": y_true.tolist(),
        "metrics": {},
    }

    pred_full = lambda xb: _predict_p1_full(model, xb, edge_index=edge_index, device=device)

    if model_kind == "base":
        with torch.no_grad():
            nf_list = []
            for g in graphs:
                gx = g.x.to(device)
                nf = model.cnn(gx)
                nf_list.append(nf.detach().cpu().numpy())
        x_nf = np.stack(nf_list, axis=0)

        class _ModelShim(torch.nn.Module):
            def __init__(self, gnn_module):
                super().__init__()
                self.gnn = gnn_module
                self.cnn = torch.nn.Identity()

        model_nf = _ModelShim(model.gnn).to(device)
        model_nf.eval()

        results["metrics"]["IG_RAW"] = {}
        ig_a = ig_explainer_raw(
            model=model,
            inputs=x_raw,
            targets=y_true,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
            thr=thr,
        )

        results["metrics"]["IG_RAW"]["Continuity_LocalLipschitz"] = feature_local_lipschitz_estimate(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_ig,
            perturb_std=custom_lip_std_ig,
            perturb_alpha=custom_lip_alpha_ig,
            thr=thr,
        )
        results["metrics"]["IG_RAW"]["Correctness_ParamRandomisation_corr_spearman"] = feature_parameter_randomisation_sanity(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
            thr=thr,
        )
        # results["metrics"]["IG_RAW"]["FaithfulnessCorrelation"] = faithfulness_correlation_metric_feature(
        #     predict_p1_fn=pred_full,
        #     x_batch=x_raw,
        #     y_batch=y_true,
        #     a_batch=ig_a,
        #     nr_runs=faith_nr_runs,
        #     subset_frac=faith_subset_frac,
        #     baseline_value=0.0,
        #     corr_kind="pearson",
        #     positive_only=False,
        #     use_logit_drop=True,
        #     block_len=faith_block_len,
        #     min_blocks=faith_min_blocks,
        # )
        # results["metrics"]["IG_RAW"]["TopKDeletion_drop"] = feature_topk_block_deletion_drop(
        #     predict_p1_fn=pred_full,
        #     x_batch=x_raw,
        #     y_batch=y_true,
        #     a_batch=ig_a,
        #     k_blocks_per_channel=ig_del_topk_per_channel,
        #     k_frac=ig_del_frac,
        #     baseline_value=0.0,
        #     block_len=ig_del_block_len,
        #     positive_only=False,
        #     abs_val=False,
        #     disallow_overlap=True,
        #     use_logit_drop=True,
        # )

        results["metrics"]["IG_RAW"]["TopKDeletion_drop"] = feature_topk_block_deletion_drop_global(
            predict_p1_fn=pred_full,
            x_batch=x_raw,
            y_batch=y_true,
            a_batch=ig_a,
            k_blocks_total=None,          # derive from k_frac
            k_frac=ig_del_frac,
            baseline_value=0.0,
            block_len=ig_del_block_len,
            positive_only=False,
            abs_val=False,
            disallow_overlap=True,
            only_positive_blocks=False,
            use_logit_drop=True,
        )

        results["metrics"]["GNN_EDGE"] = {}
        results["metrics"]["GNN_EDGE"]["Continuity_LocalLipschitz"] = edge_local_lipschitz_estimate(
            x_nf=x_nf,
            model=model_nf,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_edge,
            perturb_std=custom_lip_std_edge,
            perturb_alpha=custom_lip_alpha_edge,
            epochs=edge_epochs,
            thr=thr,
        )
        results["metrics"]["GNN_EDGE"]["Correctness_ParamRandomisation_corr_spearman"] = edge_parameter_randomisation_sanity(
            x_nf=x_nf,
            gnn=model.gnn,
            edge_index=edge_index,
            device=device,
            epochs=edge_epochs,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
        )

        edge_masks = gnn_edge_explainer(
            model=model_nf,
            inputs=x_nf,
            targets=y_true,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
            epochs=edge_epochs,
            thr=thr,
        )
        # results["metrics"]["GNN_EDGE"]["FaithfulnessCorrelation"] = faithfulness_correlation_metric_edge(
        #     gnn=model.gnn,
        #     x_batch=x_nf,
        #     y_batch=y_true,
        #     a_edge_batch=edge_masks,
        #     edge_index=edge_index,
        #     device=device,
        #     nr_runs=faith_nr_runs,
        #     subset_frac=faith_subset_frac,
        #     corr_kind="pearson",
        #     positive_only=False,
        #     use_logit_drop=True,
        # )
        results["metrics"]["GNN_EDGE"]["TopKDeletion_drop"] = edge_topk_deletion_drop(
            gnn=model.gnn,
            x_batch=x_nf,
            a_edge_batch=edge_masks,
            edge_index=edge_index,
            device=device,
            k_frac=faith_subset_frac,
            positive_only=False,
            use_logit_drop=True,
        )

    elif model_kind == "senn":
        results["metrics"]["FOCUS_MAP"] = {}
        focus_a = focusmap_explainer_raw(
            model=model,
            inputs=x_raw,
            targets=y_true,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
        )
        results["metrics"]["FOCUS_MAP"]["Continuity_LocalLipschitz"] = focusmap_local_lipschitz_estimate(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_ig,
            perturb_std=custom_lip_std_ig,
            perturb_alpha=custom_lip_alpha_ig,
        )
        results["metrics"]["FOCUS_MAP"]["Correctness_ParamRandomisation_corr_spearman"] = focusmap_parameter_randomisation_sanity(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
        )
        # results["metrics"]["FOCUS_MAP"]["FaithfulnessCorrelation"] = faithfulness_correlation_metric_feature(
        #     predict_p1_fn=pred_full,
        #     x_batch=x_raw,
        #     y_batch=y_true,
        #     a_batch=focus_a,
        #     nr_runs=faith_nr_runs,
        #     subset_frac=faith_subset_frac,
        #     baseline_value=0.0,
        #     corr_kind="pearson",
        #     positive_only=False,
        #     use_logit_drop=True,
        #     block_len=faith_block_len,
        #     min_blocks=faith_min_blocks,
        # )
        # results["metrics"]["FOCUS_MAP"]["TopKDeletion_drop"] = feature_topk_block_deletion_drop(
        #     predict_p1_fn=pred_full,
        #     x_batch=x_raw,
        #     y_batch=y_true,
        #     a_batch=focus_a,
        #     k_blocks_per_channel=ig_del_topk_per_channel,
        #     k_frac=ig_del_frac,
        #     baseline_value=0.0,
        #     block_len=ig_del_block_len,
        #     positive_only=False,
        #     abs_val=False,
        #     disallow_overlap=True,
        #     use_logit_drop=True,
        # )
        results["metrics"]["FOCUS_MAP"]["TopKDeletion_drop"] = feature_topk_block_deletion_drop_global(
            predict_p1_fn=pred_full,
            x_batch=x_raw,
            y_batch=y_true,
            a_batch=focus_a,
            k_blocks_total=None,          # derive from k_frac
            k_frac=ig_del_frac,
            baseline_value=0.0,
            block_len=ig_del_block_len,
            positive_only=False,
            abs_val=False,
            disallow_overlap=True,
            only_positive_blocks=False,
            use_logit_drop=True,
        )

    elif model_kind == "senn_fixed":
        results["metrics"]["FOCUS_MAP"] = {}
        focus_a = focusmap_explainer_raw(
            model=model,
            inputs=x_raw,
            targets=y_true,
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
        )
        results["metrics"]["FOCUS_MAP"]["Continuity_LocalLipschitz"] = focusmap_local_lipschitz_estimate(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_ig,
            perturb_std=custom_lip_std_ig,
            perturb_alpha=custom_lip_alpha_ig,
        )
        results["metrics"]["FOCUS_MAP"]["Correctness_ParamRandomisation_corr_spearman"] = focusmap_parameter_randomisation_sanity(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
        )
        # results["metrics"]["FOCUS_MAP"]["FaithfulnessCorrelation"] = faithfulness_correlation_metric_feature(
        #     predict_p1_fn=pred_full,
        #     x_batch=x_raw,
        #     y_batch=y_true,
        #     a_batch=focus_a,
        #     nr_runs=faith_nr_runs,
        #     subset_frac=faith_subset_frac,
        #     baseline_value=0.0,
        #     corr_kind="pearson",
        #     positive_only=False,
        #     use_logit_drop=True,
        #     block_len=faith_block_len,
        #     min_blocks=faith_min_blocks,
        # )
        # results["metrics"]["FOCUS_MAP"]["TopKDeletion_drop"] = feature_topk_block_deletion_drop(
        #     predict_p1_fn=pred_full,
        #     x_batch=x_raw,
        #     y_batch=y_true,
        #     a_batch=focus_a,
        #     k_blocks_per_channel=ig_del_topk_per_channel,
        #     k_frac=ig_del_frac,
        #     baseline_value=0.0,
        #     block_len=ig_del_block_len,
        #     positive_only=False,
        #     abs_val=False,
        #     disallow_overlap=True,
        #     use_logit_drop=True,
        # )
        # results["metrics"]["FOCUS_MAP"]["TopKDeletion_drop"] = feature_topk_block_deletion_drop_global(
        #     predict_p1_fn=pred_full,
        #     x_batch=x_raw,
        #     y_batch=y_true,
        #     a_batch=focus_a,
        #     k_blocks_total=None,          # derive from k_frac
        #     k_frac=ig_del_frac,
        #     baseline_value=0.0,
        #     block_len=ig_del_block_len,
        #     positive_only=False,
        #     abs_val=False,
        #     disallow_overlap=True,
        #     only_positive_blocks=False,
        #     use_logit_drop=True,
        # )
    else:
        raise ValueError("model_kind must be 'base' or 'senn'")

    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "custom_metrics_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, indent=2)

    return results


# -------------------------
# Main pipeline
# -------------------------


if __name__ == "__main__":

    RUN_CUSTOM_METRICS = True
    EEG_FS = int(os.getenv("EEG_FS", "32"))
    EDGE_RMA_INTERACTION = os.getenv("EDGE_RMA_INTERACTION", "abs_product")
    LIPSCHITZ_NR_SAMPLES_IG = int(os.getenv("LIPSCHITZ_NR_SAMPLES_IG", "25"))
    LIPSCHITZ_NR_SAMPLES_GNN = int(os.getenv("LIPSCHITZ_NR_SAMPLES_GNN", "5"))
    GNNEXPL_EPOCHS_QUANT = int(os.getenv("GNNEXPL_EPOCHS_QUANT", "80"))
    IG_STEPS = int(os.getenv("IG_STEPS", "32"))

    log = "459966"
    model_type = "base"

    
    # log = "459967"
    # model_type = "senn"

    # log = "461551" #3e-5
    # model_type = "senn"

    # log = "460527" #1e-5
    # model_type = "senn"

    

    log = "460528" #3e-6
    model_type = "senn"

    
    log = "460529" #1e-6
    model_type = "senn"


    log = "460531" #3e-7
    model_type = "senn"
    log = "459968" #SENN raw x 1e-4
    model_type ="senn"

    log = "470501" #SENN raw x 1e-4
    model_type ="senn_fixed"


    MODEL_KIND = os.getenv("MODEL_KIND", model_type).strip().lower()  # 'base' or 'senn'

    fold = "5" #Fold 5 is in all cases most representative of cv mean
    INT_PLOT = False

    DATA_FOLDER = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets\CV_Folds/"
    HISTORY_DIR = f"./History_{log}"
    MODEL_DIR = f"./Saved_models_{log}"
    RESULTS_DIR = os.path.join(f"./Results_{log}","Explainability_metrics") # RESULTS_DIR = os.path.join(f"./Results_{log}", MODEL_KIND)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fold_dir = os.path.join(DATA_FOLDER, f"fold_{fold}")

    default_subdir = f"GAT_CV_10_{fold}" #if MODEL_KIND == "base" else f"SENN_CV_10_{fold}"
    MODEL_SUBDIR = os.getenv("MODEL_SUBDIR", default_subdir)
    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR)

    ckpt = torch.load(os.path.join(model_path, "best_auprc.pt"), weights_only=False)
    norm = ckpt["normalization"]
    threshold = float(np.round(ckpt["metrics"]["threshold"], 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(MODEL_KIND, ckpt, return_explanations=(MODEL_KIND == "senn" or MODEL_KIND=="senn_fixed")).to(device)
    model.eval()

    print("... loading and batching data ... ")
    x_test = np.load(os.path.join(fold_dir, "testdata.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(fold_dir, "testlabels.npy"), mmap_mode="r")
    mask_path = os.path.join(fold_dir, "testmasks.npy")
    y_mask_test = np.load(mask_path, mmap_mode="r") if os.path.exists(mask_path) else None

    mean = norm["mean"]
    std = norm["std"]
    # x_test = (x_test - mean) / std

    batch_size = 50
    # batch_size = 250
    batch_size = 20
    idx_yes_seiz = np.where(y_test == 1)[0]
    idx_no_seiz = np.where(y_test == 0)[0]
    print(len(idx_no_seiz), len(idx_yes_seiz))

    if RUN_CUSTOM_METRICS:
        fs = EEG_FS
        t_window = len(x_test[0][0]) / fs
        t_overlap = 10
        t_overlap_seiz = 11
        thin_skip = int(t_window / (t_window - t_overlap))
        thin_skip_seiz = int(t_window / (t_window - t_overlap_seiz))

        thin_idx_no_seiz = idx_no_seiz[0:batch_size * thin_skip:thin_skip] #batch size
        thin_idx_yes_seiz = idx_yes_seiz[0:batch_size * thin_skip_seiz:thin_skip_seiz]  #batch size

        thin_idx_no_seiz = idx_no_seiz[0::thin_skip] #Full test set
        thin_idx_yes_seiz = idx_yes_seiz[0::thin_skip_seiz] #full test set

        # batch_size = min(len(thin_idx_no_seiz), len(thin_idx_yes_seiz)) # full valdiation set based on how large smallest class is (so we have same statistical power / do not bias RMA)
        # # BUT since valdiation is of different patients we still do the full sizes 
        
        # thin_idx_no_seiz = idx_no_seiz[0:batch_size * thin_skip:thin_skip] #batch size
        # thin_idx_yes_seiz = idx_yes_seiz[0:batch_size * thin_skip_seiz:thin_skip_seiz]  #batch size

        print(len(thin_idx_no_seiz), len(thin_idx_yes_seiz))

        x_batched_no_seiz = x_test[thin_idx_no_seiz]
        y_batched_no_seiz = y_test[thin_idx_no_seiz]
        m_batched_no_seiz = y_mask_test[thin_idx_no_seiz] if y_mask_test is not None else None

        x_batched_yes_seiz = x_test[thin_idx_yes_seiz]
        y_batched_yes_seiz = y_test[thin_idx_yes_seiz]
        m_batched_yes_seiz = y_mask_test[thin_idx_yes_seiz] if y_mask_test is not None else None

        testset_no_seiz = MyUtils.prepare_graphs_labels(x_batched_no_seiz, y_batched_no_seiz, Model.adj, masks=m_batched_no_seiz)
        testset_yes_seiz = MyUtils.prepare_graphs_labels(x_batched_yes_seiz, y_batched_yes_seiz, Model.adj, masks=m_batched_yes_seiz)
    else:
        start_idx = idx_yes_seiz[5]
        x_batched = x_test[start_idx:start_idx + batch_size]
        y_batched = y_test[start_idx:start_idx + batch_size]
        m_batched = y_mask_test[start_idx:start_idx + batch_size] if y_mask_test is not None else None
        testset = MyUtils.prepare_graphs_labels(x_batched, y_batched, Model.adj, masks=m_batched)

    if RUN_CUSTOM_METRICS:
        print("... running custom metric quantification ...")
        tic = time.time()

        def _run_split_metrics(split_name: str, graphs: Sequence[Data]) -> Dict[str, Any]:
            if len(graphs) == 0:
                print(f"[CustomMetrics] {split_name}: no graphs found, skipping.")
                return {"meta": {"split": split_name, "n_samples": 0}, "metrics": {}}

            split_dir = os.path.join(RESULTS_DIR, split_name)
            os.makedirs(split_dir, exist_ok=True)

            cres_split = run_custom_metrics_co12(
                model=model,
                graphs=graphs,
                thr=threshold,
                results_dir=split_dir,
                device=device,
                eeg_fs=EEG_FS,
                edge_rma_interaction=EDGE_RMA_INTERACTION,
                model_kind=MODEL_KIND,
            )
            print(f"[CustomMetrics] {split_name}: saved to {os.path.join(split_dir, 'custom_metrics_results.json')}")
            return cres_split

        cres_no = _run_split_metrics("non_seizure", testset_no_seiz)
        cres_yes = _run_split_metrics("seizure", testset_yes_seiz)

        global_rma: Dict[str, Any] = {"meta": {}}
        global_key = "IG_RAW" if MODEL_KIND == "base" else "FOCUS_MAP"
        global_rma[global_key] = {}

        try:
            graphs_all: List[Data] = list(testset_no_seiz) + list(testset_yes_seiz)
            edge_index_all = graphs_all[0].edge_index.to(device)
            x_raw_all = np.stack([g.x.detach().cpu().numpy() for g in graphs_all], axis=0)
            y_true_all = np.array([int(g.y.detach().cpu().view(-1)[0].item()) for g in graphs_all], dtype=int)

            s_time_all = None
            if all(hasattr(g, "y_mask") for g in graphs_all):
                try:
                    s_time_all = np.stack([_to_numpy(getattr(g, "y_mask")).squeeze() for g in graphs_all], axis=0)
                except Exception:
                    s_time_all = None

            global_rma["meta"] = {
                "n_samples": int(len(graphs_all)),
                "has_temporal_gt": bool(s_time_all is not None),
                "eeg_fs": int(EEG_FS),
                "rma_s_threshold": float(os.getenv("RMA_S_THRESHOLD", "0.5")),
                "edge_rma_interaction": str(EDGE_RMA_INTERACTION),
                "model_kind": MODEL_KIND,
            }

            if s_time_all is not None:
                if MODEL_KIND == "base":
                    a_all = ig_explainer_raw(
                        model=model,
                        inputs=x_raw_all,
                        targets=y_true_all,
                        abs=False,
                        normalise=False,
                        edge_index=edge_index_all,
                        device=device,
                        thr=threshold,
                    )
                    # print(a_all.max())
                else:
                    a_all = focusmap_explainer_raw(
                        model=model,
                        inputs=x_raw_all,
                        targets=y_true_all,
                        abs=False,
                        normalise=False,
                        edge_index=edge_index_all,
                        device=device,
                    )

                rma_global = relevance_mass_accuracy_temporal_global(
                    a_batch=a_all,
                    s_batch=s_time_all,
                    abs_val=False,
                    positive_only=True,
                    eps=1e-12,
                    fs=int(EEG_FS),
                    s_threshold=float(global_rma["meta"]["rma_s_threshold"]),
                    normalize_per_window=False,
                )
                # rma_global = topk_intersection_temporal_global(
                #     a_batch=a_all,
                #     s_batch=s_time_all,
                #     k=192,  # choose this
                #     abs_val=False,
                #     concept_influence=False,
                #     fs=int(EEG_FS),
                #     s_threshold=float(global_rma["meta"]["rma_s_threshold"]),
                #     positive_only=True,   # optional; set False if you want exact Quantus abs-only style
                # )
                global_rma[global_key] = {
                    "RMA_global": float(rma_global),
                    "y_true": y_true_all.tolist(),
                }

        except Exception as e:
            print("RMA coherence failed")
            global_rma["error"] = str(e)

        combined_results = {
            "non_seizure": cres_no,
            "seizure": cres_yes,
            "global_rma": global_rma,
        }
        out_json = os.path.join(RESULTS_DIR, "custom_metrics_results_split.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(combined_results), f, indent=2)
        print(f"... combined custom metric results saved to {out_json}")

        split_metrics = {
            "Non-seizure": cres_no.get("metrics", {}),
            "Seizure": cres_yes.get("metrics", {}),
        }

        print("\nCustom metrics summary (mean ± std):")
        for split_name, metrics in split_metrics.items():
            print(f"\n[{split_name}]")
            if len(metrics) == 0:
                print("  (no metrics)")
                continue

            for explainer, expl_metrics in metrics.items():
                print(f"  [{explainer}]")
                for metric_name, values in expl_metrics.items():
                    nums = _extract_numeric_values(values)
                    if len(nums) == 0:
                        print(f"    - {metric_name}: no numeric values")
                    else:
                        arr = np.asarray(nums, dtype=float)
                        print(f"    - {metric_name}: {arr.mean():.4f} ± {arr.std(ddof=0):.4f}  (n={len(arr)})")

        rows = []
        for condition, metrics in split_metrics.items():
            for explainer, expl_metrics in metrics.items():
                for metric_name, values in expl_metrics.items():
                    nums = _extract_numeric_values(values)
                    if len(nums) == 0:
                        continue
                    for v in nums:
                        rows.append({
                            "Condition": str(condition),
                            "Explainer": str(explainer),
                            "Metric": str(metric_name),
                            "Value": float(v),
                        })

        df_metrics = pd.DataFrame(rows)
        out_csv = os.path.join(RESULTS_DIR, "custom_metrics_flat_split.csv")
        df_metrics.to_csv(out_csv, index=False)
        print(f"... flattened metric values saved to {out_csv}")

        if MODEL_KIND == "base":
            row_keys = [
                ("IG", ["IG_RAW", "IG", "IntegratedGradients"]),
                ("GNNExplainer", ["GNN_EDGE", "GNNEXPLAINER", "GNNExplainer", "EDGE"]),
            ]
        else:
            row_keys = [
                ("FocusMap", ["FOCUS_MAP", "SENN_FOCUS_MAP", "FOCUSMAP"]),
            ]

        def _pick_row_key(candidates, metrics_non, metrics_seiz):
            for c in candidates:
                if (c in metrics_non) or (c in metrics_seiz):
                    return c
            return None

        metrics_non = split_metrics["Non-seizure"]
        metrics_seiz = split_metrics["Seizure"]

        picked_rows = []
        for display_name, candidates in row_keys:
            k = _pick_row_key(candidates, metrics_non, metrics_seiz)
            if k is not None:
                picked_rows.append((display_name, k))

        global_rma_block = {}
        try:
            global_rma_block = (combined_results.get("global_rma", {}) if isinstance(combined_results, dict) else {})
        except Exception:
            print("RMA results loading failed")
            global_rma_block = {}

        GLOBAL_RMA_METRIC_NAME = "Coherence_RelevanceMassAccuracy_global"

        if len(picked_rows) == 0:
            print("[Plot] Could not find explainer keys in split results; skipping boxplot.")
        else:
            all_metric_names = sorted({
                m
                for _, k in picked_rows
                for src in (metrics_non, metrics_seiz)
                for m in src.get(k, {}).keys()
                if not str(m).endswith("_error")
            })

            have_any_global = any(
                (expl_key in global_rma_block) and (global_rma_block.get(expl_key) is not None)
                for _, expl_key in picked_rows
            )
            if have_any_global and (GLOBAL_RMA_METRIC_NAME not in all_metric_names):
                all_metric_names.append(GLOBAL_RMA_METRIC_NAME)

            preferred_metric_order = [
                "Continuity_LocalLipschitz",
                "Correctness_ParamRandomisation_corr_spearman",
                "FaithfulnessCorrelation",
                "TopKDeletion_drop",
                GLOBAL_RMA_METRIC_NAME,
            ]
            metric_names = [m for m in preferred_metric_order if m in all_metric_names]
            metric_names += [m for m in all_metric_names if m not in metric_names]

            if len(metric_names) == 0:
                print("[Plot] No metric columns found; skipping boxplot.")
            else:
                n_rows = len(picked_rows)
                n_cols = len(metric_names)

                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(max(14, 4.2 * n_cols), max(5, 3.8 * n_rows)),
                    squeeze=False,
                    sharey=False,
                )

                for r, (disp_name, expl_key) in enumerate(picked_rows):
                    for c, metric_name in enumerate(metric_names):
                        ax = axes[r, c]

                        if metric_name == GLOBAL_RMA_METRIC_NAME:
                            gblock = global_rma_block.get(expl_key, {})
                            gval = None
                            for k_try in ["RMA_global", "global", "value", GLOBAL_RMA_METRIC_NAME]:
                                if isinstance(gblock, dict) and (k_try in gblock):
                                    gval = gblock.get(k_try)
                                    break
                            if gval is None and isinstance(gblock, (float, int)):
                                gval = gblock

                            if gval is None:
                                ax.text(0.5, 0.5, "No global RMA", ha="center", va="center", transform=ax.transAxes)
                                ax.set_xticks([1])
                                ax.set_xticklabels(["Global"], fontsize=8)
                                ax.grid(alpha=0.15, axis="y")
                            else:
                                gval = float(gval)
                                ax.bar([1], [gval])
                                ax.set_xticks([1])
                                ax.set_xticklabels(["Global"], fontsize=8)
                                ax.grid(alpha=0.25, axis="y")
                                ax.text(0.5, 0.98, f"{gval:.4f}", transform=ax.transAxes, ha="center", va="top", fontsize=9)

                            if r == 0:
                                ax.set_title(metric_name, fontsize=10)
                            if c == 0:
                                ax.set_ylabel(disp_name, fontsize=10)
                            continue

                        vals_non = _extract_numeric_values(metrics_non.get(expl_key, {}).get(metric_name, []))
                        vals_seiz = _extract_numeric_values(metrics_seiz.get(expl_key, {}).get(metric_name, []))

                        plot_data = []
                        positions = []
                        if len(vals_non) > 0:
                            plot_data.append(np.asarray(vals_non, dtype=float))
                            positions.append(1)
                        if len(vals_seiz) > 0:
                            plot_data.append(np.asarray(vals_seiz, dtype=float))
                            positions.append(2)

                        if len(plot_data) > 0:
                            ax.boxplot(plot_data, positions=positions, widths=0.55, showmeans=True)
                            ax.grid(alpha=0.25, axis="y")
                            ax.set_xticks([1, 2])
                            ax.set_xticklabels([f"Non-seiz\n(n={len(vals_non)})", f"Seiz\n(n={len(vals_seiz)})"], fontsize=8)

                            stats_txt_parts = []
                            if len(vals_non) > 0:
                                arrn = np.asarray(vals_non, dtype=float)
                                stats_txt_parts.append(f"μN={arrn.mean():.3f} σN={arrn.std(ddof=0):.3f}")
                            if len(vals_seiz) > 0:
                                arrs = np.asarray(vals_seiz, dtype=float)
                                stats_txt_parts.append(f"μS={arrs.mean():.3f} σS={arrs.std(ddof=0):.3f}")

                            ax.text(0.5, 0.98, "\n".join(stats_txt_parts), transform=ax.transAxes, ha="center", va="top", fontsize=8)
                        else:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            ax.set_xticks([1, 2])
                            ax.set_xticklabels(["Non-seiz", "Seiz"], fontsize=8)
                            ax.grid(alpha=0.15, axis="y")

                        if r == 0:
                            ax.set_title(metric_name, fontsize=10)
                        if c == 0:
                            ax.set_ylabel(disp_name, fontsize=10)

                fig.suptitle(
                    f"Custom Metrics ({MODEL_KIND.upper()} | split boxplots + Global RMA)",
                    y=1.02,
                    fontsize=12,
                )
                fig.tight_layout()

                out_png = os.path.join(RESULTS_DIR, "custom_metrics_grid_split_plus_global_rma.png")
                fig.savefig(out_png, dpi=200, bbox_inches="tight")
                print(f"... plot saved to {out_png}")

                toc = time.time() - tic
                print(f"time elapsed for metrics {int(toc)}s")
                plt.show()

    if INT_PLOT:
        testset = list(testset_yes_seiz) if RUN_CUSTOM_METRICS else testset
        x_batched = np.stack([g.x.detach().cpu().numpy() for g in testset], axis=0)
        y_batched = np.array([int(g.y.detach().cpu().item()) for g in testset], dtype=int)

        print("... running inference ... ")
        y_prob, _ = MyUtils.run_inference(testset, model)
        pred_batch = (y_prob >= threshold).astype(int)

        temporal_expl = []
        spatial_edge_expl = []
        spatial_node_expl = []

        print("... computing explanations for plotting ... ")
        for graph in tqdm(testset):
            graph = graph.to(device)
            if MODEL_KIND == "base":
                exp_ig = MyUtils.calculateIG(model, graph, thr=threshold, target_key="logit")
                temporal_expl.append(exp_ig.node_mask.detach().cpu().numpy())

                exp_gnn = MyUtils.calculateGNNexpl(model, graph, thr=threshold, epochs=GNNEXPL_EPOCHS_QUANT)
                spatial_edge_expl.append(exp_gnn.edge_mask.detach().cpu().numpy())
                spatial_node_expl.append(exp_gnn.node_mask.detach().cpu().numpy())
            else:
                batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
                out = model(graph.x, graph.edge_index, batch)
                temporal_expl.append(_extract_model_output(out, "explanation").detach().cpu().numpy())

        temporal_expl = np.asarray(temporal_expl)
        spatial_edge_expl = np.asarray(spatial_edge_expl) if len(spatial_edge_expl) > 0 else None
        spatial_node_expl = np.asarray(spatial_node_expl) if len(spatial_node_expl) > 0 else None

        MyUtils.plot_data_int_expl(
            x_batched,
            y_batched,
            edge_index=testset[0].edge_index,
            pred_batch=pred_batch,
            temp_expl=temporal_expl,
            spat_node_expl=spatial_node_expl,
            spat_edge_expl=spatial_edge_expl,
        )
