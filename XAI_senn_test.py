"""
XAI.py (Custom metrics only, clean single-output pipeline)

This script:
1) Loads a trained CNN-GAT (PyG) model and a chosen CV fold.
2) Generates:
   - Temporal explanations: Integrated Gradients on raw node signals (graph_data.x).
   - Spatial explanations: GNNExplainer on CNN node-features (model.cnn(x)) and edges.
3) Quantifies explanation quality using custom metrics only (no Quantus dependency):
   - Continuity  -> Relative Input Stability (RIS)
   - Correctness -> parameter randomisation sanity correlation
   - Coherence   -> global temporal coherence (mean pairwise Jaccard overlap of top-k IG time-blocks across channels)
                  (+ optional Top-K intersection vs channel-level ground truth if provided)
   - FaithfulnessCorrelation -> perturbation-based correlation metric
   - TopKDeletion_drop -> necessity via deleting top-k IG time-blocks (per channel)
   - OutputCompleteness_TargetEvidenceDeletion_IROF_AOC -> IROF-style target-class evidence deletion curve

Notes:
- Single-output binary model is used directly (no 2-class wrappers).
- MyUtils.calculateIG and MyUtils.calculateGNNexpl are used for explanations.
- Legacy local Lipschitz continuity helpers are kept in this file for backward reference, but continuity is now evaluated with RIS.
"""

import os
import json
import argparse
import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="", help="Saved_models_* directory for one model run")
    parser.add_argument("--data_folder", default="", help="Path to Datasets/CV_Folds")
    parser.add_argument("--results_dir", default="", help="Output directory; wrapper passes Results_*/Explainability_metrics")
    parser.add_argument("--history_dir", default="", help="Optional History_* directory; currently kept for symmetry with Validation.py")
    parser.add_argument("--fold", default="", help="CV fold to explain, e.g. 5")
    parser.add_argument("--checkpoint_name", default="best_auprc.pt")
    parser.add_argument("--model_kind", default="auto", help="auto, base, senn, senn_fixed, senn_fixedconcepttheta")
    parser.add_argument("--is_trivial", default="auto", help="auto/true/false. true builds senn_trivialfixed but evaluates as senn_fixed")
    parser.parse_args()
    raise SystemExit(0)

import copy
import gc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import Models_senn as Model
import MyUtils_senn_test as MyUtils
from data_utils import load_fold_arrays, prepare_graphs_labels
from io_utils import to_jsonable
from model_utils import build_model as _shared_build_model
from model_utils import extract_model_output as _shared_extract_model_output
from model_utils import normalize_model_kind
from torch_geometric.data import Data
from tqdm import tqdm


# ============================================================
# USER SETTINGS
# ============================================================
RUN_CUSTOM_METRICS = True
EEG_FS = 32
EDGE_RMA_INTERACTION = "abs_product"
LIPSCHITZ_NR_SAMPLES_IG = 25
LIPSCHITZ_NR_SAMPLES_GNN = 5
GNNEXPL_EPOCHS_QUANT = 80
IG_STEPS = 32

RIS_NR_IG = LIPSCHITZ_NR_SAMPLES_IG
RIS_STD_IG = 0.05
RIS_ALPHA_IG = 0.05
RIS_NR_EDGE = LIPSCHITZ_NR_SAMPLES_GNN
RIS_STD_EDGE = 0.05
RIS_ALPHA_EDGE = 0.05
RIS_EPS_MIN = 1e-8
RIS_INPUT_REL_EPS = 1e-8
RIS_EXPL_REL_EPS = 1e-8
RIS_NORM_P = 2.0

CUSTOM_MPRT_ROUNDS = 3
FAITH_CORR_NR_RUNS = 40
FAITH_CORR_SUBSET_FRAC = 0.10
FAITH_BLOCK_LEN = None
FAITH_MIN_BLOCKS = 4
IG_DEL_BLOCK_LEN = None
IG_DEL_TOPK_PER_CHANNEL = None
IG_DEL_FRAC = None
OUTPUT_COMPLETENESS_METRIC_NAME = "OutputCompleteness_TargetEvidenceDeletion_IROF_AOC"
OUTPUT_COMPLETENESS_MAX_FRAC = 0.50
OUTPUT_COMPLETENESS_STEPS = 10
OUTPUT_COMPLETENESS_EPS = 1e-8
RMA_S_THRESHOLD = 0.5
RMA_BALANCE_MODE = "by_window_count"

DEFAULT_LOG = "485352"
DEFAULT_MODEL_KIND = "senn"
DEFAULT_FOLD = "7"
INT_PLOT = False

DEFAULT_DATA_FOLDER = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets\CV_Folds/"
DEFAULT_HISTORY_DIR = f"./History_{DEFAULT_LOG}"
DEFAULT_MODEL_DIR = f"./Saved_models_{DEFAULT_LOG}"
DEFAULT_RESULTS_DIR = os.path.join(f"./Results_{DEFAULT_LOG}", "Explainability_metrics")
# ============================================================


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
    return to_jsonable(obj)


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
    return _shared_extract_model_output(output, key)


def _build_model(model_kind: str, ckpt: Dict[str, Any], return_explanations: bool = False) -> torch.nn.Module:
    return _shared_build_model(
        model_kind,
        ckpt=ckpt,
        return_explanations=return_explanations,
    )

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
    Extracts logit values
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
        # p1 = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
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


def _target_label_sign(y: Any) -> float:
    """Return +1 for seizure target logits and -1 for non-seizure target logits."""
    return 1.0 if int(y) == 1 else -1.0


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _target_probabilities_from_logits(logits: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """Return P(correct target class) from seizure logits and binary labels."""
    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y_batch, dtype=int).reshape(-1)
    if logits.size != y.size:
        raise ValueError(f"Shape mismatch in target probability selection: logits={logits.shape}, y={y.shape}")
    p_seizure = _sigmoid_stable(logits)
    return np.where(y == 1, p_seizure, 1.0 - p_seizure)


def _normalised_target_probability_drop(
    original_target_prob: float,
    perturbed_target_prob: float,
    eps: float = OUTPUT_COMPLETENESS_EPS,
) -> float:
    """IROF-style relative target-probability degradation.

    The normalized score curve is F(x_pert)_y / F(x)_y. The area over that
    curve is therefore 1 - F(x_pert)_y / F(x)_y at each perturbation level.
    """
    denom = max(float(original_target_prob), float(eps))
    return float(1.0 - (float(perturbed_target_prob) / denom))


def _prepare_deletion_fracs(
    deletion_fracs: Optional[Sequence[float]] = None,
    *,
    max_frac: float = OUTPUT_COMPLETENESS_MAX_FRAC,
    n_steps: int = OUTPUT_COMPLETENESS_STEPS,
) -> np.ndarray:
    """Return sorted perturbation fractions in (0, 1] for deletion-curve metrics."""
    if deletion_fracs is None:
        max_frac = float(np.clip(max_frac, 1e-6, 1.0))
        n_steps = int(max(1, n_steps))
        fracs = np.linspace(max_frac / n_steps, max_frac, n_steps)
    else:
        fracs = np.asarray(list(deletion_fracs), dtype=float)

    fracs = fracs[np.isfinite(fracs)]
    fracs = fracs[fracs > 0.0]
    if fracs.size == 0:
        return np.asarray([float(np.clip(max_frac, 1e-6, 1.0))], dtype=float)
    fracs = np.clip(fracs, 1e-6, 1.0)
    return np.unique(np.sort(fracs))


def _area_over_perturbation_curve(fracs: np.ndarray, drops: Sequence[float]) -> float:
    """Average area over a perturbation-drop curve; higher is better."""
    x = np.asarray(fracs, dtype=float).reshape(-1)
    y = np.asarray(list(drops), dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])
    denom = float(x[-1]) if x[-1] > 0 else 1.0
    return float(np.trapz(y, x) / denom)


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
    output_key: str = "explanation",
    **kwargs,
) -> np.ndarray:
    """
    Extract an intrinsic SENN explanation tensor from the model output dictionary.

    Examples:
      - SENN_raw / fixed node concepts: output_key="explanation"
      - fixed edge concepts:            output_key="explanation_edge"
      - pure relevance scores:          output_key="theta_x" or "theta_x_edge"
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
            a = _extract_model_output(out, output_key).detach().cpu().numpy()

            if abs:
                a = np.abs(a)
            if normalise:
                a = _normalise_by_absmax(a)

            a_list.append(a)

    return np.stack(a_list, axis=0)


def focusmap_explainer_channel(
    model,
    inputs,
    targets,
    abs: bool = False,
    normalise: bool = False,
    output_key: str = "explanation",
    **kwargs,
) -> np.ndarray:
    a = focusmap_explainer_raw(
        model,
        inputs,
        targets,
        abs=False,
        normalise=False,
        output_key=output_key,
        **kwargs,
    )
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
    output_key: str = "explanation",
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
            output_key=output_key,
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
                output_key=output_key,
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
    output_key: str = "explanation",
) -> List[float]:
    x_raw = np.asarray(x_raw, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)

    a_ref = focusmap_explainer_raw(
        model=model,
        inputs=x_raw,
        targets=y_batch,
        abs=False,
        normalise=False,
        output_key=output_key,
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
            output_key=output_key,
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



def _predict_binary_label_full(
    model: torch.nn.Module,
    x_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    thr: float = 0.5,
) -> np.ndarray:
    """Predict binary labels from the *full* model using probability threshold ``thr``."""
    x = torch.tensor(np.asarray(x_batch), dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            batch = torch.zeros(x.shape[1], dtype=torch.long, device=device)
            model_out = model(x[i], edge_index, batch)
            try:
                prob = _extract_model_output(model_out, "prob").view(-1)[0]
            except KeyError:
                logit = _extract_model_output(model_out, "logit").view(-1)[0]
                prob = torch.sigmoid(logit)
            probs.append(float(prob.item()))
    probs = np.asarray(probs, dtype=float)
    return (probs >= float(thr)).astype(int)


def _relative_change_lp(
    ref: np.ndarray,
    other: np.ndarray,
    p: float = 2.0,
    scale_eps: float = 1e-8,
) -> float:
    """Lp norm of the elementwise relative change ``(ref-other)/ref`` with epsilon stabilisation."""
    ref = np.asarray(ref, dtype=float)
    other = np.asarray(other, dtype=float)

    if ref.shape != other.shape:
        raise ValueError(f"Relative change expects matching shapes, got {ref.shape} vs {other.shape}")

    denom = np.maximum(np.abs(ref), float(scale_eps))
    rel = (ref - other) / denom
    rel = np.nan_to_num(rel, nan=0.0, posinf=1e12, neginf=-1e12)
    flat = rel.reshape(-1)
    if flat.size == 0:
        return 0.0

    ord_val = np.inf if (isinstance(p, float) and np.isinf(p)) else p
    return float(np.linalg.norm(flat, ord=ord_val))


def gnn_edge_explainer_raw_inputs(
    model: torch.nn.Module,
    inputs: Any,
    targets: Any,
    abs: bool = False,
    normalise: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    GNNExplainer edge mask evaluated from the *raw* EEG input.

    This keeps the perturbation domain and the RIS denominator in raw-input space,
    while MyUtils.calculateGNNexpl internally maps the raw sample through ``model.cnn``.
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

    out = []
    for i in range(x_np.shape[0]):
        g = Data(x=torch.tensor(x_np[i], dtype=torch.float32), edge_index=edge_index).to(device)
        exp = MyUtils.calculateGNNexpl(model, g, thr=thr, epochs=epochs)

        a = exp.edge_mask.detach().cpu().numpy()
        if abs:
            a = np.abs(a)
        if normalise:
            a = _normalise_by_absmax(a)
        out.append(a)

    return np.stack(out, axis=0)


def relative_input_stability_estimate(
    *,
    x_raw: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    explain_fn: Callable[..., np.ndarray],
    nr_samples: int = 20,
    perturb_std: float = 0.03,
    perturb_alpha: Optional[float] = None,
    thr: float = 0.5,
    explain_kwargs: Optional[Dict[str, Any]] = None,
    p: float = 2.0,
    eps_min: float = 1e-8,
    input_rel_eps: float = 1e-8,
    expl_rel_eps: float = 1e-8,
    require_same_prediction: bool = True,
) -> List[float]:
    """
    Relative Input Stability (RIS).

    RIS(x, x', e_x, e_x') = ||(e_x - e_x') / e_x||_p / max(||(x - x') / x||_p, eps_min)

    The perturbation is always sampled in raw-input space. When ``require_same_prediction``
    is True, only perturbations with the same *predicted* class as the reference sample are
    considered, matching the RIS definition from the paper.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    if x_raw.ndim != 3:
        raise ValueError(f"Expected x_raw with shape (B,C,T), got {x_raw.shape}")

    rng = np.random.default_rng(0)
    std_raw = _compute_feature_std(x_raw)
    explain_kwargs = dict(explain_kwargs or {})

    yhat_ref = _predict_binary_label_full(
        model=model,
        x_batch=x_raw,
        edge_index=edge_index,
        device=device,
        thr=thr,
    )

    scores: List[float] = []
    for i in range(x_raw.shape[0]):
        x0 = x_raw[i:i + 1]
        yhat0 = int(yhat_ref[i])

        a0 = explain_fn(
            model=model,
            inputs=x0,
            targets=np.array([yhat0], dtype=int),
            abs=False,
            normalise=False,
            edge_index=edge_index,
            device=device,
            thr=thr,
            **explain_kwargs,
        )[0]

        ratios = []
        n_iter=0
        n_skip=0
        #We want nr_samples iterations to be consistent across calls, but set a maximum of 100 so we do not get stuck
        while n_iter < int(nr_samples) and n_skip<100:
            if perturb_alpha is not None:
                noise = _sample_relative_noise(std_raw, float(perturb_alpha), rng)[None, ...]
            else:
                noise = rng.normal(loc=0.0, scale=float(perturb_std), size=x0.shape)

            x1 = x0 + noise

            if require_same_prediction:
                yhat1 = _predict_binary_label_full(
                    model=model,
                    x_batch=x1,
                    edge_index=edge_index,
                    device=device,
                    thr=thr,
                )[0]
                if int(yhat1) != yhat0:
                    # print("RIS pertubation gave different pred: skip")
                    n_skip +=1
                    continue

            a1 = explain_fn(
                model=model,
                inputs=x1,
                targets=np.array([yhat0], dtype=int),
                abs=False,
                normalise=False,
                edge_index=edge_index,
                device=device,
                thr=thr,
                **explain_kwargs,
            )[0]

            num = _relative_change_lp(a0, a1, p=p, scale_eps=expl_rel_eps)
            den = _relative_change_lp(x0, x1, p=p, scale_eps=input_rel_eps)
            ratios.append(float(num / max(den, float(eps_min))))

            n_iter+=1

        scores.append(float(np.max(ratios)) if len(ratios) else float("nan"))

    return scores


def feature_relative_input_stability(
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
    p: float = 2.0,
    eps_min: float = 1e-8,
    input_rel_eps: float = 1e-8,
    expl_rel_eps: float = 1e-8,
) -> List[float]:
    """RIS for IG explanations on the base model."""
    return relative_input_stability_estimate(
        x_raw=x_raw,
        model=model,
        edge_index=edge_index,
        device=device,
        explain_fn=ig_explainer_raw,
        nr_samples=nr_samples,
        perturb_std=perturb_std,
        perturb_alpha=perturb_alpha,
        thr=thr,
        explain_kwargs=None,
        p=p,
        eps_min=eps_min,
        input_rel_eps=input_rel_eps,
        expl_rel_eps=expl_rel_eps,
        require_same_prediction=True,
    )


def gnn_edge_relative_input_stability(
    *,
    x_raw: np.ndarray,
    y_batch: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    nr_samples: int = 5,
    perturb_std: float = 0.05,
    perturb_alpha: Optional[float] = None,
    epochs: int = 80,
    thr: float = 0.5,
    p: float = 2.0,
    eps_min: float = 1e-8,
    input_rel_eps: float = 1e-8,
    expl_rel_eps: float = 1e-8,
) -> List[float]:
    """RIS for GNNExplainer edge masks, with perturbations measured in raw-input space."""
    return relative_input_stability_estimate(
        x_raw=x_raw,
        model=model,
        edge_index=edge_index,
        device=device,
        explain_fn=gnn_edge_explainer_raw_inputs,
        nr_samples=nr_samples,
        perturb_std=perturb_std,
        perturb_alpha=perturb_alpha,
        thr=thr,
        explain_kwargs={"epochs": int(epochs)},
        p=p,
        eps_min=eps_min,
        input_rel_eps=input_rel_eps,
        expl_rel_eps=expl_rel_eps,
        require_same_prediction=True,
    )


def focusmap_relative_input_stability(
    *,
    x_raw: np.ndarray,
    y_batch: np.ndarray,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    nr_samples: int = 20,
    perturb_std: float = 0.03,
    perturb_alpha: Optional[float] = None,
    output_key: str = "explanation",
    thr: float = 0.5,
    p: float = 2.0,
    eps_min: float = 1e-8,
    input_rel_eps: float = 1e-8,
    expl_rel_eps: float = 1e-8,
) -> List[float]:
    """RIS for intrinsic SENN explanations (node or edge branch)."""
    return relative_input_stability_estimate(
        x_raw=x_raw,
        model=model,
        edge_index=edge_index,
        device=device,
        explain_fn=focusmap_explainer_raw,
        nr_samples=nr_samples,
        perturb_std=perturb_std,
        perturb_alpha=perturb_alpha,
        thr=thr,
        explain_kwargs={"output_key": output_key},
        p=p,
        eps_min=eps_min,
        input_rel_eps=input_rel_eps,
        expl_rel_eps=expl_rel_eps,
        require_same_prediction=True,
    )

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
    only_positive_blocks: bool = False,
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
        if only_positive_blocks and (not np.isfinite(best_v) or best_v <= 0.0):
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
        only_positive_blocks: if True, only select blocks with score > 0. If
            positive_only=False, this is a net signed block-sum criterion.

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
    only_positive_blocks: Optional[bool] = None,
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

    if only_positive_blocks is None:
        only_positive_blocks = bool(positive_only)

    out = np.zeros((C, T), dtype=bool)
    for c in range(C):
        out[c] = _topk_time_blocks_mask_1d(
            a[c],
            k_blocks=int(k_blocks_per_channel),
            block_len=L,
            disallow_overlap=disallow_overlap,
            only_positive_blocks=bool(only_positive_blocks),
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
            only_positive_blocks=positive_only,
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
        rank_source = _rank_contributions_for_deletion(a_i, positive_only=positive_only, abs_val=False)
        valid_idx = np.flatnonzero(np.isfinite(rank_source))
        if valid_idx.size == 0:
            drops.append(0.0)
            continue

        k = max(1, int(np.ceil(k_frac * E)))
        k = min(k, int(valid_idx.size))
        idx = valid_idx[np.argsort(-rank_source[valid_idx])[:k]]

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


def feature_target_evidence_deletion_irof_aoc(
    *,
    predict_logit_fn: Callable[[np.ndarray], np.ndarray],
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: np.ndarray,
    deletion_fracs: Optional[Sequence[float]] = None,
    baseline_value: float = 0.0,
    block_len: Optional[int] = None,
    disallow_overlap: bool = True,
) -> List[float]:
    """IROF-style target-class evidence deletion for signed raw EEG explanations.

    For seizure windows, positive seizure-logit attributions are deleted. For
    non-seizure windows, negative seizure-logit attributions are deleted by
    ranking ``-attribution``. The score is the area over the normalized
    target-probability curve, 1 - P_y(x_pert) / P_y(x). Contiguous EEG blocks
    are ranked by their net signed target-evidence mass; only blocks with net
    target-evidence mass > 0 are selected.
    """
    x_batch = np.asarray(x_batch, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    a_batch = np.asarray(a_batch, dtype=float)
    if y_batch.size != x_batch.shape[0]:
        raise ValueError(f"Shape mismatch: x_batch has {x_batch.shape[0]} samples, y_batch has {y_batch.size}")

    fracs = _prepare_deletion_fracs(deletion_fracs)
    logits_orig = np.asarray(predict_logit_fn(x_batch), dtype=float).reshape(-1)
    target_prob_orig = _target_probabilities_from_logits(logits_orig, y_batch)

    out: List[float] = []
    for i in range(x_batch.shape[0]):
        x_i = x_batch[i].copy()
        a_i = np.asarray(a_batch[i], dtype=float).copy()
        if x_i.ndim != 2 or a_i.ndim != 2:
            raise ValueError(
                f"Target-evidence deletion expects x and a shaped (C,T); got x={x_i.shape}, a={a_i.shape}"
            )

        C, T = int(x_i.shape[0]), int(x_i.shape[1])
        L = _default_block_len(T) if block_len is None else int(block_len)
        L = int(max(1, min(L, T)))

        sign = _target_label_sign(y_batch[i])
        rank_attr = sign * np.nan_to_num(a_i, nan=0.0, posinf=0.0, neginf=0.0)

        drops: List[float] = []
        for frac in fracs:
            k_pts_total = int(np.ceil(float(frac) * C * T))
            k_blocks = int(max(1, np.ceil(k_pts_total / max(1, L))))
            mask = _topk_time_blocks_mask_global(
                rank_attr,
                k_blocks_total=k_blocks,
                block_len=L,
                positive_only=False,
                abs_val=False,
                disallow_overlap=disallow_overlap,
                only_positive_blocks=True,
            )

            if not mask.any():
                drops.append(0.0)
                continue

            x_pert = x_i.copy()
            x_pert[mask] = float(baseline_value)
            logit_pert = float(predict_logit_fn(x_pert[None, ...])[0])
            target_prob_pert = float(_target_probabilities_from_logits([logit_pert], [y_batch[i]])[0])
            drops.append(_normalised_target_probability_drop(float(target_prob_orig[i]), target_prob_pert))

        out.append(_area_over_perturbation_curve(fracs, drops))

    return [float(v) for v in out]


def edge_target_evidence_deletion_irof_aoc(
    *,
    gnn: torch.nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_edge_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    deletion_fracs: Optional[Sequence[float]] = None,
    signed_attributions: bool = True,
) -> List[float]:
    """IROF-style target-class edge deletion AOC.

    If ``signed_attributions`` is True, ranking uses signed target evidence
    (+a for seizure, -a for non-seizure). For unsigned masks such as
    GNNExplainer, ranking uses mask magnitude. Scoring uses relative
    target-probability degradation.
    """
    x_batch = np.asarray(x_batch, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    a_edge_batch = np.asarray(a_edge_batch, dtype=float)
    if y_batch.size != x_batch.shape[0]:
        raise ValueError(f"Shape mismatch: x_batch has {x_batch.shape[0]} samples, y_batch has {y_batch.size}")

    fracs = _prepare_deletion_fracs(deletion_fracs)
    logits_orig = _predict_p1_gnn(gnn, x_batch, edge_index=edge_index, device=device)
    target_prob_orig = _target_probabilities_from_logits(logits_orig, y_batch)

    ei = edge_index.detach().cpu().numpy()
    E = int(ei.shape[1])

    out: List[float] = []
    for i in range(x_batch.shape[0]):
        a_i = np.asarray(a_edge_batch[i], dtype=float).reshape(-1)
        E_eff = min(E, int(a_i.size))
        if E_eff <= 0:
            out.append(float("nan"))
            continue

        sign = _target_label_sign(y_batch[i])
        if signed_attributions:
            rank_source = _rank_contributions_for_deletion(sign * a_i[:E_eff], positive_only=True, abs_val=False)
        else:
            rank_source = np.abs(np.nan_to_num(a_i[:E_eff], nan=0.0, posinf=0.0, neginf=0.0))

        valid_idx = np.flatnonzero(np.isfinite(rank_source))
        if valid_idx.size == 0:
            out.append(0.0)
            continue

        ranked_idx = valid_idx[np.argsort(-rank_source[valid_idx])]
        drops: List[float] = []
        for frac in fracs:
            k = int(max(1, np.ceil(float(frac) * E_eff)))
            k = min(k, int(ranked_idx.size))
            if k <= 0:
                drops.append(0.0)
                continue

            idx = ranked_idx[:k]
            keep = np.ones(E, dtype=bool)
            keep[idx] = False
            edge_index_pert = torch.tensor(ei[:, keep], dtype=torch.long, device=device)

            logit_pert = float(
                _predict_p1_gnn(
                    gnn,
                    x_batch[i][None, ...],
                    edge_index=edge_index_pert,
                    device=device,
                )[0]
            )
            target_prob_pert = float(_target_probabilities_from_logits([logit_pert], [y_batch[i]])[0])
            drops.append(_normalised_target_probability_drop(float(target_prob_orig[i]), target_prob_pert))

        out.append(_area_over_perturbation_curve(fracs, drops))

    return [float(v) for v in out]



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

def _coerce_sample_labels(
    s_batch: np.ndarray,
    batch_size: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert sample/window labels to a binary vector of shape ``(B,)``.

    Expected usage is with ``y_true`` / sample-level window labels. For convenience,
    this helper also tolerates ``(B, 1)`` arrays and temporal masks ``(B, T)`` by
    marking a window as seizure when any element exceeds ``threshold``.
    """
    s = np.asarray(s_batch)

    if s.ndim == 0:
        return np.full(batch_size, int(float(s) >= threshold), dtype=int)

    if s.ndim == 1:
        if s.size != batch_size:
            raise ValueError(f"Expected s_batch with B={batch_size}, got shape {s.shape}")
        vals = s.astype(float)
    else:
        if s.shape[0] != batch_size:
            raise ValueError(f"Expected s_batch first dimension B={batch_size}, got shape {s.shape}")
        flat = s.reshape(batch_size, -1).astype(float)
        if flat.shape[1] == 1:
            vals = flat[:, 0]
        else:
            vals = (np.nanmax(flat, axis=1) >= float(threshold)).astype(float)

    vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0)
    return (vals >= float(threshold)).astype(int)


def relevance_mass_accuracy_sample_global(
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
    """Global sample-level Relevance Mass Accuracy.

    This version uses sample/window labels ``s_batch = y_true`` instead of temporal
    masks. For each 12-second window ``i`` we first aggregate the *total positive
    relevance mass* over all explanation dimensions belonging to that sample and then
    compute the balanced global relevance mass assigned to seizure windows.

    It therefore works for raw temporal explanations (IG, raw SENN focus maps) as well
    as graph / concept-space explanations such as GNNExplainer edge masks and the fixed
    SENN node/edge concept maps.
    """
    a = np.asarray(a_batch, dtype=float)
    if a.ndim < 2:
        raise ValueError(f"Expected a_batch shaped (B, ...), got {a.shape}")

    batch_size = int(a.shape[0])
    y_sample = _coerce_sample_labels(s_batch, batch_size=batch_size, threshold=s_threshold)

    if abs_val:
        a = np.abs(a)
    if positive_only:
        a = np.maximum(a, 0.0)

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mass_i = a.reshape(batch_size, -1).sum(axis=1)

    if normalize_per_window:
        mass_i = mass_i / (mass_i + float(eps))

    num_i = mass_i * y_sample.astype(float)
    den_i = mass_i + float(eps)

    has_seiz = y_sample.astype(bool)
    n_seiz = int(has_seiz.sum())
    n_nonseiz = int((~has_seiz).sum())

    if balance_mode == "pooled":
        return float(num_i.sum() / (den_i.sum() + float(eps)))

    if balance_mode == "by_window_count":
        if n_seiz == 0 and n_nonseiz == 0:
            return 0.0
        w_seiz = 0.0 if n_seiz == 0 else 1.0 / float(n_seiz)
        w_non = 0.0 if n_nonseiz == 0 else 1.0 / float(n_nonseiz)
        weights = np.where(has_seiz, w_seiz, w_non)
        return float((weights * num_i).sum() / ((weights * den_i).sum() + float(eps)))

    if balance_mode == "seizure_only":
        if n_seiz == 0:
            return 0.0
        return float(num_i[has_seiz].sum() / (den_i[has_seiz].sum() + float(eps)))

    raise ValueError(f"Unknown balance_mode={balance_mode!r}")


def relevance_mass_accuracy_sample_global_details(
    *,
    a_batch: np.ndarray,
    s_batch: np.ndarray,
    abs_val: bool = False,
    positive_only: bool = True,
    eps: float = 1e-12,
    fs: int = 32,
    s_threshold: float = 0.5,
    normalize_per_window = False,
    balance_mode: str = "by_window_count",
) -> Dict[str, Any]:
    """Convenience wrapper returning the scalar sample-RMA plus class-wise totals."""
    a = np.asarray(a_batch, dtype=float)
    if a.ndim < 2:
        raise ValueError(f"Expected a_batch shaped (B, ...), got {a.shape}")

    batch_size = int(a.shape[0])
    y_sample = _coerce_sample_labels(s_batch, batch_size=batch_size, threshold=s_threshold)

    work = np.abs(a) if abs_val else np.asarray(a, dtype=float)
    if positive_only:
        work = np.maximum(work, 0.0)
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    mass_i = work.reshape(batch_size, -1).sum(axis=1)

    if normalize_per_window:
        mass_i = mass_i / (mass_i + float(eps))

    rma_global = relevance_mass_accuracy_sample_global(
        a_batch=a_batch,
        s_batch=s_batch,
        abs_val=abs_val,
        positive_only=positive_only,
        eps=eps,
        fs=fs,
        s_threshold=s_threshold,
        normalize_per_window=normalize_per_window,
        balance_mode=balance_mode,
    )

    seiz_mask = y_sample == 1
    non_mask = ~seiz_mask
    return {
        "RMA_global": float(rma_global),
        "Coherence_RelevanceMassAccuracy_sample_global": float(rma_global),
        "balance_mode": str(balance_mode),
        "n_samples": int(batch_size),
        "n_seizure": int(seiz_mask.sum()),
        "n_nonseizure": int(non_mask.sum()),
        "positive_mass_total": float(mass_i.sum()),
        "positive_mass_seizure": float(mass_i[seiz_mask].sum()) if np.any(seiz_mask) else 0.0,
        "positive_mass_nonseizure": float(mass_i[non_mask].sum()) if np.any(non_mask) else 0.0,
        "y_true": y_sample.astype(int).tolist(),
    }





def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _rank_contributions_for_deletion(a: np.ndarray, positive_only: bool = False, abs_val: bool = False) -> np.ndarray:
    """Prepare contribution scores for top-k deletion ranking.

    When ``positive_only`` is True we *strictly filter* to positive evidence by
    assigning all non-positive entries ``-inf``. This prevents the top-k selector
    from deleting arbitrary zero/negative entries merely to fill k, which is
    especially important on non-seizure windows where there may be no positive
    evidence at all.

    ``positive_only`` takes precedence over ``abs_val`` by design.
    """
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if positive_only:
        return np.where(a > 0.0, a, -np.inf)
    if abs_val:
        a = np.abs(a)
    return a


def _select_topk_flat_indices(
    rank_source: np.ndarray,
    *,
    k_frac: float = 0.10,
    k_total: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Select top-k flat indices from a ranking tensor.

    Entries with non-finite scores (e.g. ``-inf`` from strict positive-only
    filtering) are excluded from selection. If no valid candidates remain, an
    empty index array is returned.
    """
    flat_rank = np.asarray(rank_source, dtype=float).reshape(-1)
    n_feat = int(flat_rank.size)
    if n_feat == 0:
        return np.empty((0,), dtype=int), n_feat

    valid = np.isfinite(flat_rank)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.empty((0,), dtype=int), n_feat

    k = int(max(1, np.ceil(float(k_frac) * n_feat))) if k_total is None else int(max(1, k_total))
    k = min(k, int(valid_idx.size))
    if k <= 0:
        return np.empty((0,), dtype=int), n_feat

    top_local = np.argsort(-flat_rank[valid_idx])[:k]
    return valid_idx[top_local], n_feat


def _fixed_node_logit_from_fmap(f_node: np.ndarray) -> float:
    f_node = np.asarray(f_node, dtype=float)
    node_score = f_node.sum(axis=-1)
    return float(node_score.mean())


def _fixed_edge_logit_from_fmap(f_edge: np.ndarray) -> float:
    f_edge = np.asarray(f_edge, dtype=float)
    edge_score = f_edge.sum(axis=-1)
    return float(edge_score.mean())


def _single_fixed_concept_outputs(
    model: torch.nn.Module,
    x_single: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(np.asarray(x_single), dtype=torch.float32, device=device)
        batch = torch.zeros(x_t.shape[0], dtype=torch.long, device=device)
        out = model(x_t, edge_index, batch)
    return {
        "logit": float(_extract_model_output(out, "logit").detach().cpu().view(-1)[0].item()),
        "logit_node": float(_extract_model_output(out, "logit_node").detach().cpu().view(-1)[0].item()),
        "logit_edge": float(_extract_model_output(out, "logit_edge").detach().cpu().view(-1)[0].item()),
        "F_node": _extract_model_output(out, "explanation").detach().cpu().numpy(),
        "F_edge": _extract_model_output(out, "explanation_edge").detach().cpu().numpy(),
    }



def _single_fixedconcepttheta_state(
    model: torch.nn.Module,
    x_single: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Forward one concept-theta sample and keep tensors required for concept ablation."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(np.asarray(x_single), dtype=torch.float32, device=device)
        batch = torch.zeros(x_t.shape[0], dtype=torch.long, device=device)
        out = model(x_t, edge_index, batch)
        if not isinstance(out, dict) or ("h_x" not in out) or ("h_x_edge" not in out):
            raise KeyError("Concept-theta deletion expects model outputs to include 'h_x' and 'h_x_edge'.")
        h_node = out["h_x"].detach().clone()
        h_edge = out["h_x_edge"].detach().clone()

    return {
        "batch": batch,
        "h_node": h_node,
        "h_edge": h_edge,
        "logit": float(_extract_model_output(out, "logit").detach().cpu().view(-1)[0].item()),
        "logit_node": float(_extract_model_output(out, "logit_node").detach().cpu().view(-1)[0].item()),
        "logit_edge": float(_extract_model_output(out, "logit_edge").detach().cpu().view(-1)[0].item()),
        "F_node": _extract_model_output(out, "explanation").detach().cpu().numpy(),
        "F_edge": _extract_model_output(out, "explanation_edge").detach().cpu().numpy(),
    }


def _recompute_fixedconcepttheta_from_concepts(
    model: torch.nn.Module,
    h_node: torch.Tensor,
    h_edge: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
) -> Dict[str, float]:
    """Recompute logits after concept-space ablation for SENN_fixedconcepts_concepttheta."""
    model.eval()
    with torch.no_grad():
        theta_node, theta_edge, _ = model.relevance(h_node, h_edge, edge_index, batch)
        edge_batch = batch[edge_index[0]]
        aggr_out = model.aggregrator(
            h_node=h_node,
            theta_node=theta_node,
            batch_node=batch,
            h_edge=h_edge,
            theta_edge=theta_edge,
            batch_edge=edge_batch,
        )

    return {
        "logit": float(aggr_out["logit"].detach().cpu().view(-1)[0].item()),
        "logit_node": float(aggr_out["logit_node"].detach().cpu().view(-1)[0].item()),
        "logit_edge": float(aggr_out["logit_edge"].detach().cpu().view(-1)[0].item()),
    }


def fixed_node_topk_deletion_drop(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.10,
    k_total: Optional[int] = None,
    positive_only: bool = False,
    abs_val: bool = False,
    use_logit_drop: bool = True,
) -> List[float]:
    x_batch = np.asarray(x_batch, dtype=float)
    drops: List[float] = []

    for i in range(x_batch.shape[0]):
        out_i = _single_fixed_concept_outputs(model, x_batch[i], edge_index=edge_index, device=device)
        f_node = np.asarray(out_i["F_node"], dtype=float)
        rank_source = _rank_contributions_for_deletion(f_node, positive_only=positive_only, abs_val=abs_val)

        idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=k_frac, k_total=k_total)
        if n_feat == 0:
            drops.append(float("nan"))
            continue
        if idx.size == 0:
            drops.append(0.0)
            continue

        f_node_masked = f_node.reshape(-1).copy()
        f_node_masked[idx] = 0.0
        f_node_masked = f_node_masked.reshape(f_node.shape)

        new_logit = _fixed_node_logit_from_fmap(f_node_masked) + float(out_i["logit_edge"])
        old_logit = float(out_i["logit"])

        if use_logit_drop:
            drop = old_logit - new_logit
        else:
            drop = float(_sigmoid_np(old_logit) - _sigmoid_np(new_logit))
        drops.append(float(drop))

    return drops


def fixed_edge_topk_deletion_drop(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.10,
    k_total: Optional[int] = None,
    positive_only: bool = False,
    abs_val: bool = False,
    use_logit_drop: bool = True,
) -> List[float]:
    x_batch = np.asarray(x_batch, dtype=float)
    drops: List[float] = []

    for i in range(x_batch.shape[0]):
        out_i = _single_fixed_concept_outputs(model, x_batch[i], edge_index=edge_index, device=device)
        f_edge = np.asarray(out_i["F_edge"], dtype=float)
        rank_source = _rank_contributions_for_deletion(f_edge, positive_only=positive_only, abs_val=abs_val)

        idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=k_frac, k_total=k_total)
        if n_feat == 0:
            drops.append(float("nan"))
            continue
        if idx.size == 0:
            drops.append(0.0)
            continue

        f_edge_masked = f_edge.reshape(-1).copy()
        f_edge_masked[idx] = 0.0
        f_edge_masked = f_edge_masked.reshape(f_edge.shape)

        new_logit = float(out_i["logit_node"]) + _fixed_edge_logit_from_fmap(f_edge_masked)
        old_logit = float(out_i["logit"])

        if use_logit_drop:
            drop = old_logit - new_logit
        else:
            drop = float(_sigmoid_np(old_logit) - _sigmoid_np(new_logit))
        drops.append(float(drop))

    return drops



def fixedconcepttheta_node_topk_deletion_drop(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.10,
    k_total: Optional[int] = None,
    positive_only: bool = False,
    abs_val: bool = False,
    use_logit_drop: bool = True,
    baseline_value: float = 0.0,
) -> List[float]:
    """Concept-space top-k deletion for SENN_fixedconcepts_concepttheta node concepts.

    Ranking uses the current node focus map for comparability with the legacy fixed-concept
    metric, while deletion is performed on h_node followed by a full recomputation of the
    concept-driven relevance network and aggregator.
    """
    x_batch = np.asarray(x_batch, dtype=float)
    drops: List[float] = []

    for i in range(x_batch.shape[0]):
        state_i = _single_fixedconcepttheta_state(model, x_batch[i], edge_index=edge_index, device=device)
        f_node = np.asarray(state_i["F_node"], dtype=float)
        rank_source = _rank_contributions_for_deletion(f_node, positive_only=positive_only, abs_val=abs_val)

        idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=k_frac, k_total=k_total)
        if n_feat == 0:
            drops.append(float("nan"))
            continue
        if idx.size == 0:
            drops.append(0.0)
            continue

        h_node_masked = state_i["h_node"].reshape(-1).clone()
        h_node_masked[idx] = float(baseline_value)
        h_node_masked = h_node_masked.reshape(state_i["h_node"].shape)

        recomputed = _recompute_fixedconcepttheta_from_concepts(
            model=model,
            h_node=h_node_masked,
            h_edge=state_i["h_edge"],
            edge_index=edge_index,
            batch=state_i["batch"],
        )

        old_logit = float(state_i["logit"])
        new_logit = float(recomputed["logit"])

        if use_logit_drop:
            drop = old_logit - new_logit
        else:
            drop = float(_sigmoid_np(old_logit) - _sigmoid_np(new_logit))
        drops.append(float(drop))

    return drops


def fixedconcepttheta_edge_topk_deletion_drop(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.10,
    k_total: Optional[int] = None,
    positive_only: bool = False,
    abs_val: bool = False,
    use_logit_drop: bool = True,
    baseline_value: float = 0.0,
) -> List[float]:
    """Concept-space top-k deletion for SENN_fixedconcepts_concepttheta edge concepts."""
    x_batch = np.asarray(x_batch, dtype=float)
    drops: List[float] = []

    for i in range(x_batch.shape[0]):
        state_i = _single_fixedconcepttheta_state(model, x_batch[i], edge_index=edge_index, device=device)
        f_edge = np.asarray(state_i["F_edge"], dtype=float)
        rank_source = _rank_contributions_for_deletion(f_edge, positive_only=positive_only, abs_val=abs_val)

        idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=k_frac, k_total=k_total)
        if n_feat == 0:
            drops.append(float("nan"))
            continue
        if idx.size == 0:
            drops.append(0.0)
            continue

        h_edge_masked = state_i["h_edge"].reshape(-1).clone()
        h_edge_masked[idx] = float(baseline_value)
        h_edge_masked = h_edge_masked.reshape(state_i["h_edge"].shape)

        recomputed = _recompute_fixedconcepttheta_from_concepts(
            model=model,
            h_node=state_i["h_node"],
            h_edge=h_edge_masked,
            edge_index=edge_index,
            batch=state_i["batch"],
        )

        old_logit = float(state_i["logit"])
        new_logit = float(recomputed["logit"])

        if use_logit_drop:
            drop = old_logit - new_logit
        else:
            drop = float(_sigmoid_np(old_logit) - _sigmoid_np(new_logit))
        drops.append(float(drop))

    return drops


def fixed_node_target_evidence_deletion_irof_aoc(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    deletion_fracs: Optional[Sequence[float]] = None,
) -> List[float]:
    """IROF-style target-evidence AOC for fixed node concept contribution maps."""
    x_batch = np.asarray(x_batch, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    if y_batch.size != x_batch.shape[0]:
        raise ValueError(f"Shape mismatch: x_batch has {x_batch.shape[0]} samples, y_batch has {y_batch.size}")

    fracs = _prepare_deletion_fracs(deletion_fracs)
    out: List[float] = []

    for i in range(x_batch.shape[0]):
        out_i = _single_fixed_concept_outputs(model, x_batch[i], edge_index=edge_index, device=device)
        f_node = np.asarray(out_i["F_node"], dtype=float)
        target_prob_orig = float(_target_probabilities_from_logits([out_i["logit"]], [y_batch[i]])[0])
        sign = _target_label_sign(y_batch[i])
        rank_source = _rank_contributions_for_deletion(sign * f_node, positive_only=True, abs_val=False)

        drops: List[float] = []
        for frac in fracs:
            idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=float(frac), k_total=None)
            if n_feat == 0:
                drops.append(float("nan"))
                continue
            if idx.size == 0:
                drops.append(0.0)
                continue

            f_node_masked = f_node.reshape(-1).copy()
            f_node_masked[idx] = 0.0
            f_node_masked = f_node_masked.reshape(f_node.shape)

            new_logit = _fixed_node_logit_from_fmap(f_node_masked) + float(out_i["logit_edge"])
            target_prob_pert = float(_target_probabilities_from_logits([new_logit], [y_batch[i]])[0])
            drops.append(_normalised_target_probability_drop(target_prob_orig, target_prob_pert))

        out.append(_area_over_perturbation_curve(fracs, drops))

    return [float(v) for v in out]


def fixed_edge_target_evidence_deletion_irof_aoc(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    deletion_fracs: Optional[Sequence[float]] = None,
) -> List[float]:
    """IROF-style target-evidence AOC for fixed edge concept contribution maps."""
    x_batch = np.asarray(x_batch, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    if y_batch.size != x_batch.shape[0]:
        raise ValueError(f"Shape mismatch: x_batch has {x_batch.shape[0]} samples, y_batch has {y_batch.size}")

    fracs = _prepare_deletion_fracs(deletion_fracs)
    out: List[float] = []

    for i in range(x_batch.shape[0]):
        out_i = _single_fixed_concept_outputs(model, x_batch[i], edge_index=edge_index, device=device)
        f_edge = np.asarray(out_i["F_edge"], dtype=float)
        target_prob_orig = float(_target_probabilities_from_logits([out_i["logit"]], [y_batch[i]])[0])
        sign = _target_label_sign(y_batch[i])
        rank_source = _rank_contributions_for_deletion(sign * f_edge, positive_only=True, abs_val=False)

        drops: List[float] = []
        for frac in fracs:
            idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=float(frac), k_total=None)
            if n_feat == 0:
                drops.append(float("nan"))
                continue
            if idx.size == 0:
                drops.append(0.0)
                continue

            f_edge_masked = f_edge.reshape(-1).copy()
            f_edge_masked[idx] = 0.0
            f_edge_masked = f_edge_masked.reshape(f_edge.shape)

            new_logit = float(out_i["logit_node"]) + _fixed_edge_logit_from_fmap(f_edge_masked)
            target_prob_pert = float(_target_probabilities_from_logits([new_logit], [y_batch[i]])[0])
            drops.append(_normalised_target_probability_drop(target_prob_orig, target_prob_pert))

        out.append(_area_over_perturbation_curve(fracs, drops))

    return [float(v) for v in out]


def fixedconcepttheta_node_target_evidence_deletion_irof_aoc(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    deletion_fracs: Optional[Sequence[float]] = None,
    baseline_value: float = 0.0,
) -> List[float]:
    """IROF-style target-evidence AOC for concept-space node ablation in concept-theta SENN."""
    x_batch = np.asarray(x_batch, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    if y_batch.size != x_batch.shape[0]:
        raise ValueError(f"Shape mismatch: x_batch has {x_batch.shape[0]} samples, y_batch has {y_batch.size}")

    fracs = _prepare_deletion_fracs(deletion_fracs)
    out: List[float] = []

    for i in range(x_batch.shape[0]):
        state_i = _single_fixedconcepttheta_state(model, x_batch[i], edge_index=edge_index, device=device)
        f_node = np.asarray(state_i["F_node"], dtype=float)
        target_prob_orig = float(_target_probabilities_from_logits([state_i["logit"]], [y_batch[i]])[0])
        sign = _target_label_sign(y_batch[i])
        rank_source = _rank_contributions_for_deletion(sign * f_node, positive_only=True, abs_val=False)

        drops: List[float] = []
        for frac in fracs:
            idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=float(frac), k_total=None)
            if n_feat == 0:
                drops.append(float("nan"))
                continue
            if idx.size == 0:
                drops.append(0.0)
                continue

            h_node_masked = state_i["h_node"].reshape(-1).clone()
            h_node_masked[idx] = float(baseline_value)
            h_node_masked = h_node_masked.reshape(state_i["h_node"].shape)

            recomputed = _recompute_fixedconcepttheta_from_concepts(
                model=model,
                h_node=h_node_masked,
                h_edge=state_i["h_edge"],
                edge_index=edge_index,
                batch=state_i["batch"],
            )
            target_prob_pert = float(_target_probabilities_from_logits([recomputed["logit"]], [y_batch[i]])[0])
            drops.append(_normalised_target_probability_drop(target_prob_orig, target_prob_pert))

        out.append(_area_over_perturbation_curve(fracs, drops))

    return [float(v) for v in out]


def fixedconcepttheta_edge_target_evidence_deletion_irof_aoc(
    *,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    deletion_fracs: Optional[Sequence[float]] = None,
    baseline_value: float = 0.0,
) -> List[float]:
    """IROF-style target-evidence AOC for concept-space edge ablation in concept-theta SENN."""
    x_batch = np.asarray(x_batch, dtype=float)
    y_batch = np.asarray(y_batch, dtype=int).reshape(-1)
    if y_batch.size != x_batch.shape[0]:
        raise ValueError(f"Shape mismatch: x_batch has {x_batch.shape[0]} samples, y_batch has {y_batch.size}")

    fracs = _prepare_deletion_fracs(deletion_fracs)
    out: List[float] = []

    for i in range(x_batch.shape[0]):
        state_i = _single_fixedconcepttheta_state(model, x_batch[i], edge_index=edge_index, device=device)
        f_edge = np.asarray(state_i["F_edge"], dtype=float)
        target_prob_orig = float(_target_probabilities_from_logits([state_i["logit"]], [y_batch[i]])[0])
        sign = _target_label_sign(y_batch[i])
        rank_source = _rank_contributions_for_deletion(sign * f_edge, positive_only=True, abs_val=False)

        drops: List[float] = []
        for frac in fracs:
            idx, n_feat = _select_topk_flat_indices(rank_source, k_frac=float(frac), k_total=None)
            if n_feat == 0:
                drops.append(float("nan"))
                continue
            if idx.size == 0:
                drops.append(0.0)
                continue

            h_edge_masked = state_i["h_edge"].reshape(-1).clone()
            h_edge_masked[idx] = float(baseline_value)
            h_edge_masked = h_edge_masked.reshape(state_i["h_edge"].shape)

            recomputed = _recompute_fixedconcepttheta_from_concepts(
                model=model,
                h_node=state_i["h_node"],
                h_edge=h_edge_masked,
                edge_index=edge_index,
                batch=state_i["batch"],
            )
            target_prob_pert = float(_target_probabilities_from_logits([recomputed["logit"]], [y_batch[i]])[0])
            drops.append(_normalised_target_probability_drop(target_prob_orig, target_prob_pert))

        out.append(_area_over_perturbation_curve(fracs, drops))

    return [float(v) for v in out]


def plot_fixed_concept_explanations(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    edge_index: torch.Tensor,
    node_expl: np.ndarray,
    edge_expl: np.ndarray,
    pred_batch: Optional[np.ndarray] = None,
    concept_names: Optional[Sequence[str]] = None,
    show_plots: bool = True,
):
    """
    Static viewer for SENN fixed concepts.

    Creates two rows similar in spirit to the base setup:
      row 1: node-concept heatmap   [channels x fixed concepts]
      row 2: edge-concept graph     [edge importance on montage graph]
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.colors import Normalize

    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch).astype(int)
    node_expl = np.asarray(node_expl, dtype=float)
    edge_expl = np.asarray(edge_expl, dtype=float)
    pred_batch = None if pred_batch is None else np.asarray(pred_batch).astype(int)

    if node_expl.ndim != 3:
        raise ValueError(f"Expected node_expl shape (B,N,K_node), got {node_expl.shape}")
    if edge_expl.ndim < 2:
        raise ValueError(f"Expected edge_expl shape (B,E,...) or (B,E), got {edge_expl.shape}")

    if edge_expl.ndim == 3 and edge_expl.shape[-1] == 1:
        edge_expl = edge_expl[..., 0]

    B, N, K_node = node_expl.shape
    channel_names = [
        "Fp1-T3", "T3-O1", "Fp1-C3", "C3-O1", "Fp2-C4", "C4-O2",
        "Fp2-T4", "T4-O2", "T3-C3", "C3-Cz", "Cz-C4", "C4-T4"
    ]
    if concept_names is None:
        concept_names = ["δ rel. power", "θ rel. power", "α rel. power", "β rel. power", "Rhythmicity", "SNLEO"]
    concept_names = list(concept_names)
    if len(concept_names) != K_node:
        concept_names = [f"Concept {i+1}" for i in range(K_node)]

    node_pos = {
        0: (-2, 2), 1: (-2, 0), 2: (-1, 2), 3: (-1, 0),
        4: (1, 2), 5: (1, 0), 6: (2, 2), 7: (2, 0),
        8: (-1.5, 1), 9: (-0.5, 1), 10: (0.5, 1), 11: (1.5, 1),
    }

    ei = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    directed_edges = [(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])]
    undirected_map = {}
    for idx, (u, v) in enumerate(directed_edges):
        key = (u, v) if u <= v else (v, u)
        undirected_map.setdefault(key, []).append(idx)
    undirected_edges = list(undirected_map.keys())
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(undirected_edges)

    class _Viewer:
        def __init__(self):
            self.idx = 0
            self.fig, self.axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1.2, 0.8]})
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.draw()

        def on_key(self, event):
            if event.key == "right":
                self.idx = min(self.idx + 1, B - 1)
                self.draw()
            elif event.key == "left":
                self.idx = max(self.idx - 1, 0)
                self.draw()

        def draw(self):
            for ax in self.axes.ravel():
                ax.clear()

            ax_heat = self.axes[0, 0]
            ax_info = self.axes[0, 1]
            ax_graph = self.axes[1, 0]
            ax_summary = self.axes[1, 1]

            node_map = node_expl[self.idx]
            edge_map_dir = edge_expl[self.idx].reshape(-1)
            edge_map = np.zeros(len(undirected_edges), dtype=float)
            for k, e in enumerate(undirected_edges):
                edge_map[k] = float(np.mean(edge_map_dir[undirected_map[e]]))

            node_vis = np.maximum(node_map, 0.0)
            im = ax_heat.imshow(node_vis, aspect="auto", cmap="coolwarm")
            ax_heat.set_yticks(np.arange(N))
            ax_heat.set_yticklabels(channel_names[:N])
            ax_heat.set_xticks(np.arange(K_node))
            ax_heat.set_xticklabels(concept_names, rotation=45, ha="right")
            ax_heat.set_title("Node fixed-concept explanation")
            self.fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

            node_strength = np.maximum(node_vis.sum(axis=1), 0.0)
            node_norm = Normalize(vmin=0.0, vmax=max(float(node_strength.max()), 1e-9))
            node_color = node_norm(node_strength)
            edge_norm = Normalize(vmin=0.0, vmax=max(float(np.maximum(edge_map, 0.0).max()), 1e-9))

            for (u, v), w in zip(undirected_edges, np.maximum(edge_map, 0.0)):
                nx.draw_networkx_edges(
                    G,
                    node_pos,
                    ax=ax_graph,
                    edgelist=[(u, v)],
                    width=0.5 + 4.0 * edge_norm(w),
                    alpha=0.15 + 0.85 * edge_norm(w),
                    edge_color="black",
                )
            nx.draw_networkx_nodes(
                G,
                node_pos,
                ax=ax_graph,
                node_size=1000 + 2200 * node_color,
                node_color=node_color,
                cmap=plt.cm.coolwarm,
                vmin=0,
                vmax=1,
            )
            nx.draw_networkx_labels(
                G,
                node_pos,
                ax=ax_graph,
                labels={i: channel_names[i] for i in range(N)},
                font_size=9,
                font_weight="bold",
                font_color="white",
            )
            ax_graph.set_title("Edge fixed-concept explanation")
            ax_graph.axis("off")

            ax_info.axis("off")
            top_nodes = np.argsort(-node_strength)[:5]
            top_concepts = np.argsort(-node_vis.mean(axis=0))[:min(5, K_node)]
            info_lines = [
                f"Window: {self.idx + 1}/{B}",
                f"Label: {int(y_batch[self.idx])}",
            ]
            if pred_batch is not None:
                info_lines.append(f"Prediction: {int(pred_batch[self.idx])}")
            info_lines.append("")
            info_lines.append("Top nodes:")
            for rank, n in enumerate(top_nodes, 1):
                info_lines.append(f"  {rank}. {channel_names[n]}: {node_strength[n]:.3f}")
            info_lines.append("")
            info_lines.append("Top concepts:")
            for rank, k in enumerate(top_concepts, 1):
                info_lines.append(f"  {rank}. {concept_names[k]}: {node_vis[:, k].mean():.3f}")
            ax_info.text(0.02, 0.98, "\n".join(info_lines), va="top", ha="left", fontsize=10)

            ax_summary.axis("off")
            top_edges = np.argsort(-np.maximum(edge_map, 0.0))[:5]
            summary_lines = ["Top edges:"]
            for rank, eidx in enumerate(top_edges, 1):
                u, v = undirected_edges[eidx]
                summary_lines.append(f"  {rank}. {channel_names[u]} ↔ {channel_names[v]}: {edge_map[eidx]:.3f}")
            summary_lines.append("")
            summary_lines.append("Controls:")
            summary_lines.append("  ← / → : previous/next window")
            ax_summary.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=10)

            self.fig.suptitle("SENN fixed concepts: node and edge explanations", y=0.98)
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

    viewer = _Viewer()
    if show_plots:
        plt.show()
    return viewer.fig, viewer.axes

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

    custom_lip_nr_ig = int(RIS_NR_IG)
    custom_lip_std_ig = float(RIS_STD_IG)
    custom_lip_alpha_ig = float(RIS_ALPHA_IG)
    custom_lip_nr_edge = int(RIS_NR_EDGE)
    custom_lip_std_edge = float(RIS_STD_EDGE)
    custom_lip_alpha_edge = float(RIS_ALPHA_EDGE)
    custom_mprt_rounds = int(CUSTOM_MPRT_ROUNDS)

    faith_nr_runs = int(FAITH_CORR_NR_RUNS)
    faith_subset_frac = float(FAITH_CORR_SUBSET_FRAC)
    faith_block_len = None if FAITH_BLOCK_LEN is None else int(FAITH_BLOCK_LEN)
    faith_min_blocks = int(FAITH_MIN_BLOCKS)

    ig_del_block_len = faith_block_len if IG_DEL_BLOCK_LEN is None else int(IG_DEL_BLOCK_LEN)
    ig_del_topk_per_channel = None if IG_DEL_TOPK_PER_CHANNEL is None else int(IG_DEL_TOPK_PER_CHANNEL)
    ig_del_frac = faith_subset_frac if IG_DEL_FRAC is None else float(IG_DEL_FRAC)
    output_del_fracs = _prepare_deletion_fracs()

    edge_epochs = int(GNNEXPL_EPOCHS_QUANT)
    rma_s_threshold = float(RMA_S_THRESHOLD)
    ris_eps_min = float(RIS_EPS_MIN)
    ris_input_rel_eps = float(RIS_INPUT_REL_EPS)
    ris_expl_rel_eps = float(RIS_EXPL_REL_EPS)
    ris_norm_p = np.inf if str(RIS_NORM_P).lower() in {"inf", "+inf", "np.inf"} else float(RIS_NORM_P)

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
            "continuity_metric": "RelativeInputStability",
            "ris_norm_p": "inf" if np.isinf(ris_norm_p) else float(ris_norm_p),
            "ris_eps_min": float(ris_eps_min),
            "ris_input_rel_eps": float(ris_input_rel_eps),
            "ris_expl_rel_eps": float(ris_expl_rel_eps),
            "ris_require_same_prediction": True,
            "ris_nr_samples_ig": int(custom_lip_nr_ig),
            "ris_std_ig": float(custom_lip_std_ig),
            "ris_alpha_ig": None if custom_lip_alpha_ig is None else float(custom_lip_alpha_ig),
            "ris_nr_samples_edge": int(custom_lip_nr_edge),
            "ris_std_edge": float(custom_lip_std_edge),
            "ris_alpha_edge": None if custom_lip_alpha_edge is None else float(custom_lip_alpha_edge),
            "lipschitz_legacy_available": True,
            "gnnexpl_epochs_quant": int(edge_epochs),
            "faith_corr_nr_runs": int(faith_nr_runs),
            "faith_corr_subset_frac": float(faith_subset_frac),
            "faith_block_len": None if faith_block_len is None else int(faith_block_len),
            "faith_min_blocks": int(faith_min_blocks),
            "topk_deletion_positive_only": True,
            "output_completeness_metric": OUTPUT_COMPLETENESS_METRIC_NAME,
            "output_completeness_target_score": "seizure: sigmoid(logit); non_seizure: 1 - sigmoid(logit)",
            "output_completeness_normalisation": "IROF-style relative drop: 1 - P_target(x_perturbed) / P_target(x)",
            "output_completeness_curve_fracs": [float(v) for v in output_del_fracs],
            "output_completeness_higher_is_better": True,
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

        results["metrics"]["IG_RAW"]["Continuity_RelativeInputStability"] = feature_relative_input_stability(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_ig,
            perturb_std=custom_lip_std_ig,
            perturb_alpha=custom_lip_alpha_ig,
            thr=thr,
            p=ris_norm_p,
            eps_min=ris_eps_min,
            input_rel_eps=ris_input_rel_eps,
            expl_rel_eps=ris_expl_rel_eps,
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
            positive_only=True,
            abs_val=False,
            disallow_overlap=True,
            only_positive_blocks=True,
            use_logit_drop=True,
        )
        results["metrics"]["IG_RAW"][OUTPUT_COMPLETENESS_METRIC_NAME] = feature_target_evidence_deletion_irof_aoc(
            predict_logit_fn=pred_full,
            x_batch=x_raw,
            y_batch=y_true,
            a_batch=ig_a,
            deletion_fracs=output_del_fracs,
            baseline_value=0.0,
            block_len=ig_del_block_len,
            disallow_overlap=True,
        )

        results["metrics"]["GNN_EDGE"] = {}
        results["metrics"]["GNN_EDGE"]["Continuity_RelativeInputStability"] = gnn_edge_relative_input_stability(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_edge,
            perturb_std=custom_lip_std_edge,
            perturb_alpha=custom_lip_alpha_edge,
            epochs=edge_epochs,
            thr=thr,
            p=ris_norm_p,
            eps_min=ris_eps_min,
            input_rel_eps=ris_input_rel_eps,
            expl_rel_eps=ris_expl_rel_eps,
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
            positive_only=True,
            use_logit_drop=True,
        )
        results["metrics"]["GNN_EDGE"][OUTPUT_COMPLETENESS_METRIC_NAME] = edge_target_evidence_deletion_irof_aoc(
            gnn=model.gnn,
            x_batch=x_nf,
            y_batch=y_true,
            a_edge_batch=edge_masks,
            edge_index=edge_index,
            device=device,
            deletion_fracs=output_del_fracs,
            signed_attributions=False,
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
        results["metrics"]["FOCUS_MAP"]["Continuity_RelativeInputStability"] = focusmap_relative_input_stability(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_ig,
            perturb_std=custom_lip_std_ig,
            perturb_alpha=custom_lip_alpha_ig,
            thr=thr,
            p=ris_norm_p,
            eps_min=ris_eps_min,
            input_rel_eps=ris_input_rel_eps,
            expl_rel_eps=ris_expl_rel_eps,
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
            positive_only=True,
            abs_val=False,
            disallow_overlap=True,
            only_positive_blocks=True,
            use_logit_drop=True,
        )
        results["metrics"]["FOCUS_MAP"][OUTPUT_COMPLETENESS_METRIC_NAME] = feature_target_evidence_deletion_irof_aoc(
            predict_logit_fn=pred_full,
            x_batch=x_raw,
            y_batch=y_true,
            a_batch=focus_a,
            deletion_fracs=output_del_fracs,
            baseline_value=0.0,
            block_len=ig_del_block_len,
            disallow_overlap=True,
        )

    elif model_kind == "senn_fixed":
        results["metrics"]["FIXED_NODE"] = {}
        results["metrics"]["FIXED_EDGE"] = {}

        results["metrics"]["FIXED_NODE"]["Continuity_RelativeInputStability"] = focusmap_relative_input_stability(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_ig,
            perturb_std=custom_lip_std_ig,
            perturb_alpha=custom_lip_alpha_ig,
            output_key="explanation",
            thr=thr,
            p=ris_norm_p,
            eps_min=ris_eps_min,
            input_rel_eps=ris_input_rel_eps,
            expl_rel_eps=ris_expl_rel_eps,
        )
        results["metrics"]["FIXED_NODE"]["Correctness_ParamRandomisation_corr_spearman"] = focusmap_parameter_randomisation_sanity(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
            output_key="explanation",
        )
        results["metrics"]["FIXED_NODE"]["TopKDeletion_drop"] = fixed_node_topk_deletion_drop(
            model=model,
            x_batch=x_raw,
            edge_index=edge_index,
            device=device,
            k_frac=ig_del_frac,
            positive_only=True,
            abs_val=False,
            use_logit_drop=True,
        )
        results["metrics"]["FIXED_NODE"][OUTPUT_COMPLETENESS_METRIC_NAME] = fixed_node_target_evidence_deletion_irof_aoc(
            model=model,
            x_batch=x_raw,
            y_batch=y_true,
            edge_index=edge_index,
            device=device,
            deletion_fracs=output_del_fracs,
        )

        results["metrics"]["FIXED_EDGE"]["Continuity_RelativeInputStability"] = focusmap_relative_input_stability(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_edge,
            perturb_std=custom_lip_std_edge,
            perturb_alpha=custom_lip_alpha_edge,
            output_key="explanation_edge",
            thr=thr,
            p=ris_norm_p,
            eps_min=ris_eps_min,
            input_rel_eps=ris_input_rel_eps,
            expl_rel_eps=ris_expl_rel_eps,
        )
        results["metrics"]["FIXED_EDGE"]["Correctness_ParamRandomisation_corr_spearman"] = focusmap_parameter_randomisation_sanity(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
            output_key="explanation_edge",
        )
        results["metrics"]["FIXED_EDGE"]["TopKDeletion_drop"] = fixed_edge_topk_deletion_drop(
            model=model,
            x_batch=x_raw,
            edge_index=edge_index,
            device=device,
            k_frac=faith_subset_frac,
            positive_only=True,
            abs_val=False,
            use_logit_drop=True,
        )
        results["metrics"]["FIXED_EDGE"][OUTPUT_COMPLETENESS_METRIC_NAME] = fixed_edge_target_evidence_deletion_irof_aoc(
            model=model,
            x_batch=x_raw,
            y_batch=y_true,
            edge_index=edge_index,
            device=device,
            deletion_fracs=output_del_fracs,
        )

    elif model_kind == "senn_fixedconcepttheta" or model_kind =="logisticconcepts": #both these models have similar explanation structure
        results["metrics"]["FIXED_NODE"] = {}
        results["metrics"]["FIXED_EDGE"] = {}

        results["metrics"]["FIXED_NODE"]["Continuity_RelativeInputStability"] = focusmap_relative_input_stability(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_ig,
            perturb_std=custom_lip_std_ig,
            perturb_alpha=custom_lip_alpha_ig,
            output_key="explanation",
            thr=thr,
            p=ris_norm_p,
            eps_min=ris_eps_min,
            input_rel_eps=ris_input_rel_eps,
            expl_rel_eps=ris_expl_rel_eps,
        )
        results["metrics"]["FIXED_NODE"]["Correctness_ParamRandomisation_corr_spearman"] = focusmap_parameter_randomisation_sanity(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
            output_key="explanation",
        )
        results["metrics"]["FIXED_NODE"]["TopKDeletion_drop"] = fixedconcepttheta_node_topk_deletion_drop(
            model=model,
            x_batch=x_raw,
            edge_index=edge_index,
            device=device,
            k_frac=ig_del_frac,
            positive_only=True,
            abs_val=False,
            use_logit_drop=True,
            baseline_value=0.0,
        )
        results["metrics"]["FIXED_NODE"][OUTPUT_COMPLETENESS_METRIC_NAME] = fixedconcepttheta_node_target_evidence_deletion_irof_aoc(
            model=model,
            x_batch=x_raw,
            y_batch=y_true,
            edge_index=edge_index,
            device=device,
            deletion_fracs=output_del_fracs,
            baseline_value=0.0,
        )

        results["metrics"]["FIXED_EDGE"]["Continuity_RelativeInputStability"] = focusmap_relative_input_stability(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            nr_samples=custom_lip_nr_edge,
            perturb_std=custom_lip_std_edge,
            perturb_alpha=custom_lip_alpha_edge,
            output_key="explanation_edge",
            thr=thr,
            p=ris_norm_p,
            eps_min=ris_eps_min,
            input_rel_eps=ris_input_rel_eps,
            expl_rel_eps=ris_expl_rel_eps,
        )
        results["metrics"]["FIXED_EDGE"]["Correctness_ParamRandomisation_corr_spearman"] = focusmap_parameter_randomisation_sanity(
            x_raw=x_raw,
            y_batch=y_true,
            model=model,
            edge_index=edge_index,
            device=device,
            corr_kind="spearman",
            n_random_models=custom_mprt_rounds,
            output_key="explanation_edge",
        )
        results["metrics"]["FIXED_EDGE"]["TopKDeletion_drop"] = fixedconcepttheta_edge_topk_deletion_drop(
            model=model,
            x_batch=x_raw,
            edge_index=edge_index,
            device=device,
            k_frac=faith_subset_frac,
            positive_only=True,
            abs_val=False,
            use_logit_drop=True,
            baseline_value=0.0,
        )
        results["metrics"]["FIXED_EDGE"][OUTPUT_COMPLETENESS_METRIC_NAME] = fixedconcepttheta_edge_target_evidence_deletion_irof_aoc(
            model=model,
            x_batch=x_raw,
            y_batch=y_true,
            edge_index=edge_index,
            device=device,
            deletion_fracs=output_del_fracs,
            baseline_value=0.0,
        )
    else:
        raise ValueError("model_kind must be 'base', 'senn', 'senn_fixed', or 'senn_fixedconcepttheta'")

    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "custom_metrics_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, indent=2)

    return results




def _xai_str_to_bool_auto(value: Any) -> Optional[bool]:
    """Parse simple bool flags while allowing 'auto'/None."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"", "auto", "none", "null"}:
        return None
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean/auto value: {value!r}")


def _xai_model_kind_from_checkpoint(ckpt: Dict[str, Any], fallback: str = "base") -> Tuple[str, str]:
    """
    Return (metric_model_kind, build_model_kind).

    metric_model_kind is used by run_custom_metrics_co12 and plotting branches.
    build_model_kind is passed to _build_model.

    Trivial fixed concepts reuse the fixed-concept metric branch but need the
    SENN_trivialfixedconcepts class for construction.
    """
    try:
        model_type = normalize_model_kind(ckpt.get("model_type", fallback))
    except ValueError:
        model_type = normalize_model_kind(fallback)

    if model_type == "base":
        return "base", "base"
    if model_type == "senn_rawx":
        return "senn", "senn_rawx"
    if model_type == "senn_fixed":
        return "senn_fixed", "senn_fixed"
    if model_type == "senn_trivialfixed":
        return "senn_fixed", "senn_trivialfixed"
    if model_type == "senn_fixedconcepttheta":
        return "senn_fixedconcepttheta", "senn_fixedconcepttheta"
    if model_type == "logisticconcepts":
        return "logisticconcepts", "logisticconcepts"

    # Keep old behaviour for checkpoints without a known model_type.
    fallback = normalize_model_kind(fallback)
    return fallback, fallback


def _parse_xai_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="", help="Saved_models_* directory for one model run")
    parser.add_argument("--data_folder", default="", help="Path to Datasets/CV_Folds")
    parser.add_argument("--results_dir", default="", help="Output directory; wrapper passes Results_*/Explainability_metrics")
    parser.add_argument("--history_dir", default="", help="Optional History_* directory; currently kept for symmetry with Validation.py")
    parser.add_argument("--fold", default="", help="CV fold to explain, e.g. 5")
    parser.add_argument("--checkpoint_name", default="best_auprc.pt")
    parser.add_argument("--model_kind", default="auto", help="auto, base, senn, senn_fixed, senn_fixedconcepttheta")
    parser.add_argument("--is_trivial", default="auto", help="auto/true/false. true builds senn_trivialfixed but evaluates as senn_fixed")
    return parser.parse_args()


# -------------------------
# Main pipeline
# -------------------------


if __name__ == "__main__":
    args = _parse_xai_args()

    DATA_FOLDER = os.path.abspath(os.path.expanduser(args.data_folder)) if args.data_folder else DEFAULT_DATA_FOLDER
    HISTORY_DIR = os.path.abspath(os.path.expanduser(args.history_dir)) if args.history_dir else DEFAULT_HISTORY_DIR
    MODEL_DIR = os.path.abspath(os.path.expanduser(args.model_dir)) if args.model_dir else DEFAULT_MODEL_DIR
    RESULTS_DIR = os.path.abspath(os.path.expanduser(args.results_dir)) if args.results_dir else DEFAULT_RESULTS_DIR
    fold = str(args.fold) if args.fold else DEFAULT_FOLD
    CHECKPOINT_NAME = str(args.checkpoint_name)
    REQUESTED_MODEL_KIND = str(args.model_kind or "auto").strip().lower()
    IS_TRIVIAL_OVERRIDE = _xai_str_to_bool_auto(args.is_trivial)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    fold_dir = os.path.join(DATA_FOLDER, f"fold_{fold}")

    MODEL_SUBDIR = f"GAT_CV_10_{fold}"
    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR)

    ckpt_path = os.path.join(model_path, CHECKPOINT_NAME)
    ckpt = torch.load(ckpt_path, weights_only=False)

    if REQUESTED_MODEL_KIND in {"", "auto"}:
        MODEL_KIND, BUILD_MODEL_KIND = _xai_model_kind_from_checkpoint(ckpt, fallback=DEFAULT_MODEL_KIND)
    else:
        normalized_requested = normalize_model_kind(REQUESTED_MODEL_KIND)
        BUILD_MODEL_KIND = normalized_requested
        MODEL_KIND = "senn" if normalized_requested == "senn_rawx" else normalized_requested

    if IS_TRIVIAL_OVERRIDE is True:
        BUILD_MODEL_KIND = "senn_trivialfixed"
        MODEL_KIND = "senn_fixed"
    elif IS_TRIVIAL_OVERRIDE is False and BUILD_MODEL_KIND == "senn_trivialfixed":
        BUILD_MODEL_KIND = "senn_fixed"
        MODEL_KIND = "senn_fixed"

    print(f"XAI model_dir      : {MODEL_DIR}")
    print(f"XAI data_folder    : {DATA_FOLDER}")
    print(f"XAI results_dir    : {RESULTS_DIR}")
    print(f"XAI fold           : {fold}")
    print(f"XAI checkpoint     : {ckpt_path}")
    print(f"XAI metric kind    : {MODEL_KIND}")
    print(f"XAI build kind     : {BUILD_MODEL_KIND}")

    norm = ckpt["normalization"]
    threshold = float(np.round(ckpt["metrics"]["threshold"], 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(
        BUILD_MODEL_KIND,
        ckpt,
        return_explanations=MODEL_KIND in {"senn", "senn_fixed", "senn_fixedconcepttheta", "logisticconcepts"},
    ).to(device)
    if BUILD_MODEL_KIND == "senn_trivialfixed":
        print("trivial model is being build")
    model.eval()

    print("... loading and batching data ... ")
    x_test, y_test = load_fold_arrays(DATA_FOLDER, fold, split="test", mmap_mode="r")
    mask_path = os.path.join(fold_dir, "testmasks.npy")
    y_mask_test = np.load(mask_path, mmap_mode="r") if os.path.exists(mask_path) else None

    mean = norm["mean"]
    std = norm["std"]
    # x_test = (x_test - mean) / std

    batch_size = 50
    # batch_size = 250
    batch_size = 213
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

        testset_no_seiz = prepare_graphs_labels(x_batched_no_seiz, y_batched_no_seiz, Model.adj, masks=m_batched_no_seiz)
        testset_yes_seiz = prepare_graphs_labels(x_batched_yes_seiz, y_batched_yes_seiz, Model.adj, masks=m_batched_yes_seiz)
    else:
        start_idx = idx_yes_seiz[5]
        x_batched = x_test[start_idx:start_idx + batch_size]
        y_batched = y_test[start_idx:start_idx + batch_size]
        m_batched = y_mask_test[start_idx:start_idx + batch_size] if y_mask_test is not None else None
        testset = prepare_graphs_labels(x_batched, y_batched, Model.adj, masks=m_batched)

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

        try:
            graphs_all: List[Data] = list(testset_no_seiz) + list(testset_yes_seiz)
            if len(graphs_all) == 0:
                raise ValueError("No validation graphs available for global sample-RMA evaluation.")

            edge_index_all = graphs_all[0].edge_index.to(device)
            x_raw_all = np.stack([g.x.detach().cpu().numpy() for g in graphs_all], axis=0)
            y_true_all = np.array([int(g.y.detach().cpu().view(-1)[0].item()) for g in graphs_all], dtype=int)

            global_rma["meta"] = {
                "n_samples": int(len(graphs_all)),
                "uses_sample_labels": True,
                "sample_label_source": "y_true",
                "eeg_fs": int(EEG_FS),
                "rma_s_threshold": float(RMA_S_THRESHOLD),
                "balance_mode": str(RMA_BALANCE_MODE),
                "edge_rma_interaction": str(EDGE_RMA_INTERACTION),
                "model_kind": MODEL_KIND,
            }

            explainers_for_rma: Dict[str, np.ndarray] = {}

            if MODEL_KIND == "base":
                with torch.no_grad():
                    nf_all = []
                    for g in graphs_all:
                        gx = g.x.to(device)
                        nf_all.append(model.cnn(gx).detach().cpu().numpy())
                x_nf_all = np.stack(nf_all, axis=0)

                class _ModelShim(torch.nn.Module):
                    def __init__(self, gnn_module):
                        super().__init__()
                        self.gnn = gnn_module
                        self.cnn = torch.nn.Identity()

                model_nf_all = _ModelShim(model.gnn).to(device)
                model_nf_all.eval()

                explainers_for_rma["IG_RAW"] = ig_explainer_raw(
                    model=model,
                    inputs=x_raw_all,
                    targets=y_true_all,
                    abs=False,
                    normalise=False,
                    edge_index=edge_index_all,
                    device=device,
                    thr=threshold,
                )
                explainers_for_rma["GNN_EDGE"] = gnn_edge_explainer(
                    model=model_nf_all,
                    inputs=x_nf_all,
                    targets=y_true_all,
                    abs=False,
                    normalise=False,
                    edge_index=edge_index_all,
                    device=device,
                    epochs=GNNEXPL_EPOCHS_QUANT,
                    thr=threshold,
                )

            elif MODEL_KIND == "senn":
                explainers_for_rma["FOCUS_MAP"] = focusmap_explainer_raw(
                    model=model,
                    inputs=x_raw_all,
                    targets=y_true_all,
                    abs=False,
                    normalise=False,
                    edge_index=edge_index_all,
                    device=device,
                    output_key="explanation",
                )

            elif MODEL_KIND in {"senn_fixed", "senn_fixedconcepttheta", "logisticconcepts"}:
                explainers_for_rma["FIXED_NODE"] = focusmap_explainer_raw(
                    model=model,
                    inputs=x_raw_all,
                    targets=y_true_all,
                    abs=False,
                    normalise=False,
                    edge_index=edge_index_all,
                    device=device,
                    output_key="explanation",
                )
                explainers_for_rma["FIXED_EDGE"] = focusmap_explainer_raw(
                    model=model,
                    inputs=x_raw_all,
                    targets=y_true_all,
                    abs=False,
                    normalise=False,
                    edge_index=edge_index_all,
                    device=device,
                    output_key="explanation_edge",
                )
            else:
                raise ValueError(f"Unsupported MODEL_KIND for sample-RMA: {MODEL_KIND!r}")

            for expl_key, a_all in explainers_for_rma.items():
                global_rma[expl_key] = relevance_mass_accuracy_sample_global_details(
                    a_batch=a_all,
                    s_batch=y_true_all,
                    abs_val=False,
                    positive_only=True,
                    eps=1e-12,
                    fs=int(EEG_FS),
                    s_threshold=float(global_rma["meta"]["rma_s_threshold"]),
                    normalize_per_window=False,
                    balance_mode=str(global_rma["meta"]["balance_mode"]),
                )
                print(
                    f"[Global sample-RMA] {expl_key}: "
                    f"{global_rma[expl_key]['Coherence_RelevanceMassAccuracy_sample_global']:.4f}"
                )

        except Exception as e:
            print("Sample-level RMA coherence failed")
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
        elif MODEL_KIND == "senn_fixed" or MODEL_KIND == "senn_fixedconcepttheta" or MODEL_KIND == 'logisticconcepts':
            row_keys = [
                ("Fixed node concepts", ["FIXED_NODE", "NODE_CONCEPTS", "NODE"]),
                ("Fixed edge concepts", ["FIXED_EDGE", "EDGE_CONCEPTS", "EDGE"]),
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

        GLOBAL_RMA_METRIC_NAME = "Coherence_RelevanceMassAccuracy_sample_global"

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
                "Continuity_RelativeInputStability",
                "Continuity_LocalLipschitz",
                "Correctness_ParamRandomisation_corr_spearman",
                "FaithfulnessCorrelation",
                "TopKDeletion_drop",
                OUTPUT_COMPLETENESS_METRIC_NAME,
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
                            for k_try in [GLOBAL_RMA_METRIC_NAME, "RMA_global", "global", "value"]:
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
                # plt.show() # do not show for cluster compatability
                plt.close(fig)

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
                if MODEL_KIND == "senn_fixed" or MODEL_KIND == "senn_fixedconcepttheta":
                    spatial_edge_expl.append(_extract_model_output(out, "explanation_edge").detach().cpu().numpy())

        temporal_expl = np.asarray(temporal_expl)
        spatial_edge_expl = np.asarray(spatial_edge_expl) if len(spatial_edge_expl) > 0 else None
        spatial_node_expl = np.asarray(spatial_node_expl) if len(spatial_node_expl) > 0 else None

        if MODEL_KIND == "senn_fixed" or MODEL_KIND == "senn_fixedconcepttheta":
            plot_fixed_concept_explanations(
                x_batched,
                y_batched,
                edge_index=testset[0].edge_index,
                node_expl=temporal_expl,
                edge_expl=spatial_edge_expl,
                pred_batch=pred_batch,
            )
        else:
            MyUtils.plot_data_int_expl(
                x_batched,
                y_batched,
                edge_index=testset[0].edge_index,
                pred_batch=pred_batch,
                temp_expl=temporal_expl,
                spat_node_expl=spatial_node_expl,
                spat_edge_expl=spatial_edge_expl,
            )
