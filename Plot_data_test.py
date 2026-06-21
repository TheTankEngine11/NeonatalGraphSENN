from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

if "--no-show" in sys.argv and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import numpy as np
import torch
from tqdm import tqdm

import Models_senn as Model
from data_utils import load_fold_arrays, prepare_graphs_labels, thin_overlapping_windows
from io_utils import checkpoint_path, ensure_dir, load_checkpoint
from model_utils import build_model, extract_prob, normalize_model_kind
from MyUtils_senn_test import calculateGNNexpl, calculateIG


# ---------------------------------------------------------------------------
# Thesis plotting style
# ---------------------------------------------------------------------------
THESIS_TEXT_WIDTH_IN = 15.0 / 2.54
# Interactive figures should be sized for the screen, not for direct thesis export.
# Saved PNG/PDF files still use savefig.dpi below.
INTERACTIVE_TWO_PANEL_FIGSIZE = (10.4, 6.6)
INTERACTIVE_THREE_PANEL_FIGSIZE = (11.2, 6.9)
THESIS_TWO_PANEL_FIGSIZE = (THESIS_TEXT_WIDTH_IN, 4.6)
THESIS_THREE_PANEL_FIGSIZE = (THESIS_TEXT_WIDTH_IN, 5.2)

INTERACTIVE_TWO_PANEL_FIGSIZE = THESIS_TWO_PANEL_FIGSIZE
INTERACTIVE_THREE_PANEL_FIGSIZE = THESIS_THREE_PANEL_FIGSIZE

INTERACTIVE_FIGURE_DPI = 300

NON_SEIZURE_COLOR = "#56B4E9"  # Okabe-Ito sky blue
SEIZURE_COLOR = "#E69F00"      # Okabe-Ito orange
NON_SEIZURE_COLOR = "tab:blue"  # Okabe-Ito sky blue
SEIZURE_COLOR = "tab:orange"      # Okabe-Ito orange
NODE_COLOR = "#009E73"         # Okabe-Ito bluish green
EDGE_COLOR = "#CC79A7"         # Okabe-Ito reddish purple
UNKNOWN_COLOR = "#9e9e9e"
EEG_TRACE_COLOR = "#4d4d4d"

EVIDENCE_NEUTRAL_COLOR = "#d0d0d0"
EVIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "positive_seizure_evidence",
    [NON_SEIZURE_COLOR, SEIZURE_COLOR,'tab:red'],
)
# EVIDENCE_CMAP=cm.bwr
NODE_CMAP = EVIDENCE_CMAP
EDGE_CMAP = EVIDENCE_CMAP
CONCEPT_COMPONENT_COLORS = {
    "h": "#6f6f6f",
    "theta": NODE_COLOR,
    "F": SEIZURE_COLOR,
}

DEFAULT_FS = 32
DEFAULT_FOLD = "7"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_WINDOWS = 30
DEFAULT_PRE_CONTEXT = 15
DEFAULT_DATA_FOLDER = (
    r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis"
    r"\Project_InterpretableGNN\Datasets\CV_Folds/"
)

FIXED_CONCEPT_NAMES = [
    "RBP delta",
    "RBP theta",
    "RBP alpha",
    "RBP beta",
    "Rhythmicity",
    "SNLEO",
]

CHANNEL_NAMES = [
    "Fp1-T3",
    "T3-O1",
    "Fp1-C3",
    "C3-O1",
    "Fp2-C4",
    "C4-O2",
    "Fp2-T4",
    "T4-O2",
    "T3-C3",
    "C3-Cz",
    "Cz-C4",
    "C4-T4",
]

NODE_POS = {
    0: (-2.0, 2.0),
    1: (-2.0, 0.0),
    2: (-1.0, 2.0),
    3: (-1.0, 0.0),
    4: (1.0, 2.0),
    5: (1.0, 0.0),
    6: (2.0, 2.0),
    7: (2.0, 0.0),
    8: (-1.5, 1.0),
    9: (-0.5, 1.0),
    10: (0.5, 1.0),
    11: (1.5, 1.0),
}

GRAPH_X_PAD = 0.3
GRAPH_Y_PAD_BOTTOM = 0.5
GRAPH_Y_PAD_TOP = 0.5


def apply_thesis_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception as exc:
        print(f"[WARN] SciencePlots style unavailable ({exc}); using local thesis style.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Pagella", "Palatino Linotype", "Palatino", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Palatino Linotype",
            "mathtext.it": "Palatino Linotype:italic",
            "mathtext.bf": "Palatino Linotype:bold",
            "text.usetex": False,
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.titlesize": 10.0,
            "figure.dpi": INTERACTIVE_FIGURE_DPI,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


apply_thesis_plot_style()


# ---------------------------------------------------------------------------
# Final thesis model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelRun:
    key: str
    label: str
    model_kind: str
    model_dir: Path
    uses_posthoc: bool = False

    @property
    def is_fixed_concept(self) -> bool:
        return normalize_model_kind(self.model_kind) in {"senn_fixed", "senn_fixedconcepttheta"}

    @property
    def returns_fmap(self) -> bool:
        return normalize_model_kind(self.model_kind) != "base"


ARCHIVE_MAIN = Path("./ArchiveModelsMainResults")

FINAL_MODELS: dict[str, ModelRun] = {
    "stgat": ModelRun(
        key="stgat",
        label="ST-GAT + post hoc",
        model_kind="base",
        model_dir=ARCHIVE_MAIN / "Saved_models_491483_MTbase_LR2e-2_WD1e-3",
        uses_posthoc=True,
    ),
    "senn-ic": ModelRun(
        key="senn-ic",
        label="SENN-IC",
        model_kind="senn_rawx",
        model_dir=ARCHIVE_MAIN / "Saved_models_492092_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-4",
    ),
    "senn-fc-theta-x": ModelRun(
        key="senn-fc-theta-x",
        label=r"SENN-FC-$\theta(x)$",
        model_kind="senn_fixed",
        model_dir=ARCHIVE_MAIN / "Saved_models_486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0",
    ),
    "senn-fc-theta-h": ModelRun(
        key="senn-fc-theta-h",
        label=r"SENN-FC-$\theta(h)$",
        model_kind="senn_fixedconcepttheta",
        model_dir=ARCHIVE_MAIN
        / "Saved_models_486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0",
    ),
}

MODEL_ALIASES = {
    "base": "stgat",
    "gat": "stgat",
    "stgat": "stgat",
    "st-gat": "stgat",
    "posthoc": "stgat",
    "senn": "senn-ic",
    "sennic": "senn-ic",
    "senn-ic": "senn-ic",
    "senn_rawx": "senn-ic",
    "rawx": "senn-ic",
    "fcx": "senn-fc-theta-x",
    "theta-x": "senn-fc-theta-x",
    "theta(x)": "senn-fc-theta-x",
    "senn-fc-theta-x": "senn-fc-theta-x",
    "senn_fixed": "senn-fc-theta-x",
    "fch": "senn-fc-theta-h",
    "theta-h": "senn-fc-theta-h",
    "theta(h)": "senn-fc-theta-h",
    "senn-fc-theta-h": "senn-fc-theta-h",
    "senn_fixedconcepttheta": "senn-fc-theta-h",
}


@dataclass
class WindowSubset:
    x: np.ndarray
    y: np.ndarray
    original_indices: np.ndarray
    local_indices: np.ndarray
    masks: Optional[np.ndarray]


@dataclass
class ExplanationBundle:
    run: ModelRun
    fold: str
    split: str
    fs: int
    threshold: float
    x: np.ndarray
    y: np.ndarray
    original_indices: np.ndarray
    local_indices: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    edge_index: np.ndarray
    node_time: np.ndarray
    node_graph: np.ndarray
    edge_scores: Optional[np.ndarray] = None
    node_concepts: Optional[np.ndarray] = None
    edge_concepts: Optional[np.ndarray] = None
    node_h: Optional[np.ndarray] = None
    node_theta: Optional[np.ndarray] = None
    node_f: Optional[np.ndarray] = None
    edge_h: Optional[np.ndarray] = None
    edge_theta: Optional[np.ndarray] = None
    edge_f: Optional[np.ndarray] = None


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _lookup_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def resolve_model_run(model_name: str) -> ModelRun:
    raw = str(model_name).strip()
    if raw in FINAL_MODELS:
        return FINAL_MODELS[raw]
    if raw in MODEL_ALIASES:
        return FINAL_MODELS[MODEL_ALIASES[raw]]

    compact_aliases = {_lookup_key(k): v for k, v in MODEL_ALIASES.items()}
    compact_models = {_lookup_key(k): k for k in FINAL_MODELS}
    compact = _lookup_key(raw)
    if compact in compact_aliases:
        return FINAL_MODELS[compact_aliases[compact]]
    if compact in compact_models:
        return FINAL_MODELS[compact_models[compact]]

    valid = ", ".join(FINAL_MODELS)
    raise ValueError(f"Unknown model '{model_name}'. Valid model keys: {valid}")


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    arr = _as_numpy(value).astype(float, copy=False).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def _positive(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def _clean_signed(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _magnitude_norm(values: Any) -> tuple[Normalize, str]:
    arr = np.abs(_clean_signed(values)).reshape(-1)
    nonzero = arr[np.isfinite(arr) & (arr > 0)]
    if nonzero.size == 0:
        return Normalize(vmin=0.0, vmax=1.0), "linear"

    vmax = float(np.percentile(nonzero, 90))
    vmax = max(vmax, float(np.max(nonzero)), 1e-12)
    p10 = float(np.percentile(nonzero, 10))
    p10 = max(p10, 1e-12)
    dynamic_range = vmax / p10

    if dynamic_range >= 100.0:
        return SymLogNorm(linthresh=p10, vmin=p10, vmax=vmax, base=10), "symlog"
    return Normalize(vmin=0.0, vmax=vmax), "linear"


def _norm_scalar(norm: Normalize, value: float) -> float:
    try:
        scaled = float(norm(float(value)))
    except Exception:
        return 0.0
    if not np.isfinite(scaled):
        return 0.0
    return float(np.clip(scaled, 0.0, 1.0))


def _checkpoint_threshold(ckpt: dict[str, Any]) -> float:
    metrics = ckpt.get("metrics", {})
    for key in ("threshold", "best_threshold", "optimal_threshold"):
        if key in metrics:
            return float(metrics[key])
    return 0.5


def _single_graph_batch(graph, device: torch.device) -> torch.Tensor:
    return torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)


def _project_node_values_to_time(values: Any, n_nodes: int, time_len: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(values, dtype=float))

    if arr.ndim == 1:
        if arr.size == n_nodes:
            return np.repeat(arr[:, None], time_len, axis=1)
        if arr.size == n_nodes * time_len:
            return arr.reshape(n_nodes, time_len)

    if arr.ndim == 2:
        if arr.shape == (n_nodes, time_len):
            return arr
        if arr.shape == (time_len, n_nodes):
            return arr.T
        if arr.shape[0] == n_nodes:
            node_scores = arr.reshape(n_nodes, -1).sum(axis=1)
            return np.repeat(node_scores[:, None], time_len, axis=1)

    if arr.ndim > 2 and arr.shape[0] == n_nodes:
        node_scores = arr.reshape(n_nodes, -1).sum(axis=1)
        return np.repeat(node_scores[:, None], time_len, axis=1)

    raise ValueError(f"Could not project node explanation with shape {arr.shape} to ({n_nodes}, {time_len}).")


def _aggregate_node_values(values: Any, n_nodes: int, time_len: int) -> np.ndarray:
    node_time = _project_node_values_to_time(values, n_nodes=n_nodes, time_len=time_len)
    return node_time.mean(axis=1)


def _aggregate_edge_values(values: Any, n_edges: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(values, dtype=float))
    if arr.ndim == 0:
        return np.repeat(float(arr), n_edges)
    if arr.ndim == 1:
        if arr.size == n_edges:
            return arr
        if arr.size == 1:
            return np.repeat(float(arr[0]), n_edges)
    if arr.ndim >= 2 and arr.shape[0] == n_edges:
        return arr.reshape(n_edges, -1).sum(axis=1)
    if arr.ndim >= 2 and arr.shape[-1] == n_edges:
        return arr.reshape(-1, n_edges).sum(axis=0)
    raise ValueError(f"Could not aggregate edge explanation with shape {arr.shape} to {n_edges} edges.")


def _mean_concept_vector(values: Any, n_items: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(values, dtype=float))
    if arr.ndim == 0:
        return np.array([float(arr)])
    if arr.ndim == 1:
        if arr.size == n_items:
            return np.array([float(np.mean(arr))])
        return arr.astype(float)
    if arr.shape[0] == n_items:
        return arr.reshape(n_items, -1).mean(axis=0)
    return arr.reshape(-1)


def _undirected_edge_groups(edge_index: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_index = np.asarray(edge_index, dtype=int)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    groups: dict[tuple[int, int], list[int]] = {}
    for edge_id, (src, dst) in enumerate(edge_index.T):
        if int(src) == int(dst):
            continue
        key = tuple(sorted((int(src), int(dst))))
        groups.setdefault(key, []).append(edge_id)
    return groups


def _concept_x_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1e-3, 1.0
    abs_nonzero = np.abs(finite[np.abs(finite) > 0])
    if abs_nonzero.size == 0:
        return 1e-3, 1.0
    linthresh = max(float(np.percentile(abs_nonzero, 10)), 1e-8)
    xmax = max(float(np.percentile(abs_nonzero, 99)), float(np.max(abs_nonzero)), linthresh * 10)
    return linthresh, xmax


def _sparse_symlog_ticks(left: float, right: float, linthresh: float, max_ticks: int = 6) -> np.ndarray:
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return np.array([], dtype=float)

    max_abs = max(abs(float(left)), abs(float(right)), float(linthresh), 1e-12)
    min_exp = int(np.floor(np.log10(max(float(linthresh), 1e-12))))
    max_exp = int(np.ceil(np.log10(max_abs)))
    magnitudes = np.array([10.0**exp for exp in range(min_exp, max_exp + 1)], dtype=float)

    ticks = []
    for mag in magnitudes[::-1]:
        value = -float(mag)
        if left <= value <= right:
            ticks.append(value)
    for mag in magnitudes:
        value = float(mag)
        if left <= value <= right:
            ticks.append(value)

    ticks_arr = np.array(sorted(set(ticks)), dtype=float)
    while ticks_arr.size > max_ticks:
        ticks_arr = np.delete(ticks_arr, int(np.argmin(np.abs(ticks_arr))))
    return ticks_arr


def select_thinned_windows(
    y_thin: np.ndarray,
    keep_idx: np.ndarray,
    max_windows: int,
    pre_context: int,
    selection: str,
    start: Optional[int] = None,
) -> np.ndarray:
    n_windows = len(y_thin)
    if n_windows == 0:
        raise ValueError("No windows available after thinning.")
    if max_windows <= 0 or max_windows > n_windows:
        max_windows = n_windows

    if start is not None:
        start_idx = int(np.clip(start, 0, n_windows - 1))
    elif selection == "first":
        start_idx = 0
    else:
        y_int = np.asarray(y_thin, dtype=int)
        transitions = np.flatnonzero(np.diff(y_int) == 1) + 1
        seizure_idx = np.flatnonzero(y_int == 1)
        if selection == "transition" and transitions.size > 0:
            center = int(transitions[0])
        elif seizure_idx.size > 0:
            center = int(seizure_idx[0])
        else:
            center = 0
        start_idx = max(center - int(pre_context), 0)

    end_idx = min(start_idx + int(max_windows), n_windows)
    start_idx = max(0, end_idx - int(max_windows))
    local_indices = np.arange(start_idx, end_idx, dtype=int)
    print(
        "Selected thinned validation windows "
        f"{int(local_indices[0])}-{int(local_indices[-1])} "
        f"(original indices {int(keep_idx[local_indices[0]])}-{int(keep_idx[local_indices[-1]])})."
    )
    return local_indices


def load_window_subset(
    data_folder: str | os.PathLike[str],
    fold: str,
    split: str,
    fs: int,
    max_windows: int,
    pre_context: int,
    selection: str,
    start: Optional[int],
) -> WindowSubset:
    x_all, y_all = load_fold_arrays(data_folder, fold, split=split, mmap_mode="r")
    fold_dir = Path(data_folder) / f"fold_{fold}"
    mask_path = fold_dir / f"{split}masks.npy"
    masks_all = np.load(mask_path, mmap_mode="r") if mask_path.exists() else None

    x_thin, y_thin, keep_idx = thin_overlapping_windows(x_all, y_all, fs=fs, return_indices=True)
    local_indices = select_thinned_windows(
        y_thin=np.asarray(y_thin),
        keep_idx=np.asarray(keep_idx),
        max_windows=max_windows,
        pre_context=pre_context,
        selection=selection,
        start=start,
    )

    masks = None
    if masks_all is not None:
        masks = np.asarray(masks_all[np.asarray(keep_idx)[local_indices]])

    return WindowSubset(
        x=np.asarray(x_thin[local_indices], dtype=float),
        y=np.asarray(y_thin[local_indices], dtype=int),
        original_indices=np.asarray(keep_idx)[local_indices].astype(int),
        local_indices=local_indices.astype(int),
        masks=masks,
    )


def collect_explanations(
    run: ModelRun,
    subset: WindowSubset,
    fold: str,
    split: str,
    checkpoint_name: str,
    fs: int,
    device: torch.device,
    gnn_epochs: int,
    skip_edge_explainer: bool,
) -> ExplanationBundle:
    ckpt_file = checkpoint_path(run.model_dir, fold, checkpoint_name=checkpoint_name)
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"Missing checkpoint for {run.label}: {ckpt_file}")

    ckpt = load_checkpoint(ckpt_file, map_location="cpu")
    threshold = _checkpoint_threshold(ckpt)
    model = build_model(
        run.model_kind,
        ckpt=ckpt,
        return_explanations=run.returns_fmap,
        device=device,
    )
    model.eval()

    graphs = prepare_graphs_labels(subset.x, subset.y, Model.adj, masks=subset.masks)
    edge_index = graphs[0].edge_index.detach().cpu().numpy()
    n_windows, n_nodes, time_len = subset.x.shape
    n_edges = edge_index.shape[1]

    probs = np.zeros(n_windows, dtype=float)
    preds = np.zeros(n_windows, dtype=int)
    node_time = np.zeros((n_windows, n_nodes, time_len), dtype=float)
    node_graph = np.zeros((n_windows, n_nodes), dtype=float)
    edge_scores = np.zeros((n_windows, n_edges), dtype=float) if run.uses_posthoc or run.is_fixed_concept else None

    node_h_values: list[np.ndarray] = []
    node_theta_values: list[np.ndarray] = []
    node_f_values: list[np.ndarray] = []
    edge_h_values: list[np.ndarray] = []
    edge_theta_values: list[np.ndarray] = []
    edge_f_values: list[np.ndarray] = []

    desc = f"Explaining {run.label}"
    for idx, graph in enumerate(tqdm(graphs, desc=desc)):
        graph = graph.to(device)
        batch = _single_graph_batch(graph, device)

        with torch.no_grad():
            out = model(graph.x, graph.edge_index, batch)
            prob = _first_float(extract_prob(out), default=np.nan)
        probs[idx] = prob
        preds[idx] = int(prob >= threshold) if np.isfinite(prob) else 0

        if run.uses_posthoc:
            ig = calculateIG(model, graph, thr=threshold, target_key="logit")
            cur_node_time = _project_node_values_to_time(
                _clean_signed(ig.node_mask.detach().cpu().numpy()),
                n_nodes=n_nodes,
                time_len=time_len,
            )
            node_time[idx] = cur_node_time
            node_graph[idx] = cur_node_time.mean(axis=1)

            if edge_scores is not None and not skip_edge_explainer:
                gnn_exp = calculateGNNexpl(
                    model,
                    graph,
                    thr=threshold,
                    epochs=gnn_epochs,
                    target_key="logit",
                )
                if getattr(gnn_exp, "edge_mask", None) is not None:
                    edge_mask = _positive(
                        _aggregate_edge_values(gnn_exp.edge_mask.detach().cpu().numpy(), n_edges=n_edges)
                    )
                    # GNNExplainer edge masks are non-negative importance masks. For a
                    # one-sided seizure-evidence plot, keep them only when the graph logit
                    # supports the seizure class.
                    edge_scores[idx] = edge_mask if prob >= threshold else 0.0
            continue

        with torch.no_grad():
            out = model(graph.x, graph.edge_index, batch)

        if run.is_fixed_concept:
            cur_node_fmap = _clean_signed(out["explanation"].detach().cpu().numpy())
            cur_node_scores = cur_node_fmap.reshape(n_nodes, -1).sum(axis=1)
            node_graph[idx] = cur_node_scores
            node_time[idx] = np.repeat(cur_node_scores[:, None], time_len, axis=1)
            node_h_values.append(_mean_concept_vector(out["h_x"].detach().cpu().numpy(), n_items=n_nodes))
            node_theta_values.append(_mean_concept_vector(out["theta_x"].detach().cpu().numpy(), n_items=n_nodes))
            node_f_values.append(_mean_concept_vector(cur_node_fmap, n_items=n_nodes))

            if edge_scores is not None and "explanation_edge" in out:
                cur_edge_fmap = _clean_signed(out["explanation_edge"].detach().cpu().numpy())
                edge_scores[idx] = _aggregate_edge_values(cur_edge_fmap, n_edges=n_edges)
                edge_f_values.append(_mean_concept_vector(cur_edge_fmap, n_items=n_edges))
            if "h_x_edge" in out:
                edge_h_values.append(_mean_concept_vector(out["h_x_edge"].detach().cpu().numpy(), n_items=n_edges))
            if "theta_x_edge" in out:
                edge_theta_values.append(
                    _mean_concept_vector(out["theta_x_edge"].detach().cpu().numpy(), n_items=n_edges)
                )
        else:
            cur_node_time = _project_node_values_to_time(
                _clean_signed(out["explanation"].detach().cpu().numpy()),
                n_nodes=n_nodes,
                time_len=time_len,
            )
            node_time[idx] = cur_node_time
            node_graph[idx] = cur_node_time.sum(axis=1)

    node_h_arr = np.vstack(node_h_values) if node_h_values else None
    node_theta_arr = np.vstack(node_theta_values) if node_theta_values else None
    node_f_arr = np.vstack(node_f_values) if node_f_values else None
    edge_h_arr = np.vstack(edge_h_values) if edge_h_values else None
    edge_theta_arr = np.vstack(edge_theta_values) if edge_theta_values else None
    edge_f_arr = np.vstack(edge_f_values) if edge_f_values else None

    return ExplanationBundle(
        run=run,
        fold=str(fold),
        split=str(split),
        fs=int(fs),
        threshold=float(threshold),
        x=np.asarray(subset.x, dtype=float),
        y=np.asarray(subset.y, dtype=int),
        original_indices=np.asarray(subset.original_indices, dtype=int),
        local_indices=np.asarray(subset.local_indices, dtype=int),
        probs=probs,
        preds=preds,
        edge_index=edge_index,
        node_time=node_time,
        node_graph=node_graph,
        edge_scores=edge_scores,
        node_concepts=node_h_arr,
        edge_concepts=edge_h_arr,
        node_h=node_h_arr,
        node_theta=node_theta_arr,
        node_f=node_f_arr,
        edge_h=edge_h_arr,
        edge_theta=edge_theta_arr,
        edge_f=edge_f_arr,
    )


class InteractiveExplanationFigure:
    def __init__(self, bundle: ExplanationBundle, output_dir: Optional[Path] = None):
        self.bundle = bundle
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.current_idx = 0

        node_values = _positive(
            np.concatenate(
                [
                    bundle.node_time.reshape(-1),
                    bundle.node_graph.reshape(-1),
                ]
            )
        )
        self.node_color_norm, self.node_scale_label = _magnitude_norm(node_values)
        self.node_magnitude_norm = self.node_color_norm

        self.edge_color_norm: Optional[Normalize] = None
        self.edge_magnitude_norm: Optional[Normalize] = None
        self.edge_scale_label = "linear"
        if bundle.edge_scores is not None:
            edge_values = _positive(bundle.edge_scores)
            if np.any(edge_values > 0):
                self.edge_color_norm, self.edge_scale_label = _magnitude_norm(edge_values)
                self.edge_magnitude_norm = self.edge_color_norm

            if self.edge_scale_label == "linear" and bundle.run.uses_posthoc:
                self.edge_color_norm = Normalize(vmin=0.0, vmax=1.0)
                self.edge_magnitude_norm = self.edge_color_norm

        figsize = INTERACTIVE_THREE_PANEL_FIGSIZE if bundle.run.is_fixed_concept else INTERACTIVE_TWO_PANEL_FIGSIZE
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        layout_engine = self.fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(w_pad=0.025, h_pad=0.025, wspace=0.08, hspace=0.08)
        if bundle.run.is_fixed_concept:
            gs = self.fig.add_gridspec(
                2,
                2,
                height_ratios=[1.35, 1.0],
                width_ratios=[1.02, 1.0],
            )
            self.ax_time = self.fig.add_subplot(gs[0, :])
            self.ax_graph = self.fig.add_subplot(gs[1, 0])
            self.ax_concepts = self.fig.add_subplot(gs[1, 1])
        else:
            gs = self.fig.add_gridspec(
                2,
                2,
                height_ratios=[1.35, 1.0],
                width_ratios=[1.05, 0.95],
            )
            self.ax_time = self.fig.add_subplot(gs[0, :])
            self.ax_graph = self.fig.add_subplot(gs[1, 0])
            self.ax_concepts = None

        self.edge_groups = _undirected_edge_groups(bundle.edge_index)
        self.selection_patch = None
        self._draw_time_axis()
        self._add_colorbars()
        self.update(16)
        self._connect_events()

    def _draw_time_axis(self) -> None:
        bundle = self.bundle
        x = bundle.x
        expl = bundle.node_time
        n_windows, n_channels, time_len = x.shape
        total_len = n_windows * time_len
        t_axis = np.arange(total_len, dtype=float) / bundle.fs
        x_series = np.concatenate([x[i] for i in range(n_windows)], axis=1)
        expl_series = _positive(np.concatenate([expl[i] for i in range(n_windows)], axis=1))

        amp = float(np.nanpercentile(np.abs(x_series), 95)) if x_series.size else 1.0
        spacing = max(amp * 3.0, 1.0)
        offsets = np.arange(n_channels, dtype=float) * spacing

        for channel_idx in range(n_channels):
            y_series = x_series[channel_idx] + offsets[channel_idx]
            if total_len > 1:
                points = np.column_stack([t_axis, y_series]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=NODE_CMAP, norm=self.node_color_norm)
                lc.set_array(expl_series[channel_idx, :-1])
                lc.set_linewidth(0.8)
                lc.set_zorder(2.0)
                self.ax_time.add_collection(lc)
            else:
                self.ax_time.plot(t_axis, y_series, color=EEG_TRACE_COLOR, linewidth=0.8)

        top_trace_y = offsets[-1] + spacing
        track_height = 0.18 * spacing
        true_y = top_trace_y + 0.25 * spacing
        pred_y = top_trace_y + 0.58 * spacing
        window_width = time_len / bundle.fs

        for idx in range(n_windows):
            x0 = idx * window_width
            true_color = SEIZURE_COLOR if int(bundle.y[idx]) == 1 else NON_SEIZURE_COLOR
            pred_color = SEIZURE_COLOR if int(bundle.preds[idx]) == 1 else NON_SEIZURE_COLOR
            self.ax_time.add_patch(
                Rectangle(
                    (x0, true_y),
                    window_width,
                    track_height,
                    facecolor=true_color,
                    edgecolor="none",
                    alpha=0.70,
                    zorder=3.0,
                )
            )
            self.ax_time.add_patch(
                Rectangle(
                    (x0, pred_y),
                    window_width,
                    track_height,
                    facecolor=pred_color,
                    edgecolor="none",
                    alpha=0.70,
                    zorder=3.0,
                )
            )

        self.ax_time.text(
            -0.01,
            true_y + -1 +0 * track_height,
            "True",
            transform=self.ax_time.get_yaxis_transform(),
            ha="right",
            va="center",
        )
        self.ax_time.text(
            -0.01,
            pred_y + 1 +0. * track_height,
            "Pred.",
            transform=self.ax_time.get_yaxis_transform(),
            ha="right",
            va="center",
        )

        labels = CHANNEL_NAMES[:n_channels]
        self.ax_time.set_yticks(offsets)
        self.ax_time.set_yticklabels(labels)
        self.ax_time.set_xlim(0.0, total_len / bundle.fs)
        self.ax_time.set_ylim(-0.75 * spacing, pred_y + 0.65 * spacing)
        self.ax_time.set_xlabel("Time in selected validation sequence (s)", labelpad=1.0)
        self.ax_time.set_ylabel("Channel")
        self.ax_time.set_title("EEG signal coloured by positive seizure evidence", pad=1.5)
        self.ax_time.grid(axis="x", color="0.88", linewidth=0.45)

        legend_handles = [
            Patch(facecolor=NON_SEIZURE_COLOR, edgecolor="none", label="Non-seizure"),
            Patch(facecolor=SEIZURE_COLOR, edgecolor="none", label="Seizure"),
        ]
        self.ax_time.legend(
            handles=legend_handles,
            loc="upper right",
            ncol=2,
            handlelength=1.0,
            columnspacing=0.7,
            borderaxespad=0.15,
            frameon=True,
            framealpha=0.7,
            fancybox=False,
        )

    def _add_colorbars(self) -> None:
        node_label = f"Node evidence\nReLU(F) ({self.node_scale_label})"
        node_cbar = self.fig.colorbar(
            ScalarMappable(norm=self.node_color_norm, cmap=NODE_CMAP),
            ax=self.ax_time,
            pad=0.012,
            fraction=0.026,
        )
        node_cbar.set_label(node_label)

        if self.edge_color_norm is not None:
            edge_cbar = self.fig.colorbar(
                ScalarMappable(norm=self.edge_color_norm, cmap=EDGE_CMAP),
                ax=self.ax_graph,
                orientation="horizontal",
                pad=0.055,
                fraction=0.085,
                shrink=0.78,
                aspect=22,
            )
            edge_cbar.set_label(f"Edge evidence ReLU(F) ({self.edge_scale_label})", labelpad=1.0)
            edge_base_formatter = edge_cbar.ax.xaxis.get_major_formatter()
            if self.edge_scale_label == "linear":
                edge_cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
                edge_cbar.ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
            else:
                edge_base_formatter = edge_cbar.ax.xaxis.get_major_formatter()
                edge_cbar.ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: "" if x == 0.0 else edge_base_formatter(x, pos))
                )

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_key(self, event) -> None:
        if event.key in {"right", "d", "n"}:
            self.update(self.current_idx + 1)
        elif event.key in {"left", "a", "p"}:
            self.update(self.current_idx - 1)
        elif event.key == "home":
            self.update(0)
        elif event.key == "end":
            self.update(len(self.bundle.y) - 1)
        elif event.key == "up":
            self._jump_to_label(label=1, direction=1)
        elif event.key == "down":
            self._jump_to_label(label=1, direction=-1)
        elif event.key == "e":
            self.save_current()
        elif event.key == "q":
            plt.close(self.fig)

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax_time or event.xdata is None:
            return
        time_len = self.bundle.x.shape[-1]
        window_width = time_len / self.bundle.fs
        idx = int(np.floor(float(event.xdata) / window_width))
        self.update(idx)

    def _jump_to_label(self, label: int, direction: int) -> None:
        y = np.asarray(self.bundle.y, dtype=int)
        if direction >= 0:
            candidates = np.flatnonzero((np.arange(len(y)) > self.current_idx) & (y == label))
        else:
            candidates = np.flatnonzero((np.arange(len(y)) < self.current_idx) & (y == label))
        if candidates.size == 0:
            return
        self.update(int(candidates[0] if direction >= 0 else candidates[-1]))

    def update(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.bundle.y) - 1))
        self.current_idx = idx
        self._update_selection_patch()
        self._draw_graph(idx)
        if self.ax_concepts is not None:
            self._draw_concepts(idx)

        y_true = int(self.bundle.y[idx])
        y_pred = int(self.bundle.preds[idx])
        prob = float(self.bundle.probs[idx])
        original_idx = int(self.bundle.original_indices[idx]) #window {idx + 1}/{len(self.bundle.y)} {original_idx}
        title = (
            f"{self.bundle.run.label} | fold {self.bundle.fold} | "
            f"window {original_idx}| "
            f"y={y_true}, pred={y_pred}, p={prob:.3f}, thr={self.bundle.threshold:.3f}"
        )
        self.fig.suptitle(title)
        self.fig.canvas.draw_idle()

    def _update_selection_patch(self) -> None:
        if self.selection_patch is not None:
            self.selection_patch.remove()
        time_len = self.bundle.x.shape[-1]
        window_width = time_len / self.bundle.fs
        x0 = self.current_idx * window_width
        self.selection_patch = self.ax_time.axvspan(
            x0,
            x0 + window_width,
            facecolor="0.70",
            edgecolor="0.20",
            linewidth=0.6,
            alpha=0.20,
            zorder=1.5,
        )

    def _draw_graph(self, idx: int) -> None:
        self.ax_graph.clear()
        bundle = self.bundle
        node_scores = _positive(bundle.node_graph[idx])
        edge_scores = _clean_signed(bundle.edge_scores[idx]) if bundle.edge_scores is not None else None

        segments = []
        edge_colors = []
        edge_widths = []
        for (src, dst), edge_ids in self.edge_groups.items():
            segments.append([NODE_POS[src], NODE_POS[dst]])
            if edge_scores is not None and self.edge_color_norm is not None and self.edge_magnitude_norm is not None:
                value = float(np.mean(edge_scores[np.asarray(edge_ids, dtype=int)]))
                value = float(max(value, 0.0))
                scaled = _norm_scalar(self.edge_magnitude_norm, value)
                edge_colors.append(EDGE_CMAP(self.edge_color_norm(value)))
                edge_widths.append(0.45 + 3.4 * scaled)
            else:
                edge_colors.append(UNKNOWN_COLOR)
                edge_widths.append(0.65)

        if segments:
            edge_alpha = 0.90 if edge_scores is not None and self.edge_color_norm is not None else 0.42
            lc = LineCollection(segments, colors=edge_colors, linewidths=edge_widths, alpha=edge_alpha, zorder=1)
            self.ax_graph.add_collection(lc)

        xs = np.array([NODE_POS[i][0] for i in range(len(node_scores))], dtype=float)
        ys = np.array([NODE_POS[i][1] for i in range(len(node_scores))], dtype=float)
        scaled_nodes = np.array([_norm_scalar(self.node_magnitude_norm, value) for value in node_scores], dtype=float)
        node_sizes = 110.0 + 360.0 * scaled_nodes
        node_colors = [NODE_CMAP(self.node_color_norm(float(value))) for value in node_scores]

        self.ax_graph.scatter(
            xs,
            ys,
            s=node_sizes,
            c=node_colors,
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )
        for node_idx, (x_pos, y_pos) in NODE_POS.items():
            if node_idx >= len(node_scores):
                continue
            label = CHANNEL_NAMES[node_idx] if node_idx < len(CHANNEL_NAMES) else str(node_idx)
            self.ax_graph.text(
                x_pos,
                y_pos - 0.27,
                label,
                ha="center",
                va="top",
                fontsize=6.4,
                color="0.15",
                zorder=4,
            )

        edge_note = "seizure edge evidence" if edge_scores is not None and self.edge_color_norm is not None else "no edge mask"
        self.ax_graph.set_title(f"Graph representation\n({edge_note})", pad=3.0)
        self.ax_graph.set_aspect("equal", adjustable="box")
        if xs.size:
            self.ax_graph.set_xlim(float(np.min(xs)) - GRAPH_X_PAD, float(np.max(xs)) + GRAPH_X_PAD)
            self.ax_graph.set_ylim(float(np.min(ys)) - GRAPH_Y_PAD_BOTTOM, float(np.max(ys)) + GRAPH_Y_PAD_TOP)
        self.ax_graph.axis("off")

    def _draw_concepts(self, idx: int) -> None:
        if self.ax_concepts is None:
            return
        self.ax_concepts.clear()
        bundle = self.bundle
        if bundle.node_h is None or bundle.node_theta is None or bundle.node_f is None:
            self.ax_concepts.text(0.5, 0.5, "No fixed-concept decomposition", ha="center", va="center")
            self.ax_concepts.axis("off")
            return

        node_h = np.asarray(bundle.node_h[idx], dtype=float).reshape(-1)
        node_theta = np.asarray(bundle.node_theta[idx], dtype=float).reshape(-1)
        node_f = np.asarray(bundle.node_f[idx], dtype=float).reshape(-1)

        edge_h = np.asarray(bundle.edge_h[idx], dtype=float).reshape(-1) if bundle.edge_h is not None else np.array([])
        edge_theta = (
            np.asarray(bundle.edge_theta[idx], dtype=float).reshape(-1)
            if bundle.edge_theta is not None
            else np.array([])
        )
        edge_f = np.asarray(bundle.edge_f[idx], dtype=float).reshape(-1) if bundle.edge_f is not None else np.array([])

        labels = FIXED_CONCEPT_NAMES[: len(node_h)]
        h_values = node_h
        theta_values = node_theta
        f_values = node_f

        if edge_h.size > 0:
            edge_labels = [
                "Edge concept" if edge_h.size == 1 else f"Edge concept {edge_idx + 1}"
                for edge_idx in range(edge_h.size)
            ]
            labels = labels + edge_labels
            h_values = np.concatenate([h_values, edge_h])
            theta_values = np.concatenate([theta_values, edge_theta]) if edge_theta.size else np.concatenate(
                [theta_values, np.full(edge_h.shape, np.nan)]
            )
            f_values = np.concatenate([f_values, edge_f]) if edge_f.size else np.concatenate(
                [f_values, np.full(edge_h.shape, np.nan)]
            )

        values_by_component = {
            "h": h_values,
            "theta": theta_values,
            "F": f_values,
        }
        y_pos = np.arange(len(labels), dtype=float)
        bar_height = 0.21
        offsets = {"h": -bar_height, "theta": 0.0, "F": bar_height}

        for component, values in values_by_component.items():
            self.ax_concepts.barh(
                y_pos + offsets[component],
                values,
                height=bar_height * 0.9,
                color=CONCEPT_COMPONENT_COLORS[component],
                alpha=0.88,
                edgecolor="none",
                label={"h": r"$h$", "theta": r"$\theta$", "F": r"$F$"}[component],
            )

        self.ax_concepts.axvline(0.0, color="0.35", linewidth=0.6)
        self.ax_concepts.set_yticks(y_pos)
        self.ax_concepts.set_yticklabels(labels)
        self.ax_concepts.tick_params(axis="y", pad=1.0)
        self.ax_concepts.invert_yaxis()
        self.ax_concepts.set_xlabel("Mean graph-level value", labelpad=1.0)
        self.ax_concepts.set_title(r"Concept decomposition: $h$, $\theta$, and $F$", pad=3.0)
        self.ax_concepts.grid(axis="x", color="0.88", linewidth=0.45)

        all_concepts = np.concatenate(
            [
                bundle.node_h.reshape(-1),
                bundle.node_theta.reshape(-1),
                bundle.node_f.reshape(-1),
                bundle.edge_h.reshape(-1) if bundle.edge_h is not None else np.array([], dtype=float),
                bundle.edge_theta.reshape(-1) if bundle.edge_theta is not None else np.array([], dtype=float),
                bundle.edge_f.reshape(-1) if bundle.edge_f is not None else np.array([], dtype=float),
            ]
        )
        linthresh, xmax = _concept_x_scale(all_concepts)
        self.ax_concepts.set_xscale("symlog", linthresh=linthresh, base=10)
        min_val = float(np.nanmin(all_concepts)) if all_concepts.size else 0.0
        left = -1.05 * xmax if min_val < 0 else 0.0
        right = 1.05 * xmax
        self.ax_concepts.set_xlim(left, right)
        concept_ticks = _sparse_symlog_ticks(left, right, linthresh)
        if concept_ticks.size:
            self.ax_concepts.set_xticks(concept_ticks)
        self.ax_concepts.legend(
            loc="lower right",
            ncol=3,
            handlelength=1.0,
            columnspacing=0.6,
            borderaxespad=0.25,
        )
        concept_base_formatter = self.ax_concepts.xaxis.get_major_formatter()
        self.ax_concepts.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, pos: "" if np.isclose(x, 0.0) else concept_base_formatter(x, pos)
            )
        )
    def save_current(self) -> None:
        if self.output_dir is None:
            self.output_dir = Path("./Results_interactive_plots")
        ensure_dir(self.output_dir)
        stem = (
            f"{_safe_name(self.bundle.run.key)}_fold{self.bundle.fold}_"
            f"window{self.current_idx:03d}_original{int(self.bundle.original_indices[self.current_idx])}"
        )
        png_path = self.output_dir / f"{stem}.png"
        pdf_path = self.output_dir / f"{stem}.pdf"
        self.fig.savefig(png_path, bbox_inches="tight")
        self.fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved current view to {png_path} and {pdf_path}")


def plot_interactive_explanations(
    bundle: ExplanationBundle,
    output_dir: Optional[Path] = None,
    show: bool = True,
    save_initial: bool = False,
) -> InteractiveExplanationFigure:
    view = InteractiveExplanationFigure(bundle=bundle, output_dir=output_dir)
    print("Interactive controls: left/right move windows, up/down jump seizure windows, click the time axis, e saves, q closes.")
    if save_initial:
        view.save_current()
    if show:
        plt.show()
    return view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive final-model EEG/XAI plotter. The validation sequence is first thinned "
            "with the same non-overlap convention used by the XAI statistics scripts."
        )
    )
    parser.add_argument("--model", default="senn-ic", help="Model key, alias, or 'all'.")
    parser.add_argument("--list-models", action="store_true", help="Print available final models and exit.")
    parser.add_argument("--data-folder", default=DEFAULT_DATA_FOLDER, help="Path to Datasets/CV_Folds.")
    parser.add_argument("--fold", default=DEFAULT_FOLD, help="CV fold to plot.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Data split name, usually 'test'.")
    parser.add_argument("--checkpoint-name", default="best_auprc.pt", help="Checkpoint file inside GAT_CV_10_<fold>.")
    parser.add_argument("--fs", type=int, default=DEFAULT_FS, help="EEG sampling frequency.")
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS, help="Number of thinned windows to plot.")
    parser.add_argument("--pre-context", type=int, default=DEFAULT_PRE_CONTEXT, help="Windows shown before first seizure/transition.")
    parser.add_argument(
        "--selection",
        choices=["transition", "first-seizure", "first"],
        default="transition",
        help="How to choose the plotted thinned validation segment.",
    )
    parser.add_argument("--start", type=int, default=None, help="Manual start index in the thinned validation set.")
    parser.add_argument("--gnn-epochs", type=int, default=200, help="GNNExplainer epochs for the ST-GAT edge mask.")
    parser.add_argument(
        "--skip-edge-explainer",
        action="store_true",
        help="Skip the slow ST-GAT GNNExplainer edge mask and draw neutral graph edges.",
    )
    parser.add_argument("--device", default=None, help="Torch device. Defaults to CUDA when available.")
    parser.add_argument("--output-dir", default="./Results_interactive_plots", help="Directory for saved interactive views.")
    parser.add_argument("--save", action="store_true", help="Save the first view as PNG/PDF.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive matplotlib window.")
    return parser.parse_args()


def _print_model_registry() -> None:
    print("Available final models:")
    for key, run in FINAL_MODELS.items():
        print(f"  {key:18s} {run.label:24s} {run.model_dir}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        _print_model_registry()
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_runs: Sequence[ModelRun]
    if str(args.model).strip().lower() == "all":
        model_runs = list(FINAL_MODELS.values())
    else:
        model_runs = [resolve_model_run(args.model)]

    subset = load_window_subset(
        data_folder=args.data_folder,
        fold=str(args.fold),
        split=str(args.split),
        fs=int(args.fs),
        max_windows=int(args.max_windows),
        pre_context=int(args.pre_context),
        selection=str(args.selection),
        start=args.start,
    )

    output_dir = Path(args.output_dir)
    for run in model_runs:
        print(f"\nModel: {run.label}")
        print(f"Checkpoint root: {run.model_dir}")
        bundle = collect_explanations(
            run=run,
            subset=subset,
            fold=str(args.fold),
            split=str(args.split),
            checkpoint_name=str(args.checkpoint_name),
            fs=int(args.fs),
            device=device,
            gnn_epochs=int(args.gnn_epochs),
            skip_edge_explainer=bool(args.skip_edge_explainer),
        )
        plot_interactive_explanations(
            bundle=bundle,
            output_dir=output_dir,
            show=not bool(args.no_show),
            save_initial=bool(args.save),
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
