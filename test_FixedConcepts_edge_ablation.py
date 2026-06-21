import os
import json
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, fbeta_score, cohen_kappa_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import Models_senn as Model
import MyUtils_senn_test as MyUtils
from metrics_utils import auprg

THESIS_TEXT_WIDTH_IN = 15.0 / 2.54
ABLATION_2X2_FIGSIZE = (THESIS_TEXT_WIDTH_IN, 4.05)
ABLATION_AUX_WIDTH_IN = THESIS_TEXT_WIDTH_IN
NON_SEIZURE_COLOR = "#56B4E9"  # blue
SEIZURE_COLOR = "#E69F00"      # orange
GLOBAL_COLOR = "#009E73"       # muted green
EDGE_COLOR = "#CC79A7"
UNKNOWN_COLOR = "#9e9e9e"      # neutral grey
ABLATION_MODEL_COLORS = [
    "#7D7D7D",  # dark grey
    "#A6761D",  # muted brown/gold
    ]
# ABLATION_MODEL_COLORS = [
#     "tab:green",
#     "tab:red",
#     ]


def apply_thesis_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception as exc:
        print(f"[WARN] SciencePlots style unavailable ({exc}); using local thesis style.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "Palatino Linotype", "Palatino", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
        "mathtext.bf": "Palatino Linotype:bold",
        "text.usetex": False,
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.2,
        "figure.titlesize": 10.0,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(
            color=[NON_SEIZURE_COLOR, SEIZURE_COLOR, GLOBAL_COLOR, "#009E73", "#CC79A7", "#E69F00"]
        ),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


apply_thesis_plot_style()

"""
Sequential inference-time ablation study comparing SENN concept variants.

What this script does
---------------------
1. Runs inference-time concept ablations for SENN-IC and both SENN-FC models.
2. Uses full-path semantics for the comparison.
3. Saves one grouped bar plot with one bar per model for each ablation condition.

Goal
----
Test whether the fixed concepts themselves materially constrain prediction,
or whether the learned relevance networks carry most of the discriminative power.

SENN-IC ablation modes act on h_x (shape [B, N, T]).
SENN-FC node ablation modes act on h_x (shape [B, N, K_node]).
SENN-FC edge ablation modes act on h_x_edge (shape [B, E, K_edge]).

Included ablation modes
-----------------------
- none
- global_mean: reference-set per-channel/per-edge mean
- per_graph_mean
- shuffle_within_graph
- zero
"""


# ============================================================
# USER SETTINGS
# ============================================================
LOG = "470501"
FOLD = "7"
FOLDS = [str(i) for i in range(10)]  # Use [FOLD] for a quick single-fold run.
CKPT_NAME = "best_auprc.pt"
MODEL_SUBDIR = f"GAT_CV_10_{FOLD}"

# Ablation paths:
#   "carrier_only": keep theta fixed from the unablated sample and only replace h in F = theta * h
#   "full_path":    recompute theta from ablated concepts where the architecture allows it

DATA_FOLDER = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets\CV_Folds/"

# Model variants included in the single comparison plot:
#   "senn_rawx"              = SENN-IC / identity concepts
#   "senn_fixed"            = SENN-FC-theta(x): theta from raw EEG x
#   "senn_fixedconcepttheta" = SENN-FC-theta(h): theta from fixed concepts h
MODEL_RUNS = [
    # {
    #     "label": "SENN-IC",
    #     "model_kind": "senn_rawx",
    #     "model_dir": "./ArchiveModelsMainResults/Saved_models_492092_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-4/",
    #     "ablation_path": "full_path",
    # },
    {
        "label": "SENN-FC-$\\theta(x)$",
        "model_kind": "senn_fixed",
        "model_dir": "./ArchiveModelsMainResults/Saved_models_486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0/",
        "ablation_path": "full_path",
    },
    {
        "label": "SENN-FC-$\\theta(h)$",
        "model_kind": "senn_fixedconcepttheta",
        "model_dir": "./ArchiveModelsMainResults/Saved_models_486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0/",
        "ablation_path": "full_path",
    },
]

# "both" means all available concepts: identity concepts for SENN-IC, node+edge concepts for SENN-FC.
COMPARISON_BRANCH = "both"

REFERENCE_SPLIT = "train"   # reference split for the global_mean baseline
EVAL_SPLIT = "test"

RESULTS_BASE = os.path.join(".", "Results_Charact")
RESULTS_ROOT = os.path.join(
    RESULTS_BASE,
    "node_edge_ablation",
    "SENN-IC_FC-x_FC-h_full_path",
)

# Active model globals are set from MODEL_RUNS during main().
MODEL_KIND = MODEL_RUNS[0]["model_kind"]
MODEL_DIR = MODEL_RUNS[0]["model_dir"]
ABLATION_PATH = MODEL_RUNS[0]["ablation_path"]
MODEL_OUTPUT_NAME = MODEL_RUNS[0]["label"]
ABLATION_PATH_OUTPUT_NAME = "full"

ALLOW_REFERENCE_EVAL_LEAKAGE = False
USE_FULL_EVAL_SPLIT = True
EVAL_SUBSET_SIZE = 1500
CROSS_FOLD_ERROR = "std"  # "sem" gives compact fold-level error bars; use "std" for spread.

ABLATION_MODES = [
    "none",
    "global_mean",
    "per_graph_mean",
    "shuffle_within_graph",
    "zero",
]

LOADER_BATCH_SIZE = 128
NUM_WORKERS = 0
PIN_MEM = False
RANDOM_SEED = 42

SAVE_PLOTS = True
SHOW_PLOTS = False
SAVE_CROSS_FOLD_TABLES = False
PANEL_READY_EXPORT = True
# ============================================================


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_active_fold(fold: str) -> None:
    """Update the few legacy globals used by the existing single-fold code."""
    global FOLD, MODEL_SUBDIR
    FOLD = str(fold)
    MODEL_SUBDIR = f"GAT_CV_10_{FOLD}"


def normalize_model_kind(model_kind: str) -> str:
    aliases = {
        "senn_ic": "senn_rawx",
        "senn-ic": "senn_rawx",
        "senn_raw": "senn_rawx",
        "sennrawx": "senn_rawx",
        "rawx": "senn_rawx",
        "senn_fixed": "senn_fixed",
        "sennfixed": "senn_fixed",
        "senn_fc_x": "senn_fixed",
        "senn-fc-x": "senn_fixed",
        "senn_fixedconcepttheta": "senn_fixedconcepttheta",
        "senn_fixed_concepttheta": "senn_fixedconcepttheta",
        "sennfixed_concepttheta": "senn_fixedconcepttheta",
        "senn_fc_h": "senn_fixedconcepttheta",
        "senn-fc-h": "senn_fixedconcepttheta",
    }
    key = str(model_kind).strip().lower()
    return aliases.get(key, key)


def model_has_edge_concepts(model_kind: str) -> bool:
    return normalize_model_kind(model_kind) in {"senn_fixed", "senn_fixedconcepttheta"}


def set_active_model_run(model_run: Dict[str, str]) -> None:
    """Update legacy globals for the model currently being evaluated."""
    global MODEL_KIND, MODEL_DIR, ABLATION_PATH, MODEL_OUTPUT_NAME, ABLATION_PATH_OUTPUT_NAME
    MODEL_KIND = normalize_model_kind(model_run["model_kind"])
    MODEL_DIR = model_run["model_dir"]
    ABLATION_PATH = model_run.get("ablation_path", "full_path")
    MODEL_OUTPUT_NAME = model_run["label"]
    ABLATION_PATH_OUTPUT_NAME = {
        "carrier_only": "carrier",
        "full_path": "full",
    }.get(ABLATION_PATH, ABLATION_PATH)


def build_model(ckpt: dict, model_kind: str) -> torch.nn.Module:
    """Build one of the SENN variants used by this ablation script."""
    model_kind = normalize_model_kind(model_kind)

    if model_kind == "senn_rawx":
        model = Model.SENN_raw(
            global_min=ckpt.get("global_min", None),
            return_node_scores=False,
            return_fmap=True,
        )
    elif model_kind == "senn_fixed":
        model = Model.SENN_fixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=True,
        )
    elif model_kind == "senn_fixedconcepttheta":
        model = Model.SENN_fixedconcepts_concepttheta(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=True,
        )
    else:
        raise ValueError(
            "MODEL_KIND must be 'senn_rawx', 'senn_fixed', or 'senn_fixedconcepttheta', "
            f"got {model_kind!r}"
        )

    model.load_state_dict(ckpt["model_state_dict"])
    return model


def validate_config() -> None:
    if COMPARISON_BRANCH not in {"node", "edge", "both"}:
        raise ValueError("COMPARISON_BRANCH must be 'node', 'edge', or 'both'")
    if len(MODEL_RUNS) == 0:
        raise ValueError("MODEL_RUNS must contain at least one model")
    for model_run in MODEL_RUNS:
        model_kind = normalize_model_kind(model_run["model_kind"])
        ablation_path = str(model_run.get("ablation_path", "full_path")).strip().lower()
        if model_kind not in {"senn_rawx", "senn_fixed", "senn_fixedconcepttheta"}:
            raise ValueError(
                "MODEL_RUNS model_kind values must be 'senn_rawx', 'senn_fixed', "
                f"or 'senn_fixedconcepttheta'; got {model_kind!r}"
            )
        if ablation_path not in {"carrier_only", "full_path"}:
            raise ValueError("MODEL_RUNS ablation_path values must be 'carrier_only' or 'full_path'")
        if model_kind == "senn_rawx" and COMPARISON_BRANCH == "edge":
            raise ValueError("SENN-IC has no edge concepts; use COMPARISON_BRANCH='node' or 'both'")
        if model_kind == "senn_fixed" and ablation_path == "full_path":
            print(
                "WARNING: SENN-FC-x has theta(x), not theta(h). "
                "Full-path is equivalent to carrier-only for this model."
            )
    if CROSS_FOLD_ERROR not in {"sem", "std"}:
        raise ValueError("CROSS_FOLD_ERROR must be 'sem' or 'std'")
    if len(FOLDS) == 0:
        raise ValueError("FOLDS must contain at least one fold")

    if not ALLOW_REFERENCE_EVAL_LEAKAGE:
        if str(REFERENCE_SPLIT).strip().lower() == "test":
            raise ValueError(
                "Do not compute global_mean concept baselines on the test split. "
                "Use REFERENCE_SPLIT='train' (preferred) or 'val'."
            )
        if REFERENCE_SPLIT == EVAL_SPLIT:
            raise ValueError(
                "REFERENCE_SPLIT equals EVAL_SPLIT. This leaks evaluation-set concept "
                "statistics into the global_mean ablation. Use train/val as reference, "
                "or set ALLOW_REFERENCE_EVAL_LEAKAGE=True only for debugging."
            )


def load_split_arrays(split: str, norm: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
    fold_dir = os.path.join(DATA_FOLDER, f"fold_{FOLD}")
    x = np.load(os.path.join(fold_dir, f"{split}data.npy"), mmap_mode="r")
    y = np.load(os.path.join(fold_dir, f"{split}labels.npy"), mmap_mode="r")
    if norm is not None:
        x = (x - norm["mean"]) / norm["std"]
    return x, y


def prepare_split_dataset(split: str, norm: Optional[dict], use_full: bool = True, subset_size: int = 1500):
    x, y = load_split_arrays(split, norm)
    dataset = MyUtils.prepare_graphs_labels(x, y, Model.adj)

    idx_seiz = np.where(y == 1)[0]
    idx_non = np.where(y == 0)[0]

    fs = 32
    t_window = x.shape[-1] / fs
    t_overlap_non = 10
    t_overlap_seiz = 11

    skip_non = int(t_window / (t_window - t_overlap_non))
    skip_seiz = int(t_window / (t_window - t_overlap_seiz))
    skip_non = max(skip_non, 1)
    skip_seiz = max(skip_seiz, 1)

    keep_idx = np.sort(np.concatenate([idx_non[0::skip_non], idx_seiz[0::skip_seiz]]))
    x = x[keep_idx]
    y = y[keep_idx]

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    dataset = MyUtils.prepare_graphs_labels(x, y, Model.adj)
    return dataset, y


def _reshape_per_graph(tensor: torch.Tensor, batch_size_actual: int) -> torch.Tensor:
    total_items = tensor.shape[0]
    if total_items % batch_size_actual != 0:
        raise ValueError(
            f"Cannot reshape first dimension {total_items} into batch size {batch_size_actual}"
        )
    items_per_graph = total_items // batch_size_actual
    return tensor.view(batch_size_actual, items_per_graph, *tensor.shape[1:])


def _conceptize_batch(
    model: torch.nn.Module,
    model_kind: str,
    batch,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    model_kind = normalize_model_kind(model_kind)
    if model_kind == "senn_rawx":
        return model.conceptizer(batch.x), None
    h_node, h_edge = model.conceptizer(batch.x, batch.edge_index)
    return h_node, h_edge


def _identity_concepts_to_raw(model: torch.nn.Module, h_node: torch.Tensor) -> torch.Tensor:
    """Map SENN-IC shifted identity concepts back to the raw signal scale."""
    shift = getattr(getattr(model, "conceptizer", None), "shift", None)
    if shift is None:
        return h_node
    return h_node - shift.to(device=h_node.device, dtype=h_node.dtype)


@torch.no_grad()
def compute_reference_means(
    model: torch.nn.Module,
    model_kind: str,
    reference_dataset,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute reference-set per-node/per-edge concept baselines.

    Returns:
        global_node_mean: shape [N_nodes, K_node or T]
        global_edge_mean: shape [E_edges, K_edge], or None for SENN-IC
    """
    loader = DataLoader(
        reference_dataset,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEM,
        num_workers=NUM_WORKERS,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    model = model.to(device)
    model.eval()

    node_sum = None
    edge_sum = None
    graph_count = 0

    edge_label = "node/edge" if model_has_edge_concepts(model_kind) else "identity"
    print(f"... computing reference {edge_label} means on '{REFERENCE_SPLIT}' split ...")
    for batch in tqdm(loader):
        batch = batch.to(device, non_blocking=True)
        h_node, h_edge = _conceptize_batch(model, model_kind, batch)

        if h_node.ndim != 2:
            raise ValueError(f"Expected h_node shape [N_total, K_node/T], got {tuple(h_node.shape)}")
        batch_size_actual = int(batch.y.view(-1).shape[0])
        edge_batch = None
        if h_edge is not None:
            if h_edge.ndim != 2:
                raise ValueError(f"Expected h_edge shape [E_total, K_edge], got {tuple(h_edge.shape)}")
            edge_batch = batch.batch[batch.edge_index[0]]
            if not torch.equal(batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]]):
                raise RuntimeError("Cross-graph edges detected in batched edge_index.")
            _check_batched_graph_layout(batch, h_node, h_edge, edge_batch)
        else:
            _check_batched_node_layout(batch, h_node)

        h_node_graph = _reshape_per_graph(h_node, batch_size_actual)
        node_batch_sum = h_node_graph.sum(dim=0)  # [N_nodes, K_node]

        if node_sum is not None and node_sum.shape != node_batch_sum.shape:
            raise RuntimeError(
                f"Reference node mean shape changed from {tuple(node_sum.shape)} "
                f"to {tuple(node_batch_sum.shape)}"
            )
        node_sum = node_batch_sum if node_sum is None else node_sum + node_batch_sum

        if h_edge is not None:
            h_edge_graph = _reshape_per_graph(h_edge, batch_size_actual)
            edge_batch_sum = h_edge_graph.sum(dim=0)  # [E_edges, K_edge]
            if edge_sum is not None and edge_sum.shape != edge_batch_sum.shape:
                raise RuntimeError(
                    f"Reference edge mean shape changed from {tuple(edge_sum.shape)} "
                    f"to {tuple(edge_batch_sum.shape)}"
                )
            edge_sum = edge_batch_sum if edge_sum is None else edge_sum + edge_batch_sum
        graph_count += batch_size_actual

    if graph_count == 0:
        raise RuntimeError("No concepts found while computing reference means.")

    global_node_mean = (node_sum / graph_count).detach().cpu().numpy()
    global_edge_mean = None if edge_sum is None else (edge_sum / graph_count).detach().cpu().numpy()

    print(f"Reference node mean shape = {global_node_mean.shape}")
    if global_edge_mean is not None:
        print(f"Reference edge mean shape = {global_edge_mean.shape}")
    return global_node_mean, global_edge_mean


def ablate_node_concepts(
    h_node_graph: torch.Tensor,
    mode: str,
    global_mean_vec: Optional[np.ndarray],
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    h_node_graph shape: [B, N, K_node]
    global_mean mode expects reference-set per-channel mean shape [N, K_node].
    """
    if mode == "none":
        return h_node_graph

    if mode == "global_mean":
        if global_mean_vec is None:
            raise ValueError("global_mean node ablation requested but global_mean_vec is None")
        mean_t = torch.as_tensor(global_mean_vec, dtype=h_node_graph.dtype, device=h_node_graph.device)
        if tuple(mean_t.shape) != tuple(h_node_graph.shape[1:]):
            raise ValueError(
                f"Node global_mean shape {tuple(mean_t.shape)} must match "
                f"h_node_graph.shape[1:]={tuple(h_node_graph.shape[1:])}"
            )
        return mean_t.unsqueeze(0).expand_as(h_node_graph)

    if mode == "per_graph_mean":
        per_graph_mean = h_node_graph.mean(dim=1, keepdim=True)
        return per_graph_mean.expand_as(h_node_graph)

    if mode == "shuffle_within_graph":
        out = h_node_graph.clone()
        B, N, _ = out.shape
        for b in range(B):
            perm = torch.as_tensor(rng.permutation(N), device=out.device)
            out[b] = out[b, perm]
        return out

    if mode == "zero":
        return torch.zeros_like(h_node_graph)

    raise ValueError(f"Unknown node ablation mode: {mode}")


def ablate_edge_concepts(
    h_edge_graph: torch.Tensor,
    mode: str,
    global_mean_vec: Optional[np.ndarray],
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    h_edge_graph shape: [B, E, K_edge]
    global_mean mode expects reference-set per-edge mean shape [E, K_edge].
    """
    if mode == "none":
        return h_edge_graph

    if mode == "global_mean":
        if global_mean_vec is None:
            raise ValueError("global_mean edge ablation requested but global_mean_vec is None")
        mean_t = torch.as_tensor(global_mean_vec, dtype=h_edge_graph.dtype, device=h_edge_graph.device)
        if tuple(mean_t.shape) != tuple(h_edge_graph.shape[1:]):
            raise ValueError(
                f"Edge global_mean shape {tuple(mean_t.shape)} must match "
                f"h_edge_graph.shape[1:]={tuple(h_edge_graph.shape[1:])}"
            )
        return mean_t.unsqueeze(0).expand_as(h_edge_graph)

    if mode == "per_graph_mean":
        per_graph_mean = h_edge_graph.mean(dim=1, keepdim=True)
        return per_graph_mean.expand_as(h_edge_graph)

    if mode == "shuffle_within_graph":
        out = h_edge_graph.clone()
        B, E, _ = out.shape
        for b in range(B):
            perm = torch.as_tensor(rng.permutation(E), device=out.device)
            out[b] = out[b, perm]
        return out

    if mode == "zero":
        return torch.zeros_like(h_edge_graph)

    raise ValueError(f"Unknown edge ablation mode: {mode}")


def _assert_same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch: {name_a}{tuple(a.shape)} vs {name_b}{tuple(b.shape)}")


def _check_batched_node_layout(batch, h_node: torch.Tensor) -> None:
    """
    Check that fixed EEG graph nodes are contiguous and equally sized per graph.
    """
    y = batch.y.view(-1)
    batch_size_actual = int(y.shape[0])

    if h_node.shape[0] % batch_size_actual != 0:
        raise RuntimeError(f"Node count {h_node.shape[0]} is not divisible by batch size {batch_size_actual}")

    node_counts = torch.bincount(batch.batch, minlength=batch_size_actual)
    if not torch.all(node_counts == node_counts[0]):
        raise RuntimeError(f"Unequal node counts per graph: {node_counts.detach().cpu().tolist()}")


def _check_batched_graph_layout(batch, h_node: torch.Tensor, h_edge: torch.Tensor, edge_batch: torch.Tensor) -> None:
    """
    This script reshapes [total_items, K] to [B, items_per_graph, K]. That is safe
    for PyG batches when every graph has the same number of nodes/edges and PyG has
    concatenated graphs contiguously, which is the case for this fixed EEG graph.
    """
    y = batch.y.view(-1)
    batch_size_actual = int(y.shape[0])

    if h_node.shape[0] % batch_size_actual != 0:
        raise RuntimeError(f"Node count {h_node.shape[0]} is not divisible by batch size {batch_size_actual}")
    if h_edge.shape[0] % batch_size_actual != 0:
        raise RuntimeError(f"Edge count {h_edge.shape[0]} is not divisible by batch size {batch_size_actual}")

    node_counts = torch.bincount(batch.batch, minlength=batch_size_actual)
    edge_counts = torch.bincount(edge_batch, minlength=batch_size_actual)
    if not torch.all(node_counts == node_counts[0]):
        raise RuntimeError(f"Unequal node counts per graph: {node_counts.detach().cpu().tolist()}")
    if not torch.all(edge_counts == edge_counts[0]):
        raise RuntimeError(f"Unequal edge counts per graph: {edge_counts.detach().cpu().tolist()}")


def _compute_relevance(
    model: torch.nn.Module,
    model_kind: str,
    batch,
    h_node: torch.Tensor,
    h_edge: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Return theta_node, theta_edge for a SENN concept variant.
    - senn_rawx: theta = theta(x), recomputed from ablated identity concepts when passed in.
    - senn_fixed: theta = theta(x), generated from raw EEG.
    - senn_fixedconcepttheta: theta = theta(h), generated from fixed concepts.
    """
    model_kind = normalize_model_kind(model_kind)
    if model_kind == "senn_rawx":
        raw_for_relevance = _identity_concepts_to_raw(model, h_node)
        theta_node = model.relevance(raw_for_relevance, batch.edge_index, batch.batch)
        theta_node = model.upscaler(theta_node)
        theta_edge = None
    elif model_kind == "senn_fixed":
        theta_node, theta_edge = model.relevance(batch.x, batch.edge_index, batch.batch)
    elif model_kind == "senn_fixedconcepttheta":
        if h_edge is None:
            raise ValueError("SENN-FC-theta(h) requires edge concepts for relevance computation")
        rel_out = model.relevance(h_node, h_edge, batch.edge_index, batch.batch)
        theta_node, theta_edge = rel_out[0], rel_out[1]
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r}")

    _assert_same_shape("h_node", h_node, "theta_node", theta_node)
    if h_edge is not None and theta_edge is not None:
        _assert_same_shape("h_edge", h_edge, "theta_edge", theta_edge)
    return theta_node, theta_edge


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auprg(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(auprg(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "auprg": safe_auprg(y_true, y_prob),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f2": float(fbeta_score(y_true, y_pred, zero_division=0,beta=2)),
        "positive_rate": float(y_pred.mean()),
    }


@torch.no_grad()
def evaluate_ablation(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    threshold: float,
    branch: str,
    mode: str,
    global_node_mean_vec: Optional[np.ndarray],
    global_edge_mean_vec: Optional[np.ndarray],
    rng: np.random.Generator,
    model_kind: str,
    ablation_path: str,
) -> Dict[str, np.ndarray | Dict[str, float]]:
    """
    branch: "node", "edge", or "both".

    MODEL_KIND="senn_rawx" / SENN-IC:
        ABLATION_PATH="full_path": recompute theta from ablated identity concepts.

    MODEL_KIND="senn_fixed" / SENN-FC-theta(x):
        theta is always computed from raw EEG x and kept fixed while h is ablated.

    MODEL_KIND="senn_fixedconcepttheta" / SENN-FC-theta(h):
        ABLATION_PATH="carrier_only": theta_real = theta(h_real), aggregate h_ab with theta_real.
        ABLATION_PATH="full_path":    theta_ab   = theta(h_ab),   aggregate h_ab with theta_ab.
    """
    model_kind = normalize_model_kind(model_kind)
    ablation_path = str(ablation_path).strip().lower()
    if branch not in {"node", "edge", "both"}:
        raise ValueError('branch must be "node", "edge", or "both"')
    if ablation_path not in {"carrier_only", "full_path"}:
        raise ValueError("ablation_path must be 'carrier_only' or 'full_path'")
    if branch == "edge" and not model_has_edge_concepts(model_kind):
        raise ValueError("SENN-IC has no edge concepts to ablate")

    loader = DataLoader(
        dataset,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEM,
        num_workers=NUM_WORKERS,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    model = model.to(device)
    model.eval()

    collected: Dict[str, List[np.ndarray]] = {
        "y_true": [],
        "y_prob": [],
        "logit": [],
        "logit_node": [],
        "logit_edge": [],
        "node_ratio": [],
        "edge_ratio": [],
        "mean_h_node": [],
        "mean_abs_theta_node": [],
        "mean_abs_f_node": [],
        "mean_h_edge": [],
        "mean_abs_theta_edge": [],
        "mean_abs_f_edge": [],
    }

    print(f"... evaluating {branch} ablation mode: {mode} | model={model_kind} | path={ablation_path} ...")
    for batch in tqdm(loader):
        batch = batch.to(device, non_blocking=True)

        h_node_real, h_edge_real = _conceptize_batch(model, model_kind, batch)
        batch_size_actual = int(batch.y.view(-1).shape[0])
        edge_batch = None

        if h_edge_real is not None:
            edge_batch = batch.batch[batch.edge_index[0]]
            if not torch.equal(batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]]):
                raise RuntimeError("Cross-graph edges detected in batched edge_index.")
            _check_batched_graph_layout(batch, h_node_real, h_edge_real, edge_batch)
        else:
            _check_batched_node_layout(batch, h_node_real)

        h_node_graph = _reshape_per_graph(h_node_real, batch_size_actual)
        h_edge_graph = None if h_edge_real is None else _reshape_per_graph(h_edge_real, batch_size_actual)

        apply_node = branch in {"node", "both"}
        apply_edge = h_edge_graph is not None and branch in {"edge", "both"}

        if apply_node:
            h_node_graph_ab = ablate_node_concepts(
                h_node_graph=h_node_graph,
                mode=mode,
                global_mean_vec=global_node_mean_vec,
                rng=rng,
            )
        else:
            h_node_graph_ab = h_node_graph

        if h_edge_graph is None:
            h_edge_graph_ab = None
        elif apply_edge:
            h_edge_graph_ab = ablate_edge_concepts(
                h_edge_graph=h_edge_graph,
                mode=mode,
                global_mean_vec=global_edge_mean_vec,
                rng=rng,
            )
        else:
            h_edge_graph_ab = h_edge_graph

        h_node_ab = h_node_graph_ab.reshape_as(h_node_real)
        h_edge_ab = None if h_edge_graph_ab is None else h_edge_graph_ab.reshape_as(h_edge_real)

        # Choose theta path.
        if model_kind in {"senn_rawx", "senn_fixedconcepttheta"} and ablation_path == "full_path":
            # theta_ab = theta(h_ab)
            theta_node, theta_edge = _compute_relevance(model, model_kind, batch, h_node_ab, h_edge_ab)
        else:
            # carrier-only for theta(h): theta_real = theta(h_real)
            # theta(x) model: theta is independent of h, so this is the only meaningful path.
            theta_node, theta_edge = _compute_relevance(model, model_kind, batch, h_node_real, h_edge_real)

        theta_node_graph = _reshape_per_graph(theta_node, batch_size_actual)
        theta_edge_graph = None if theta_edge is None else _reshape_per_graph(theta_edge, batch_size_actual)

        if model_kind == "senn_rawx":
            aggr_out = model.aggregrator(
                h_x=h_node_ab,
                theta_x=theta_node,
                batch=batch.batch,
            )
        else:
            aggr_out = model.aggregrator(
                h_node=h_node_ab,
                theta_node=theta_node,
                batch_node=batch.batch,
                h_edge=h_edge_ab,
                theta_edge=theta_edge,
                batch_edge=edge_batch,
            )

        prob = aggr_out["prob"].detach().cpu().numpy().ravel()
        logit = aggr_out["logit"].detach().cpu().numpy().ravel()
        logit_node = aggr_out.get("logit_node", aggr_out["logit"]).detach().cpu().numpy().ravel()
        if "logit_edge" in aggr_out:
            logit_edge = aggr_out["logit_edge"].detach().cpu().numpy().ravel()
        else:
            logit_edge = np.zeros_like(logit)
        y_true = batch.y.view(-1).detach().cpu().numpy().ravel()

        f_node_graph = h_node_graph_ab * theta_node_graph

        mean_h_node = h_node_graph_ab.mean(dim=(1, 2)).detach().cpu().numpy().ravel()
        mean_abs_theta_node = theta_node_graph.abs().mean(dim=(1, 2)).detach().cpu().numpy().ravel()
        mean_abs_f_node = f_node_graph.abs().mean(dim=(1, 2)).detach().cpu().numpy().ravel()

        if h_edge_graph_ab is not None and theta_edge_graph is not None:
            f_edge_graph = h_edge_graph_ab * theta_edge_graph
            mean_h_edge = h_edge_graph_ab.mean(dim=(1, 2)).detach().cpu().numpy().ravel()
            mean_abs_theta_edge = theta_edge_graph.abs().mean(dim=(1, 2)).detach().cpu().numpy().ravel()
            mean_abs_f_edge = f_edge_graph.abs().mean(dim=(1, 2)).detach().cpu().numpy().ravel()
        else:
            mean_h_edge = np.zeros(batch_size_actual, dtype=float)
            mean_abs_theta_edge = np.zeros(batch_size_actual, dtype=float)
            mean_abs_f_edge = np.zeros(batch_size_actual, dtype=float)

        denom = np.abs(logit_node) + np.abs(logit_edge) + 1e-12
        node_ratio = np.abs(logit_node) / denom
        edge_ratio = np.abs(logit_edge) / denom

        collected["y_true"].append(y_true)
        collected["y_prob"].append(prob)
        collected["logit"].append(logit)
        collected["logit_node"].append(logit_node)
        collected["logit_edge"].append(logit_edge)
        collected["node_ratio"].append(node_ratio)
        collected["edge_ratio"].append(edge_ratio)
        collected["mean_h_node"].append(mean_h_node)
        collected["mean_abs_theta_node"].append(mean_abs_theta_node)
        collected["mean_abs_f_node"].append(mean_abs_f_node)
        collected["mean_h_edge"].append(mean_h_edge)
        collected["mean_abs_theta_edge"].append(mean_abs_theta_edge)
        collected["mean_abs_f_edge"].append(mean_abs_f_edge)

    out_np: Dict[str, np.ndarray] = {
        key: np.concatenate(values, axis=0) for key, values in collected.items()
    }
    out_np["metrics"] = compute_metrics(out_np["y_true"], out_np["y_prob"], threshold)
    return out_np


def attach_baseline_shifts(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]]) -> None:
    """Add per-sample baseline/delta arrays and aggregate shift metrics in-place."""
    if "none" not in results:
        raise KeyError("Ablation results must include mode='none' to compute shifts")

    baseline_logit = np.asarray(results["none"]["logit"], dtype=float)
    baseline_prob = np.asarray(results["none"]["y_prob"], dtype=float)

    for mode, res in results.items():
        logit = np.asarray(res["logit"], dtype=float)
        prob = np.asarray(res["y_prob"], dtype=float)
        if logit.shape != baseline_logit.shape:
            raise RuntimeError(f"Logit shape mismatch for mode={mode}: {logit.shape} vs {baseline_logit.shape}")
        if prob.shape != baseline_prob.shape:
            raise RuntimeError(f"Probability shape mismatch for mode={mode}: {prob.shape} vs {baseline_prob.shape}")

        # Distribution convention: ablated minus unablated.
        delta_logit = logit - baseline_logit
        delta_prob = prob - baseline_prob

        # Interpretation convention: positive drop means evidence decreased under ablation.
        logit_drop = baseline_logit - logit
        prob_drop = baseline_prob - prob

        res["baseline_logit"] = baseline_logit.copy()
        res["baseline_y_prob"] = baseline_prob.copy()
        res["delta_logit"] = delta_logit
        res["logit_drop"] = logit_drop
        res["delta_prob"] = delta_prob
        res["prob_drop"] = prob_drop

        m = res["metrics"]
        m["mean_delta_logit"] = float(np.mean(delta_logit))
        m["median_delta_logit"] = float(np.median(delta_logit))
        m["mean_abs_delta_logit"] = float(np.mean(np.abs(delta_logit)))
        m["mean_logit_drop"] = float(np.mean(logit_drop))
        m["mean_delta_prob"] = float(np.mean(delta_prob))
        m["median_delta_prob"] = float(np.median(delta_prob))
        m["mean_abs_delta_prob"] = float(np.mean(np.abs(delta_prob)))
        m["mean_prob_drop"] = float(np.mean(prob_drop))


def _ensure_dir(path: str) -> str:
    """Create a directory if needed and return it as a string path."""
    os.makedirs(path, exist_ok=True)
    return path


def _save_plot_png_pdf(fig, save_path: str, dpi: int = 300) -> None:
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")


def save_run_metadata(
    out_dir: str,
    ckpt_path: str,
    threshold: float,
    global_node_mean_vec: np.ndarray,
    global_edge_mean_vec: Optional[np.ndarray],
) -> str:
    """Save run-level metadata once at the results root."""
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "run_metadata.json")
    metadata = {
        "model_kind": MODEL_KIND,
        "model_dir": MODEL_DIR,
        "model_subdir": MODEL_SUBDIR,
        "ckpt_name": CKPT_NAME,
        "ckpt_path": ckpt_path,
        "fold": FOLD,
        "ablation_path": ABLATION_PATH,
        "reference_split": REFERENCE_SPLIT,
        "eval_split": EVAL_SPLIT,
        "allow_reference_eval_leakage": ALLOW_REFERENCE_EVAL_LEAKAGE,
        "use_full_eval_split": USE_FULL_EVAL_SPLIT,
        "eval_subset_size": EVAL_SUBSET_SIZE,
        "loader_batch_size": LOADER_BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "ablation_modes": ABLATION_MODES,
        "threshold": float(threshold),
        "global_node_mean_vec": np.asarray(global_node_mean_vec, dtype=float).tolist(),
        "global_edge_mean_vec": (
            None
            if global_edge_mean_vec is None
            else np.asarray(global_edge_mean_vec, dtype=float).tolist()
        ),
        "delta_logit_definition": "logit_ablation - logit_baseline_none",
        "logit_drop_definition": "logit_baseline_none - logit_ablation",
        "delta_prob_definition": "prob_ablation - prob_baseline_none",
        "prob_drop_definition": "prob_baseline_none - prob_ablation",
        "output_structure": {
            "root": "run-level metadata and file manifest",
            "node": "node-only concept ablations",
            "edge": "edge-only concept ablations",
            "both": "simultaneous node+edge concept ablations",
            "branch_files": [
                "summary.csv",
                "summary.json",
                "arrays.npz",
                "barplots.png",
                "ratios.png",
                "prob_shift.png",
                "logit_shift.png",
            ],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def save_created_files_manifest(paths: List[str], out_dir: str) -> str:
    """Save a plain-text manifest of all files created in this run."""
    _ensure_dir(out_dir)
    manifest_path = os.path.join(out_dir, "created_files.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(f"{os.path.abspath(path)}\n")
    return manifest_path


def save_results_csv(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], out_dir: str, branch: str) -> str:
    import csv

    _ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "summary.csv")
    metric_names = [
        "auroc", "auprc", "auprg", "kappa", "f2", "positive_rate",
        "mean_delta_prob", "mean_abs_delta_prob", "mean_prob_drop",
        "mean_delta_logit", "mean_abs_delta_logit", "mean_logit_drop",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode",
            *metric_names,
            "mean_node_ratio",
            "mean_edge_ratio",
            "mean_h_node",
            "mean_abs_theta_node",
            "mean_abs_f_node",
            "mean_h_edge",
            "mean_abs_theta_edge",
            "mean_abs_f_edge",
        ])
        for mode, res in results.items():
            m = res["metrics"]
            writer.writerow([
                mode,
                *(m.get(name, float("nan")) for name in metric_names),
                float(np.mean(res["node_ratio"])),
                float(np.mean(res["edge_ratio"])),
                float(np.mean(res["mean_h_node"])),
                float(np.mean(res["mean_abs_theta_node"])),
                float(np.mean(res["mean_abs_f_node"])),
                float(np.mean(res["mean_h_edge"])),
                float(np.mean(res["mean_abs_theta_edge"])),
                float(np.mean(res["mean_abs_f_edge"])),
            ])
    return csv_path


def save_results_json(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], out_dir: str, branch: str) -> str:
    _ensure_dir(out_dir)
    json_path = os.path.join(out_dir, "summary.json")
    serializable = {
        "metadata": {
            "model_kind": MODEL_KIND,
            "ablation_path": ABLATION_PATH,
            "branch": branch,
            "reference_split": REFERENCE_SPLIT,
            "eval_split": EVAL_SPLIT,
            "delta_logit_definition": "logit_ablation - logit_baseline_none",
            "logit_drop_definition": "logit_baseline_none - logit_ablation",
            "delta_prob_definition": "prob_ablation - prob_baseline_none",
        },
        "modes": {},
    }
    for mode, res in results.items():
        serializable["modes"][mode] = {
            "metrics": res["metrics"],
            "mean_node_ratio": float(np.mean(res["node_ratio"])),
            "mean_edge_ratio": float(np.mean(res["edge_ratio"])),
            "mean_h_node": float(np.mean(res["mean_h_node"])),
            "mean_abs_theta_node": float(np.mean(res["mean_abs_theta_node"])),
            "mean_abs_f_node": float(np.mean(res["mean_abs_f_node"])),
            "mean_h_edge": float(np.mean(res["mean_h_edge"])),
            "mean_abs_theta_edge": float(np.mean(res["mean_abs_theta_edge"])),
            "mean_abs_f_edge": float(np.mean(res["mean_abs_f_edge"])),
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return json_path


def summarize_branch_results(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Keep only fold-level scalar summaries needed for cross-fold reporting."""
    summary: Dict[str, Dict[str, float]] = {}
    for mode, res in results.items():
        metrics = {
            key: float(value)
            for key, value in res["metrics"].items()
        }
        metrics.update({
            "mean_node_ratio": float(np.mean(res["node_ratio"])),
            "mean_edge_ratio": float(np.mean(res["edge_ratio"])),
            "mean_h_node": float(np.mean(res["mean_h_node"])),
            "mean_abs_theta_node": float(np.mean(res["mean_abs_theta_node"])),
            "mean_abs_f_node": float(np.mean(res["mean_abs_f_node"])),
            "mean_h_edge": float(np.mean(res["mean_h_edge"])),
            "mean_abs_theta_edge": float(np.mean(res["mean_abs_theta_edge"])),
            "mean_abs_f_edge": float(np.mean(res["mean_abs_f_edge"])),
        })
        summary[mode] = metrics
    return summary


def _stats_across_folds(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "error": float("nan"), "n": 0}

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    error = float(std / np.sqrt(n)) if CROSS_FOLD_ERROR == "sem" else std
    return {"mean": mean, "std": std, "error": error, "n": n}


def _ordered_fold_ids(fold_summaries: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    return sorted(
        fold_summaries.keys(),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )


def _ordered_summary_modes(fold_summaries: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    seen = set()
    for fold_summary in fold_summaries.values():
        seen.update(fold_summary.keys())
    ordered = [mode for mode in ABLATION_MODES if mode in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def _ordered_summary_metrics(fold_summaries: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    seen = set()
    for fold_summary in fold_summaries.values():
        for mode_summary in fold_summary.values():
            seen.update(mode_summary.keys())

    preferred = [
        "auroc", "auprc", "auprg", "kappa", "f2", "positive_rate",
        "mean_delta_prob", "mean_abs_delta_prob", "mean_prob_drop",
        "mean_delta_logit", "mean_abs_delta_logit", "mean_logit_drop",
        "mean_node_ratio", "mean_edge_ratio",
        "mean_h_node", "mean_abs_theta_node", "mean_abs_f_node",
        "mean_h_edge", "mean_abs_theta_edge", "mean_abs_f_edge",
    ]
    ordered = [metric for metric in preferred if metric in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def final_barplot_filename(branch: str) -> str:
    branch_label = {
        "both": "Both",
        "node": "Node",
        "edge": "Edge",
    }.get(branch, str(branch).capitalize())
    return f"{MODEL_OUTPUT_NAME}_Ablation{branch_label}_{ABLATION_PATH_OUTPUT_NAME}.png"


def _aggregate_branch_summary(
    fold_summaries: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    fold_ids = _ordered_fold_ids(fold_summaries)
    modes = _ordered_summary_modes(fold_summaries)
    metrics = _ordered_summary_metrics(fold_summaries)

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in modes:
        aggregated[mode] = {}
        for metric in metrics:
            values = [
                fold_summaries[fold].get(mode, {}).get(metric, float("nan"))
                for fold in fold_ids
            ]
            aggregated[mode][metric] = _stats_across_folds(values)
    return aggregated


def _aggregate_branch_delta_summary(
    fold_summaries: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    fold_ids = _ordered_fold_ids(fold_summaries)
    modes = _ordered_summary_modes(fold_summaries)
    metrics = _ordered_summary_metrics(fold_summaries)

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in modes:
        aggregated[mode] = {}
        for metric in metrics:
            values = []
            for fold in fold_ids:
                fold_summary = fold_summaries[fold]
                mode_value = fold_summary.get(mode, {}).get(metric, float("nan"))
                baseline_value = fold_summary.get("none", {}).get(metric, float("nan"))
                values.append(mode_value - baseline_value)
            aggregated[mode][metric] = _stats_across_folds(values)
    return aggregated


def save_cross_fold_summary_csv(
    fold_summaries: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: str,
    branch: str,
) -> str:
    import csv

    _ensure_dir(out_dir)
    stem = os.path.splitext(final_barplot_filename(branch))[0]
    csv_path = os.path.join(out_dir, f"{stem}_summary.csv")
    metrics = _ordered_summary_metrics(fold_summaries)
    aggregated = _aggregate_branch_summary(fold_summaries)

    fieldnames = ["branch", "mode", "folds"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_{CROSS_FOLD_ERROR}"])
        if CROSS_FOLD_ERROR != "std":
            fieldnames.append(f"{metric}_std")
        fieldnames.append(f"{metric}_n")

    fold_label = ",".join(_ordered_fold_ids(fold_summaries))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode, mode_stats in aggregated.items():
            row = {"branch": branch, "mode": mode, "folds": fold_label}
            for metric in metrics:
                stats = mode_stats[metric]
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_{CROSS_FOLD_ERROR}"] = stats["error"]
                if CROSS_FOLD_ERROR != "std":
                    row[f"{metric}_std"] = stats["std"]
                row[f"{metric}_n"] = stats["n"]
            writer.writerow(row)
    return csv_path


def save_cross_fold_summary_json(
    fold_summaries: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: str,
    branch: str,
) -> str:
    _ensure_dir(out_dir)
    stem = os.path.splitext(final_barplot_filename(branch))[0]
    json_path = os.path.join(out_dir, f"{stem}_summary.json")
    payload = {
        "metadata": {
            "model_kind": MODEL_KIND,
            "ablation_path": ABLATION_PATH,
            "branch": branch,
            "folds": _ordered_fold_ids(fold_summaries),
            "reference_split": REFERENCE_SPLIT,
            "eval_split": EVAL_SPLIT,
            "error_bar": CROSS_FOLD_ERROR,
            "error_bar_definition": "standard error of the fold means" if CROSS_FOLD_ERROR == "sem" else "standard deviation of the fold means",
        },
        "modes": _aggregate_branch_summary(fold_summaries),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return json_path


def plot_cross_fold_metric_bars(
    fold_summaries: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: str,
    branch: str,
) -> str:
    _ensure_dir(out_dir)
    modes = _ordered_summary_modes(fold_summaries)
    metric_names = ["auroc", "auprc", "kappa", "f2"]
    aggregated = _aggregate_branch_summary(fold_summaries)

    fig, axes = plt.subplots(2, 2, figsize=ABLATION_2X2_FIGSIZE, squeeze=False)
    axes = axes.ravel()

    x = np.arange(len(modes))
    for ax, metric in zip(axes, metric_names):
        vals = [aggregated[mode][metric]["mean"] for mode in modes]
        errs = [aggregated[mode][metric]["error"] for mode in modes]
        ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=20, ha="right")
        ax.set_title(metric.upper())
        ax.grid(alpha=0.25, axis="y", linewidth=0.4)
        ax.set_ylim(0, 1)

    n_folds = len(_ordered_fold_ids(fold_summaries))
    error_label = "SEM" if CROSS_FOLD_ERROR == "sem" else "SD"
    fig.suptitle(
        f"Fixed {branch} concept ablation across {n_folds} folds: mean +/- {error_label}",
        y=1.01,
    )
    fig.tight_layout()

    save_path = os.path.join(out_dir, final_barplot_filename(branch))
    _save_plot_png_pdf(fig, save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def _metric_label(metric: str) -> str:
    return {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "auprg": "AUPRG",
        "kappa": "Kappa",
        "f2": "F2",
    }.get(metric, metric.upper())


def _metric_delta_label(metric: str) -> str:
    return f"$\\Delta$ {_metric_label(metric)}"


def _set_metric_ylim(ax, vals: List[float], errs: List[float], metric: str) -> None:
    vals_arr = np.asarray(vals, dtype=float)
    errs_arr = np.asarray(errs, dtype=float)
    errs_arr = np.where(np.isfinite(errs_arr), errs_arr, 0.0)
    finite = np.isfinite(vals_arr)
    if not finite.any():
        ax.set_ylim(0, 1)
        return

    lower = float(np.nanmin(vals_arr[finite] - errs_arr[finite]))
    upper = float(np.nanmax(vals_arr[finite] + errs_arr[finite]))
    lower = min(0.0, lower - 0.03) if metric in {"auprg", "kappa"} else 0.0
    upper = max(1.0, upper + 0.03)
    ax.set_ylim(lower, upper)


def _set_delta_metric_ylim(ax, vals: List[float], errs: List[float]) -> None:
    vals_arr = np.asarray(vals, dtype=float)
    errs_arr = np.asarray(errs, dtype=float)
    errs_arr = np.where(np.isfinite(errs_arr), errs_arr, 0.0)
    finite = np.isfinite(vals_arr)
    if not finite.any():
        ax.set_ylim(-1, 1)
        return

    lower = float(np.nanmin(vals_arr[finite] - errs_arr[finite]))
    upper = float(np.nanmax(vals_arr[finite] + errs_arr[finite]))
    pad = max(0.03, 0.08 * max(abs(lower), abs(upper), 1e-12))
    ax.set_ylim(min(0.0, lower - pad), max(0.0, upper + pad))


def combined_barplot_filename(branch: str) -> str:
    branch_label = {
        "both": "AllConcepts",
        "node": "NodeConcepts",
        "edge": "EdgeConcepts",
    }.get(branch, str(branch).capitalize())
    return f"SENN-IC_SENN-FC-x_SENN-FC-h_{branch_label}_{ABLATION_PATH_OUTPUT_NAME}.png"


def combined_delta_barplot_filename(branch: str) -> str:
    branch_label = {
        "both": "AllConcepts",
        "node": "NodeConcepts",
        "edge": "EdgeConcepts",
    }.get(branch, str(branch).capitalize())
    return f"SENN-IC_SENN-FC-x_SENN-FC-h_{branch_label}_{ABLATION_PATH_OUTPUT_NAME}_delta.png"


def plot_combined_cross_model_metric_bars(
    model_fold_summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: str,
    branch: str,
) -> str:
    _ensure_dir(out_dir)
    model_labels = [run["label"] for run in MODEL_RUNS if run["label"] in model_fold_summaries]
    seen_modes = set()
    for fold_summaries in model_fold_summaries.values():
        seen_modes.update(_ordered_summary_modes(fold_summaries))
    modes = [mode for mode in ABLATION_MODES if mode in seen_modes]
    modes.extend(sorted(seen_modes.difference(modes)))

    metric_names = ["auroc", "auprc", "kappa", "f2"]
    aggregated_by_model = {
        label: _aggregate_branch_summary(model_fold_summaries[label])
        for label in model_labels
    }

    fig, axes = plt.subplots(2, 2, figsize=ABLATION_2X2_FIGSIZE, squeeze=False)
    axes = axes.ravel()

    x = np.arange(len(modes))
    width = min(0.24, 0.8 / max(len(model_labels), 1))
    offsets = (np.arange(len(model_labels)) - (len(model_labels) - 1) / 2.0) * width

    for ax, metric in zip(axes, metric_names):
        all_vals: List[float] = []
        all_errs: List[float] = []
        for model_idx, (offset, label) in enumerate(zip(offsets, model_labels)):
            aggregated = aggregated_by_model[label]
            vals = [
                aggregated.get(mode, {}).get(metric, {}).get("mean", float("nan"))
                for mode in modes
            ]
            errs = [
                aggregated.get(mode, {}).get(metric, {}).get("error", float("nan"))
                for mode in modes
            ]
            ax.bar(
                x + offset,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
                color=ABLATION_MODEL_COLORS[model_idx % len(ABLATION_MODEL_COLORS)],
                label=label,
            )
            all_vals.extend(vals)
            all_errs.extend(errs)

        ax.set_xticks(x)
        mode_names = {"none": "None",
                  "zero": "Zero",
                  "global_mean": "Global mean",
                  "per_graph_mean": "Per graph mean",
                  "shuffle_within_graph": "Shuffle within graph"}
        ax.set_xticklabels([mode_names[m] for m in modes], rotation=20, ha="right")
        
        ax.set_title(_metric_label(metric))
        ax.grid(alpha=0.25, axis="y", linewidth=0.4)
        _set_metric_ylim(ax, all_vals, all_errs, metric)
        ax.set_ylim(0,1)

    error_label = "SEM" if CROSS_FOLD_ERROR == "sem" else "SD"
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle(
        "Concept ablation performance",
        y=0.985,
        fontweight="normal",
    )
    if not PANEL_READY_EXPORT:
        fig.text(
            0.5,
            0.905,
            f"Bars show mean performance across folds; error bars indicate {error_label}.",
            ha="center",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
        )
        layout_top = 0.89
    else:
        layout_top = 0.91
    fig.tight_layout(rect=[0, 0, 1, layout_top])

    save_path = os.path.join(out_dir, combined_barplot_filename(branch))
    _save_plot_png_pdf(fig, save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_combined_cross_model_metric_delta_bars(
    model_fold_summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: str,
    branch: str,
) -> str:
    _ensure_dir(out_dir)
    model_labels = [run["label"] for run in MODEL_RUNS if run["label"] in model_fold_summaries]
    seen_modes = set()
    for fold_summaries in model_fold_summaries.values():
        seen_modes.update(_ordered_summary_modes(fold_summaries))
    modes = [mode for mode in ABLATION_MODES if mode in seen_modes]
    modes.extend(sorted(seen_modes.difference(modes)))

    metric_names = ["auroc", "auprc", "kappa", "f2"]
    aggregated_by_model = {
        label: _aggregate_branch_delta_summary(model_fold_summaries[label])
        for label in model_labels
    }

    fig, axes = plt.subplots(2, 2, figsize=ABLATION_2X2_FIGSIZE, squeeze=False)
    axes = axes.ravel()

    x = np.arange(len(modes))
    width = min(0.24, 0.8 / max(len(model_labels), 1))
    offsets = (np.arange(len(model_labels)) - (len(model_labels) - 1) / 2.0) * width

    for ax, metric in zip(axes, metric_names):
        all_vals: List[float] = []
        all_errs: List[float] = []
        for model_idx, (offset, label) in enumerate(zip(offsets, model_labels)):
            aggregated = aggregated_by_model[label]
            vals = [
                aggregated.get(mode, {}).get(metric, {}).get("mean", float("nan"))
                for mode in modes
            ]
            errs = [
                aggregated.get(mode, {}).get(metric, {}).get("error", float("nan"))
                for mode in modes
            ]
            ax.bar(
                x + offset,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                error_kw={"elinewidth": 1, "capthick": 1},
                color=ABLATION_MODEL_COLORS[model_idx % len(ABLATION_MODEL_COLORS)],
                label=label,
            )
            all_vals.extend(vals)
            all_errs.extend(errs)

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_xticks(x)
        mode_names = {"none": "None",
                  "zero": "Zero",
                  "global_mean": "Global mean",
                  "per_graph_mean": "Per graph mean",
                  "shuffle_within_graph": "Shuffle within graph"}
        ax.set_xticklabels([mode_names[m] for m in modes], rotation=20, ha="right")

        ax.set_title(_metric_delta_label(metric))
        ax.grid(alpha=0.25, axis="y", linewidth=0.4)
        _set_delta_metric_ylim(ax, all_vals, all_errs)
        ax.set_ylim(bottom=-1)

    error_label = "SEM" if CROSS_FOLD_ERROR == "sem" else "SD"

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle(
        "Concept ablation effect",
        y=0.985,
        fontweight="normal",
    )
    if not PANEL_READY_EXPORT:
        fig.text(
            0.5,
            0.905,
            f"Bars show mean change from no ablation across folds; error bars indicate {error_label}.",
            ha="center",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
        )
        layout_top = 0.89
    else:
        layout_top = 0.91
    fig.tight_layout(rect=[0, 0, 1, layout_top])


    save_path = os.path.join(out_dir, combined_delta_barplot_filename(branch))
    _save_plot_png_pdf(fig, save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def save_results_npz(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], out_dir: str, branch: str) -> str:
    """Save per-sample arrays, including baseline logits and delta_logit/logit_drop."""
    _ensure_dir(out_dir)
    npz_path = os.path.join(out_dir, "arrays.npz")
    arrays = {}
    array_keys = [
        "y_true", "y_prob", "logit", "baseline_logit", "baseline_y_prob",
        "delta_logit", "logit_drop", "delta_prob", "prob_drop",
        "logit_node", "logit_edge", "node_ratio", "edge_ratio",
        "mean_h_node", "mean_abs_theta_node", "mean_abs_f_node",
        "mean_h_edge", "mean_abs_theta_edge", "mean_abs_f_edge",
    ]
    for mode, res in results.items():
        safe_mode = mode.replace("/", "_")
        for key in array_keys:
            if key in res:
                arrays[f"{safe_mode}__{key}"] = np.asarray(res[key])
    np.savez_compressed(npz_path, **arrays)
    return npz_path


def plot_metric_bars(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], out_dir: str, branch: str) -> str:
    _ensure_dir(out_dir)
    modes = list(results.keys())
    metric_names = ["auroc", "auprc", "kappa", "f2"]

    fig, axes = plt.subplots(2, 2, figsize=ABLATION_2X2_FIGSIZE, squeeze=False)
    axes = axes.ravel()
    
    for ax, metric in zip(axes, metric_names):
        vals = [results[m]["metrics"][metric] for m in modes]
        ax.bar(np.arange(len(modes)), vals)
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(modes, rotation=20, ha="right")
        ax.set_title(metric.upper())
        ax.grid(alpha=0.25, axis="y", linewidth=0.4)
        
        ax.set_ylim(0, 1)

    fig.suptitle(f"Concept ablation aceross 10 folds: mean +/- SD", y=1.01)
    fig.tight_layout()

    save_path = os.path.join(out_dir, "barplots.png")
    _save_plot_png_pdf(fig, save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_ratio_hist(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], out_dir: str, branch: str) -> str:
    _ensure_dir(out_dir)
    modes = list(results.keys())

    if branch in {"node", "edge"}:
        n_cols = 2
        n_rows = int(np.ceil(len(modes) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(ABLATION_AUX_WIDTH_IN, 2.15 * n_rows),
            squeeze=False,
        )
        axes = axes.ravel()

        ratio_key = f"{branch}_ratio"
        label = rf"|logit_{branch}| / (|logit_node| + |logit_edge|)"

        for ax, mode in zip(axes, modes):
            vals = results[mode][ratio_key]
            y_true = results[mode]["y_true"]
            ax.hist(vals[y_true == 0], bins=60, alpha=0.5, density=True, label="Non-seiz")
            ax.hist(vals[y_true == 1], bins=60, alpha=0.5, density=True, label="Seiz")
            ax.set_title(mode)
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.grid(alpha=0.2, axis="y", linewidth=0.4)
            ax.legend()

        for k in range(len(modes), len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"{branch.capitalize()}-branch dominance under ablation", y=1.01)
    elif branch == "both":
        n_rows = len(modes)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(ABLATION_AUX_WIDTH_IN, 2.1 * n_rows),
            squeeze=False,
        )
        ratio_specs = [
            ("node_ratio", r"|logit_node| / (|logit_node| + |logit_edge|)", "Node dominance"),
            ("edge_ratio", r"|logit_edge| / (|logit_node| + |logit_edge|)", "Edge dominance"),
        ]

        for row, mode in enumerate(modes):
            y_true = results[mode]["y_true"]
            for col, (ratio_key, xlabel, title_prefix) in enumerate(ratio_specs):
                ax = axes[row, col]
                vals = results[mode][ratio_key]
                ax.hist(vals[y_true == 0], bins=60, alpha=0.5, density=True, label="Non-seiz")
                ax.hist(vals[y_true == 1], bins=60, alpha=0.5, density=True, label="Seiz")
                ax.set_title(f"{mode} — {title_prefix}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Density")
                ax.grid(alpha=0.2, axis="y", linewidth=0.4)
                ax.legend()

        fig.suptitle("Node and edge dominance under joint ablation", y=1.01)
    else:
        raise ValueError('branch must be "node", "edge", or "both"')

    fig.tight_layout()
    save_path = os.path.join(out_dir, "ratios.png")
    _save_plot_png_pdf(fig, save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_prob_shift_vs_baseline(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], out_dir: str, branch: str) -> Optional[str]:
    _ensure_dir(out_dir)
    if "none" not in results:
        return None

    baseline_prob = results["none"]["y_prob"]
    modes = [m for m in results.keys() if m != "none"]
    if not modes:
        return None

    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(ABLATION_AUX_WIDTH_IN, 2.0 * len(modes)),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, mode in zip(axes, modes):
        delta = results[mode]["y_prob"] - baseline_prob
        y_true = results[mode]["y_true"]
        ax.hist(delta[y_true == 0], bins=60, alpha=0.5, density=True, label="Non-seiz")
        ax.hist(delta[y_true == 1], bins=60, alpha=0.5, density=True, label="Seiz")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{mode}: probability shift vs baseline")
        ax.set_xlabel(r"p_ablation - p_baseline")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2, axis="y", linewidth=0.4)
        ax.legend()

    fig.tight_layout()
    save_path = os.path.join(out_dir, f"prob_shift.png")
    _save_plot_png_pdf(fig, save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_logit_shift_vs_baseline(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], out_dir: str, branch: str) -> Optional[str]:
    _ensure_dir(out_dir)
    if "none" not in results:
        return None

    modes = [m for m in results.keys() if m != "none"]
    if not modes:
        return None

    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(ABLATION_AUX_WIDTH_IN, 2.0 * len(modes)),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, mode in zip(axes, modes):
        delta = results[mode]["delta_logit"]  # logit_ablation - logit_baseline
        y_true = results[mode]["y_true"]
        ax.hist(delta[y_true == 0], bins=60, alpha=0.5, density=True, label="Non-seiz")
        ax.hist(delta[y_true == 1], bins=60, alpha=0.5, density=True, label="Seiz")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{mode}: logit shift vs baseline")
        ax.set_xlabel(r"delta_logit = logit_ablation - logit_baseline")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2, axis="y", linewidth=0.4)
        ax.legend()

    fig.tight_layout()
    save_path = os.path.join(out_dir, f"logit_shift.png")
    _save_plot_png_pdf(fig, save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def print_summary(results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]], branch: str) -> None:
    print(f"\n===== {branch.upper()} CONCEPT ABLATION SUMMARY =====")
    for mode, res in results.items():
        m = res["metrics"]
        print(
            f"{mode:<20} | "
            f"AUROC={m['auroc']:.4f} | "
            f"AUPRC={m['auprc']:.4f} | "
            f"kappa={m['kappa']:.4f} | "
            f"F2={m['f2']:.4f} | "
            f"mean node ratio={np.mean(res['node_ratio']):.4f} | "
            f"mean edge ratio={np.mean(res['edge_ratio']):.4f}"
        )

    if "none" in results:
        base = results["none"]["metrics"]
        print(f"\n===== {branch.upper()} DROP VS BASELINE (mode='none') =====")
        for mode, res in results.items():
            if mode == "none":
                continue
            m = res["metrics"]
            print(
                f"{mode:<20} | "
                f"ΔAUROC={m['auroc'] - base['auroc']:+.4f} | "
                f"ΔAUPRC={m['auprc'] - base['auprc']:+.4f} | "
                f"Δkappa={m['kappa'] - base['kappa']:+.4f} | "
                f"ΔF2={m['f2'] - base['f2']:+.4f} | "
                f"mean Δprob={m.get('mean_delta_prob', float('nan')):+.5f} | "
                f"mean Δlogit={m.get('mean_delta_logit', float('nan')):+.5f} | "
                f"mean logit_drop={m.get('mean_logit_drop', float('nan')):+.5f}"
            )


def run_branch_study(
    branch: str,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    threshold: float,
    global_node_mean_vec: np.ndarray,
    global_edge_mean_vec: Optional[np.ndarray],
    rng_seed: int,
    model_kind: str,
    ablation_path: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray | Dict[str, float]]], List[str]]:
    results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]] = {}
    for idx, mode in enumerate(ABLATION_MODES):
        # Re-seed per mode for reproducible shuffles.
        rng = np.random.default_rng(rng_seed + idx)
        results[mode] = evaluate_ablation(
            model=model,
            dataset=dataset,
            device=device,
            threshold=threshold,
            branch=branch,
            mode=mode,
            global_node_mean_vec=global_node_mean_vec,
            global_edge_mean_vec=global_edge_mean_vec,
            rng=rng,
            model_kind=model_kind,
            ablation_path=ablation_path,
        )

    attach_baseline_shifts(results)
    print_summary(results, branch=branch)

    created_files: List[str] = []
    return results, created_files


def run_single_fold(
    fold: str,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:
    set_active_fold(fold)
    _ensure_dir(RESULTS_ROOT)
    print(f"\n===== RUNNING {MODEL_OUTPUT_NAME} FOLD {FOLD} =====")
    print("Keeping fold outputs in memory for the final cross-fold plot.")

    ckpt_path = os.path.join(MODEL_DIR, MODEL_SUBDIR, CKPT_NAME)
    ckpt = torch.load(ckpt_path, weights_only=False)
    threshold = float(np.round(ckpt["metrics"]["threshold"], 2))
    norm = None

    model = build_model(ckpt, MODEL_KIND)

    reference_dataset, _ = prepare_split_dataset(
        split=REFERENCE_SPLIT,
        norm=norm,
        use_full=True,
        subset_size=EVAL_SUBSET_SIZE,
    )
    eval_dataset, _ = prepare_split_dataset(
        split=EVAL_SPLIT,
        norm=norm,
        use_full=USE_FULL_EVAL_SPLIT,
        subset_size=EVAL_SUBSET_SIZE,
    )

    global_node_mean_vec, global_edge_mean_vec = compute_reference_means(
        model=model,
        model_kind=MODEL_KIND,
        reference_dataset=reference_dataset,
        device=device,
    )

    all_created_files: List[str] = []
    fold_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    rng_offset = {
        "node": 0,
        "edge": 1000,
        "both": 2000,
    }[COMPARISON_BRANCH]

    branch_results, branch_files = run_branch_study(
        branch=COMPARISON_BRANCH,
        model=model,
        dataset=eval_dataset,
        device=device,
        threshold=threshold,
        global_node_mean_vec=global_node_mean_vec,
        global_edge_mean_vec=global_edge_mean_vec,
        rng_seed=RANDOM_SEED + rng_offset,
        model_kind=MODEL_KIND,
        ablation_path=ABLATION_PATH,
    )
    fold_summaries[COMPARISON_BRANCH] = summarize_branch_results(branch_results)
    all_created_files.extend(branch_files)

    return fold_summaries, all_created_files


def main() -> None:
    validate_config()
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fold_ids = [str(fold) for fold in FOLDS]
    model_fold_summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    all_created_files: List[str] = []

    for model_run in MODEL_RUNS:
        set_active_model_run(model_run)
        model_fold_summaries[MODEL_OUTPUT_NAME] = {}
        print(
            f"\n===== MODEL COMPARISON MEMBER: {MODEL_OUTPUT_NAME} "
            f"({MODEL_KIND}, {ABLATION_PATH}) ====="
        )
        for fold in fold_ids:
            fold_summaries, fold_files = run_single_fold(fold=fold, device=device)
            all_created_files.extend(fold_files)
            model_fold_summaries[MODEL_OUTPUT_NAME][str(fold)] = fold_summaries[COMPARISON_BRANCH]

    _ensure_dir(RESULTS_ROOT)
    print(f"\nSaving final cross-fold figures under: {os.path.abspath(RESULTS_ROOT)}")

    if SAVE_CROSS_FOLD_TABLES:
        print("SAVE_CROSS_FOLD_TABLES is disabled for the combined comparison table in this script version.")
    if SAVE_PLOTS:
        all_created_files.append(
            plot_combined_cross_model_metric_bars(
                model_fold_summaries=model_fold_summaries,
                out_dir=RESULTS_ROOT,
                branch=COMPARISON_BRANCH,
            )
        )
        all_created_files.append(
            plot_combined_cross_model_metric_delta_bars(
                model_fold_summaries=model_fold_summaries,
                out_dir=RESULTS_ROOT,
                branch=COMPARISON_BRANCH,
            )
        )

    print("\nSaved outputs:")
    for path in all_created_files:
        print(path)


if __name__ == "__main__":
    main()
