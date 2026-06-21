import gc
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import Models_senn as Model
import MyUtils_senn_test as MyUtils


# ============================================================
# USER SETTINGS
# ============================================================
LOG = "459967"
FOLD = "5"
MODEL_KIND = "senn"          # "base", "senn", or fixed aliases below
EXPLAINER_TYPE = "focusmap"  # base: "ig" | raw/fixed SENN: "focusmap"
IS_TRIVIAL = False

LOG = "470501"
FOLD = "5"
MODEL_KIND = "senn_fixed"          # "base", "senn", or fixed aliases below
EXPLAINER_TYPE = "focusmap"  # base: "ig" | raw/fixed SENN: "focusmap"
IS_TRIVIAL = False

LOG = "481616"
FOLD = "5"
MODEL_KIND = "senn_fixed"          # "base", "senn", or fixed aliases below
EXPLAINER_TYPE = "focusmap"  # base: "ig" | raw/fixed SENN: "focusmap"
IS_TRIVIAL = True

LOG = "482221" #SENN trivialfixed
MODEL_KIND = "senn_fixed_concepttheta" # note that we use the senn fixed type, but we must specifically build the trivial model, this is now manually done in the build model function
IS_TRIVIAL = False

# LOG = "485352" #SENN trivialfixed
# FOLD = "0"
# MODEL_KIND = "senn" # note that we use the senn fixed type, but we must specifically build the trivial model, this is now manually done in the build model function
# IS_TRIVIAL = False

CKPT_NAME = "best_auprc.pt"

DATA_FOLDER = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets\CV_Folds/"
FOLD = "7"
MODEL_SUBDIR = f"GAT_CV_10_{FOLD}"
MODEL_KIND = "senn_fixed_concepttheta" 
MODEL_DIR = f"./ArchiveModelsMainResults/Saved_models_486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0"
RESULTS_DIR = f"./Results_Performance/Results_486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0"

# MODEL_KIND = "senn_fixed"  
# MODEL_DIR = f"./ArchiveModelsMainResults/Saved_models_486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0"
# RESULTS_DIR = f"./Results_Performance/Results_486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0"
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_FULL_TESTSET = True
PLOT_BATCH_SIZE = 1500
LOADER_BATCH_SIZE = 128
NUM_WORKERS = 0
PIN_MEM = False

SAVE_PLOT = True
SHOW_PLOT = False
BINS = 100
CLIP_PERCENTILES = (1, 99)

IG_TARGET_KEY = "logit"
IG_ABS = False
COMPUTE_INPUT_GRADS = True
GRAD_TARGET_KEY = "logit"
GRAD_PLOT_BINS = 80

FIXED_MODEL_ALIASES = {"senn_fixed", "fixed_senn","senn_fixed_concepttheta", "senn_fixedconcepts", "fixed"}
FIXED_CONCEPT_NAMES = [
    "RBP delta",
    "RBP theta",
    "RBP alpha",
    "RBP beta",
    "Rhythmicity",
    "SNLEO",
]
# ============================================================



def _extract_model_output(output, key: str):
    if torch.is_tensor(output):
        return output

    if not isinstance(output, dict):
        raise TypeError(f"Expected tensor or dict output, got {type(output)!r}")

    aliases = {
        "prob": ["prob", "probability", "probs", "probabilities"],
        "logit": ["logit", "logits"],
        "explanation": ["explanation", "focus_map", "F_map"],
        "explanation_edge": ["explanation_edge", "F_edge"],
    }.get(key, [key])

    for name in aliases:
        if name in output:
            return output[name]

    raise KeyError(f"Could not find '{key}' in model output. Available keys: {sorted(output.keys())}")



def build_model(model_kind: str, ckpt: dict):
    model_kind = model_kind.strip().lower()

    if model_kind == "base":
        model = Model.EEG_GAT_Model()
    elif model_kind == "senn":
        model = Model.SENN_raw(
            global_min=ckpt.get("global_min", None),
            return_node_scores=False,
            return_fmap=True,
        )
    elif model_kind == "senn_fixed" and not IS_TRIVIAL:
        model = Model.SENN_fixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=True,
        )
    elif model_kind == "senn_fixed" and IS_TRIVIAL:
        print("trivial model loading")
        model = Model.SENN_trivialfixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=True,
        )
    elif model_kind == "senn_fixed_concepttheta":
        print("concepttheta model loading")
        model = Model.SENN_fixedconcepts_concepttheta(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=True,
        )
    else:
        raise ValueError("MODEL_KIND must be 'base', 'senn', or a fixed-concept SENN alias")

    model.load_state_dict(ckpt["model_state_dict"])
    return model




def _load_fold_data(fold: str, norm: Optional[dict] = None):
    fold_dir = os.path.join(DATA_FOLDER, f"fold_{fold}")

    x_data = np.load(os.path.join(fold_dir, "testdata.npy"), mmap_mode="r")
    y_data = np.load(os.path.join(fold_dir, "testlabels.npy"), mmap_mode="r")

    if norm is not None:
        x_data = (x_data - norm["mean"]) / norm["std"]

    idx_non = np.where(y_data == 0)[0]
    idx_seiz = np.where(y_data == 1)[0]

    # --- timing setup ---
    fs = 32  # Hz
    t_window = x_data.shape[-1] / fs  # e.g. 384 / 32 = 12 sec

    t_overlap_non = 10
    t_overlap_seiz = 11

    step_non = t_window - t_overlap_non
    step_seiz = t_window - t_overlap_seiz

    skip_non = int(t_window / step_non)
    skip_seiz = int(t_window / step_seiz)

    thin_idx_non = idx_non[::skip_non]
    thin_idx_seiz = idx_seiz[::skip_seiz]

    
    keep_idx = np.sort(np.concatenate([thin_idx_non, thin_idx_seiz]))

    return x_data[keep_idx], y_data[keep_idx]
# def _load_fold_data(fold: str, norm: Optional[dict] = None):
#     fold_dir = os.path.join(DATA_FOLDER, f"fold_{fold}")
#     x_data = np.load(os.path.join(fold_dir, "testdata.npy"), mmap_mode="r")
#     y_data = np.load(os.path.join(fold_dir, "testlabels.npy"), mmap_mode="r")

#     if norm is not None:
#         x_data = (x_data - norm["mean"]) / norm["std"]

#     if USE_FULL_TESTSET:
#         return x_data, y_data

#     idx_non = np.where(y_data == 0)[0]
#     idx_seiz = np.where(y_data == 1)[0]
#     n = min(len(idx_non), len(idx_seiz), PLOT_BATCH_SIZE)
#     idx_balanced = np.concatenate([idx_non[:n], idx_seiz[:n]])
#     np.random.shuffle(idx_balanced)
#     return x_data[idx_balanced], y_data[idx_balanced]



def load_data_and_model():
    fold_dir = os.path.join(DATA_FOLDER, f"fold_{FOLD}")
    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR)

    ckpt = torch.load(os.path.join(model_path, CKPT_NAME), weights_only=False)
    norm = ckpt["normalization"]
    threshold = float(np.round(ckpt["metrics"]["threshold"], 2))

    model = build_model(MODEL_KIND, ckpt)
    model.eval()

    print("... loading test data ...")
    np.random.seed(43)
    

    # x_train = np.load(os.path.join(fold_dir,'traindata.npy'),mmap_mode='r')
    # y_train = np.load(os.path.join(fold_dir,'trainlabels.npy'),mmap_mode='r')
    x_test  = np.load(os.path.join(fold_dir,'testdata.npy'),mmap_mode='r')
    y_test  = np.load(os.path.join(fold_dir,'testlabels.npy'),mmap_mode='r')

    
    idx_yes_seiz = np.where(y_test == 1)[0]
    idx_no_seiz = np.where(y_test == 0)[0]
    fs = 32 #Hz
    t_window = len(x_test[0][0]) / fs
    t_overlap = 10
    t_overlap_seiz = 11
    thin_skip = int(t_window / (t_window - t_overlap))
    thin_skip_seiz = int(t_window / (t_window - t_overlap_seiz))
    thin_idx_no_seiz = idx_no_seiz[0::thin_skip] #Full test set 
    thin_idx_yes_seiz = idx_yes_seiz[0::thin_skip_seiz] #full test set

    keep_idx = np.sort(np.concatenate([thin_idx_no_seiz, thin_idx_yes_seiz]))

    x_test = x_test[keep_idx]
    y_test = y_test[keep_idx]



    testset = MyUtils.prepare_graphs_labels(x_test,y_test,Model.adj)
    

    
    return model, testset, threshold, ckpt


@torch.no_grad()
def run_inference(model, testset, device, threshold):
    loader = DataLoader(
        testset,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEM,
        num_workers=NUM_WORKERS,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    all_probs = []
    all_labels = []

    model = model.to(device)
    model.eval()

    print("... running inference ...")
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch.x, batch.edge_index, batch.batch)
        prob = _extract_model_output(out, "prob").detach().cpu()
        all_probs.append(prob)
        all_labels.append(batch.y.detach().cpu())

    y_prob = torch.cat(all_probs).numpy().ravel()
    y_true = torch.cat(all_labels).numpy().ravel()
    pred = (y_prob >= threshold).astype(int)
    return y_prob, y_true, pred



def _project_fixed_node_fmap_to_time(node_fmap: np.ndarray, time_len: int) -> np.ndarray:
    node_scores = np.asarray(node_fmap, dtype=float).sum(axis=-1)
    return np.repeat(node_scores[:, None], time_len, axis=1)



def compute_temporal_explanations(model, testset, device, threshold):
    model = model.to(device)
    model.eval()

    temporal_expl = []
    print(f"... computing temporal explanations for MODEL_KIND='{MODEL_KIND}' with EXPLAINER_TYPE='{EXPLAINER_TYPE}' ...")

    for graph in tqdm(testset):
        graph = graph.to(device)

        if MODEL_KIND == "base":
            if EXPLAINER_TYPE != "ig":
                raise ValueError("For MODEL_KIND='base', only EXPLAINER_TYPE='ig' is supported.")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            exp = MyUtils.calculateIG(model, graph, thr=threshold, target_key=IG_TARGET_KEY)
            a = exp.node_mask.detach().cpu().numpy()
            if IG_ABS:
                a = np.abs(a)
            temporal_expl.append(a)

        elif MODEL_KIND == "senn":
            if EXPLAINER_TYPE != "focusmap":
                raise ValueError("For raw SENN use EXPLAINER_TYPE='focusmap'.")
            with torch.no_grad():
                batch_vec = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
                out = model(graph.x, graph.edge_index, batch_vec)
                temporal_expl.append(_extract_model_output(out, "explanation").detach().cpu().numpy())

        elif MODEL_KIND in FIXED_MODEL_ALIASES:
            if EXPLAINER_TYPE not in {"focusmap", "projection", "node_score"}:
                raise ValueError("For fixed-concept SENN use EXPLAINER_TYPE in {'focusmap', 'projection', 'node_score'}." )
            with torch.no_grad():
                batch_vec = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
                out = model(graph.x, graph.edge_index, batch_vec)
                node_fmap = _extract_model_output(out, "explanation").detach().cpu().numpy()
                temporal_expl.append(_project_fixed_node_fmap_to_time(node_fmap, graph.x.shape[-1]))
        else:
            raise ValueError("Unsupported MODEL_KIND")

    return np.stack(temporal_expl, axis=0)


def print_sample_rma_coherence(explanations: np.ndarray, y_true: np.ndarray, name: str = "explanation"):
    y = np.asarray(y_true, dtype=int).reshape(-1)
    a = np.maximum(np.nan_to_num(np.asarray(explanations, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if a.shape[0] != y.size:
        raise ValueError(f"Expected {y.size} explanations, got shape {a.shape}")
    mass = a.reshape(y.size, -1).sum(axis=1)
    seiz = y == 1
    weights = np.where(seiz, 1.0 / max(1, int(seiz.sum())), 1.0 / max(1, int((~seiz).sum())))
    coherence = float((weights * mass * seiz).sum() / ((weights * mass).sum() + 1e-12))
    print(f"{name} coherence (sample-RMA): {coherence:.4f}")



def compute_input_gradients(model, testset, device, target_key="logit"):
    model = model.to(device)
    model.eval()
    grads_all = []

    print(f"... computing input gradients wrt '{target_key}' ...")

    for graph in tqdm(testset):
        graph = graph.to(device)
        x = graph.x.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        batch_vec = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        out = model(x, graph.edge_index, batch_vec)
        target = _extract_model_output(out, target_key)
        if target.ndim > 0:
            target = target.sum()

        grad_x = torch.autograd.grad(
            outputs=target,
            inputs=x,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads_all.append(grad_x.detach().cpu().numpy())

    grads = np.stack(grads_all, axis=0)
    grad_abs = np.mean(np.abs(grads), axis=(1, 2))
    grad_var = np.var(grads, axis=(1, 2))
    grad_dt = np.diff(grads, axis=2)
    grad_diff = np.mean(np.abs(grad_dt), axis=(1, 2))
    return grads, grad_abs, grad_var, grad_diff



def plot_explanation_distribution(
    temporal_expl: np.ndarray,
    y_true: np.ndarray,
    results_dir: str,
    model_kind: str,
    explainer_type: str,
    clip_percentiles=(0, 100),
    bins=100,
    save_plot=True,
    show_plot=True,
):
    print("... plotting relevance distribution ...")

    expl_flat = temporal_expl.reshape(len(y_true), -1)
    expl_non = expl_flat[y_true == 0].ravel()
    expl_seiz = expl_flat[y_true == 1].ravel()

    low, high = np.percentile(expl_flat, clip_percentiles)
    expl_non_clip = np.clip(expl_non, low, high)
    expl_seiz_clip = np.clip(expl_seiz, low, high)

    plt.figure(figsize=(7, 4))
    plt.hist(expl_non_clip, bins=bins, alpha=0.5, density=True, label="Non-seiz")
    plt.hist(expl_seiz_clip, bins=bins, alpha=0.5, density=True, label="Seiz")

    if model_kind == "base":
        title_name = "IG"
    elif model_kind in FIXED_MODEL_ALIASES:
        title_name = "Projected fixed SENN node score"
    else:
        title_name = "Focus map"

    plt.title(f"{title_name} distribution ({clip_percentiles[0]}-{clip_percentiles[1]}% clipped)")
    plt.xlabel("Explanation value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        save_name = f"relevance_distribution_{model_kind}_{explainer_type}_fold{FOLD}_log{LOG}.png"
        save_path = os.path.join(results_dir, save_name)
        plt.savefig(save_path, dpi=300)
        print(f"saved plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print("\nExplanation summary:")
    print(f"  all      : mean={expl_flat.mean():.6f}, std={expl_flat.std():.6f}, min={expl_flat.min():.6f}, max={expl_flat.max():.6f}")
    print(f"  non-seiz : mean={expl_non.mean():.6f}, std={expl_non.std():.6f}, min={expl_non.min():.6f}, max={expl_non.max():.6f}")
    print(f"  seiz     : mean={expl_seiz.mean():.6f}, std={expl_seiz.std():.6f}, min={expl_seiz.min():.6f}, max={expl_seiz.max():.6f}")



def plot_gradient_diagnostics(grads: np.ndarray, y_true: np.ndarray, results_dir: str, bins=80, save_plot=True, show_plot=True):
    print("... plotting gradient diagnostics ...")

    grads_flat = grads.reshape(len(y_true), -1)
    grad_non = grads_flat[y_true == 0].ravel()
    grad_seiz = grads_flat[y_true == 1].ravel()
    low, high = np.percentile(grads_flat, [1, 99])
    grad_non_clip = np.clip(grad_non, low, high)
    grad_seiz_clip = np.clip(grad_seiz, low, high)

    grad_abs = np.mean(np.abs(grads), axis=(1, 2))
    grad_var = np.var(grads, axis=(1, 2))
    grad_dt = np.diff(grads, axis=2)
    grad_rough = np.mean(np.abs(grad_dt), axis=(1, 2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(grad_non_clip, bins=bins, alpha=0.5, density=True, label="Non-seiz")
    axes[0, 0].hist(grad_seiz_clip, bins=bins, alpha=0.5, density=True, label="Seiz")
    axes[0, 0].set_title("Input gradient distribution (1-99% clipped)")
    axes[0, 0].set_xlabel("Gradient value")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    axes[0, 1].hist(grad_abs[y_true == 0], bins=bins, alpha=0.5, density=True, label="Non-seiz")
    axes[0, 1].hist(grad_abs[y_true == 1], bins=bins, alpha=0.5, density=True, label="Seiz")
    axes[0, 1].set_title("Per-sample mean |∇x f(x)|")
    axes[0, 1].set_xlabel("Mean absolute gradient")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()

    axes[1, 0].hist(grad_var[y_true == 0], bins=bins, alpha=0.5, density=True, label="Non-seiz")
    axes[1, 0].hist(grad_var[y_true == 1], bins=bins, alpha=0.5, density=True, label="Seiz")
    axes[1, 0].set_title("Per-sample Var(∇x f(x))")
    axes[1, 0].set_xlabel("Gradient variance")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()

    axes[1, 1].hist(grad_rough[y_true == 0], bins=bins, alpha=0.5, density=True, label="Non-seiz")
    axes[1, 1].hist(grad_rough[y_true == 1], bins=bins, alpha=0.5, density=True, label="Seiz")
    axes[1, 1].set_title("Per-sample temporal roughness of ∇x f(x)")
    axes[1, 1].set_xlabel("Mean |Δ_t ∇x f(x)|")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    plt.tight_layout()

    if save_plot:
        save_name = f"input_gradient_diagnostics_{MODEL_KIND}_{FOLD}_log{LOG}.png"
        save_path = os.path.join(results_dir, save_name)
        plt.savefig(save_path, dpi=300)
        print(f"saved gradient plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print("\nGradient summary:")
    print(f"Mean |grad| non-seiz: {grad_abs[y_true == 0].mean():.6f}")
    print(f"Mean |grad| seiz    : {grad_abs[y_true == 1].mean():.6f}")
    print(f"Var(grad) non-seiz  : {grad_var[y_true == 0].mean():.6f}")
    print(f"Var(grad) seiz      : {grad_var[y_true == 1].mean():.6f}")
    print(f"Rough non-seiz      : {grad_rough[y_true == 0].mean():.6f}")
    print(f"Rough seiz          : {grad_rough[y_true == 1].mean():.6f}")



def _reshape_per_graph(tensor_np: np.ndarray, batch_size_actual: int) -> np.ndarray:
    total_items = tensor_np.shape[0]
    if total_items % batch_size_actual != 0:
        raise ValueError(f"Cannot reshape first dimension {total_items} into batch size {batch_size_actual}")
    items_per_graph = total_items // batch_size_actual
    return tensor_np.reshape(batch_size_actual, items_per_graph, *tensor_np.shape[1:])



def compute_fixed_concepts_only(conceptizer, testset, device):
    conceptizer = conceptizer.to(device)
    conceptizer.eval()

    all_node_concepts = []
    all_edge_concepts = []
    all_labels = []

    print("... computing fixed concepts only ...")

    test_loader = DataLoader(
        testset,
        batch_size=16,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
    )

    with torch.no_grad():
        for graph in tqdm(test_loader):
            graph = graph.to(device)
            node_feats, edge_feats = conceptizer(graph.x, graph.edge_index)
            y_batch = graph.y.view(-1)
            batch_size_actual = y_batch.shape[0]

            node_feats = _reshape_per_graph(node_feats.detach().cpu().numpy(), batch_size_actual)
            edge_feats = _reshape_per_graph(edge_feats.detach().cpu().numpy(), batch_size_actual)

            all_node_concepts.append(node_feats)
            all_edge_concepts.append(edge_feats)
            all_labels.append(y_batch.detach().cpu().numpy())

    node_concepts = np.concatenate(all_node_concepts, axis=0)
    edge_concepts = np.concatenate(all_edge_concepts, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return node_concepts, edge_concepts, y_true



def compute_fixed_senn_outputs(model, testset, device) -> Dict[str, np.ndarray]:
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        testset,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEM,
        num_workers=NUM_WORKERS,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    collected = {
        "h_x": [],
        "theta_x": [],
        "explanation": [],
        "h_x_edge": [],
        "theta_x_edge": [],
        "explanation_edge": [],
        "y_true": [],
    }

    print("... computing fixed SENN activations / relevances / F_maps ...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device, non_blocking=True)
            out = model(batch.x, batch.edge_index, batch.batch)
            batch_size_actual = batch.y.view(-1).shape[0]

            collected["h_x"].append(_reshape_per_graph(out["h_x"].detach().cpu().numpy(), batch_size_actual))
            collected["theta_x"].append(_reshape_per_graph(out["theta_x"].detach().cpu().numpy(), batch_size_actual))
            collected["explanation"].append(_reshape_per_graph(_extract_model_output(out, "explanation").detach().cpu().numpy(), batch_size_actual))
            collected["y_true"].append(batch.y.view(-1).detach().cpu().numpy())

            if "h_x_edge" in out:
                collected["h_x_edge"].append(_reshape_per_graph(out["h_x_edge"].detach().cpu().numpy(), batch_size_actual))
            if "theta_x_edge" in out:
                collected["theta_x_edge"].append(_reshape_per_graph(out["theta_x_edge"].detach().cpu().numpy(), batch_size_actual))
            if "explanation_edge" in out:
                collected["explanation_edge"].append(_reshape_per_graph(_extract_model_output(out, "explanation_edge").detach().cpu().numpy(), batch_size_actual))

    out_np: Dict[str, np.ndarray] = {}
    for key, values in collected.items():
        if len(values) > 0:
            out_np[key] = np.concatenate(values, axis=0)
    return out_np



def plot_fixed_concept_boxplots(concepts, y_true, results_dir, concept_names=None, reduce_nodes="mean", save_plot=True, show_plot=True):
    print("... plotting fixed concept boxplots ...")
    if concept_names is None:
        concept_names = FIXED_CONCEPT_NAMES

    if reduce_nodes == "mean":
        concept_vals = concepts.mean(axis=1)
        mode_txt = "mean over channels"
    elif reduce_nodes == "median":
        concept_vals = np.median(concepts, axis=1)
        mode_txt = "median over channels"
    elif reduce_nodes == "flatten":
        concept_vals = concepts
        mode_txt = "all channel values"
    else:
        raise ValueError("reduce_nodes must be 'mean', 'median', or 'flatten'")

    K = concepts.shape[-1]
    n_cols = 3
    n_rows = int(np.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.0 * n_rows), squeeze=False)

    for k in range(K):
        ax = axes[k // n_cols, k % n_cols]
        if reduce_nodes == "flatten":
            vals_non = concept_vals[y_true == 0, :, k].reshape(-1)
            vals_seiz = concept_vals[y_true == 1, :, k].reshape(-1)
        else:
            vals_non = concept_vals[y_true == 0, k]
            vals_seiz = concept_vals[y_true == 1, k]

        ax.boxplot([vals_non, vals_seiz], widths=0.55, showmeans=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Non-seiz\n(n={len(vals_non)})", f"Seiz\n(n={len(vals_seiz)})"], fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        ax.set_title(concept_names[k], fontsize=10)
        ax.set_ylabel("Concept value", fontsize=9)

    for k in range(K, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle(f"Fixed concepts by class ({mode_txt})", y=1.02, fontsize=12)
    fig.tight_layout()

    if save_plot:
        save_path = os.path.join(results_dir, f"fixed_concepts_boxplots_fold{FOLD}_log{LOG}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved plot to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)



def plot_edge_concept_boxplots(edge_concepts, y_true, results_dir, concept_names=None, reduce_edges="mean", save_plot=True, show_plot=True):
    print("... plotting edge concept boxplots ...")
    if concept_names is None:
        concept_names = ["Imaginary coherence"]

    if reduce_edges == "mean":
        edge_vals = edge_concepts.mean(axis=1)
        mode_txt = "mean over edges"
    elif reduce_edges == "median":
        edge_vals = np.median(edge_concepts, axis=1)
        mode_txt = "median over edges"
    elif reduce_edges == "flatten":
        edge_vals = edge_concepts
        mode_txt = "all edge values"
    else:
        raise ValueError("reduce_edges must be 'mean', 'median', or 'flatten'")

    K = edge_concepts.shape[-1]
    fig, axes = plt.subplots(1, K, figsize=(5.2 * K, 4.2), squeeze=False)
    axes = axes[0]
    for k in range(K):
        ax = axes[k]
        if reduce_edges == "flatten":
            vals_non = edge_vals[y_true == 0, :, k].reshape(-1)
            vals_seiz = edge_vals[y_true == 1, :, k].reshape(-1)
        else:
            vals_non = edge_vals[y_true == 0, k]
            vals_seiz = edge_vals[y_true == 1, k]
        ax.boxplot([vals_non, vals_seiz], widths=0.55, showmeans=True)
        ax.grid(alpha=0.25, axis="y")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Non-seiz\n(n={len(vals_non)})", f"Seiz\n(n={len(vals_seiz)})"], fontsize=8)
        ax.set_title(concept_names[k], fontsize=10)
        ax.set_ylabel("Concept value", fontsize=9)

    fig.suptitle(f"Edge concepts by class ({mode_txt})", y=1.02, fontsize=12)
    fig.tight_layout()

    if save_plot:
        save_path = os.path.join(results_dir, f"edge_concepts_boxplots_fold{FOLD}_log{LOG}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved edge concept plot to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)



def plot_component_triplets_by_concept(
    activations: np.ndarray,
    relevances: np.ndarray,
    fmaps: np.ndarray,
    y_true: np.ndarray,
    results_dir: str,
    concept_names,
    prefix: str,
    bins: int = 80,
    clip_percentiles=(1, 99),
    save_plot: bool = True,
    show_plot: bool = True,
):
    K = activations.shape[-1]
    fig, axes = plt.subplots(K, 3, figsize=(14, 3.2 * K), squeeze=False)
    metric_names = ["Activation h(x)", "Relevance θ(x)", "F_map = h(x)·θ(x)"]
    tensors = [activations, relevances, fmaps]

    print(f"... plotting {prefix} activation / relevance / F_map distributions ...")

    for k in range(K):
        for col, values in enumerate(tensors):
            ax = axes[k, col]
            vals_non = values[y_true == 0, :, k].reshape(-1)
            vals_seiz = values[y_true == 1, :, k].reshape(-1)
            combined = np.concatenate([vals_non, vals_seiz]) if (len(vals_non) + len(vals_seiz)) > 0 else np.array([0.0])
            low, high = np.percentile(combined, clip_percentiles)
            vals_non_clip = np.clip(vals_non, low, high)
            vals_seiz_clip = np.clip(vals_seiz, low, high)

            ax.hist(vals_non_clip, bins=bins, alpha=0.5, density=True, label="Non-seiz")
            ax.hist(vals_seiz_clip, bins=bins, alpha=0.5, density=True, label="Seiz")
            ax.grid(alpha=0.2)
            if k == 0:
                ax.set_title(metric_names[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(concept_names[k], fontsize=10)
            if k == K - 1:
                ax.set_xlabel("Value")
            if k == 0 and col == 2:
                ax.legend(fontsize=9)

    fig.suptitle(f"{prefix.capitalize()} concept distributions stratified by seizure / non-seizure", y=1.01, fontsize=13)
    fig.tight_layout()

    if save_plot:
        save_path = os.path.join(results_dir, f"{prefix}_distributions.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved plot to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"\n{prefix.capitalize()} summary by concept:")
    for idx, name in enumerate(concept_names):
        act_non = activations[y_true == 0, :, idx].reshape(-1)
        act_seiz = activations[y_true == 1, :, idx].reshape(-1)
        rel_non = relevances[y_true == 0, :, idx].reshape(-1)
        rel_seiz = relevances[y_true == 1, :, idx].reshape(-1)
        fmap_non = fmaps[y_true == 0, :, idx].reshape(-1)
        fmap_seiz = fmaps[y_true == 1, :, idx].reshape(-1)
        print(
            f"  {name:<14} | h non/seiz = {act_non.mean(): .4f}/{act_seiz.mean(): .4f} | "
            f"θ non/seiz = {rel_non.mean(): .4f}/{rel_seiz.mean(): .4f} | "
            f"F non/seiz = {fmap_non.mean(): .4f}/{fmap_seiz.mean(): .4f}"
        )



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, testset, threshold, _ = load_data_and_model()
    y_prob, y_true, pred = run_inference(model, testset, device, threshold)
    temporal_expl = compute_temporal_explanations(model, testset, device, threshold)

    print(f"temporal_expl shape: {temporal_expl.shape}")
    print(f"y_true shape       : {y_true.shape}")
    print(f"y_prob shape       : {y_prob.shape}")
    print_sample_rma_coherence(temporal_expl, y_true, "temporal")

    plot_explanation_distribution(
        temporal_expl=temporal_expl,
        y_true=y_true,
        results_dir=RESULTS_DIR,
        model_kind=MODEL_KIND,
        explainer_type=EXPLAINER_TYPE,
        clip_percentiles=CLIP_PERCENTILES,
        bins=BINS,
        save_plot=SAVE_PLOT,
        show_plot=SHOW_PLOT,
    )

    if COMPUTE_INPUT_GRADS:
        grads, _, _, _ = compute_input_gradients(model=model, testset=testset, device=device, target_key=GRAD_TARGET_KEY)
        print(f"grads shape        : {grads.shape}")
        plot_gradient_diagnostics(
            grads=grads,
            y_true=y_true,
            results_dir=RESULTS_DIR,
            bins=GRAD_PLOT_BINS,
            save_plot=SAVE_PLOT,
            show_plot=SHOW_PLOT,
        )



def main2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR)
    ckpt = torch.load(os.path.join(model_path, CKPT_NAME), weights_only=False)
    norm = ckpt["normalization"]

    print("... loading test data for fixed concept analysis ...")
    x_sel, y_sel = _load_fold_data(FOLD, norm=None)
    testset = MyUtils.prepare_graphs_labels(x_sel, y_sel, Model.adj)

    print("... building FixedConceptizer ...")
    conceptizer = Model.FixedConceptizer()
    node_concepts, edge_concepts, y_true = compute_fixed_concepts_only(conceptizer=conceptizer, testset=testset, device=device)

    print(f"node_concepts shape: {node_concepts.shape}")
    print(f"edge_concepts shape: {edge_concepts.shape}")
    print(f"labels shape       : {y_true.shape}")

    plot_fixed_concept_boxplots(
        concepts=node_concepts,
        y_true=y_true,
        results_dir=RESULTS_DIR,
        concept_names=FIXED_CONCEPT_NAMES,
        reduce_nodes="median",
        save_plot=SAVE_PLOT,
        show_plot=SHOW_PLOT,
    )
    plot_edge_concept_boxplots(
        edge_concepts=edge_concepts,
        y_true=y_true,
        results_dir=RESULTS_DIR,
        concept_names=["Imaginary coherence"],
        reduce_edges="median",
        save_plot=SAVE_PLOT,
        show_plot=SHOW_PLOT,
    )

    print("... building trained fixed-concept SENN ...")
    try:
        
        fixed_model = build_model(MODEL_KIND, ckpt)
        
    except Exception as exc:
        raise RuntimeError(
            "main2 now expects LOG/FOLD to point to a trained fixed-concept SENN checkpoint. "
            "Set LOG and FOLD to that model before running main2()."
        ) from exc

    fixed_outputs = compute_fixed_senn_outputs(model=fixed_model, testset=testset, device=device)
    print(f"h_x shape           : {fixed_outputs['h_x'].shape}")
    print(f"theta_x shape       : {fixed_outputs['theta_x'].shape}")
    print(f"explanation shape   : {fixed_outputs['explanation'].shape}")
    print_sample_rma_coherence(fixed_outputs["explanation"], fixed_outputs["y_true"], "node")

    plot_component_triplets_by_concept(
        activations=fixed_outputs["h_x"],
        relevances=fixed_outputs["theta_x"],
        fmaps=fixed_outputs["explanation"],
        y_true=fixed_outputs["y_true"],
        results_dir=RESULTS_DIR,
        concept_names=FIXED_CONCEPT_NAMES,
        prefix="node",
        bins=BINS,
        clip_percentiles=CLIP_PERCENTILES,
        save_plot=SAVE_PLOT,
        show_plot=SHOW_PLOT,
    )

    if all(key in fixed_outputs for key in ["h_x_edge", "theta_x_edge", "explanation_edge"]):
        print(f"h_x_edge shape      : {fixed_outputs['h_x_edge'].shape}")
        print(f"theta_x_edge shape  : {fixed_outputs['theta_x_edge'].shape}")
        print(f"F_edge shape        : {fixed_outputs['explanation_edge'].shape}")
        print_sample_rma_coherence(fixed_outputs["explanation_edge"], fixed_outputs["y_true"], "edge")
        plot_component_triplets_by_concept(
            activations=fixed_outputs["h_x_edge"],
            relevances=fixed_outputs["theta_x_edge"],
            fmaps=fixed_outputs["explanation_edge"],
            y_true=fixed_outputs["y_true"],
            results_dir=RESULTS_DIR,
            concept_names=["Imaginary coherence"],
            prefix="edge",
            bins=BINS,
            clip_percentiles=CLIP_PERCENTILES,
            save_plot=SAVE_PLOT,
            show_plot=SHOW_PLOT,
        )


if __name__ == "__main__":
    # main()
    main2()
