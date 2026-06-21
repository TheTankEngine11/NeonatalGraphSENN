import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.algorithm import CaptumExplainer
from captum.attr import IntegratedGradients
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import networkx as nx
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
from data_utils import prepare_graphs_labels as _prepare_graphs_labels
from model_utils import build_model as _build_model
from model_utils import extract_model_output as _extract_model_output_shared

def build_model(model_kind: str, ckpt: Dict[str, Any], return_explanations: bool = False) -> torch.nn.Module:
    """Compatibility wrapper around model_utils.build_model."""
    return _build_model(model_kind, ckpt=ckpt, return_explanations=return_explanations)


def prepare_graphs_labels(features, labels, adj_matrix, masks=None):
    """Compatibility wrapper around data_utils.prepare_graphs_labels."""
    return _prepare_graphs_labels(features, labels, adj_matrix, masks=masks)



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Args:
            patience (int): How many epochs to wait after last time AUROC improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        return

    def __call__(self, current_auroc):
        if self.best_score is None:
            self.best_score = current_auroc
        elif current_auroc < self.best_score + self.min_delta:
            self.counter += 1
            #print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_auroc
            self.counter = 0
        
        return
    
    # End class

class EarlyStoppingValLoss:
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Args:
            patience (int): How many epochs to wait after last time AUROC improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        return

    def __call__(self, current_val):
        if self.best_score is None:
            self.best_score = current_val
        elif current_val > self.best_score - self.min_delta:
            self.counter += 1
            #print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_val
            self.counter = 0
        
        return
    
    # End class

def _ensure_graph_batch(graph_data: Data) -> Optional[torch.Tensor]:
    if hasattr(graph_data, "batch") and graph_data.batch is not None:
        return graph_data.batch
    return torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=graph_data.x.device)


def _extract_output_tensor(output, key: str = "logit") -> torch.Tensor:
    return _extract_model_output_shared(output, key)


class _ModelOutputWrapper(nn.Module):
    """
    Wrap a model that returns a dictionary so PyG/Captum explainers receive a single tensor as foward output instead of a dict.
    """
    def __init__(self, model: nn.Module, output_key: str = "logit"):
        super().__init__()
        self.model = model
        self.output_key = output_key

    @property
    def cnn(self):
        return getattr(self.model, "cnn")

    @property
    def gnn(self):
        return getattr(self.model, "gnn")

    def forward(self, x, edge_index, batch=None):
        out = self.model(x, edge_index, batch)
        return _extract_output_tensor(out, self.output_key)


def calculateIG(model, graph_data, thr=None, target_key="logit", **kwargs):
    """
    Integrated Gradients on the raw node-time input.

    For the updated models, the forward pass returns a dictionary. By default we explain
    the pre-activation graph logit because that is the quantity you want to compare
    against the SENN focus map.
    """
    wrapped_model = _ModelOutputWrapper(model, output_key=target_key)

    explainer = Explainer(
        model=wrapped_model,
        algorithm=CaptumExplainer(IntegratedGradients),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    batch = _ensure_graph_batch(graph_data)
    explanation = explainer(
        x=graph_data.x,
        edge_index=graph_data.edge_index,
        batch=batch,
    )
    return explanation


def calculateGNNexpl(model, graph_data, thr=None, epochs: int = 200, target_key="logit", **kwargs):
    """
    GNNExplainer on the graph head in CNN-node-feature space.

    The base GNN head already returns the raw graph logit, so the explanation is naturally
    computed for the pre-sigmoid decision quantity.
    """
    with torch.no_grad():
        node_features = model.cnn(graph_data.x)

    explainer = Explainer(
        model=model.gnn,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    batch = _ensure_graph_batch(graph_data)
    explanation = explainer(
        x=node_features,
        edge_index=graph_data.edge_index,
        batch=batch,
    )
    return explanation

def run_inference(testdata, model):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    model.eval()

    num_workers = 0
    pin_mem = False
    batch_size = 128

    test_loader = DataLoader(
        testdata,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            out = model(batch.x, batch.edge_index, batch.batch)

            if isinstance(out, dict):
                probs = _extract_output_tensor(out, "prob")
            else:
                probs = out

            all_probs.append(probs.detach().cpu())
            all_labels.append(batch.y.detach().cpu())

    y_prob = torch.cat(all_probs).cpu().numpy().ravel()
    y_test = torch.cat(all_labels).cpu().numpy().ravel()

    return y_prob, y_test


def plot_data_int_expl(
    x_batch, y_batch, edge_index, show_plots=True,
    pred_batch=None, temp_expl=None,
    spat_edge_expl=None, spat_node_expl=None,
    fs=32
):
    """
    Temporal: SIGNED IG is rectified with ReLU (max(IG, 0)).
             Negative values become 0 and thus are plotted as blue.
             Overlaps are averaged (mean) per time sample.
             Colormap uses scaled on vmax (99th percentile) + optional symlog scaling
             to avoid a single outlier washing everything out.
             IG data itself is NOT normalized to [0,1] (only the colormap scaling is robust).

    Spatial: GNNExplainer node/edge masks are assumed already in [0,1] (not re-normalized).
    Includes info panel with window times, label, pred, and top nodes/edges.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize, SymLogNorm, LinearSegmentedColormap
    from matplotlib import cm

    # -----------------------------
    # Helpers
    # -----------------------------
    def _to_np(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def _truncate_cmap(cmap_obj, minval=0.0, maxval=1.0, n=256):
        """Return a truncated colormap. minval/maxval are in [0,1]."""
        colors = cmap_obj(np.linspace(minval, maxval, n))
        return LinearSegmentedColormap.from_list("trunc_cmap", colors)

    # -----------------------------
    # 0) Robust numpy conversion
    # -----------------------------
    x_batch = _to_np(x_batch)
    y_batch = _to_np(y_batch).astype(int)
    pred_batch_np = _to_np(pred_batch)
    temp_expl = _to_np(temp_expl)
    spat_edge_expl = _to_np(spat_edge_expl)
    spat_node_expl = _to_np(spat_node_expl)

    B, C, T = x_batch.shape

    channel_names = [
        "Fp1-T3", "T3-O1", "Fp1-C3", "C3-O1", "Fp2-C4", "C4-O2",
        "Fp2-T4", "T4-O2", "T3-C3", "C3-Cz", "Cz-C4", "C4-T4"
    ]

    # -----------------------------
    # 1) Window placement + overlap handling (mean over overlaps)
    # -----------------------------
    overlap_lengths = np.zeros(B, dtype=int)
    for b in range(B):
        overlap_lengths[b] = int((11 if y_batch[b] == 1 else 10) * fs)

    overlap_lengths = np.clip(overlap_lengths, 0, T - 1)
    strides = (T - overlap_lengths).astype(int)
    strides = np.clip(strides, 1, T)

    batch_indices = []
    starts = np.zeros(B, dtype=int)
    for b in range(B):
        if b > 0:
            starts[b] = starts[b - 1] + strides[b - 1]
        batch_indices.append((int(starts[b]), int(starts[b] + T)))

    total_samples = batch_indices[-1][1]
    t_axis = np.arange(total_samples) / fs

    # accumulators
    x_sum = np.zeros((C, total_samples), dtype=float)
    ig_sum = np.zeros((C, total_samples), dtype=float)
    pred_sum = np.zeros(total_samples, dtype=float)
    gt_sum = np.zeros(total_samples, dtype=float)
    counts = np.zeros(total_samples, dtype=float)

    for b in range(B):
        start, end = batch_indices[b]
        x_sum[:, start:end] += x_batch[b]

        if temp_expl is not None:
            # ReLU-rectified IG: keep only positive contributions
            ig_w = temp_expl[b]
            ig_w = np.nan_to_num(ig_w, nan=0.0, posinf=0.0, neginf=0.0)
            ig_w = np.maximum(ig_w, 0.0)  # <-- ReLU rectifier
            ig_sum[:, start:end] += ig_w

        if pred_batch_np is not None:
            pred_sum[start:end] += float(pred_batch_np[b])

        gt_sum[start:end] += float(y_batch[b])
        counts[start:end] += 1.0

    mask = counts > 0
    x_mean = np.zeros_like(x_sum)
    ig_mean = np.zeros_like(ig_sum)
    pred_mean = np.zeros_like(pred_sum)
    gt_mean = np.zeros_like(gt_sum)

    x_mean[:, mask] = x_sum[:, mask] / counts[mask]
    ig_mean[:, mask] = ig_sum[:, mask] / counts[mask]
    pred_mean[mask] = pred_sum[mask] / counts[mask]
    gt_mean[mask] = gt_sum[mask] / counts[mask]

    # -----------------------------
    # 2) Figure layout
    # -----------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8])

    ax_main = fig.add_subplot(gs[0, :])
    ax_graph = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis("off")

    cax_graph = fig.add_axes([0.55, 0.12, 0.015, 0.32])
    cax_info = fig.add_axes([0.95, 0.12, 0.015, 0.32])

    # -----------------------------
    # 3) Temporal plot (color by rectified IG)
    #    Colormap: white->red (truncate bwr to white->red half)
    # -----------------------------
    channel_spacing = 5
    top_y = C * channel_spacing

    cmap_pos = _truncate_cmap(cm.bwr, 0.5, 1.0)  # white -> red
    cmap_pos = _truncate_cmap(cm.OrRd)  # white -> red
    cmap_pos = _truncate_cmap(cm.bwr)  # white -> red
    cmap_pos = _truncate_cmap(cm.coolwarm)  # 

    if temp_expl is not None:
        ig_all = np.nan_to_num(ig_mean, nan=0.0, posinf=0.0, neginf=0.0)
        nonzero = ig_all[ig_all > 0]

        if nonzero.size == 0:
            vmax_display = 1.0
            norm_attr = Normalize(vmin=0.0, vmax=vmax_display)
            cbar_label = "ReLU(IG) (raw units)"
        else:
            # Robust vmax so a single outlier doesn't wash out the rest
            vmax_display = float(np.percentile(nonzero, 99))
            vmax_display = max(vmax_display, 1e-20)

            # Optional symlog-like handling for huge dynamic range (still works since values >= 0)
            p10 = float(np.percentile(nonzero, 10))
            p10 = max(p10, 1e-20)
            dynamic = vmax_display / p10

            if dynamic > 10:
                norm_attr = SymLogNorm(linthresh=p10, vmin=0.0, vmax=vmax_display, base=10)
                cbar_label = "ReLU(IG) (raw units, symlog; vmax=99th pct)"
            else:
                norm_attr = Normalize(vmin=0.0, vmax=vmax_display)
                cbar_label = "ReLU(IG) (raw units; vmax=99th pct)"
    else:
        norm_attr = Normalize(vmin=0.0, vmax=1.0)
        cbar_label = "ReLU(IG) (raw units)"

    for ch in range(C):
        offset = ch * channel_spacing
        points = np.array([t_axis, x_mean[ch] + offset]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap_pos, norm=norm_attr)
        if temp_expl is not None and ig_mean.shape[1] > 1:
            lc.set_array(ig_mean[ch, :-1])  # rectified, >=0
        else:
            lc.set_array(np.array([0.0]))
        lc.set_linewidth(1.0)
        ax_main.add_collection(lc)

    ax_main.set_yticks([ch * channel_spacing for ch in range(C)])
    ax_main.set_yticklabels(channel_names)

    sm = plt.cm.ScalarMappable(cmap=cmap_pos, norm=norm_attr)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_main, pad=0.01)
    cbar.set_label(cbar_label)

    # -----------------------------
    # 3b) GT & Pred tracks (overlap-aware)
    # -----------------------------
    def _track_colors(values):
        cols = []
        for v in values:
            if v >= 0.95:
                cols.append((1, 0, 0, 1))  # red
            elif v <= 0.05:
                cols.append((0, 0, 1, 1))  # blue
            else:
                cols.append((1, 1, 0, 1))  # yellow: mixed overlap region
        return cols

    gt_y_pos = top_y + 10
    pred_y_pos = top_y + 14

    ax_main.scatter(
        t_axis, np.full_like(t_axis, gt_y_pos, dtype=float),
        c=_track_colors(gt_mean), s=6, marker="s", linewidths=0
    )

    if pred_batch_np is not None:
        ax_main.scatter(
            t_axis, np.full_like(t_axis, pred_y_pos, dtype=float),
            c=_track_colors(pred_mean), s=6, marker="s", linewidths=0
        )

    ax_main.text(t_axis[0] if len(t_axis) else 0, gt_y_pos + 0.8, "GT", fontsize=10, fontweight="bold")
    if pred_batch_np is not None:
        ax_main.text(t_axis[0] if len(t_axis) else 0, pred_y_pos + 0.8, "Pred", fontsize=10, fontweight="bold")

    ax_main.set_xlim(0, t_axis[-1] if len(t_axis) else 0)
    ax_main.set_ylim(-channel_spacing, top_y + channel_spacing + 18)
    ax_main.set_xlabel("Time (s)")
    ax_main.set_title("Interactive spatiotemporal explanation: ReLU(IG) (temporal) + GNNExplainer (graph)")

    # -----------------------------
    # 3c) Node norm based on MEAN ReLU(IG) over windows (global across batch)
    #     This improves contrast for node plotting because node scores are means.
    # -----------------------------
    if temp_expl is not None:
        tmp = np.nan_to_num(temp_expl, nan=0.0, posinf=0.0, neginf=0.0)   # (B, C, T)
        tmp = np.maximum(tmp, 0.0)                                        # ReLU
        node_all_means = tmp.mean(axis=2)                                 # (B, C) mean over time
        node_nonzero = node_all_means[node_all_means > 0]

        if node_nonzero.size == 0:
            node_norm = Normalize(vmin=0.0, vmax=1.0)
            node_cbar_label = "Mean ReLU(IG) per node (raw)"
        else:
            node_vmax = float(np.percentile(node_nonzero, 99))
            node_vmax = max(node_vmax, 1e-20)

            node_p10 = float(np.percentile(node_nonzero, 10))
            node_p10 = max(node_p10, 1e-20)

            node_dynamic = node_vmax / node_p10
            if node_dynamic > 0:#1e3:
                node_norm = SymLogNorm(linthresh=node_p10, vmin=0.0, vmax=node_vmax, base=10)
                node_cbar_label = "Mean ReLU(IG) per node (raw, symlog; vmax=99th pct)"
            else:
                node_norm = Normalize(vmin=0.0, vmax=node_vmax)
                node_cbar_label = "Mean ReLU(IG) per node (raw; vmax=99th pct)"
    else:
        node_norm = Normalize(vmin=0.0, vmax=1.0)
        node_cbar_label = "Mean ReLU(IG) per node (raw)"

    # -----------------------------
    # 4) Graph setup (directed -> undirected merge for display)
    # -----------------------------
    node_pos = {
        0: (-2, 2), 1: (-2, 0), 2: (-1, 2), 3: (-1, 0),
        4: (1, 2), 5: (1, 0), 6: (2, 2), 7: (2, 0),
        8: (-1.5, 1), 9: (-0.5, 1), 10: (0.5, 1),  11: (1.5, 1),
    }


    edge_index_np = _to_np(edge_index).astype(int)
    directed_edges = [(int(edge_index_np[0, i]), int(edge_index_np[1, i]))
                      for i in range(edge_index_np.shape[1])]

    undirected_map = {}
    for idx, (u, v) in enumerate(directed_edges):
        key = (u, v) if u <= v else (v, u)
        undirected_map.setdefault(key, []).append(idx)

    undirected_edges = list(undirected_map.keys())

    G = nx.Graph()
    G.add_nodes_from(range(C))
    G.add_edges_from(undirected_edges)

    # -----------------------------
    # 5) Draw graph for selected window (GNNExplainer masks untouched)
    # -----------------------------
    def draw_graph_for_batch(batch_idx: int):
        ax_graph.clear()
        cax_graph.cla()

         # --- NODE importance from temporal IG (RAW mean ReLU(IG) for this window) ---
        if temp_expl is not None:
            ig_win = np.asarray(temp_expl[batch_idx])  # expected (C, T)
            ig_win = np.nan_to_num(ig_win, nan=0.0, posinf=0.0, neginf=0.0)
            ig_win = np.maximum(ig_win, 0.0)  # ReLU

            ig_win = np.squeeze(ig_win)

            # Ensure shape (C, T)
            if ig_win.ndim == 1:
                node_raw = ig_win
            else:
                if ig_win.shape[0] != C and ig_win.shape[-1] == C:
                    ig_win = ig_win.T
                node_raw = ig_win.mean(axis=1)  # RAW mean ReLU(IG) per node

            # Map RAW values via node_norm -> [0,1] for plotting with networkx
            node_vis = np.array(node_norm(node_raw))
            node_vis = np.clip(node_vis, 0.0, 1.0)

        # Fallback if no IG was passed in
        elif spat_node_expl is not None:
            node_mask_raw = np.asarray(spat_node_expl[batch_idx])  # (C, F)
            k = 3  # top 3 of features
            node_mask = np.sort(node_mask_raw, axis=1)[:, -k:].mean(axis=1)
            node_mask = np.clip(node_mask, 0.0, 1.0)
            node_raw = node_mask
            node_vis = node_mask

        else:
            node_raw = np.zeros(C, dtype=float)
            node_vis = np.zeros(C, dtype=float)

        if spat_edge_expl is not None:
            edge_mask_dir = np.asarray(spat_edge_expl[batch_idx]).astype(float)
            edge_mask = np.zeros(len(undirected_edges), dtype=float)
            for k, e in enumerate(undirected_edges):
                idxs = undirected_map[e]
                edge_mask[k] = float(np.mean(edge_mask_dir[idxs]))
            edge_mask = np.clip(edge_mask, 0.0, 1.0)
        else:
            edge_mask = np.ones(len(undirected_edges), dtype=float)

        for (u, v), w in zip(undirected_edges, edge_mask):
            nx.draw_networkx_edges(
                G, node_pos, ax=ax_graph,
                edgelist=[(u, v)],
                width=0.5 + 4.0 * w,
                alpha=0.15 + 0.85 * w,
                edge_color="black",
            )

        nx.draw_networkx_nodes(
            G, node_pos, ax=ax_graph,
            node_size=1000 + 2600 * node_vis//5,
            node_color=node_vis,
            cmap=plt.cm.coolwarm,
            vmin=0, vmax=1,
            alpha=1.00,
        )

        nx.draw_networkx_labels(
            G, node_pos, ax=ax_graph,
            labels={i: channel_names[i] for i in range(C)},
            font_size=10, font_weight="bold",font_color="white",
        )

        sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=node_norm)
        sm_nodes.set_array([])
        cb = plt.colorbar(sm_nodes, cax=cax_graph)
        cb.set_label(node_cbar_label)

        ax_graph.set_title(f"Spatial explanation - window {batch_idx}")
        ax_graph.axis("off")

        

         # --- NODE importance from GNNexpainer(RAW mean ReLU(IG) for this window) ---
         # # garph based on GNNexplainer node features
        ax_info.clear()
        cax_info.cla()
        if spat_node_expl is not None:
            node_mask_raw = np.asarray(spat_node_expl[batch_idx])  # (C, F)
            k = 3  # top 3 of features
            node_mask = np.sort(node_mask_raw, axis=1)[:, -k:].mean(axis=1)
            node_mask = np.clip(node_mask, 0.0, 1.0)
            node_raw = node_mask
            node_vis = node_mask
            node_norm2 = Normalize(vmin=0.0, vmax=1.0)

        else:
            node_raw = np.zeros(C, dtype=float)
            node_vis = np.zeros(C, dtype=float)
            node_norm2 = Normalize(vmin=0.0, vmax=1.0)

        if spat_edge_expl is not None:
            edge_mask_dir = np.asarray(spat_edge_expl[batch_idx]).astype(float)
            edge_mask = np.zeros(len(undirected_edges), dtype=float)
            for k, e in enumerate(undirected_edges):
                idxs = undirected_map[e]
                edge_mask[k] = float(np.mean(edge_mask_dir[idxs]))
            edge_mask = np.clip(edge_mask, 0.0, 1.0)
        else:
            edge_mask = np.ones(len(undirected_edges), dtype=float)

        for (u, v), w in zip(undirected_edges, edge_mask):
            nx.draw_networkx_edges(
                G, node_pos, ax=ax_info,
                edgelist=[(u, v)],
                width=0.5 + 4.0 * w,
                alpha=0.15 + 0.85 * w,
                edge_color="black",
            )

        nx.draw_networkx_nodes(
            G, node_pos, ax=ax_info,
            node_size=1000 + 2600 * node_vis//5,
            node_color=node_vis,
            cmap=plt.cm.coolwarm,
            vmin=0, vmax=1,
            alpha=1.00,
        )

        nx.draw_networkx_labels(
            G, node_pos, ax=ax_info,
            labels={i: channel_names[i] for i in range(C)},
            font_size=10, font_weight="bold",font_color="white",
        )

        sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=node_norm2)
        sm_nodes.set_array([])
        cb = plt.colorbar(sm_nodes, cax=cax_info)
        cb.set_label(node_cbar_label)

        ax_info.set_title(f"Spatial explanation - window {batch_idx}")
        ax_info.axis("off")
        # # ----- Info panel -----

        # ax_info.clear()
        # ax_info.axis("off")

        # s, e = batch_indices[batch_idx]
        # info_text = (
        #     f"Window: {batch_idx}\n"
        #     f"Time: {s / fs:.2f}s – {e / fs:.2f}s\n"
        #     f"Label: {int(y_batch[batch_idx])}\n"
        # )
        # if pred_batch_np is not None:
        #     info_text += f"Pred (window scalar): {float(pred_batch_np[batch_idx]):.3f}\n"

        # info_text += "\nTop nodes (mean IG ReLU, window):\n"
        # top_nodes = np.argsort(node_raw)[::-1][:5]
        # for i, n in enumerate(top_nodes):
        #     info_text += f"  {i+1}. {channel_names[n]}: {node_raw[n]:.3f}| mean GNNexpl score: {spat_node_expl[batch_idx][n].mean():.3f}\n"
        #     # info_text += f"  {i+1}. {channel_names[n]}: {node_raw[n]:.3f}\n"

        # info_text += "\nTop edges (GNNExplainer):\n"
        # top_edges = np.argsort(edge_mask)[::-1][:5]
        # for i, eid in enumerate(top_edges):
        #     a, b = undirected_edges[eid]
        #     info_text += f"  {i+1}. {channel_names[a]} ↔ {channel_names[b]}: {edge_mask[eid]:.3f}\n"

        # info_text += "\nControls:\n  ← / →  : previous/next window\n  click  : select window at time"

        # ax_info.text(
        #     0.05, 0.95, info_text,
        #     va="top", fontsize=10,
        #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        # )




    # -----------------------------
    # 6) Interaction
    # -----------------------------
    centers = np.array([(s + e) / 2 for s, e in batch_indices], dtype=float)

    class InteractionContext:
        def __init__(self):
            self.current_batch = 0
            self.highlight_rect = ax_main.axvspan(0, 0, color="gray", alpha=0.2)
            self.update_view()

        def update_view(self):
            draw_graph_for_batch(self.current_batch)
            s, e = batch_indices[self.current_batch]
            self.highlight_rect.remove()
            self.highlight_rect = ax_main.axvspan(s / fs, e / fs, color="gray", alpha=0.2)
            fig.canvas.draw_idle()

        def on_key(self, event):
            if event.key == "right":
                self.current_batch = min(self.current_batch + 1, B - 1)
                self.update_view()
            elif event.key == "left":
                self.current_batch = max(self.current_batch - 1, 0)
                self.update_view()

        def on_click(self, event):
            if event.inaxes != ax_main or event.xdata is None:
                return
            click_sample = float(event.xdata) * fs
            candidates = [b for b, (s, e) in enumerate(batch_indices) if s <= click_sample <= e]
            if candidates:
                self.current_batch = max(candidates, key=lambda b: batch_indices[b][0])
            else:
                self.current_batch = int(np.argmin(np.abs(centers - click_sample)))
            self.update_view()

    tracker = InteractionContext()
    fig.canvas.mpl_connect("key_press_event", tracker.on_key)
    fig.canvas.mpl_connect("button_press_event", tracker.on_click)

    if show_plots:
        plt.show()

    return fig, ax_main

# ==========================
# Quantus support helpers
# ==========================
# NOTE: These helpers do NOT change existing behaviour of calculateIG/calculateGNNexpl.
# They are additional utilities to make Quantus evaluation easier/cleaner.


def calculateGNNexpl_nodefeatures(gnn_model, node_features, edge_index, batch=None, epochs: int = 200, **kwargs):
    """Run PyG GNNExplainer on *precomputed* node_features.

    Why this exists:
      Quantus metrics often assume that x_batch (the thing being perturbed) and a_batch (the explanation)
      live in the same feature space. Your original calculateGNNexpl() deliberately computes node_features
      via model.cnn(...) and then explains model.gnn(...) in that space.

      This helper lets you keep that exact design, but pass node_features directly so external tooling
      can perturb node_features and re-run the explainer consistently.

    Args:
        gnn_model: typically model.gnn
        node_features: Tensor [num_nodes, num_node_features] (e.g., [12, 24])
        edge_index: Tensor [2, num_edges]
        batch: Optional Tensor [num_nodes] mapping nodes to graphs; if None, treated as single graph.
        epochs: GNNExplainer optimisation epochs.

    Returns:
        torch_geometric.explain.Explanation with .node_mask and .edge_mask
    """
    explainer = Explainer(
        model=gnn_model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    explanation = explainer(
        x=node_features,
        edge_index=edge_index,
        batch=batch,
    )
    return explanation


def edge_mask_from_channel_gt(edge_index: torch.Tensor, gt_channels: Sequence[int], mode: str = "either") -> np.ndarray:
    """Create an *edge-level* binary mask from a set of ground-truth channels (nodes).

    If you only have channel-level ground truth but want to evaluate edge explanations, you can derive
    a reference mask over edges:
      - mode='either': an edge is relevant if *either* endpoint is in gt_channels.
      - mode='both':   an edge is relevant if *both* endpoints are in gt_channels.

    Returns:
        np.ndarray shape (num_edges,) with values in {0,1}.
    """
    if edge_index is None:
        raise ValueError("edge_index is None")
    gt = set(int(i) for i in gt_channels)
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()

    if mode not in ("either", "both"):
        raise ValueError("mode must be 'either' or 'both'")

    if mode == "either":
        mask = np.array([(s in gt) or (d in gt) for s, d in zip(src, dst)], dtype=int)
    else:
        mask = np.array([(s in gt) and (d in gt) for s, d in zip(src, dst)], dtype=int)

    return mask
