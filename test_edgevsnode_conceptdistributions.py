import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

import Models_senn as Model
import MyUtils_senn_test as MyUtils

# =========================
# USER SETTINGS
# =========================
LOG = "470501"
IS_TRIVIAL = False
LOG = "481616"
IS_TRIVIAL = True
FOLD = "5"
CKPT_NAME = "best_auprc.pt"
MODEL_SUBDIR = f"GAT_CV_10_{FOLD}"

DATA_FOLDER = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets\CV_Folds"
MODEL_DIR = f"./Saved_models_{LOG}"

USE_FULL_TESTSET = True
BATCH_SIZE = 128

# Optional: set False if you only want summary tables
SHOW_PLOTS = True

# EEG channel names from your model file
CHANNEL_NAMES = [
    "Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2",
    "Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"
]


def safe_normalize(x, norm_dict):
    mean = norm_dict.get("mean", np.nan)
    std = norm_dict.get("std", np.nan)

    # Your training code sometimes stores NaN normalization if global_dataset=False.
    if np.isfinite(mean) and np.isfinite(std) and std > 0:
        return (x - mean) / std
    return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Load checkpoint + model
    # -------------------------
    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR)
    ckpt = torch.load(os.path.join(model_path, CKPT_NAME), weights_only=False)
    if not IS_TRIVIAL:
        model = Model.SENN_fixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=True,   # needed for explanation_edge
        ).to(device)
    else:
        model = Model.SENN_trivialfixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=True,   # needed for explanation_edge
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -------------------------
    # Load data
    # -------------------------
    fold_dir = os.path.join(DATA_FOLDER, f"fold_{FOLD}")
    x_test = np.load(os.path.join(fold_dir, "testdata.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(fold_dir, "testlabels.npy"), mmap_mode="r")

    # x_test = safe_normalize(x_test, ckpt.get("normalization", {}))

    if not USE_FULL_TESTSET:
        idx_non = np.where(y_test == 0)[0]
        idx_seiz = np.where(y_test == 1)[0]
        n = min(len(idx_non), len(idx_seiz), 500)
        idx = np.concatenate([idx_non[:n], idx_seiz[:n]])
        np.random.shuffle(idx)
        x_test = x_test[idx]
        y_test = y_test[idx]

    testset = MyUtils.prepare_graphs_labels(x_test, y_test, Model.adj)
    loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # edge names
    edge_index_np = testset[0].edge_index.cpu().numpy()
    edge_names = [
        f"{CHANNEL_NAMES[s]} -> {CHANNEL_NAMES[t]}"
        for s, t in zip(edge_index_np[0], edge_index_np[1])
    ]
    num_edges = len(edge_names)

    # -------------------------
    # Collect outputs
    # -------------------------
    sample_rows = []
    edge_rows = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)

            B = batch.y.shape[0]

            # shapes:
            # h_x_edge        : (E_total, 1)
            # theta_x_edge    : (E_total, 1)
            # explanation_edge: (E_total, 1)
            # logit_edge      : (B, 1)
            # logit_node      : (B, 1)
            h_edge = out["h_x_edge"].view(B, num_edges, -1).cpu().numpy()[..., 0]
            theta_edge = out["theta_x_edge"].view(B, num_edges, -1).cpu().numpy()[..., 0]
            expl_edge = out["explanation_edge"].view(B, num_edges, -1).cpu().numpy()[..., 0]

            logit_edge = out["logit_edge"].view(B).cpu().numpy()
            logit_node = out["logit_node"].view(B).cpu().numpy()
            logit_total = out["logit"].view(B).cpu().numpy()
            prob = out["prob"].view(B).cpu().numpy()
            y = batch.y.view(B).cpu().numpy().astype(int)

            for i in range(B):
                # per-sample summaries
                abs_node = abs(logit_node[i])
                abs_edge = abs(logit_edge[i])
                edge_share_abs = abs_edge / (abs_node + abs_edge + 1e-12)

                top_idx = int(np.argmax(np.abs(expl_edge[i])))

                sample_rows.append({
                    "label": y[i],
                    "prob": prob[i],
                    "logit_total": logit_total[i],
                    "logit_node": logit_node[i],
                    "logit_edge": logit_edge[i],
                    "edge_share_abs": edge_share_abs,

                    "coh_mean": float(np.mean(h_edge[i])),
                    "coh_median": float(np.median(h_edge[i])),
                    "coh_max": float(np.max(h_edge[i])),

                    "theta_mean": float(np.mean(theta_edge[i])),
                    "theta_abs_mean": float(np.mean(np.abs(theta_edge[i]))),
                    "theta_maxabs": float(np.max(np.abs(theta_edge[i]))),

                    "contrib_mean": float(np.mean(expl_edge[i])),
                    "contrib_abs_mean": float(np.mean(np.abs(expl_edge[i]))),
                    "contrib_maxabs": float(np.max(np.abs(expl_edge[i]))),

                    "top_edge_name": edge_names[top_idx],
                    "top_edge_contrib": float(expl_edge[i, top_idx]),
                    "top_edge_coh": float(h_edge[i, top_idx]),
                    "top_edge_theta": float(theta_edge[i, top_idx]),
                })

                # per-edge rows
                for e in range(num_edges):
                    edge_rows.append({
                        "label": y[i],
                        "edge_name": edge_names[e],
                        "coherence": float(h_edge[i, e]),
                        "theta": float(theta_edge[i, e]),
                        "abs_theta": float(abs(theta_edge[i, e])),
                        "contribution": float(expl_edge[i, e]),
                        "abs_contribution": float(abs(expl_edge[i, e])),
                        "logit_edge": float(logit_edge[i]),
                        "logit_total": float(logit_total[i]),
                    })

    df_samples = pd.DataFrame(sample_rows)
    df_edges = pd.DataFrame(edge_rows)

    # -------------------------
    # Print summary
    # -------------------------
    print("\n=== Per-sample summary by class ===")
    print(
        df_samples.groupby("label")[
            [
                "coh_mean", "coh_median", "coh_max",
                "theta_mean", "theta_abs_mean", "theta_maxabs",
                "contrib_mean", "contrib_abs_mean", "contrib_maxabs",
                "logit_node", "logit_edge", "edge_share_abs"
            ]
        ].agg(["mean", "std", "median"])
    )

    print("\n=== Spearman correlations per class ===")
    for lab in [0, 1]:
        sub = df_samples[df_samples["label"] == lab]
        corr = sub[
            ["coh_mean", "theta_abs_mean", "contrib_abs_mean", "logit_edge", "edge_share_abs", "prob"]
        ].corr(method="spearman")
        print(f"\nLabel = {lab}")
        print(corr.round(3))

    print("\n=== Top edges by mean |contribution| per class ===")
    top_edges = (
        df_edges.groupby(["label", "edge_name"])["abs_contribution"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    print(top_edges.groupby("label").head(10))

    # -------------------------
    # Quick plots
    # -------------------------
    if SHOW_PLOTS:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        # 1) coherence by class
        axes[0, 0].hist(df_samples[df_samples.label == 0]["coh_mean"], bins=50, alpha=0.5, label="Non-seiz")
        axes[0, 0].hist(df_samples[df_samples.label == 1]["coh_mean"], bins=50, alpha=0.5, label="Seiz")
        axes[0, 0].set_title("Mean coherence per sample")
        axes[0, 0].legend()

        # 2) theta magnitude by class
        axes[0, 1].hist(df_samples[df_samples.label == 0]["theta_abs_mean"], bins=50, alpha=0.5, label="Non-seiz")
        axes[0, 1].hist(df_samples[df_samples.label == 1]["theta_abs_mean"], bins=50, alpha=0.5, label="Seiz")
        axes[0, 1].set_title("Mean |edge relevance| per sample")
        axes[0, 1].legend()

        # 3) contribution magnitude by class
        axes[0, 2].hist(df_samples[df_samples.label == 0]["contrib_abs_mean"], bins=50, alpha=0.5, label="Non-seiz")
        axes[0, 2].hist(df_samples[df_samples.label == 1]["contrib_abs_mean"], bins=50, alpha=0.5, label="Seiz")
        axes[0, 2].set_title("Mean |edge contribution| per sample")
        axes[0, 2].legend()

        # 4) coherence vs relevance
        for lab, name in [(0, "Non-seiz"), (1, "Seiz")]:
            sub = df_samples[df_samples.label == lab]
            axes[1, 0].scatter(sub["coh_mean"], sub["theta_abs_mean"], s=10, alpha=0.5, label=name)
        axes[1, 0].set_xlabel("Mean coherence")
        axes[1, 0].set_ylabel("Mean |theta_edge|")
        axes[1, 0].set_title("Coherence vs relevance")
        axes[1, 0].legend()

        # 5) coherence vs contribution
        for lab, name in [(0, "Non-seiz"), (1, "Seiz")]:
            sub = df_samples[df_samples.label == lab]
            axes[1, 1].scatter(sub["coh_mean"], sub["contrib_abs_mean"], s=10, alpha=0.5, label=name)
        axes[1, 1].set_xlabel("Mean coherence")
        axes[1, 1].set_ylabel("Mean |h*theta|")
        axes[1, 1].set_title("Coherence vs contribution")
        axes[1, 1].legend()

        # 6) edge branch dominance
        axes[1, 2].hist(df_samples[df_samples.label == 0]["edge_share_abs"], bins=50, alpha=0.5, label="Non-seiz")
        axes[1, 2].hist(df_samples[df_samples.label == 1]["edge_share_abs"], bins=50, alpha=0.5, label="Seiz")
        axes[1, 2].set_title("|logit_edge| / (|logit_node| + |logit_edge|)")
        axes[1, 2].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()