#!/usr/bin/env python3
"""
Summarize hyperparameter sweep results across Saved_models_* run folders.

This script scans all saved-model run directories, groups checkpoints by run,
extracts model type and hyperparameters from checkpoint metadata,
aggregates fold-level metrics across the 10 CV folds, and creates:

- leaderboard CSV across all runs
- best-per-model CSV
- per-run per-fold CSVs
- LR x WD heatmaps per model type

Two evaluation modes are supported:
1) checkpoint: fast, reads metrics already stored in best_auprc.pt / best.pt
2) recompute: slow, reloads each model and recomputes metrics on the test fold

Recommended use for sweep ranking:
- rank within each model type by mean AUPRC
- tie break by median AUPRC, then lower std AUPRC
- optionally inspect mean AUPRG in recompute mode
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEFAULT_MODEL_PATTERN = "Saved_models_*"
DEFAULT_CHECKPOINT_NAME = "best_auprc.pt"
DEFAULT_OUTPUT_DIR = "Results_sweep_summary"
EPS = 1e-12


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.6g}"


def safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

def extract_hparams_from_run_name(run_name: str) -> Dict[str, object]:
    """
    Extract hyperparameters from names like:
    Saved_models_492093_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-3
    """
    float_pat = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"

    out = {
        "model_type": None,
        "lr": np.nan,
        "wd": np.nan,
        "rob_loss": np.nan,
    }

    model_match = re.search(r"_MT([^_]+)", run_name)
    if model_match:
        out["model_type"] = model_match.group(1)

    lr_match = re.search(rf"_LR({float_pat})", run_name, flags=re.IGNORECASE)
    if lr_match:
        out["lr"] = safe_float(lr_match.group(1))

    wd_match = re.search(rf"_WD({float_pat})", run_name, flags=re.IGNORECASE)
    if wd_match:
        out["wd"] = safe_float(wd_match.group(1))

    rob_match = re.search(
        rf"_(?:robloss|rob_loss|RobLoss|lambda1|l1)({float_pat})",
        run_name,
        flags=re.IGNORECASE,
    )
    if rob_match:
        out["rob_loss"] = safe_float(rob_match.group(1))

    return out

def extract_fold_index(path: str) -> Optional[int]:
    match = re.search(r"GAT_CV_10_(\d+)", path)
    return int(match.group(1)) if match else None


def natural_sort_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Scan runs and checkpoints
# -----------------------------------------------------------------------------

def find_fold_checkpoints(run_dir: Path, checkpoint_name: str) -> List[Tuple[int, Path]]:
    """Find fold checkpoints inside one run directory.

    Supports both direct layout:
        Saved_models_xxx/GAT_CV_10_0/best_auprc.pt
    and nested layouts produced by run subfolders.
    """
    matches = glob.glob(str(run_dir / "**" / checkpoint_name), recursive=True)
    fold_to_path: Dict[int, Path] = {}
    for match in matches:
        fold = extract_fold_index(match)
        if fold is None:
            continue
        fold_to_path[fold] = Path(match)
    return sorted(fold_to_path.items(), key=lambda x: x[0])



def load_checkpoint(checkpoint_path: Path) -> dict:
    return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)



def extract_hparams_from_checkpoint(checkpoint: dict) -> Tuple[Optional[str], float, float]:
    model_type = checkpoint.get("model_type")
    opt_state = checkpoint.get("optimizer_state_dict", {}) or {}
    param_groups = opt_state.get("param_groups", []) or []
    if not param_groups:
        return model_type, np.nan, np.nan

    group0 = param_groups[0]
    # Prefer initial_lr because current 'lr' may be the scheduler-updated value.
    lr = safe_float(group0.get("initial_lr", group0.get("lr", np.nan)))
    wd = safe_float(group0.get("weight_decay", np.nan))
    return model_type, lr, wd



def extract_metrics_from_checkpoint(checkpoint: dict) -> Dict[str, float]:
    metrics = checkpoint.get("metrics", {}) or {}
    out = {
        "AUPRC": safe_float(metrics.get("auprc")),
        "AUROC": safe_float(metrics.get("auroc")),
        "Kappa": safe_float(metrics.get("kappa")),
        "Threshold": safe_float(metrics.get("threshold")),
        "Recall": safe_float(metrics.get("recall")),
        "Precision": safe_float(metrics.get("precision")),
        "F1": safe_float(metrics.get("f1")),
        "Best_Epoch": safe_float(checkpoint.get("epoch", np.nan)),
    }
    return out


# -----------------------------------------------------------------------------
# Recompute mode helpers
# -----------------------------------------------------------------------------

def precision_gain(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pi = np.mean(y_true)
    precision = tp / (tp + fp + EPS)
    return (precision - pi) / ((1.0 - pi) * precision + EPS)



def recall_gain(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pi = np.mean(y_true)
    recall = tp / (tp + fn + EPS)
    return (recall - pi) / ((1.0 - pi) * recall + EPS)



def auprg(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import auc, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    pi = np.mean(y_true)

    precision_gain_values = (precision - pi) / ((1.0 - pi) * np.clip(precision, EPS, None))
    recall_gain_values = (recall - pi) / ((1.0 - pi) * np.clip(recall, EPS, None))

    mask = np.isfinite(precision_gain_values) & np.isfinite(recall_gain_values)
    precision_gain_values = precision_gain_values[mask]
    recall_gain_values = recall_gain_values[mask]

    if len(recall_gain_values) == 0:
        return np.nan

    order = np.argsort(recall_gain_values)
    precision_gain_values = precision_gain_values[order]
    recall_gain_values = recall_gain_values[order]

    non_negative = np.where(recall_gain_values >= 0.0)[0]
    if len(non_negative) == 0:
        return 0.0

    first = non_negative[0]
    if first > 0 and recall_gain_values[first] > 0.0:
        x1, y1 = recall_gain_values[first - 1], precision_gain_values[first - 1]
        x2, y2 = recall_gain_values[first], precision_gain_values[first]
        y_at_zero = y1 + (0.0 - x1) * (y2 - y1) / (x2 - x1 + EPS)
        recall_gain_values = np.concatenate(([0.0], recall_gain_values[first:]))
        precision_gain_values = np.concatenate(([y_at_zero], precision_gain_values[first:]))
    else:
        recall_gain_values = recall_gain_values[first:]
        precision_gain_values = precision_gain_values[first:]

    if recall_gain_values[0] > 0.0:
        recall_gain_values = np.concatenate(([0.0], recall_gain_values))
        precision_gain_values = np.concatenate(([precision_gain_values[0]], precision_gain_values))

    if recall_gain_values[-1] < 1.0:
        recall_gain_values = np.concatenate((recall_gain_values, [1.0]))
        precision_gain_values = np.concatenate((precision_gain_values, [0.0]))

    recall_gain_values, unique_idx = np.unique(recall_gain_values, return_index=True)
    precision_gain_values = precision_gain_values[unique_idx]

    return auc(recall_gain_values, precision_gain_values)



def kappa_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    thresholds = np.linspace(0.01, 0.99, 99)
    kappas = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        kappas.append(cohen_kappa_score(y_true, y_pred))
    return float(thresholds[int(np.argmax(kappas))])



def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return float(thresholds[idx])


def F2_threshold(y_true, y_prob):
    from sklearn.metrics import fbeta_score
    thresholds = np.linspace(0.01, 0.99, 99)
    f2s = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2s.append(fbeta_score(y_true, y_pred,beta=2))

    return thresholds[np.argmax(f2s)]

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        fbeta_score
    )

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp + EPS),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "AUPRG": auprg(y_true, y_prob),
        "Precision_Gain": precision_gain(y_true, y_prob, threshold),
        "Recall_Gain": recall_gain(y_true, y_prob, threshold),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true,y_pred,beta=2),
        "Threshold": threshold,
    }



def lazy_import_project_modules():
    import Models_senn as Model
    import MyUtils_senn_test as MyUtils
    from torch_geometric.loader import DataLoader

    return Model, MyUtils, DataLoader



def instantiate_model(model_type: str, Model):
    if model_type == "base":
        return Model.EEG_GAT_Model()
    if model_type == "SENNrawx":
        return Model.SENN_raw()
    if model_type == "SENNfixed":
        return Model.SENN_fixedconcepts()
    if model_type == "SENNtrivialfixed":
        return Model.SENN_trivialfixedconcepts()
    if model_type == "SENNfixed_concepttheta":
        return Model.SENN_fixedconcepts_concepttheta()
    if model_type == "LogisticConcepts":
        return Model.ConceptLogisticDual()
    raise ValueError(f"Unsupported model_type: {model_type}")



def load_fold_data(data_folder: Path, fold: int, Model, MyUtils, no_overlap: bool = True):
    fold_dir = data_folder / f"fold_{fold}"
    x_test = np.load(fold_dir / "testdata.npy", mmap_mode="r")
    y_test = np.load(fold_dir / "testlabels.npy", mmap_mode="r")

    if no_overlap:
        idx_yes_seiz = np.where(y_test == 1)[0]
        idx_no_seiz = np.where(y_test == 0)[0]
        fs = 32
        t_window = len(x_test[0][0]) / fs
        t_overlap = 10
        t_overlap_seiz = 11
        thin_skip = int(t_window / (t_window - t_overlap))
        thin_skip_seiz = int(t_window / (t_window - t_overlap_seiz))
        thin_idx_no_seiz = idx_no_seiz[0::thin_skip]
        thin_idx_yes_seiz = idx_yes_seiz[0::thin_skip_seiz]
        keep_idx = np.sort(np.concatenate([thin_idx_no_seiz, thin_idx_yes_seiz]))
        x_test = x_test[keep_idx]
        y_test = y_test[keep_idx]

    testdata = MyUtils.prepare_graphs_labels(x_test, y_test, Model.adj)
    return testdata



def recompute_fold_metrics(
    checkpoint: dict,
    fold: int,
    model_type: str,
    data_folder: Path,
    threshold_policy: str,
    fixed_threshold: float,
    device: str,
) -> Dict[str, float]:
    Model, MyUtils, DataLoader = lazy_import_project_modules()
    model = instantiate_model(model_type, Model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    testdata = load_fold_data(data_folder, fold, Model, MyUtils)
    test_loader = DataLoader(
        testdata,
        batch_size=128,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
    )

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            out = model(batch.x, batch.edge_index, batch.batch)
            all_probs.append(out["prob"].detach().cpu())
            all_labels.append(batch.y.cpu())

    y_prob = torch.cat(all_probs).numpy().ravel()
    y_true = torch.cat(all_labels).numpy().ravel()

    if threshold_policy == "saved":
        threshold = safe_float((checkpoint.get("metrics") or {}).get("threshold"), 0.5)
    elif threshold_policy == "fixed":
        threshold = float(fixed_threshold)
    elif threshold_policy == "kappa":
        threshold = kappa_threshold(y_true, y_prob)
    elif threshold_policy == "youden":
        threshold = youden_threshold(y_true, y_prob)
    elif threshold_policy == "F2":
        threshold = F2_threshold(y_true, y_prob)
    else:
        raise ValueError(f"Unsupported threshold_policy: {threshold_policy}")

    metrics = compute_metrics(y_true, y_prob, threshold)
    metrics["Best_Epoch"] = safe_float(checkpoint.get("epoch", np.nan))
    return metrics


# -----------------------------------------------------------------------------
# Aggregation and ranking
# -----------------------------------------------------------------------------

def summarize_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Dict[str, float]:
    exclude = exclude or []
    numeric_cols = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]

    summary: Dict[str, float] = {}
    for col in numeric_cols:
        summary[f"mean_{col}"] = df[col].mean()
        summary[f"median_{col}"] = df[col].median()
        summary[f"std_{col}"] = df[col].std(ddof=1) if len(df[col]) > 1 else 0.0
        summary[f"min_{col}"] = df[col].min()
        summary[f"max_{col}"] = df[col].max()
    return summary



def rank_runs_within_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    score_col = f"mean_{metric}"
    median_col = f"median_{metric}"
    std_col = f"std_{metric}"

    ranked = df.sort_values(
        by=["model_type", score_col, median_col, std_col],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).copy()
    ranked["rank_within_model"] = ranked.groupby("model_type").cumcount() + 1
    return ranked


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_heatmap_for_model(
    model_df: pd.DataFrame,
    model_type: str,
    metric: str,
    output_dir: Path,
) -> None:
    score_col = f"mean_{metric}"
    if score_col not in model_df.columns:
        return

    pivot = model_df.pivot_table(index="wd", columns="lr", values=score_col, aggfunc="median")
    if pivot.empty:
        return

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(pivot.columns)), max(4, 0.8 * len(pivot.index))))
    im = ax.imshow(pivot.values, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(score_col)

    ax.set_title(f"{model_type} | mean {metric} across folds")
    ax.set_xlabel("LR")
    ax.set_ylabel("WD")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([format_float(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([format_float(v) for v in pivot.index])

    values = pivot.values
    finite_vals = values[np.isfinite(values)]
    threshold = np.nanmedian(finite_vals) if finite_vals.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text = "nan" if not np.isfinite(val) else f"{val:.3f}"
            text_color = "white" if np.isfinite(val) and val < threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    save_path = output_dir / f"heatmap_{model_type}_{metric}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pivot.to_csv(output_dir / f"heatmap_{model_type}_{metric}.csv")


    # Try to run rob loss plot if present
    try:
        # Allow rob loss either as a column or as an index level.
        df = model_df.copy()
        possible_rob_cols = ["rob_loss", "RobLoss", "robloss", "lambda1", "l1"]

        rob_col = None

        for col in possible_rob_cols:
            if col in df.columns:
                rob_col = col
                break

        if rob_col is None:
            index_names = list(df.index.names)
            for col in possible_rob_cols:
                if col in index_names:
                    df = df.reset_index()
                    rob_col = col
                    break

        if rob_col is None:
            return

        required_cols = {rob_col, "lr", "wd", score_col}
        if not required_cols.issubset(df.columns):
            return

        df = df.copy()
        df[rob_col] = pd.to_numeric(df[rob_col], errors="coerce")
        df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
        df["wd"] = pd.to_numeric(df["wd"], errors="coerce")
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

        df = df.dropna(subset=[rob_col, "lr", "wd", score_col])

        if df.empty:
            return

        # Median over folds / repeated runs.
        rob_summary = (
            df.groupby([rob_col, "lr", "wd"], as_index=False)[score_col]
            .median()
            .sort_values([rob_col, "lr", "wd"])
        )

        if rob_summary.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        # One line per LR/WD combination.
        for (lr, wd), group in rob_summary.groupby(["lr", "wd"]):
            group = group.sort_values(rob_col)

            ax.plot(
                group[rob_col],
                group[score_col],
                marker="o",
                linewidth=1.5,
                label=f"LR={format_float(lr)}, WD={format_float(wd)}",
            )

        # A normal log axis cannot show x=0.
        # symlog gives a small linear region around zero and log scaling elsewhere.
        positive_rob = rob_summary.loc[rob_summary[rob_col] > 0, rob_col]

        if not positive_rob.empty:
            linthresh = positive_rob.min() / 10.0
            ax.set_xscale("symlog", linthresh=linthresh)
        else:
            ax.set_xscale("linear")

        unique_x = np.sort(rob_summary[rob_col].unique())
        ax.set_xticks(unique_x)
        ax.set_xticklabels([format_float(v) for v in unique_x], rotation=45, ha="right")

        ax.set_title(f"{model_type} | {metric} vs robustness loss")
        ax.set_xlabel("RobLoss")
        ax.set_ylabel(score_col)
        ax.set_ylim(-0.05,1.05)
        ax.grid(True, which="both", alpha=0.3)

        # Avoid huge unreadable legends.
        n_lines = rob_summary.groupby(["lr", "wd"]).ngroups
        if n_lines <= 12:
            ax.legend(fontsize=8)
        else:
            ax.legend(fontsize=7, ncol=2, title="LR / WD", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        rob_plot_path = output_dir / f"robloss_line_{model_type}_{metric}.png"
        plt.savefig(rob_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rob_summary.to_csv(output_dir / f"robloss_line_{model_type}_{metric}.csv", index=False)

    except Exception as e:
        print(f"lineplot failed exception: {e}")
        return
         


# -----------------------------------------------------------------------------
# Main sweep summarization
# -----------------------------------------------------------------------------

def summarize_run(
    run_dir: Path,
    checkpoint_name: str,
    mode: str,
    data_folder: Optional[Path],
    threshold_policy: str,
    fixed_threshold: float,
    device: str,
    per_run_output_dir: Path,
) -> Optional[Dict[str, object]]:
    fold_checkpoints = find_fold_checkpoints(run_dir, checkpoint_name)
    if not fold_checkpoints:
        return None

    per_fold_rows: List[Dict[str, object]] = []

    run_hparams = extract_hparams_from_run_name(run_dir.name)

    model_type = run_hparams["model_type"]
    lr = run_hparams["lr"]
    wd = run_hparams["wd"]
    rob_loss = run_hparams["rob_loss"]
    

    for fold, ckpt_path in fold_checkpoints:
        checkpoint = load_checkpoint(ckpt_path)
        model_type_fold, lr_fold, wd_fold = extract_hparams_from_checkpoint(checkpoint)

        # Prefer checkpoint metadata when present, otherwise fall back to run folder name.
        model_type_fold = model_type_fold or model_type
        lr_fold = lr_fold if not np.isnan(lr_fold) else lr
        wd_fold = wd_fold if not np.isnan(wd_fold) else wd

        model_type = model_type or model_type_fold

        if np.isnan(lr):
            lr = lr_fold
        if np.isnan(wd):
            wd = wd_fold

        if mode == "checkpoint":
            metrics = extract_metrics_from_checkpoint(checkpoint)
        elif mode == "recompute":
            if data_folder is None:
                raise ValueError("data_folder is required in recompute mode")
            metrics = recompute_fold_metrics(
                checkpoint=checkpoint,
                fold=fold,
                model_type=model_type_fold or model_type,
                data_folder=data_folder,
                threshold_policy=threshold_policy,
                fixed_threshold=fixed_threshold,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        row = {
            "run_name": run_dir.name,
            "fold": fold,
            "model_type": model_type_fold or model_type,
            "lr": lr_fold,
            "wd": wd_fold,
            "rob_loss": rob_loss,
            **metrics,
        }
        per_fold_rows.append(row)

    if not per_fold_rows:
        return None

    per_fold_df = pd.DataFrame(per_fold_rows).sort_values("fold")
    ensure_dir(per_run_output_dir)
    per_fold_df.to_csv(per_run_output_dir / f"{run_dir.name}_per_fold.csv", index=False)

    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "model_type": model_type,
        "lr": lr,
        "wd": wd,
        "rob_loss": rob_loss,
        "n_folds": int(len(per_fold_df)),
    }
    summary.update(summarize_numeric_columns(per_fold_df, exclude=["fold"]))

    with open(per_run_output_dir / f"{run_dir.name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".", help="Project root that contains Saved_models_* folders")
    parser.add_argument("--pattern", type=str, default=DEFAULT_MODEL_PATTERN, help="Glob pattern for run folders")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=DEFAULT_CHECKPOINT_NAME,
        choices=["best_auprc.pt", "best.pt"],
        help="Checkpoint type to summarize per fold",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "recompute"],
        help="checkpoint = use stored checkpoint metrics, recompute = rerun validation from saved weights",
    )
    parser.add_argument(
        "--score_metric",
        type=str,
        default="AUPRC",
        help="Metric used to rank runs and draw heatmaps, e.g. AUPRC, AUPRG, AUROC",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="Required for recompute mode; points to Datasets/CV_Folds",
    )
    parser.add_argument(
        "--threshold_policy",
        type=str,
        default="F2",
        choices=["saved", "fixed", "kappa", "youden","F2"],
        help="Threshold policy in recompute mode for threshold-dependent metrics",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Used when threshold_policy=fixed")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    per_run_output_dir = output_dir / "per_run"
    ensure_dir(output_dir)
    ensure_dir(per_run_output_dir)

    run_dirs = [Path(p) for p in glob.glob(str(root_dir / args.pattern)) if Path(p).is_dir()]
    run_dirs = sorted(run_dirs, key=lambda p: natural_sort_key(p.name))

    if not run_dirs:
        raise FileNotFoundError(f"No run folders found under {root_dir} with pattern {args.pattern}")

    summaries: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        try:
            summary = summarize_run(
                run_dir=run_dir,
                checkpoint_name=args.checkpoint_name,
                mode=args.mode,
                data_folder=(Path(args.data_folder).resolve() if args.data_folder else None),
                threshold_policy=args.threshold_policy,
                fixed_threshold=args.threshold,
                device=args.device,
                per_run_output_dir=per_run_output_dir,
            )
            if summary is not None:
                summaries.append(summary)
                print(
                    f"[OK] {run_dir.name}: model={summary['model_type']} lr={format_float(summary['lr'])} wd={format_float(summary['wd'])} "
                    f"n_folds={summary['n_folds']} mean_{args.score_metric}={summary.get(f'mean_{args.score_metric}', np.nan):.4f}"
                )
            else:
                print(f"[SKIP] {run_dir.name}: no fold checkpoints found")
        except Exception as exc:
            print(f"[ERROR] {run_dir.name}: {exc}")

    if not summaries:
        raise RuntimeError("No runs were summarized successfully.")

    leaderboard = pd.DataFrame(summaries)
    leaderboard = rank_runs_within_model(leaderboard, args.score_metric)
    score_col = f"mean_{args.score_metric}"
    median_col = f"median_{args.score_metric}"
    std_col = f"std_{args.score_metric}"
    sort_cols = ["model_type", "rank_within_model", score_col, median_col]
    leaderboard = leaderboard.sort_values(by=sort_cols, ascending=[True, True, False, False], kind="mergesort")

    leaderboard_path = output_dir / "leaderboard_all_runs.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    best_per_model = (
        leaderboard.sort_values(
            by=["model_type", score_col, median_col, std_col],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .groupby("model_type", as_index=False)
        .head(1)
        .copy()
    )
    best_per_model_path = output_dir / "best_per_model.csv"
    best_per_model.to_csv(best_per_model_path, index=False)

    for model_type, model_df in leaderboard.groupby("model_type"):
        plot_heatmap_for_model(model_df, model_type, args.score_metric, output_dir)

    print("\nDone.")
    print(f"Leaderboard:     {leaderboard_path}")
    print(f"Best per model:  {best_per_model_path}")
    print(f"Heatmaps folder: {output_dir}")
    print("\nSelection rule per model type:")
    print(f"1) highest mean {args.score_metric}")
    print(f"2) if tied: highest median {args.score_metric}")
    print(f"3) if tied: lowest std {args.score_metric}")


if __name__ == "__main__":
    main()
