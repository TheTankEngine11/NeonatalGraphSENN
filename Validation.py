#!/usr/bin/env python3
"""
Validation.py

Simple validation script for one cross-validation model run.
This script is meant to be called by RunValidationSweep.py.

Expected model layout:
    MODEL_DIR/GAT_CV_10_0/best_auprc.pt
    MODEL_DIR/GAT_CV_10_1/best_auprc.pt
    ...
    MODEL_DIR/GAT_CV_10_9/best_auprc.pt

Expected data layout:
    DATA_FOLDER/fold_0/testdata.npy
    DATA_FOLDER/fold_0/testlabels.npy
    ...
    DATA_FOLDER/fold_9/testdata.npy
    DATA_FOLDER/fold_9/testlabels.npy
"""

import argparse
import json
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader

import Models_senn as Model
import MyUtils_senn_test as MyUtils


# ============================================================
# USER SETTINGS
# ============================================================
N_FOLDS = 10
CHECKPOINT_NAME = "best_auprc.pt"

BATCH_SIZE = 128
NUM_WORKERS = 0
PIN_MEM = False

# "recompute" reloads the models and evaluates the test folds.
# "checkpoint" only reads the already stored checkpoint metrics.
MODE = "recompute"

# Threshold options: "fixed", "checkpoint", "youden", "kappa", "F2"
THRESHOLD_MODE = "F2"
FIXED_THRESHOLD = 0.5

# False means that overlapping test windows are thinned,
USE_FULL_OVERLAP_TESTSET = False

MAKE_PLOTS = True
PLOT_HISTORY = True
UPDATE_CHECKPOINT_THRESHOLD = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12
# ============================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def summarize_folds(rows):
    df = pd.DataFrame(rows)
    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.agg(["mean", "std", "median", iqr, q1, q3])
    return df, summary


def safe_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if math.isfinite(value) else np.nan


def checkpoint_path(model_dir, fold):
    return os.path.join(model_dir, f"GAT_CV_10_{fold}", CHECKPOINT_NAME)


def available_folds(model_dir):
    folds = []
    for fold in range(N_FOLDS):
        if os.path.isfile(checkpoint_path(model_dir, fold)):
            folds.append(fold)
    return folds


def load_checkpoint(model_dir, fold):
    ckpt_path = checkpoint_path(model_dir, fold)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {ckpt_path}")
    return ckpt, ckpt_path


def model_kind_from_checkpoint(ckpt):
    """
    Converts the model names stored during training to the names used by
    MyUtils.build_model().
    """
    model_type = str(ckpt.get("model_type", "base")).strip().lower()

    if model_type in ["base", "stgat", "gat"]:
        return "base"
    if model_type in ["senn", "sennrawx", "senn_rawx", "senn_raw"]:
        return "senn"
    if model_type in ["sennfixed", "senn_fixed", "senn_fixedconcepts", "fixed"]:
        return "senn_fixed"
    if model_type in ["senntrivialfixed", "senn_trivialfixed", "trivialfixed"]:
        return "senn_trivialfixed"
    if model_type in [
        "sennfixed_concepttheta",
        "senn_fixed_concepttheta",
        "senn_fixedconcepttheta",
        "senn_fixedconcepts_concepttheta",
    ]:
        return "senn_fixedconcepttheta"
    if model_type in ["LogisticConcepts", "logisticconcepts", "FixedLogisticConcepts"]:
        return "LogisticConcepts"

    raise ValueError(
        f"Unknown model_type in checkpoint: {ckpt.get('model_type')}. "
        "Add this model to model_kind_from_checkpoint() and MyUtils.build_model()."
    )


def load_fold_data(data_folder, fold):
    fold_dir = os.path.join(data_folder, f"fold_{fold}")
    x_path = os.path.join(fold_dir, "testdata.npy")
    y_path = os.path.join(fold_dir, "testlabels.npy")

    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"Missing test data: {x_path}")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"Missing test labels: {y_path}")

    x_test = np.load(x_path, mmap_mode="r")
    y_test = np.load(y_path, mmap_mode="r")

    if not USE_FULL_OVERLAP_TESTSET:
        idx_seiz = np.where(y_test == 1)[0]
        idx_non = np.where(y_test == 0)[0]

        fs = 32
        t_window = x_test.shape[-1] / fs
        t_overlap_non = 10
        t_overlap_seiz = 11

        skip_non = int(t_window / (t_window - t_overlap_non))
        skip_seiz = int(t_window / (t_window - t_overlap_seiz))
        skip_non = max(skip_non, 1)
        skip_seiz = max(skip_seiz, 1)

        keep_idx = np.sort(np.concatenate([idx_non[0::skip_non], idx_seiz[0::skip_seiz]]))
        x_test = x_test[keep_idx]
        y_test = y_test[keep_idx]

    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    testdata = MyUtils.prepare_graphs_labels(x_test, y_test, Model.adj)
    return testdata


def extract_prob(output):
    if isinstance(output, dict):
        if "prob" in output:
            return output["prob"]
        if "logit" in output:
            return torch.sigmoid(output["logit"])
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unexpected model output type: {type(output)}")


@torch.no_grad()
def run_inference(model, testdata):
    loader = DataLoader(
        testdata,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEM,
        num_workers=NUM_WORKERS,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    model = model.to(DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        out = model(batch.x, batch.edge_index, batch.batch)
        prob = extract_prob(out)
        all_probs.append(prob.detach().cpu())
        all_labels.append(batch.y.detach().cpu())

    y_prob = torch.cat(all_probs).numpy().ravel()
    y_true = torch.cat(all_labels).numpy().ravel()
    return y_prob, y_true


# ============================================================
# Metrics
# ============================================================

def precision_gain(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pi = np.mean(y_true)
    precision = tp / (tp + fp + EPS)
    return (precision - pi) / ((1.0 - pi) * precision + EPS)


def recall_gain(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pi = np.mean(y_true)
    recall = tp / (tp + fn + EPS)
    return (recall - pi) / ((1.0 - pi) * recall + EPS)


def auprg(y_true, y_prob):
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


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return thresholds[np.argmax(tpr - fpr)]


def kappa_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    kappas = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        kappas.append(cohen_kappa_score(y_true, y_pred))
    return thresholds[np.nanargmax(kappas)]


def F2_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    f2s = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2s.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    return thresholds[np.nanargmax(f2s)]


def select_threshold(y_true, y_prob, ckpt):
    if THRESHOLD_MODE == "fixed":
        return FIXED_THRESHOLD
    if THRESHOLD_MODE == "checkpoint":
        return float(ckpt.get("metrics", {}).get("threshold", FIXED_THRESHOLD))
    if THRESHOLD_MODE == "youden":
        return youden_threshold(y_true, y_prob)
    if THRESHOLD_MODE == "kappa":
        return kappa_threshold(y_true, y_prob)
    if THRESHOLD_MODE == "F2":
        return F2_threshold(y_true, y_prob)
    raise ValueError(f"Unknown THRESHOLD_MODE: {THRESHOLD_MODE}")


def safe_roc_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp + EPS),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "AUROC": safe_roc_auc(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "AUPRG": auprg(y_true, y_prob),
        "precision_gain": precision_gain(y_true, y_prob, threshold),
        "recall_gain": recall_gain(y_true, y_prob, threshold),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def metrics_from_checkpoint(ckpt):
    metrics = ckpt.get("metrics", {})
    return {
        "AUROC": safe_float(metrics.get("auroc", metrics.get("AUROC"))),
        "AUPRC": safe_float(metrics.get("auprc", metrics.get("AUPRC"))),
        "kappa": safe_float(metrics.get("kappa", metrics.get("Kappa"))),
        "threshold": safe_float(metrics.get("threshold", metrics.get("Threshold"))),
        "recall": safe_float(metrics.get("recall", metrics.get("Recall"))),
        "precision": safe_float(metrics.get("precision", metrics.get("Precision"))),
        "F1": safe_float(metrics.get("f1", metrics.get("F1"))),
        "best_epoch": safe_float(ckpt.get("epoch")),
    }


# ============================================================
# Optional plotting
# ============================================================

def plot_loss_history(history_dir, results_dir, fold):
    history_path = os.path.join(history_dir, f"history_cv_10_{fold}.json")
    if not os.path.isfile(history_path):
        print(f"History not found: {history_path}")
        return

    with open(history_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        train_loss = [entry.get("loss") or entry.get("train_loss") for entry in data]
        val_loss = [entry.get("val_loss") for entry in data]
        val_auprc = [entry.get("val_auprc") for entry in data]
    else:
        train_loss = data.get("loss") or data.get("train_loss")
        val_loss = data.get("val_loss")
        val_auprc = data.get("val_auprc")

    if train_loss is not None and val_loss is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(train_loss) + 1), train_loss, color='tab:blue',label="Training loss")
        plt.plot(range(1, len(val_loss) + 1), val_loss, color='tab:orange',label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training history fold {fold}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"training_history_fold_{fold}.png"), dpi=300)
        plt.close()

    if val_auprc is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(val_auprc) + 1), val_auprc, color='tab:red', label="Validation AUPRC")
        plt.xlabel("Epoch")
        plt.ylabel("AUPRC")
        plt.ylim([-0.05,1.05])
        plt.title(f"Validation AUPRC fold {fold}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"validation_auprc_fold_{fold}.png"), dpi=300)
        plt.close()



def plot_decisions(fold_results, fold_data,output_dir,tag):
    for i, (y_true, y_prob) in enumerate(fold_data):
        thr = fold_results[i]['threshold']

        fig, axes = plt.subplots(
            2, 1,
            figsize=(10, 6),
            sharex=True,
            gridspec_kw={'height_ratios': [1, 2]}
        )

        # =========================
        # Top subplot — Ground Truth
        # =========================
        axes[0].plot(y_true,color="black", label="Ground Truth")
        axes[0].set_ylabel("GT")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"Decision Curve Fold {i}")

        # =========================
        # Bottom subplot — Probabilities
        # =========================
        axes[1].plot(y_prob,color="blue", label="Output probability")
        axes[1].axhline(
            y=thr,
            linestyle="--",
            label="Decision threshold",color = "red"
        )

        # axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Output probability")
        axes[1].set_xlabel("Sample")
        axes[1].legend(loc="upper right")

        # --- Save ---
        save_path = os.path.join(output_dir, f"Decisions_{tag}_{str(i)}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        # print(f"decisions saved to: {save_path}")

    return

def plot_cv_curves(fold_data, output_dir, tag):
    """
    Plots mean ROC and mean PR curves with standard deviation shading.
    Adds mean PRG curves in a separate figure while keeping the existing ROC/PR figure unchanged.
    fold_data: list of tuples (y_true, y_prob)
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    precisions = []
    mean_recall = np.linspace(0, 1, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i, (y_true, y_pred) in enumerate(fold_data):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        ax1.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i} (AUC = {roc_auc:.2f})')

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_prec)
        ax2.plot(recall, precision, lw=1, alpha=0.3, label=f'Fold {i} (AP = {pr_auc:.2f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax1.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $+-$ {std_auc:.2f})', lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic", xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right", fontsize='small')

    mean_precision = np.mean(precisions, axis=0)
    mean_ap = auc(mean_recall, mean_precision)

    ax2.plot(mean_recall, mean_precision, color='b', label=f'Mean PR (AP = {mean_ap:.2f})', lw=2, alpha=.8)

    std_prec = np.std(precisions, axis=0)
    ax2.fill_between(mean_recall, np.maximum(mean_precision - std_prec, 0), np.minimum(mean_precision + std_prec, 1), color='grey', alpha=.2)

    ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Precision-Recall Curve", xlabel='Recall', ylabel='Precision')
    ax2.legend(loc="lower left", fontsize='small')

    save_path = os.path.join(output_dir, f"Curves_{tag}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    # print(f"Curves saved to: {save_path}")

    prg_curves = []
    mean_gain_recall = np.linspace(0, 1, 100)
    prg_aucs = []

    fig, ax3 = plt.subplots(1, 1, figsize=(8, 7))

    for i, (y_true, y_pred) in enumerate(fold_data):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        precision = np.asarray(precision, dtype=float)
        recall = np.asarray(recall, dtype=float)
        pi = np.mean(y_true)

        precision_gain_values = (precision - pi) / ((1.0 - pi) * np.clip(precision, EPS, None))
        recall_gain_values = (recall - pi) / ((1.0 - pi) * np.clip(recall, EPS, None))

        mask = np.isfinite(precision_gain_values) & np.isfinite(recall_gain_values)
        precision_gain_values = precision_gain_values[mask]
        recall_gain_values = recall_gain_values[mask]

        if len(recall_gain_values) == 0:
            continue

        order = np.argsort(recall_gain_values)
        precision_gain_values = precision_gain_values[order]
        recall_gain_values = recall_gain_values[order]

        non_negative = np.where(recall_gain_values >= 0.0)[0]
        if len(non_negative) == 0:
            continue

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

        # recall_gain_values = np.clip(recall_gain_values, 0.0, 1.0)
        if recall_gain_values[0] > 0.0:
            recall_gain_values = np.concatenate(([0.0], recall_gain_values))
            precision_gain_values = np.concatenate(([precision_gain_values[0]], precision_gain_values))
        if recall_gain_values[-1] < 1.0:
            recall_gain_values = np.concatenate((recall_gain_values, [1.0]))
            precision_gain_values = np.concatenate((precision_gain_values, [0.0]))

        recall_gain_values, unique_idx = np.unique(recall_gain_values, return_index=True)
        precision_gain_values = precision_gain_values[unique_idx]

        prg_auc = auc(recall_gain_values, precision_gain_values)
        interp_prec_gain = np.interp(mean_gain_recall, recall_gain_values, precision_gain_values)
        prg_curves.append(interp_prec_gain)
        prg_aucs.append(prg_auc)
        ax3.plot(recall_gain_values, precision_gain_values, lw=1, alpha=0.3, label=f'Fold {i} (AUPRG = {prg_auc:.2f})')

    if len(prg_curves) > 0:
        mean_prg = np.mean(prg_curves, axis=0)
        std_prg = np.std(prg_curves, axis=0)
        mean_prg_auc = np.mean(prg_aucs)
        std_prg_auc = np.std(prg_aucs)
        ax3.plot(mean_gain_recall, mean_prg, color='b', label=f'Mean PRG (AUPRG = {mean_prg_auc:.2f} $+-$ {std_prg_auc:.2f})', lw=2, alpha=.8)
        ax3.fill_between(mean_gain_recall, mean_prg - std_prg, mean_prg + std_prg, color='grey', alpha=.2)

    ax3.axhline(0.0, linestyle='--', lw=2, color='r', alpha=.8)
    ax3.set(xlim=[-0.05, 1.05], ylim=(-0.1,1.05), title="Precision-Recall-Gain Curve", xlabel='Recall Gain', ylabel='Precision Gain')
    ax3.legend(loc="lower left", fontsize='small')

    save_path = os.path.join(output_dir, f"PRG_Curves_{tag}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PRG curves saved to: {save_path}")

    return



# ============================================================
# Main evaluation
# ============================================================

def evaluate_from_checkpoints(model_dir, results_dir):
    rows = []
    folds = available_folds(model_dir)

    for fold in folds:
        ckpt, ckpt_path = load_checkpoint(model_dir, fold)
        model_kind = model_kind_from_checkpoint(ckpt)
        row = metrics_from_checkpoint(ckpt)
        row["fold"] = fold
        row["model_kind"] = model_kind
        row["checkpoint"] = ckpt_path
        rows.append(row)

    df, summary = summarize_folds(rows)
    save_results(df, summary, results_dir, "checkpoint")


def evaluate_from_models(model_dir, data_folder, results_dir, history_dir):
    folds = available_folds(model_dir)
    if len(folds) == 0:
        raise FileNotFoundError(f"No fold checkpoints found in {model_dir}")

    if PLOT_HISTORY:
        for fold in folds:
            plot_loss_history(history_dir, results_dir, fold)

    fold_results = []
    fold_data = []

    for fold in folds:
        print("=" * 80)
        print(f"Evaluating fold {fold}")

        ckpt, ckpt_path = load_checkpoint(model_dir, fold)
        model_kind = model_kind_from_checkpoint(ckpt)
        print(f"Checkpoint: {ckpt_path}")
        print(f"Model kind: {model_kind}")

        model = MyUtils.build_model(
            model_kind=model_kind,
            ckpt=ckpt,
            return_explanations=False,
        )
        model = model.to(DEVICE)
        model.eval()

        testdata = load_fold_data(data_folder, fold)
        y_prob, y_true = run_inference(model, testdata)
        fold_data.append((y_true, y_prob))

        threshold = float(select_threshold(y_true, y_prob, ckpt))
        metrics = compute_metrics(y_true, y_prob, threshold)
        metrics["fold"] = fold
        metrics["model_kind"] = model_kind
        # metrics["checkpoint"] = ckpt_path
        # metrics["best_epoch"] = safe_float(ckpt.get("epoch"))
        fold_results.append(metrics)

        if UPDATE_CHECKPOINT_THRESHOLD:
            ckpt["metrics"]["threshold"] = threshold
            torch.save(ckpt, ckpt_path)
            print(f"Updated threshold in: {ckpt_path}")

        print(
            f"Fold {fold}: AUPRC={metrics['AUPRC']:.4f}, "
            f"AUROC={metrics['AUROC']:.4f}, "
            f"Kappa={metrics['kappa']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Threshold={threshold:.3f}"
        )

    tag = f"recompute_{THRESHOLD_MODE}"
    df, summary = summarize_folds(fold_results)
    save_results(df, summary, results_dir, tag)
    save_fold_data(fold_data=fold_data, results_dir=results_dir,tag=tag)

    if MAKE_PLOTS:
        plot_cv_curves(fold_data, results_dir, tag)
        plot_decisions(fold_results, fold_data, results_dir, tag)


def save_results(df, summary, results_dir, tag):
    ensure_dir(results_dir)

    df_path = os.path.join(results_dir, f"per_fold_metrics_{tag}.csv")
    summary_path = os.path.join(results_dir, f"summary_metrics_{tag}.csv")

    df.to_csv(df_path, index=False)
    summary.to_csv(summary_path)

    print("\nPer-fold metrics:")
    print(df)
    print("\nSummary metrics:")
    print(summary)
    print(f"\nSaved: {df_path}")
    print(f"Saved: {summary_path}")

def save_fold_data(fold_data, results_dir, tag):
    ensure_dir(results_dir)

    payload = {}
    payload["n_folds"] = len(fold_data)

    for i, (y_true, y_prob) in enumerate(fold_data):
        payload[f"fold_{i}_y_true"] = np.asarray(y_true, dtype=np.int64)
        payload[f"fold_{i}_y_prob"] = np.asarray(y_prob, dtype=np.float32)

    save_path = os.path.join(results_dir, f"fold_predictions_{tag}.npz")
    np.savez_compressed(save_path, **payload)
    print(f"Saved fold predictions to: {save_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--history_dir", default="")
    parser.add_argument("--mode", choices=["checkpoint", "recompute"], default=MODE)
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    results_dir = os.path.abspath(os.path.expanduser(args.results_dir))
    history_dir = os.path.abspath(os.path.expanduser(args.history_dir)) if args.history_dir else ""

    ensure_dir(results_dir)

    print(f"Using device: {DEVICE}")
    print(f"Model dir  : {model_dir}")
    print(f"Data folder: {data_folder}")
    print(f"Results dir: {results_dir}")

    if args.mode == "checkpoint":
        evaluate_from_checkpoints(model_dir, results_dir)
    else:
        evaluate_from_models(model_dir, data_folder, results_dir, history_dir)


if __name__ == "__main__":
    main()




# import torch


# import os
# import json
# import glob
# import argparse
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     roc_auc_score,
#     auc,
#     precision_recall_curve,
#     average_precision_score,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     accuracy_score,
#     roc_curve,
#     cohen_kappa_score,
#     fbeta_score
# )

# from cross_validation_senn import *  # must now expose PyTorch model + architecture
# from natsort import natsorted

# import MyUtils_senn_test as MyUtils
# # import Read_Data as RD
# import Models_senn as Model
# # ------------------------------------------------------------------
# # Configuration
# # ------------------------------------------------------------------


# DATA_FOLDER = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets\CV_Folds/"
# HISTORY_DIR = "./History_432270" #AUROC monitor native torch
# MODEL_DIR = "./Saved_models_432270"
# RESULTS_DIR = "./Results_432270"
# HISTORY_DIR = "./History_434147" #AUPRC monitor native torch
# MODEL_DIR = "./Saved_models_434147"
# RESULTS_DIR = "./Results_434147"

# log = "435445" #AUPRC monitor torch Geometric -> performs similar as native for fold 0 (1CV run)
# log = "435464" #PyG 10CV
# log = "435837" #Pyg GATv2 fc l2

# log = "436083" #pyg gatv2 gobal weight decay
# log = "449243" #Annotations included and shuffled BASE MODEL
# log = "456384" #SENN raw x fold 0
# log = "456398" #SENN raw x 10cv

# MODEL_TYPE ="SENNrawx"
# # log = "456465" #SENN raw x 10cv lr on plateau
# # MODEL_TYPE ="base"
# log = "456735" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"
# # log = "456741" #SENN raw x 10cv lr cosine annealing
# # MODEL_TYPE ="SENNrawx"

# log = "457392" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"
# log = "457393" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"
# log = "457395" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"
# log = "457396" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"

# log = "457476" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"
# log = "457473" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"

# log = "457477" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"

# log = "457531" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"

# log = "457560" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="SENNrawx"

# log = "457561" #SENN raw x 10cv lr cosine annealing
# MODEL_TYPE ="base"

# # log = "457636" #SENN raw x 10cv lr cosine annealing
# # MODEL_TYPE ="base"

# # log = "457639" #SENN raw x 10cv lr cosine annealing
# # MODEL_TYPE ="SENNrawx"

# # log = "458448" #Base model fixed LR; auprc tracked
# # MODEL_TYPE ="base"

# # log = "458445" #SENN raw x model fixed LR; auprc tracked
# # MODEL_TYPE ="SENNrawx"

# log = "458468" #Base model cosine LR; auprc tracked
# MODEL_TYPE ="base"


# log = "458467" #SENN raw x model cosine LR; auprc tracked
# MODEL_TYPE ="SENNrawx"
# log = "460528" #SENN raw x model cosine LR; auprc tracked; max epoch 250
# MODEL_TYPE ="SENNrawx"

# log = "460529" #SENN raw x model cosine LR; auprc tracked; max epoch 250
# MODEL_TYPE ="SENNrawx"


# log = "460531" #SENN raw x model cosine LR; auprc tracked; max epoch 250
# MODEL_TYPE ="SENNrawx"
# log = "459970" #SENN raw x model cosine LR; auprc tracked; max epoch 250
# MODEL_TYPE ="SENNrawx"


# log = "459966" # Base model epoch 250
# MODEL_TYPE ="base"

# log = "459968" #Senn Rob loss 1e-4
# MODEL_TYPE ="SENNrawx"

# log = "459967" #SENN raw x no rob loss
# MODEL_TYPE ="SENNrawx"

# log = "460527" #SENN raw x rob loss 1e-5
# MODEL_TYPE ="SENNrawx"


# log = "461551" #SENN raw x model cosine LR; auprc tracked; max epoch 250
# MODEL_TYPE ="SENNrawx"


# ## New dataset per pateint per channel normalization
# # log = "470501" #fixed no rob loss
# # MODEL_TYPE ="SENNfixed"
# log = "468340" #base model
# MODEL_TYPE ="base"
# log = "480652" #rob loss 3e-5
# MODEL_TYPE ="SENNrawx"

# # log = "481616" 
# # MODEL_TYPE ="SENNtrivialfixed"

# # log = "482221" 
# # MODEL_TYPE ="SENNfixed_concepttheta"

# # log = "482170" 
# # MODEL_TYPE ="LogisticConcepts"


# N_FOLDS = 10

# EPOCHS = 250

# HISTORY_DIR = f"./ModelArchiveRecent/History_{log}" 
# MODEL_DIR = f"./ModelArchiveRecent/Saved_models_{log}"
# RESULTS_DIR = f"./ModelArchiveRecent/Results_{log}"


# os.makedirs(RESULTS_DIR, exist_ok=True)

# FILES = os.listdir(DATA_FOLDER)
# # FILES = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.edf')])

# #for 80 20
# # np.random.seed(44)
        
# # np.random.shuffle(FILES)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EPS = 1e-12

# # ------------------------------------------------------------------
# # Utility functions
# # ------------------------------------------------------------------

# def load_history(fold):
#     path = os.path.join(HISTORY_DIR, f"history-8020split.json")
#     path = os.path.join(HISTORY_DIR, f"history_cv_10_{fold}.json")
#     with open(path, "r") as f:
#         return pd.read_json(f)


# def get_best_epoch(history_df, monitor="val_AUROC", mode="max"):
#     if mode == "max":
#         return history_df[monitor].idxmax()
#     else:
#         return history_df[monitor].idxmin()


# def resolve_metric_name(history_df, base_name, fold):
#     fold_name = f"{base_name}_{fold}"
#     if fold_name in history_df.columns:
#         return fold_name
#     elif base_name in history_df.columns:
#         return base_name
#     else:
#         return None

# def IQR(x):
#     return x.quantile(0.75) - x.quantile(0.25)
# def q1(x):
#     return x.quantile(0.25)
# def q3(x):
#     return x.quantile(0.75)

# def summarize_folds(results):
#     df = pd.DataFrame(results)
#     summary = df.agg(["mean", "std","median",IQR,q1,q3])
    
#     return df, summary


# def find_best_model_path(fold):
#     # fold_dir = os.path.join(MODEL_DIR, "8020split")
#     fold_dir = os.path.join(MODEL_DIR, f"GAT_CV_10_{fold}")
#     # models = natsorted(glob.glob(os.path.join(fold_dir, "best_model.pt")))
#     models = os.path.join(fold_dir, "best_model.pt")
#     models = os.path.join(fold_dir, "best_model_AUPRC.pt")
#     if len(models) == 0:
#         raise FileNotFoundError(f"No models found for fold {fold}")
#     return models


# def load_fold_data(fold,no_overlap=True):
#     np.random.seed(43)
#     data_folder = DATA_FOLDER
#     fold_dir = os.path.join(data_folder, f'fold_{fold}')

#     # x_train = np.load(os.path.join(fold_dir,'traindata.npy'),mmap_mode='r')
#     # y_train = np.load(os.path.join(fold_dir,'trainlabels.npy'),mmap_mode='r')
#     x_test  = np.load(os.path.join(fold_dir,'testdata.npy'),mmap_mode='r')
#     y_test  = np.load(os.path.join(fold_dir,'testlabels.npy'),mmap_mode='r')

#     if no_overlap:
#         idx_yes_seiz = np.where(y_test == 1)[0]
#         idx_no_seiz = np.where(y_test == 0)[0]
#         fs = 32 #Hz
#         t_window = len(x_test[0][0]) / fs
#         t_overlap = 10
#         t_overlap_seiz = 11
#         thin_skip = int(t_window / (t_window - t_overlap))
#         thin_skip_seiz = int(t_window / (t_window - t_overlap_seiz))
#         thin_idx_no_seiz = idx_no_seiz[0::thin_skip] #Full test set 
#         thin_idx_yes_seiz = idx_yes_seiz[0::thin_skip_seiz] #full test set

#         keep_idx = np.sort(np.concatenate([thin_idx_no_seiz, thin_idx_yes_seiz]))

#         x_test = x_test[keep_idx]
#         y_test = y_test[keep_idx]

    

#     testdata = MyUtils.prepare_graphs_labels(x_test,y_test,Model.adj)
    

#     return testdata

# def precision_gain(y_true, y_prob, threshold=0.5):
#     y_pred = (y_prob >= threshold).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#     pi = np.mean(y_true)
#     precision = tp / (tp + fp + EPS)
#     return (precision - pi) / ((1.0 - pi) * precision + EPS)


# def recall_gain(y_true, y_prob, threshold=0.5):
#     y_pred = (y_prob >= threshold).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#     pi = np.mean(y_true)
#     recall = tp / (tp + fn + EPS)
#     return (recall - pi) / ((1.0 - pi) * recall + EPS)


# def auprg(y_true, y_prob):
#     precision, recall, _ = precision_recall_curve(y_true, y_prob)
#     precision = np.asarray(precision, dtype=float)
#     recall = np.asarray(recall, dtype=float)
#     pi = np.mean(y_true)

#     precision_gain_values = (precision - pi) / ((1.0 - pi) * np.clip(precision, EPS, None))
#     recall_gain_values = (recall - pi) / ((1.0 - pi) * np.clip(recall, EPS, None))

#     mask = np.isfinite(precision_gain_values) & np.isfinite(recall_gain_values)
#     precision_gain_values = precision_gain_values[mask]
#     recall_gain_values = recall_gain_values[mask]

#     if len(recall_gain_values) == 0:
#         return np.nan

#     order = np.argsort(recall_gain_values)
#     precision_gain_values = precision_gain_values[order]
#     recall_gain_values = recall_gain_values[order]

#     non_negative = np.where(recall_gain_values >= 0.0)[0]
#     if len(non_negative) == 0:
#         return 0.0

#     first = non_negative[0]
#     if first > 0 and recall_gain_values[first] > 0.0:
#         x1, y1 = recall_gain_values[first - 1], precision_gain_values[first - 1]
#         x2, y2 = recall_gain_values[first], precision_gain_values[first]
#         y_at_zero = y1 + (0.0 - x1) * (y2 - y1) / (x2 - x1 + EPS)
#         recall_gain_values = np.concatenate(([0.0], recall_gain_values[first:]))
#         precision_gain_values = np.concatenate(([y_at_zero], precision_gain_values[first:]))
#     else:
#         recall_gain_values = recall_gain_values[first:]
#         precision_gain_values = precision_gain_values[first:]

#     # recall_gain_values = np.clip(recall_gain_values, 0.0, 1.0)

#     if recall_gain_values[0] > 0.0:
#         recall_gain_values = np.concatenate(([0.0], recall_gain_values))
#         precision_gain_values = np.concatenate(([precision_gain_values[0]], precision_gain_values))

#     if recall_gain_values[-1] < 1.0:
#         recall_gain_values = np.concatenate((recall_gain_values, [1.0]))
#         precision_gain_values = np.concatenate((precision_gain_values, [0.0]))

#     recall_gain_values, unique_idx = np.unique(recall_gain_values, return_index=True)
#     precision_gain_values = precision_gain_values[unique_idx]

#     return auc(recall_gain_values, precision_gain_values)


# def compute_metrics(y_true, y_prob, threshold=0.5):
#     y_pred = (y_prob >= threshold).astype(int)

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     return {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, zero_division=0),
#         "recall": recall_score(y_true, y_pred),
#         "specificity": tn / (tn + fp + 1e-8),
#         "F2": fbeta_score(y_true,y_pred,beta=2),
#         "kappa": cohen_kappa_score(y_true, y_pred),
#         "AUROC": roc_auc_score(y_true, y_prob),
#         "AUPRC": average_precision_score(y_true, y_prob),
#         "precision_gain": precision_gain(y_true, y_prob, threshold),
#         "recall_gain": recall_gain(y_true, y_prob, threshold),
#         "AUPRG": auprg(y_true, y_prob),
#     }

# def youden_threshold(y_true, y_prob):
#     fpr, tpr, thresholds = roc_curve(y_true, y_prob)
#     j_scores = tpr - fpr
#     idx = np.argmax(j_scores)
#     return thresholds[idx]


# def kappa_threshold(y_true, y_prob):
#     thresholds = np.linspace(0.01, 0.99, 99)
#     kappas = []

#     for t in thresholds:
#         y_pred = (y_prob >= t).astype(int)
#         kappas.append(cohen_kappa_score(y_true, y_pred))

#     return thresholds[np.argmax(kappas)]

# def F2_threshold(y_true, y_prob):
#     thresholds = np.linspace(0.01, 0.99, 99)
#     f2s = []

#     for t in thresholds:
#         y_pred = (y_prob >= t).astype(int)
#         f2s.append(fbeta_score(y_true, y_pred,beta=2))

#     return thresholds[np.argmax(f2s)]

# def plot_decisions(fold_results, fold_data,output_dir,tag):
#     for i, (y_true, y_prob) in enumerate(fold_data):
#         thr = fold_results[i]['threshold']

#         fig, axes = plt.subplots(
#             2, 1,
#             figsize=(10, 6),
#             sharex=True,
#             gridspec_kw={'height_ratios': [1, 2]}
#         )

#         # =========================
#         # Top subplot — Ground Truth
#         # =========================
#         axes[0].plot(y_true,color="black", label="Ground Truth")
#         axes[0].set_ylabel("GT")
#         axes[0].set_ylim(-0.05, 1.05)
#         axes[0].legend(loc="upper right")
#         axes[0].set_title(f"Decision Curve Fold {i}")

#         # =========================
#         # Bottom subplot — Probabilities
#         # =========================
#         axes[1].plot(y_prob,color="blue", label="Output probability")
#         axes[1].axhline(
#             y=thr,
#             linestyle="--",
#             label="Decision threshold",color = "red"
#         )

#         # axes[1].set_ylim(0, 1)
#         axes[1].set_ylabel("Output probability")
#         axes[1].set_xlabel("Sample")
#         axes[1].legend(loc="upper right")

#         # --- Save ---
#         save_path = os.path.join(output_dir, f"Decisions_{tag}_{str(i)}.png")
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300)
#         plt.close()

#         print(f"decisions saved to: {save_path}")

#     return
# def plot_cv_curves(fold_data, output_dir, tag):
#     """
#     Plots mean ROC and mean PR curves with standard deviation shading.
#     Adds mean PRG curves in a separate figure while keeping the existing ROC/PR figure unchanged.
#     fold_data: list of tuples (y_true, y_prob)
#     """
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)

#     precisions = []
#     mean_recall = np.linspace(0, 1, 100)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

#     for i, (y_true, y_pred) in enumerate(fold_data):
#         fpr, tpr, _ = roc_curve(y_true, y_pred)
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         ax1.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i} (AUC = {roc_auc:.2f})')

#         precision, recall, _ = precision_recall_curve(y_true, y_pred)
#         pr_auc = auc(recall, precision)
#         interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
#         precisions.append(interp_prec)
#         ax2.plot(recall, precision, lw=1, alpha=0.3, label=f'Fold {i} (AP = {pr_auc:.2f})')

#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)

#     ax1.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $+-$ {std_auc:.2f})', lw=2, alpha=.8)

#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

#     ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
#     ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic", xlabel='False Positive Rate', ylabel='True Positive Rate')
#     ax1.legend(loc="lower right", fontsize='small')

#     mean_precision = np.mean(precisions, axis=0)
#     mean_ap = auc(mean_recall, mean_precision)

#     ax2.plot(mean_recall, mean_precision, color='b', label=f'Mean PR (AP = {mean_ap:.2f})', lw=2, alpha=.8)

#     std_prec = np.std(precisions, axis=0)
#     ax2.fill_between(mean_recall, np.maximum(mean_precision - std_prec, 0), np.minimum(mean_precision + std_prec, 1), color='grey', alpha=.2)

#     ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Precision-Recall Curve", xlabel='Recall', ylabel='Precision')
#     ax2.legend(loc="lower left", fontsize='small')

#     save_path = os.path.join(output_dir, f"Curves_{tag}.png")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"Curves saved to: {save_path}")

#     prg_curves = []
#     mean_gain_recall = np.linspace(0, 1, 100)
#     prg_aucs = []

#     fig, ax3 = plt.subplots(1, 1, figsize=(8, 7))

#     for i, (y_true, y_pred) in enumerate(fold_data):
#         precision, recall, _ = precision_recall_curve(y_true, y_pred)
#         precision = np.asarray(precision, dtype=float)
#         recall = np.asarray(recall, dtype=float)
#         pi = np.mean(y_true)

#         precision_gain_values = (precision - pi) / ((1.0 - pi) * np.clip(precision, EPS, None))
#         recall_gain_values = (recall - pi) / ((1.0 - pi) * np.clip(recall, EPS, None))

#         mask = np.isfinite(precision_gain_values) & np.isfinite(recall_gain_values)
#         precision_gain_values = precision_gain_values[mask]
#         recall_gain_values = recall_gain_values[mask]

#         if len(recall_gain_values) == 0:
#             continue

#         order = np.argsort(recall_gain_values)
#         precision_gain_values = precision_gain_values[order]
#         recall_gain_values = recall_gain_values[order]

#         non_negative = np.where(recall_gain_values >= 0.0)[0]
#         if len(non_negative) == 0:
#             continue

#         first = non_negative[0]
#         if first > 0 and recall_gain_values[first] > 0.0:
#             x1, y1 = recall_gain_values[first - 1], precision_gain_values[first - 1]
#             x2, y2 = recall_gain_values[first], precision_gain_values[first]
#             y_at_zero = y1 + (0.0 - x1) * (y2 - y1) / (x2 - x1 + EPS)
#             recall_gain_values = np.concatenate(([0.0], recall_gain_values[first:]))
#             precision_gain_values = np.concatenate(([y_at_zero], precision_gain_values[first:]))
#         else:
#             recall_gain_values = recall_gain_values[first:]
#             precision_gain_values = precision_gain_values[first:]

#         # recall_gain_values = np.clip(recall_gain_values, 0.0, 1.0)
#         if recall_gain_values[0] > 0.0:
#             recall_gain_values = np.concatenate(([0.0], recall_gain_values))
#             precision_gain_values = np.concatenate(([precision_gain_values[0]], precision_gain_values))
#         if recall_gain_values[-1] < 1.0:
#             recall_gain_values = np.concatenate((recall_gain_values, [1.0]))
#             precision_gain_values = np.concatenate((precision_gain_values, [0.0]))

#         recall_gain_values, unique_idx = np.unique(recall_gain_values, return_index=True)
#         precision_gain_values = precision_gain_values[unique_idx]

#         prg_auc = auc(recall_gain_values, precision_gain_values)
#         interp_prec_gain = np.interp(mean_gain_recall, recall_gain_values, precision_gain_values)
#         prg_curves.append(interp_prec_gain)
#         prg_aucs.append(prg_auc)
#         ax3.plot(recall_gain_values, precision_gain_values, lw=1, alpha=0.3, label=f'Fold {i} (AUPRG = {prg_auc:.2f})')

#     if len(prg_curves) > 0:
#         mean_prg = np.mean(prg_curves, axis=0)
#         std_prg = np.std(prg_curves, axis=0)
#         mean_prg_auc = np.mean(prg_aucs)
#         std_prg_auc = np.std(prg_aucs)
#         ax3.plot(mean_gain_recall, mean_prg, color='b', label=f'Mean PRG (AUPRG = {mean_prg_auc:.2f} $+-$ {std_prg_auc:.2f})', lw=2, alpha=.8)
#         ax3.fill_between(mean_gain_recall, mean_prg - std_prg, mean_prg + std_prg, color='grey', alpha=.2)

#     ax3.axhline(0.0, linestyle='--', lw=2, color='r', alpha=.8)
#     ax3.set(xlim=[-0.05, 1.05], ylim=[-1.05, 1.05], title="Precision-Recall-Gain Curve", xlabel='Recall Gain', ylabel='Precision Gain')
#     ax3.legend(loc="lower left", fontsize='small')

#     save_path = os.path.join(output_dir, f"PRG_Curves_{tag}.png")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"PRG curves saved to: {save_path}")

#     return

# def plot_loss_history(json_file_path, save_name='training_history_val_loss.png'):
#     """
#     Reads a JSON file containing training history and plots Training vs Validation Loss.
#     """
#     # Load the JSON data
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)

#     # Extract loss values based on the structure of the JSON
#     if isinstance(data, list):
#         # Format: [{"loss": 0.5, "val_loss": 0.6}, ...]
#         train_loss = [entry.get('loss') or entry.get('train_loss') for entry in data]
#         val_loss = [entry.get('val_loss') for entry in data]
#     else:
#         # Format: {"loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}
#         # train_loss = data.get('loss') or data.get('train_loss')
#         # val_loss = data.get('val_loss')
#         train_loss = data.get('loss') or data.get('train_loss')
#         # val_loss = data.get('val_auprc')
#         val_loss = data.get('val_loss')

#     # Create the plot
#     epochs = range(1, len(train_loss) + 1)
    
#     plt.plot(epochs, train_loss, 'b-', label='Training Loss')
#     plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
#     # plt.ylim(0,1.)
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Save the figure
#     plt.savefig(os.path.join(RESULTS_DIR,save_name), dpi=300, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#     print(f"Plot saved as {save_name}")

#     # Load the JSON data
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)

#     # Extract loss values based on the structure of the JSON
#     if isinstance(data, list):
#         # Format: [{"loss": 0.5, "val_loss": 0.6}, ...]
#         train_loss = [entry.get('loss') or entry.get('train_loss') for entry in data]
#         val_loss = [entry.get('val_loss') for entry in data]
#         val_loss = [entry.get('val_auprc') for entry in data]
#     else:
#         # Format: {"loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}
#         # train_loss = data.get('loss') or data.get('train_loss')
#         # val_loss = data.get('val_loss')
#         train_loss = data.get('loss') or data.get('train_loss')
#         val_loss = data.get('val_auprc')
#         # val_loss = data.get('val_loss')

#     # Create the plot
#     epochs = range(1, len(train_loss) + 1)
    
#     # plt.plot(epochs, train_loss, 'b-', label='Training Loss')
#     plt.plot(epochs, val_loss, 'r-', label='Validation AUPRC')
    
#     plt.title('Validation AUPRC')
#     plt.xlabel('Epochs')
#     plt.ylabel('AUPRC')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Save the figure
#     plt.savefig(os.path.join(RESULTS_DIR,f"auprc_{save_name}"), dpi=300, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#     print(f"Plot saved as auprc_{save_name}")

# # ------------------------------------------------------------------
# # Mode 1: HISTORY-BASED EVALUATION (UNCHANGED)
# # ------------------------------------------------------------------

# def evaluate_from_history():
#     model_dir = MODEL_DIR
    

    
#     all_fold_metrics = []

#     print(f"{'Fold':<8} | {'AUROC':<8} | {'Kappa':<8} | {'Threshold':<8} |{'Recall':<8} | {'F1':<8}")
#     print("-" * 55)

#     for r in range(10):
#         checkpoint_path = os.path.join(model_dir, f"GAT_CV_10_{r}", "best.pt")
        
#         if not os.path.exists(checkpoint_path):
#             print(f"Warning: Checkpoint for fold {r} not found at {checkpoint_path}")
#             continue

#         # Load the dictionary checkpoint
#         checkpoint = torch.load(checkpoint_path,weights_only=False)
        
#         # Extract metrics from the 'metrics' key we created in the trainer
#         m = checkpoint['metrics']
        
#         fold_data = {
#             'Fold': r + 1,
#             'AUROC': m.get('auroc', 0),
#             'Kappa': m.get('kappa', 0),
#             'Threshold': m.get('threshold', 0),
#             'Recall': m.get('recall', 0),
#             'Precision': m.get('precision', 0),
#             'F1': m.get('f1', 0),
#             'Best_Epoch': checkpoint.get('epoch', 0)
#         }
        
#         all_fold_metrics.append(fold_data)
        
#         print(f"Fold {r+1:<3} (Ep {fold_data['Best_Epoch']:>2}) | "
#               f"{fold_data['AUROC']:.4f} | {fold_data['Kappa']:.4f}| {fold_data['Threshold']:.4f} | "
#               f"{fold_data['Recall']:.4f} | {fold_data['F1']:.4f}")

#     # Convert to DataFrame for easy statistics
#     df = pd.DataFrame(all_fold_metrics)
    
#     # Calculate Mean and Std
#     summary_mean = df.mean(numeric_only=True).drop(['Fold', 'Best_Epoch'])
#     summary_std = df.std(numeric_only=True).drop(['Fold', 'Best_Epoch'])

#     print("-" * 55)
#     print("\nFINAL CROSS-VALIDATION SUMMARY (Mean ± Std):")
#     for metric in summary_mean.index:
#         print(f"{metric:<10}: {summary_mean[metric]:.4f} ± {summary_std[metric]:.4f}")
    
#     # Optional: Save to CSV for your records
    
   

#     df.to_csv(os.path.join(RESULTS_DIR, "per_fold_metrics_history.csv"), index=False)
#     print("\nResults saved to cv_results_summary.csv")

#     return df


# # ------------------------------------------------------------------
# # Mode 2: MODEL RECOMPUTATION (PYTORCH)
# # ------------------------------------------------------------------

# def evaluate_from_models(threshold_mode="fixed", fixed_threshold=0.5):
    
#     #--plot loss history
#     for r in range(N_FOLDS):
#         history_path = os.path.join(HISTORY_DIR,f"history_cv_10_{r}.json")
#         plot_loss_history(history_path, save_name=f'training_history_{r}.png')

#     #--calculate metrics
#     fold_results = []
#     all_fold_data = []

#     for fold in range(N_FOLDS):
#         print(f"Evaluating fold {fold}")

#         model_path = find_best_model_path(fold)
#         if MODEL_TYPE == "base":
#             print("Loading base model")
#             model = Model.EEG_GAT_Model()
#         elif MODEL_TYPE=="SENNrawx":
#             print("Loading SENN raw x model")
#             model = Model.SENN_raw()
#         elif MODEL_TYPE=="SENNfixed":
#             print("Loading SENN fixed model")
#             model = Model.SENN_fixedconcepts()
#         elif MODEL_TYPE=="SENNtrivialfixed":
#             model = Model.SENN_trivialfixedconcepts()
#         elif MODEL_TYPE=="SENNfixed_concepttheta":
#             model = Model.SENN_fixedconcepts_concepttheta()
#         elif MODEL_TYPE== "LogisticConcepts":
#             model = Model.ConceptLogisticDual()
#         else:
#             Warning("Model type not supported")

#         print(f"Loading model:{MODEL_TYPE}")
#         state_dict = torch.load(model_path, map_location=DEVICE)
#         model.load_state_dict(state_dict)
#         model = model.to(DEVICE)
        
#         # d = torch.load(os.path.join(MODEL_DIR, f"GAT_CV_10_{0}","best_auprc.pt"),map_location=DEVICE,weights_only=False)
#         # print(d["metrics"])

#         model.eval()

#         testdata = load_fold_data(fold)
#         num_workers = 0
#         pin_mem=False
#         batch_size = 128
#         test_loader =DataLoader(
#             testdata, 
#             batch_size=batch_size, 
#             shuffle=False, 
#             pin_memory=pin_mem,  # Keeps data in "page-locked" memory for faster GPU transfer
#             num_workers=num_workers, # This tells Python to use your extra cores
#             prefetch_factor=4 if num_workers > 0 else None,
#             persistent_workers=True if num_workers > 0 else False
#             )       
#         all_probs=[]
#         all_labels=[]
#         with torch.no_grad():
#             for batch in test_loader:
#                 batch = batch.to(DEVICE, non_blocking=True)
#                 out = model(batch.x,batch.edge_index,batch.batch)
#                 all_probs.append(out["prob"].cpu())
#                 # all_probs.append(out["logit"].cpu())
#                 all_labels.append(batch.y.cpu())

#         y_prob = torch.cat(all_probs).cpu().numpy().ravel()
#         y_test = torch.cat(all_labels).cpu().numpy().ravel()
        
           

#         all_fold_data.append((y_test, y_prob))

#         if threshold_mode == "youden":
#             threshold = youden_threshold(y_test, y_prob)
#         elif threshold_mode == "kappa":
#             threshold = kappa_threshold(y_test, y_prob)
#         elif threshold_mode == "F2":
#             threshold = F2_threshold(y_test, y_prob)
#         else:
#             threshold = fixed_threshold
#         #Overwrite model training part were Kappa was optimised over full validation set as initial indication-> here we use the non overlapping dataset.
#         ckpt_path = os.path.join(MODEL_DIR, f"GAT_CV_10_{fold}/best_auprc.pt")
#         # os.path.join(MODEL_DIR, f"GAT_CV_10_{fold}/recomputed_threshold.pt")
#         ckpt = torch.load(ckpt_path,weights_only=False)
#         ckpt["metrics"]["threshold"] = threshold


#         metrics = compute_metrics(y_test, y_prob, threshold)
#         metrics["fold"] = fold
#         metrics["threshold"] = threshold

#         ckpt_path = os.path.join(MODEL_DIR, f"GAT_CV_10_{fold}/best_auprc.pt")
#         # os.path.join(MODEL_DIR, f"GAT_CV_10_{fold}/recomputed_threshold.pt")
#         ckpt = torch.load(ckpt_path,weights_only=False)
#         ckpt["metrics"]["threshold"] = threshold

#         torch.save(ckpt,ckpt_path)

#         fold_results.append(metrics)

#     df, summary = summarize_folds(fold_results)

   

#     print("\nPer-fold metrics (recomputed):")
#     print(df)

#     print("\nMean ± Std across folds:")
#     print(summary)

#     tag = f"recompute_{threshold_mode}"

#     df_path = os.path.join(RESULTS_DIR, f"per_fold_metrics_{tag}.csv")
#     summary_path = os.path.join(RESULTS_DIR, f"summary_metrics_{tag}.csv")

#     df.to_csv(df_path)
#     summary.to_csv(summary_path)

#     print(f"\nSaved per-fold results to: {df_path}")
#     print(f"Saved summary results to:  {summary_path}")

#      # --- Call the new Plotting Function ---
#     print("\nGenerating ROC and PR curves...")
#     plot_cv_curves(all_fold_data, RESULTS_DIR, tag)
#     print("\nGenerating decision curves...")
#     plot_decisions(fold_results,all_fold_data,RESULTS_DIR,tag)

    

#     return df, summary


# # ------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["history", "recompute"], required=True)
#     parser.add_argument("--threshold_mode", choices=["fixed", "youden", "kappa","F2"], default="fixed")
#     parser.add_argument("--threshold", type=float, default=0.5)

#     args = parser.parse_args()

#     if args.mode == "history":
#         evaluate_from_history()
#     else:
#         evaluate_from_models(args.threshold_mode, args.threshold)

# # def summarize_cross_validation_results():
# #     model_dir = "Saved_models"
# #     model_dir = "cv_run_hist/Old focal loss/Saved_models"
# #     model_dir = "cv_run_hist/Saved_models_427698"
# #     model_dir = "Saved_models_429300"
# #     model_dir = "Saved_models_429335"
# #     model_dir = "Saved_models_429466"
# #     model_dir = "Saved_models_431626"
# #     model_dir = "Saved_models_431673" # 1 fold new preprocessing
# #     # model_dir = "Saved_models_431680" #10 fold cv new preprocessing
# #     model_dir = "Saved_models_431753" #1 fold old preprocessing
# #     model_dir = "Saved_models_431807" #1 fold old preprocessing batch norm to 0.01
# #     model_dir = "Saved_models_431853" #1 fold old preprocessing batch norm to 0.01 and Architecture residual change
# #     # model_dir = "Saved_models_431886" #1 fold old preprocessing batch norm to 0.1 and Architecture residual chang
# #     # model_dir = "Saved_models_431752" #10 fold old preprocessing old BN old archt.
# #     model_dir = "Saved_models_431899" #10 fold old preprocessing N=BN 0.01 old proc 1-30
# #     # model_dir = "Saved_models_431900" #10 fold old preprocessing N=BN 0.01 new proc 0.5-30
# #     # model_dir = "Saved_models_431919" #10 fold old preprocessing N=BN 0.01 new proc 0.5-16
# #     model_dir = "Saved_models_432038" #10 fold old preprocessing N=BN 0.01 old proc 1-16
# #     model_dir = "Saved_models_432039"
# #     model_dir = "Saved_models_432040"
# #     model_dir = "Saved_models_432041"
# #     model_dir = "Saved_models_431805"

    
# #     all_fold_metrics = []

# #     print(f"{'Fold':<8} | {'AUROC':<8} | {'Kappa':<8} | {'Threshold':<8} |{'Recall':<8} | {'F1':<8}")
# #     print("-" * 55)

# #     for r in range(10):
# #         checkpoint_path = os.path.join(model_dir, f"GAT_CV_10_{r}", "best.pt")
        
# #         if not os.path.exists(checkpoint_path):
# #             print(f"Warning: Checkpoint for fold {r} not found at {checkpoint_path}")
# #             continue

# #         # Load the dictionary checkpoint
# #         checkpoint = torch.load(checkpoint_path,weights_only=False)
        
# #         # Extract metrics from the 'metrics' key we created in the trainer
# #         m = checkpoint['metrics']
        
# #         fold_data = {
# #             'Fold': r + 1,
# #             'AUROC': m.get('auroc', 0),
# #             'Kappa': m.get('kappa', 0),
# #             'Threshold': m.get('threshold', 0),
# #             'Recall': m.get('recall', 0),
# #             'Precision': m.get('precision', 0),
# #             'F1': m.get('f1', 0),
# #             'Best_Epoch': checkpoint.get('epoch', 0)
# #         }
        
# #         all_fold_metrics.append(fold_data)
        
# #         print(f"Fold {r+1:<3} (Ep {fold_data['Best_Epoch']:>2}) | "
# #               f"{fold_data['AUROC']:.4f} | {fold_data['Kappa']:.4f}| {fold_data['Threshold']:.4f} | "
# #               f"{fold_data['Recall']:.4f} | {fold_data['F1']:.4f}")

# #     # Convert to DataFrame for easy statistics
# #     df = pd.DataFrame(all_fold_metrics)
    
# #     # Calculate Mean and Std
# #     summary_mean = df.mean(numeric_only=True).drop(['Fold', 'Best_Epoch'])
# #     summary_std = df.std(numeric_only=True).drop(['Fold', 'Best_Epoch'])

# #     print("-" * 55)
# #     print("\nFINAL CROSS-VALIDATION SUMMARY (Mean ± Std):")
# #     for metric in summary_mean.index:
# #         print(f"{metric:<10}: {summary_mean[metric]:.4f} ± {summary_std[metric]:.4f}")
    
# #     # Optional: Save to CSV for your records
# #     df.to_csv(os.path.join(model_dir,"cv_results_summary.csv"))
# #     print("\nResults saved to cv_results_summary.csv")

# # def find_optimal_threshold_youdens(labels, probs):
# #     """
# #     Calculates the optimal threshold using Youden's J statistic.
# #     J = Sensitivity + Specificity - 1
# #     """
# #     fpr, tpr, thresholds = roc_curve(labels, probs)
# #     j_scores = tpr - fpr  # This is Sensitivity + (1 - FPR) - 1
# #     best_idx = np.argmax(j_scores)
# #     return thresholds[best_idx]

# # def validate_with_youdens_j():
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     raw_data_folder = 'Datasets/zenodo_eeg/'
# #     model_dir = "Saved_models_432039"
# #     files = sorted([f for f in os.listdir(raw_data_folder) if f.endswith('.edf')])
# #     adj = Model.adj.to(device)
    
# #     all_results = []

# #     print(f"{'Fold':<5} | {'AUROC':<7} | {'Old Kappa':<9} | {'New Kappa':<9} | {'Best Thresh':<11}")
# #     print("-" * 65)

# #     for r in range(10):
# #         # 1. Reconstruct Data Split
# #         x_train_np, _, x_test_np, y_test_np = RD.read_data(
# #             raw_data_folder, files, r*4 + 1, (r+1)*4
# #         )

# #         # 2. Z-Score Normalization (using training stats to avoid leakage)
# #         mean, std = x_train_np.mean(), x_train_np.std()
# #         x_test_np = (x_test_np - mean) / std
        
# #         x_test_ts = torch.tensor(np.expand_dims(x_test_np, -1), dtype=torch.float32).to(device)
# #         y_test_ts = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)
# #         test_loader = DataLoader(TensorDataset(x_test_ts, y_test_ts), batch_size=128, shuffle=False)

# #         # 3. Load Model
# #         model = Model.EEG_GAT_Model(Model.adj).to(device)
# #         # Assuming the weight file is 'best.pt' or 'best_model.pt' as per your previous snippets
# #         model_path = os.path.join(model_dir, f"GAT_CV_10_{r}", "best_model.pt")
# #         model.load_state_dict(torch.load(model_path, map_location=device))
# #         model.eval()

# #         # 4. Inference
# #         all_probs, all_labels = [], []
# #         with torch.no_grad():
# #             for xb, yb in test_loader:
# #                 probs = model(xb)
# #                 all_probs.append(probs.cpu())
# #                 all_labels.append(yb.cpu())

# #         probs_flat = torch.cat(all_probs).numpy().ravel()
# #         labels_flat = torch.cat(all_labels).numpy().ravel()

# #         # 5. Threshold Moving
# #         best_threshold = find_optimal_threshold_youdens(labels_flat, probs_flat)
        
# #         # Compare 0.5 vs Optimal
# #         preds_fixed = (probs_flat >= 0.5).astype(int)
# #         preds_optimal = (probs_flat >= best_threshold).astype(int)

# #         kappa_fixed = cohen_kappa_score(labels_flat, preds_fixed)
# #         kappa_optimal = cohen_kappa_score(labels_flat, preds_optimal)
# #         auroc = roc_auc_score(labels_flat, probs_flat)

# #         fold_metrics = {
# #             'Fold': r + 1,
# #             'AUROC': auroc,
# #             'Kappa_0.5': kappa_fixed,
# #             'Kappa_Opt': kappa_optimal,
# #             'Threshold': best_threshold
# #         }
# #         all_results.append(fold_metrics)
        
# #         print(f"{r+1:<5} | {auroc:.4f} | {kappa_fixed:.4f}  | {kappa_optimal:.4f}  | {best_threshold:.4f}")
# #     # 1. Convert results list to DataFrame
# #     df = pd.DataFrame(all_results)
# #    # 2. Calculate Summary Statistics
# #     # We select only the metric columns for the summary
# #     metrics_to_summarize = ['AUROC', 'Kappa_0.5', 'Kappa_Opt', 'Threshold']
# #     summary_mean = df[metrics_to_summarize].mean().to_frame(name='mean').T
# #     summary_std = df[metrics_to_summarize].std().to_frame(name='std').T

# #     # 3. Combine Folds and Summary into one final table for CSV
# #     # This adds two rows at the bottom for Mean and Std
# #     df_with_summary = pd.concat([df, summary_mean, summary_std], ignore_index=True)
    
# #     # Label the new rows in the 'Fold' column
# #     df_with_summary.at[df_with_summary.index[-2], 'Fold'] = 'Mean'
# #     df_with_summary.at[df_with_summary.index[-1], 'Fold'] = 'Std'

# #     # 4. Save to CSV
# #     csv_filename = os.path.join(model_dir,"cv_results_with_youdens.csv")
# #     df_with_summary.to_csv(csv_filename, index=False)

# #     # 5. Print Detailed Summary to Console
# #     print("\nFINAL SUMMARY WITH YOUDEN'S J (Mean ± Std):")
# #     means = df[metrics_to_summarize].mean()
# #     stds = df[metrics_to_summarize].std()

# #     for metric in metrics_to_summarize:
# #         print(f"{metric:<12}: {means[metric]:.4f} ± {stds[metric]:.4f}")

# #     print(f"\nFull fold results and summary saved to: {csv_filename}")
# # if __name__ == "__main__":
    
# #     validate_with_youdens_j()

# #     # summarize_cross_validation_results()
