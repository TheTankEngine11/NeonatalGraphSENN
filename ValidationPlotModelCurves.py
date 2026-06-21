from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


THESIS_TEXT_WIDTH_IN = 15.0 / 2.54
FIGSIZE = (THESIS_TEXT_WIDTH_IN, 2.55)
EPS = 1e-12

RESULTS_ROOT = Path("./Results_Performance")
OUT_DIR = Path("./Results_statistics/Classification/Curves")
PREDICTION_FILE = "fold_predictions_recompute_F2.npz"

# Hardcoded on purpose: this script is for the final thesis comparison figure.
MODEL_RESULTS = {
    "ST-GAT": RESULTS_ROOT / "Results_491483_MTbase_LR2e-2_WD1e-3",
    "SENN-IC": RESULTS_ROOT / "Results_492092_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-4",
    r"SENN-FC-$\theta(x)$": RESULTS_ROOT / "Results_486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0",
    r"SENN-FC-$\theta(h)$": RESULTS_ROOT / "Results_486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0",
    "Logistic regression": RESULTS_ROOT / "Results_486189_MTLogisticConcepts_LR2e-2_WD1e-4",
}

OKABE_ITO_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

MPLT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
]

MODEL_COLORS = {
    model_label: MPLT_COLORS[i]
    for i, model_label in enumerate(MODEL_RESULTS)
}


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
        "font.size": 7.4,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 5.9,
        "figure.titlesize": 9.2,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def load_fold_predictions(npz_path: Path):
    data = np.load(npz_path)
    n_folds = int(data["n_folds"])
    return [
        (data[f"fold_{fold}_y_true"], data[f"fold_{fold}_y_prob"])
        for fold in range(n_folds)
    ]


def prg_curve(y_true, y_prob):
    """Validation.py PRG logic, kept here so this script can run standalone."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    
    pi = float(np.mean(y_true))
    

    precision_gain = (precision - pi) / ((1.0 - pi) * np.clip(precision, EPS, None))
    recall_gain = (recall - pi) / ((1.0 - pi) * np.clip(recall, EPS, None))

    mask = np.isfinite(precision_gain) & np.isfinite(recall_gain)
    precision_gain = precision_gain[mask]
    recall_gain = recall_gain[mask]
    if recall_gain.size == 0:
        return None, None

    order = np.argsort(recall_gain)
    recall_gain = recall_gain[order]
    precision_gain = precision_gain[order]

    non_negative = np.where(recall_gain >= 0.0)[0]
    if non_negative.size == 0:
        return None, None

    first = int(non_negative[0])
    if first > 0 and recall_gain[first] > 0.0:
        x1, y1 = recall_gain[first - 1], precision_gain[first - 1]
        x2, y2 = recall_gain[first], precision_gain[first]
        y_at_zero = y1 + (0.0 - x1) * (y2 - y1) / (x2 - x1 + EPS)
        recall_gain = np.concatenate(([0.0], recall_gain[first:]))
        precision_gain = np.concatenate(([y_at_zero], precision_gain[first:]))
    else:
        recall_gain = recall_gain[first:]
        precision_gain = precision_gain[first:]

    if recall_gain[0] > 0.0:
        recall_gain = np.concatenate(([0.0], recall_gain))
        precision_gain = np.concatenate(([precision_gain[0]], precision_gain))
    if recall_gain[-1] < 1.0:
        recall_gain = np.concatenate((recall_gain, [1.0]))
        precision_gain = np.concatenate((precision_gain, [0.0]))

    recall_gain, unique_idx = np.unique(recall_gain, return_index=True)
    precision_gain = precision_gain[unique_idx]
    return recall_gain, precision_gain,pi


def summarize_roc_curves(fold_data):
    mean_fpr = np.linspace(0, 1, 101)
    curves = []
    scores = []

    for y_true, y_prob in fold_data:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        curves.append(interp_tpr)
        scores.append(roc_auc_score(y_true, y_prob))

    curves = np.asarray(curves)
    return mean_fpr, curves.mean(axis=0), curves.std(axis=0), np.mean(scores), np.std(scores, ddof=1)


def summarize_pr_curves(fold_data):
    mean_recall = np.linspace(0, 1, 101)
    curves = []
    scores = []

    for y_true, y_prob in fold_data:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        curves.append(interp_precision)
        scores.append(average_precision_score(y_true, y_prob))

    curves = np.asarray(curves)
    return mean_recall, curves.mean(axis=0), curves.std(axis=0), np.mean(scores), np.std(scores, ddof=1)


def summarize_prg_curves(fold_data):
    mean_recall_gain = np.linspace(0, 1, 101)
    curves = []
    scores = []
    prevelances = []

    for y_true, y_prob in fold_data:
        recall_gain, precision_gain, pi = prg_curve(y_true, y_prob)
        if recall_gain is None:
            continue
        curves.append(np.interp(mean_recall_gain, recall_gain, precision_gain))
        scores.append(auc(recall_gain, precision_gain))
        prevelances.append(pi)

    curves = np.asarray(curves)

    #Set mean global prevelance for baseline in AURPC plot
    global pi_global
    pi_global = np.mean(prevelances)
    # print(pi_global)
    return (
        mean_recall_gain,
        curves.mean(axis=0),
        curves.std(axis=0),
        np.mean(scores),
        np.std(scores, ddof=1),
    )


def plot_model_comparison_curves():
    apply_thesis_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    curve_specs = [
        ("AUROC", summarize_roc_curves, axes[0], "False positive rate", "True positive rate", "ROC"),
        ("AUPRC", summarize_pr_curves, axes[1], "Recall", "Precision", "Precision-recall"),
        ("AUPRG", summarize_prg_curves, axes[2], "Recall gain", "Precision gain", "Precision-recall-gain"),
    ]

    for model_label, results_dir in MODEL_RESULTS.items():
        npz_path = results_dir / PREDICTION_FILE
        fold_data = load_fold_predictions(npz_path)
        color = MODEL_COLORS.get(model_label, None)

        for metric_name, summarizer, ax, _, _, _ in curve_specs:
            x, mean_y, sd_y, _, _ = summarizer(fold_data)
            label = model_label if metric_name == "AUROC" else None

            ax.plot(x, mean_y, color=color, label=label, linewidth=1.15)
            ax.fill_between(
                x,
                np.maximum(mean_y - sd_y, 0.0),
                np.minimum(mean_y + sd_y, 1.0),
                color=color,
                alpha=0.13,
                linewidth=0,
            )

    axes[0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="0.55",
        linewidth=0.75,
        label="Metric baseline",
    )
    axes[1].axhline(
        pi_global,
        linestyle="--",
        color="0.55",
        linewidth=0.75,
        label=None,
    )
    axes[2].axhline(
        0.0,
        linestyle="--",
        color="0.55",
        linewidth=0.75,
        label=None,
    )

    for _, _, ax, xlabel, ylabel, title in curve_specs:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="both", color="0.88", linewidth=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    prg_handles, prg_labels = axes[2].get_legend_handles_labels()
    handles.extend(prg_handles)
    labels.extend(prg_labels)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, +0.05),
        ncol=len(MODEL_RESULTS) + 2,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 1.0], w_pad=1.2)

    png_path = OUT_DIR / "model_performance_curves_mean_sd.png"
    pdf_path = OUT_DIR / "model_performance_curves_mean_sd.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_model_comparison_curves()
