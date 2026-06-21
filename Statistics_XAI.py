from pathlib import Path
import itertools
import json
import math
import os
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, wilcoxon

try:
    from scipy.stats import brunnermunzel
except Exception:  # older scipy versions
    brunnermunzel = None

THESIS_TEXT_WIDTH_IN = 15.0 / 2.54
XAI_GRID_FIGSIZE = (THESIS_TEXT_WIDTH_IN, 4.55)
XAI_RMA_FIGSIZE = (THESIS_TEXT_WIDTH_IN, 2.85)
XAI_MODEL_TITLE_SIZE = 7.3
XAI_STAR_SIZE = 8.3
NON_SEIZURE_COLOR = "#56B4E9"  # blue
SEIZURE_COLOR = "#E69F00"      # orange
GLOBAL_COLOR = "#009E73"       # muted green
EDGE_COLOR = "#CC79A7"
UNKNOWN_COLOR = "#9e9e9e"      # neutral grey
CONDITION_COLORS = {
    "Non-seizure": NON_SEIZURE_COLOR,
    "Seizure": SEIZURE_COLOR,
    "Global": GLOBAL_COLOR,
    "Unknown": UNKNOWN_COLOR,
}
EXPLANATION_TYPE_COLORS = {
    "Node": GLOBAL_COLOR,
    "Edge": EDGE_COLOR,
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
        "font.size": 7.3,
        "axes.titlesize": 8.1,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.titlesize": 9.0,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
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

# =========================
# EDIT THESE SETTINGS FOR CORRECT DIRECTORIES (GLOBAL)
# =========================
# XAI_senn_test.py normally saves to: ./Results_<log_id>/Explainability_metrics/
BASE_DIR = Path("./Results_Performance/")
# BASE_DIR = Path("./ArchiveModelsRobLossSweep/")
RESULTS_SUBDIR = "XAI_metrics"
FLAT_CSV_NAME = "custom_metrics_flat_split.csv"
SPLIT_JSON_NAME = "custom_metrics_results_split.json"

OUT_DIR = Path("./Results_statistics/XAI/Main")
# OUT_DIR = Path("./Results_statistics/XAI/RLS")
# Primary paired model-vs-model test.
PRIMARY_PAIRED_TEST = "sign"  # "sign" or "wilcoxon"
PRIMARY_PAIRED_TEST = "wilcoxon"  # "sign" or "wilcoxon"
P_THRESHOLD = 0.05

# For seizure-vs-non-seizure comparisons within a model. These are usually unpaired
# windows, so a median-difference permutation test is used by default.
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# If your XAI flat CSV has no original sample/window id, pairing across models is by
# row order within each (Condition, Explainer/ExplanationType, Metric) group. This is
# valid only if all models were run on the same fold and same sample selection/thinning.
PAIR_BY_ORDER = True

# Model dictionary: display name -> log id. Edit these to match your XAI runs.
LOGS = {
    "STGAT": "491483_MTbase_LR2e-2_WD1e-3",
    "SENN IC": "492092_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-4",
    "SENN FC theta(x)": "486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0",
    "SENN FC theta(h)": "486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0",
    "Logistic Regression": "486189_MTLogisticConcepts_LR2e-2_WD1e-4",
}

# LOGS = {
#     "SENN IC lambda=0.0": "492082_MTSENNrawx_LR2e-3_WD1e-3_robloss0.0",
#     "SENN IC lambda=1e-8": "492083_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-8",
#     "SENN IC lambda=3e-8": "492084_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-8",
#     "SENN IC lambda=1e-7": "492085_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-7",
#     "SENN IC lambda=3e-7": "492086_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-7",
#     "SENN IC lambda=1e-6": "492087_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-6",
#     "SENN IC lambda=3e-6": "492088_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-6",
#     "SENN IC lambda=1e-5": "492089_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-5",
#     "SENN IC lambda=3e-5": "492090_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-5",
#     "SENN IC lambda=1e-4": "492091_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-4",
#     "SENN IC lambda=3e-4": "492092_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-4",
#     "SENN IC lambda=1e-3": "492093_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-3",
#     "SENN IC lambda=3e-3": "496198_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-3",
#     "SENN IC lambda=1e-2": "496199_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-2",
#     "SENN IC lambda=3e-2": "496200_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-2",
#     "SENN IC lambda=1e-1": "496201_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-1",
#     "SENN IC lambda=3e-1": "496202_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-1",
#     "SENN IC lambda=1.0": "496203_MTSENNrawx_LR2e-3_WD1e-3_robloss1.0",
    
# }

MODEL_ORDER = list(LOGS.keys())
BASELINE_MODEL = MODEL_ORDER[0]

CONDITION_ORDER = ["Non-seizure", "Seizure"]
EXPLANATION_TYPE_ORDER = ["Node", "Edge"]

# Set to None to include all metrics found in the files.
METRICS_TO_PLOT = None
OUTPUTCOMPLETENESS_METRIC_NAME = "OutputCompleteness_TargetEvidenceDeletion_IROF_AOC"
CONTINUITY_METRIC_NAME = "Continuity_RelativeInputStability" 
CORRECTNESS_METRIC_NAME = "Correctness_ParamRandomisation_corr_spearman" 
RMA_METRIC_NAME = "Coherence_RelevanceMassAccuracy_sample_global"
SHOWFLIERS = False
PANEL_READY_EXPORT = True

MODEL_DISPLAY_MAP = {
    "STGAT": "ST-GAT",
    "SENN IC": "SENN-IC",
    "SENN FC theta(x)": r"SENN-FC-$\theta(x)$",
    "SENN FC theta(h)": r"SENN-FC-$\theta(h)$",
    "Logistic Regression": "Logistic regression",
}

METRIC_DISPLAY_MAP = {
    OUTPUTCOMPLETENESS_METRIC_NAME: ("Output completeness", "IROF AOC"),
    CONTINUITY_METRIC_NAME: ("Continuity", "RIS"),
    CORRECTNESS_METRIC_NAME: ("Correctness", r"MPRT Spearman $\rho$"),
    RMA_METRIC_NAME: ("Coherence", "RMA"),
}

EXPLANATION_TYPE_DISPLAY_MAP = {
    "Node": "Node explanations",
    "Edge": "Edge explanations",
}

CONDITION_DISPLAY_MAP = {
    "Non-seizure": "Non-seizure",
    "Seizure": "Seizure",
}
# Example:
# METRICS_TO_PLOT = [
#     "Continuity_RelativeInputStability",
#     "Correctness_ParamRandomisation_corr_spearman",
#     "TopKDeletion_drop",
#     "Coherence_RelevanceMassAccuracy_sample_global",
# ]
# =========================


EXPLAINER_TYPE_MAP = {
    "IG_RAW": "Node",
    "IG": "Node",
    "INTEGRATEDGRADIENTS": "Node",
    "FOCUS_MAP": "Node",
    "FOCUSMAP": "Node",
    "SENN_FOCUS_MAP": "Node",
    "FIXED_NODE": "Node",
    "NODE_CONCEPTS": "Node",
    "NODE": "Node",
    "GNN_NODE": "Node",
    "GNN_EDGE": "Edge",
    "GNNEXPLAINER": "Edge",
    "EDGE": "Edge",
    "FIXED_EDGE": "Edge",
    "EDGE_CONCEPTS": "Edge",
}

EXPLAINER_DISPLAY_MAP = {
    "IG_RAW": "IG / raw node attribution",
    "GNN_EDGE": "GNNExplainer / edge mask",
    "FOCUS_MAP": "SENN focus map",
    "FIXED_NODE": "Fixed node concepts",
    "FIXED_EDGE": "Fixed edge concepts",
}


# -------------------------
# Loading helpers
# -------------------------
def safe_name(s: Any) -> str:
    s = str(s)
    s = s.replace("θ", "theta")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "unnamed"


def save_plot_png_pdf(fig: Figure, save_path: Path, dpi: int = 300) -> None:
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved plot: {save_path}")
    print(f"[OK] Saved plot: {pdf_path}")


def _boxplot_visible_y_limits(
    data: list[np.ndarray],
    showfliers: bool,
    whis: float = 1.5,
) -> tuple[float, float] | None:
    """Return the y-range that is actually visible in a Matplotlib boxplot."""
    finite_data = []
    for values in data:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            finite_data.append(arr)

    if not finite_data:
        return None

    if showfliers:
        visible_values = np.concatenate(finite_data)
        return float(np.min(visible_values)), float(np.max(visible_values))

    stats = boxplot_stats(finite_data, whis=whis)
    lows = [float(s["whislo"]) for s in stats if np.isfinite(s["whislo"])]
    highs = [float(s["whishi"]) for s in stats if np.isfinite(s["whishi"])]
    if not lows or not highs:
        visible_values = np.concatenate(finite_data)
        return float(np.min(visible_values)), float(np.max(visible_values))

    return min(lows), max(highs)


def _nonzero_y_span(ymin: float, ymax: float) -> float:
    y_span = ymax - ymin
    if y_span > 0:
        return y_span
    return abs(ymax) if ymax != 0 else 1.0


def _metric_scale_kind(metric: str) -> str:
    if metric == CONTINUITY_METRIC_NAME:
        return "log"
    if metric == CORRECTNESS_METRIC_NAME:
        return "correlation"
    if metric == RMA_METRIC_NAME:
        return "unit"
    if metric == OUTPUTCOMPLETENESS_METRIC_NAME:
        return "output_completeness"
    return "linear"


def _prepare_plot_values(metric: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if _metric_scale_kind(metric) == "log":
        values = values[values > 0]
    return values


def _shared_metric_y_limits(
    metric_df: pd.DataFrame,
    metric: str,
    model_order: list[str],
) -> tuple[str, tuple[float, float]]:
    scale_kind = _metric_scale_kind(metric)

    if scale_kind == "correlation":
        return "linear", (-1.0, 1.0)
    if scale_kind == "unit":
        return "linear", (0.0, 1.0)

    groups: list[np.ndarray] = []
    for explanation_type in EXPLANATION_TYPE_ORDER:
        for model in model_order:
            sub = metric_df[
                (metric_df["Model"] == model)
                & (metric_df["ExplanationType"] == explanation_type)
            ]
            for condition in CONDITION_ORDER:
                vals = sub.loc[sub["Condition"] == condition, "Value"].to_numpy(dtype=float)
                vals = _prepare_plot_values(metric, vals)
                if len(vals) > 0:
                    groups.append(vals)

    if not groups:
        return "linear", (0.0, 1.0)

    visible_limits = _boxplot_visible_y_limits(groups, SHOWFLIERS)
    if visible_limits is None:
        visible_values = np.concatenate(groups)
        visible_limits = (float(np.min(visible_values)), float(np.max(visible_values)))

    visible_ymin, visible_ymax = visible_limits

    if scale_kind == "log":
        positive_values = np.concatenate(groups)
        positive_values = positive_values[positive_values > 0]
        if len(positive_values) == 0:
            return "log", (1e-3, 1.0)
        bottom = max(min(visible_ymin, float(np.min(positive_values))) / 1.35, np.finfo(float).tiny)
        top = max(visible_ymax * 2.0, bottom * 10.0)
        return "log", (bottom, top)

    yrange = _nonzero_y_span(visible_ymin, visible_ymax)
    bottom = visible_ymin - 0.10 * yrange
    top = visible_ymax + 0.20 * yrange

    if scale_kind == "output_completeness":
        top = max(top, 1.0 + 0.06 * _nonzero_y_span(bottom, max(visible_ymax, 1.0)))
        if bottom >= 0:
            bottom = min(-0.05, visible_ymin - 0.10 * yrange)

    return "linear", (bottom, top)


def _star_y_from_ylim(metric: str, ylim: tuple[float, float]) -> float:
    bottom, top = ylim
    if _metric_scale_kind(metric) == "log":
        return 10 ** (np.log10(bottom) + 0.94 * (np.log10(top) - np.log10(bottom)))
    return bottom + 0.94 * (top - bottom)


def _extract_numeric_values(obj: Any) -> list[float]:
    """Flatten nested containers to finite numeric values, mirroring XAI_senn_test.py."""
    out: list[float] = []

    def rec(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)
            return
        if isinstance(x, np.ndarray):
            for v in x.ravel():
                rec(v.item() if hasattr(v, "item") else v)
            return
        if isinstance(x, (str, bytes)) or x is None:
            return
        if isinstance(x, (bool, np.bool_)):
            out.append(float(bool(x)))
            return
        if isinstance(x, (int, float, np.integer, np.floating)):
            v = float(x)
            if np.isfinite(v):
                out.append(v)
            return

    rec(obj)
    return out


def infer_explanation_type(explainer: Any) -> str:
    key = str(explainer).strip().upper()
    if key in EXPLAINER_TYPE_MAP:
        return EXPLAINER_TYPE_MAP[key]
    if "EDGE" in key:
        return "Edge"
    if any(token in key for token in ["NODE", "IG", "FOCUS"]):
        return "Node"
    return "Unknown"


def display_explainer(explainer: Any) -> str:
    key = str(explainer).strip().upper()
    return EXPLAINER_DISPLAY_MAP.get(key, str(explainer))


def display_model(model: Any) -> str:
    return MODEL_DISPLAY_MAP.get(str(model), str(model).replace("_", " "))


def display_condition(condition: Any) -> str:
    return CONDITION_DISPLAY_MAP.get(str(condition), str(condition).replace("_", " "))


def display_explanation_type(explanation_type: Any) -> str:
    return EXPLANATION_TYPE_DISPLAY_MAP.get(str(explanation_type), str(explanation_type).replace("_", " "))


def metric_title_and_ylabel(metric: Any) -> tuple[str, str]:
    metric = str(metric)
    if metric in METRIC_DISPLAY_MAP:
        return METRIC_DISPLAY_MAP[metric]

    readable = metric.replace("_", " ")
    readable = re.sub(r"\s+", " ", readable).strip()
    return readable, readable


def results_dir_for_log(log_id: str) -> Path:
    """Return the Explainability_metrics directory for one log id."""
    candidates = [
        BASE_DIR / f"Results_{log_id}" / RESULTS_SUBDIR,
        BASE_DIR / str(log_id) / RESULTS_SUBDIR,
        BASE_DIR / f"Results_{log_id}",
        BASE_DIR / str(log_id),
    ]
    for p in candidates:
        if (p / FLAT_CSV_NAME).exists() or (p / SPLIT_JSON_NAME).exists():
            return p
    msg = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find XAI results for log_id={log_id!r}. Tried:\n{msg}\n"
        f"Expected {FLAT_CSV_NAME} or {SPLIT_JSON_NAME}."
    )


def _normalise_condition_name(name: Any) -> str:
    s = str(name).strip().replace("_", "-").lower()
    if s in {"non-seizure", "non seizure", "nonseizure", "no seizure", "no-seizure"}:
        return "Non-seizure"
    if s in {"seizure", "seizures", "yes seizure", "yes-seizure"}:
        return "Seizure"
    if s == "global":
        return "Global"
    return str(name)


def ensure_sample_index(df: pd.DataFrame) -> pd.DataFrame:
    """Use an existing sample/window id when present; otherwise reconstruct by row order."""
    df = df.copy()
    candidates = ["SampleIdx", "sample_idx", "sample_index", "WindowIdx", "window_idx", "Index", "idx"]
    found = next((c for c in candidates if c in df.columns), None)
    if found is not None and found != "SampleIdx":
        df = df.rename(columns={found: "SampleIdx"})
    if "SampleIdx" not in df.columns:
        df["SampleIdx"] = df.groupby(["Condition", "Explainer", "Metric"], sort=False).cumcount()
    df["SampleIdx"] = pd.to_numeric(df["SampleIdx"], errors="coerce").astype("Int64")
    return df


def rows_from_split_json(json_path: Path) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict[str, Any]] = []
    for split_key, condition in [("non_seizure", "Non-seizure"), ("seizure", "Seizure")]:
        metrics = data.get(split_key, {}).get("metrics", {})
        for explainer, expl_metrics in metrics.items():
            for metric, values in expl_metrics.items():
                nums = _extract_numeric_values(values)
                for sample_idx, v in enumerate(nums):
                    rows.append(
                        {
                            "Condition": condition,
                            "Explainer": str(explainer),
                            "Metric": str(metric),
                            "Value": float(v),
                            "SampleIdx": int(sample_idx),
                        }
                    )

    # Optional scalar global RMA. This is included in summaries, but not in paired
    # seizure/non-seizure tests because it is not a per-window distribution.
    global_rma = data.get("global_rma", {})
    if isinstance(global_rma, dict):
        for explainer, block in global_rma.items():
            if explainer in {"meta", "error"}:
                continue
            if not isinstance(block, dict):
                continue
            metric = RMA_METRIC_NAME
            if metric in block and block[metric] is not None:
                rows.append(
                    {
                        "Condition": "Global",
                        "Explainer": str(explainer),
                        "Metric": metric,
                        "Value": float(block[metric]),
                        "SampleIdx": 0,
                    }
                )
    return rows


def load_xai_results(model_name: str, log_id: str) -> pd.DataFrame:
    results_dir = results_dir_for_log(log_id)
    flat_csv = results_dir / FLAT_CSV_NAME
    split_json = results_dir / SPLIT_JSON_NAME

    if flat_csv.exists():
        df = pd.read_csv(flat_csv)
        required = {"Condition", "Explainer", "Metric", "Value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{flat_csv} is missing required columns: {sorted(missing)}")
        df = ensure_sample_index(df)

        # Add global RMA from the JSON if present; the flat CSV generated by XAI_senn_test.py
        # only contains split metrics.
        if split_json.exists():
            global_rows = [r for r in rows_from_split_json(split_json) if r["Condition"] == "Global"]
            if global_rows:
                df = pd.concat([df, pd.DataFrame(global_rows)], ignore_index=True)
    elif split_json.exists():
        df = pd.DataFrame(rows_from_split_json(split_json))
        if df.empty:
            raise ValueError(f"No numeric XAI metric values found in {split_json}")
    else:
        raise FileNotFoundError(f"No {FLAT_CSV_NAME} or {SPLIT_JSON_NAME} in {results_dir}")

    df["Condition"] = df["Condition"].map(_normalise_condition_name)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[np.isfinite(df["Value"])].copy()

    df["Model"] = model_name
    df["Log"] = log_id
    df["ResultsDir"] = str(results_dir)
    df["ExplanationType"] = df["Explainer"].map(infer_explanation_type)
    df["ExplainerDisplay"] = df["Explainer"].map(display_explainer)

    print(f"Loaded {len(df):>7} values for {model_name:<22} from {results_dir}")
    return df


# -------------------------
# Descriptive statistics
# -------------------------
def q1(x: pd.Series) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), 0.25))


def q3(x: pd.Series) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), 0.75))


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Model", "Log", "Condition", "ExplanationType", "Explainer", "ExplainerDisplay", "Metric"]
    summary = (
        df.groupby(group_cols, dropna=False)["Value"]
        .agg(
            n="count",
            mean="mean",
            sd=lambda s: float(np.std(np.asarray(s, dtype=float), ddof=1)) if len(s) > 1 else np.nan,
            median="median",
            q1=q1,
            q3=q3,
            min="min",
            max="max",
        )
        .reset_index()
    )
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["median_iqr"] = summary.apply(
        lambda r: f"{r['median']:.6g} [{r['q1']:.6g}, {r['q3']:.6g}]", axis=1
    )
    return summary.sort_values(["Metric", "ExplanationType", "Condition", "Model"])


# -------------------------
# Statistical tests
# -------------------------
def paired_sign_test(diffs: np.ndarray) -> tuple[float, float, str]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0, "All paired differences are zero"
    n_pos = int(np.sum(nonzero > 0))
    n = int(len(nonzero))
    res = binomtest(k=n_pos, n=n, p=0.5, alternative="two-sided")
    return float(n_pos), float(res.pvalue), ""


def wilcoxon_test(diffs: np.ndarray) -> tuple[float, float, str]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, "No valid paired values"
    if np.sum(diffs != 0) == 0:
        return 0.0, 1.0, "All paired differences are zero"
    try:
        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
        return float(res.statistic), float(res.pvalue), "exact"
    except Exception as e:
        try:
            res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
            return float(res.statistic), float(res.pvalue), f"auto; exact unavailable: {e}"
        except Exception as e2:
            return np.nan, np.nan, f"Wilcoxon failed: {e2}"


def compare_paired_values(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    diffs = y - x

    out: dict[str, Any] = {
        "n_pairs": int(len(diffs)),
        "n_nonzero": int(np.sum(diffs != 0)) if len(diffs) else 0,
        "median_1": float(np.median(x)) if len(x) else np.nan,
        "median_2": float(np.median(y)) if len(y) else np.nan,
        "q1_1": float(np.quantile(x, 0.25)) if len(x) else np.nan,
        "q3_1": float(np.quantile(x, 0.75)) if len(x) else np.nan,
        "q1_2": float(np.quantile(y, 0.25)) if len(y) else np.nan,
        "q3_2": float(np.quantile(y, 0.75)) if len(y) else np.nan,
        "mean_1": float(np.mean(x)) if len(x) else np.nan,
        "mean_2": float(np.mean(y)) if len(y) else np.nan,
        "median_diff_paired": float(np.median(diffs)) if len(diffs) else np.nan,
        "median_diff_marginal": (float(np.median(y) - np.median(x)) if len(diffs) else np.nan),
        "mean_diff": float(np.mean(diffs)) if len(diffs) else np.nan,
    }
    out["iqr_1"] = out["q3_1"] - out["q1_1"] if np.isfinite(out["q3_1"]) else np.nan
    out["iqr_2"] = out["q3_2"] - out["q1_2"] if np.isfinite(out["q3_2"]) else np.nan

    if len(diffs) == 0:
        out.update(
            {
                "statistic": np.nan,
                "pvalue": np.nan,
                "method_used": "no_data",
                "pvalue_sign": np.nan,
                "pvalue_wilcoxon": np.nan,
                "wilcoxon_note": "No valid paired values",
            }
        )
        return out

    sign_stat, p_sign, sign_note = paired_sign_test(diffs)
    w_stat, p_wilcoxon, w_note = wilcoxon_test(diffs)

    out["pvalue_sign"] = p_sign
    out["sign_statistic_n_positive"] = sign_stat
    out["sign_note"] = sign_note
    out["pvalue_wilcoxon"] = p_wilcoxon
    out["wilcoxon_statistic"] = w_stat
    out["wilcoxon_note"] = w_note

    if PRIMARY_PAIRED_TEST.lower() == "wilcoxon":
        out["statistic"] = w_stat
        out["pvalue"] = p_wilcoxon
        out["method_used"] = "wilcoxon_signed_rank"
    else:
        out["statistic"] = sign_stat
        out["pvalue"] = p_sign
        out["method_used"] = "paired_sign_test"

    return out


def get_group_values(
    df: pd.DataFrame,
    model: str,
    condition: str,
    explanation_type: str,
    metric: str,
) -> pd.DataFrame:
    sub = df[
        (df["Model"] == model)
        & (df["Condition"] == condition)
        & (df["ExplanationType"] == explanation_type)
        & (df["Metric"] == metric)
    ].copy()
    if sub.empty:
        return sub
    # If duplicate SampleIdx values exist because multiple explainers mapped to the same
    # ExplanationType, average them before paired testing. This should normally not happen.
    return sub.groupby("SampleIdx", as_index=False)["Value"].mean().sort_values("SampleIdx")


def make_pairwise_model_tests(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows = []
    metrics = sorted(df.loc[df["Condition"].isin(CONDITION_ORDER), "Metric"].dropna().unique())
    explanation_types = [t for t in EXPLANATION_TYPE_ORDER if t in set(df["ExplanationType"])]

    for condition, explanation_type, metric in itertools.product(CONDITION_ORDER, explanation_types, metrics):
        for ref_model in model_order:
            xdf = get_group_values(df, ref_model, condition, explanation_type, metric)
            if xdf.empty:
                continue
            for comp_model in model_order:
                ydf = get_group_values(df, comp_model, condition, explanation_type, metric)
                if ydf.empty:
                    continue
                aligned = pd.merge(xdf, ydf, on="SampleIdx", suffixes=("_1", "_2"), how="inner")
                if aligned.empty:
                    continue
                out = compare_paired_values(aligned["Value_1"].to_numpy(), aligned["Value_2"].to_numpy())
                out.update(
                    {
                        "condition": condition,
                        "explanation_type": explanation_type,
                        "metric": metric,
                        "reference_model": ref_model,
                        "comparator_model": comp_model,
                    }
                )
                rows.append(out)

    res = pd.DataFrame(rows)
    if not res.empty:
        res = add_bh_qvalues(res, p_col="pvalue", q_col="qvalue_bh")
        res = res.sort_values(["metric", "explanation_type", "condition", "pvalue"], na_position="last")
    return res


def permutation_median_test_unpaired(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Two-sided Monte Carlo permutation test for difference in medians: median(y)-median(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    observed = float(np.median(y) - np.median(x))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_permutations)):
        perm = rng.permutation(pooled)
        stat = float(np.median(perm[n_x:]) - np.median(perm[:n_x]))
        if abs(stat) >= abs(observed):
            count += 1
    pvalue = (count + 1) / (int(n_permutations) + 1)
    return observed, float(pvalue)


def compare_unpaired_conditions(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    out: dict[str, Any] = {
        "n_non_seizure": int(len(x)),
        "n_seizure": int(len(y)),
        "median_non_seizure": float(np.median(x)) if len(x) else np.nan,
        "median_seizure": float(np.median(y)) if len(y) else np.nan,
        "q1_non_seizure": float(np.quantile(x, 0.25)) if len(x) else np.nan,
        "q3_non_seizure": float(np.quantile(x, 0.75)) if len(x) else np.nan,
        "q1_seizure": float(np.quantile(y, 0.25)) if len(y) else np.nan,
        "q3_seizure": float(np.quantile(y, 0.75)) if len(y) else np.nan,
    }
    out["median_diff_seizure_minus_non"] = (
        out["median_seizure"] - out["median_non_seizure"]
        if np.isfinite(out["median_seizure"]) and np.isfinite(out["median_non_seizure"])
        else np.nan
    )
    out["permutation_statistic"] = np.nan
    out["pvalue_median_permutation"] = np.nan
    out["pvalue_mannwhitney"] = np.nan
    out["pvalue_brunnermunzel"] = np.nan
    out["method_used"] = "median_permutation_unpaired"
    out["note"] = ""

    if len(x) == 0 or len(y) == 0:
        out["method_used"] = "no_data"
        out["note"] = "Need both non-seizure and seizure values"
        return out

    stat, p_perm = permutation_median_test_unpaired(x, y)
    out["permutation_statistic"] = stat
    out["pvalue_median_permutation"] = p_perm
    out["pvalue"] = p_perm

    try:
        mw = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        out["pvalue_mannwhitney"] = float(mw.pvalue)
    except Exception as e:
        out["note"] += f"Mann-Whitney failed: {e}; "

    if brunnermunzel is not None:
        try:
            bm = brunnermunzel(x, y, alternative="two-sided")
            out["pvalue_brunnermunzel"] = float(bm.pvalue)
        except Exception as e:
            out["note"] += f"Brunner-Munzel failed: {e}; "

    return out


def make_condition_tests(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows = []
    metrics = sorted(df.loc[df["Condition"].isin(CONDITION_ORDER), "Metric"].dropna().unique())
    explanation_types = [t for t in EXPLANATION_TYPE_ORDER if t in set(df["ExplanationType"])]

    for model, explanation_type, metric in itertools.product(model_order, explanation_types, metrics):
        xdf = get_group_values(df, model, "Non-seizure", explanation_type, metric)
        ydf = get_group_values(df, model, "Seizure", explanation_type, metric)
        if xdf.empty or ydf.empty:
            continue
        out = compare_unpaired_conditions(xdf["Value"].to_numpy(), ydf["Value"].to_numpy())
        out.update({"model": model, "explanation_type": explanation_type, "metric": metric})
        rows.append(out)

    res = pd.DataFrame(rows)
    if not res.empty:
        res = add_bh_qvalues(res, p_col="pvalue", q_col="qvalue_bh")
        res = res.sort_values(["metric", "explanation_type", "pvalue"], na_position="last")
    return res


def add_bh_qvalues(df: pd.DataFrame, p_col: str = "pvalue", q_col: str = "qvalue_bh") -> pd.DataFrame:
    """Benjamini-Hochberg q-values over all finite p-values in df."""
    df = df.copy()
    p = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        df[q_col] = q
        return df
    p_fin = p[finite]
    order = np.argsort(p_fin)
    ranked = p_fin[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    q_fin = np.empty_like(q_ranked)
    q_fin[order] = q_ranked
    q[finite] = q_fin
    df[q_col] = q
    return df


# -------------------------
# Plotting
# -------------------------
def plot_delta_pvalue_pivot_tables(
    all_results_df: pd.DataFrame,
    output_dir: Path,
    metrics_to_plot: list[str] | None = None,
    model_order: list[str] | None = None,
    delta_col: str = "median_diff_paired",
    pvalue_col: str = "pvalue",
    p_threshold: float = P_THRESHOLD,
) -> None:
    """
    Create Statistics.py-style pivot tables, now stratified by condition and node/edge.

    Saves, per Condition x ExplanationType x Metric:
        - pivot_delta_*.csv
        - pivot_pvalue_*.csv
        - pivot_delta_pvalue_*.csv
        - pivot_delta_pvalue_*.png
    """
    if all_results_df.empty:
        print("[SKIP] No pairwise results to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_to_plot is None:
        metrics_to_plot = sorted(all_results_df["metric"].dropna().unique())

    for condition in CONDITION_ORDER:
        for explanation_type in EXPLANATION_TYPE_ORDER:
            for metric in metrics_to_plot:
                metric_df = all_results_df[
                    (all_results_df["condition"] == condition)
                    & (all_results_df["explanation_type"] == explanation_type)
                    & (all_results_df["metric"] == metric)
                ].copy()

                if metric_df.empty:
                    continue

                delta_pivot = metric_df.pivot(
                    index="reference_model",
                    columns="comparator_model",
                    values=delta_col,
                )
                p_pivot = metric_df.pivot(
                    index="reference_model",
                    columns="comparator_model",
                    values=pvalue_col,
                )

                if model_order is not None:
                    row_order = [m for m in model_order if m in delta_pivot.index]
                    col_order = [m for m in model_order if m in delta_pivot.columns]
                    delta_pivot = delta_pivot.reindex(index=row_order, columns=col_order)
                    p_pivot = p_pivot.reindex(index=row_order, columns=col_order)
                else:
                    delta_pivot = delta_pivot.sort_index(axis=0).sort_index(axis=1)
                    p_pivot = p_pivot.reindex(index=delta_pivot.index, columns=delta_pivot.columns)

                combined = pd.DataFrame(index=delta_pivot.index, columns=delta_pivot.columns)
                for ref in delta_pivot.index:
                    for comp in delta_pivot.columns:
                        delta = delta_pivot.loc[ref, comp]
                        pval = p_pivot.loc[ref, comp]
                        if pd.isna(delta) or pd.isna(pval):
                            combined.loc[ref, comp] = ""
                        else:
                            combined.loc[ref, comp] = f"{delta:+.3g} ({pval:.3g})"

                subdir = output_dir / safe_name(explanation_type) / safe_name(condition)
                subdir.mkdir(parents=True, exist_ok=True)
                stem = f"{safe_name(explanation_type)}_{safe_name(condition)}_{safe_name(metric)}"
                delta_pivot.to_csv(subdir / f"pivot_delta_{stem}.csv")
                p_pivot.to_csv(subdir / f"pivot_pvalue_{stem}.csv")
                combined.to_csv(subdir / f"pivot_delta_pvalue_{stem}.csv")

                n_rows, n_cols = combined.shape
                cell_size = 1.45
                fig_width = max(5.0, cell_size * n_cols + 2.5)
                fig_height = max(3.5, cell_size * n_rows + 2.0)

                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.set_aspect("equal", adjustable="box")
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                ax.set_xlim(-0.5, n_cols - 0.5)
                ax.set_ylim(n_rows - 0.5, -0.5)
                ax.set_xticks(np.arange(n_cols))
                ax.set_yticks(np.arange(n_rows))
                ax.set_xticklabels([display_model(m) for m in combined.columns], rotation=45, ha="right")
                ax.set_yticklabels([display_model(m) for m in combined.index])
                ax.set_xlabel("Comparator model")
                ax.set_ylabel("Reference model")
                metric_title, metric_ylabel = metric_title_and_ylabel(metric)
                ax.set_title(
                    f"{metric_title}: {metric_ylabel}\n"
                    f"{display_condition(condition)} | {display_explanation_type(explanation_type)}: "
                    r"$\Delta$ median paired difference (p)"
                )

                ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
                ax.grid(which="minor", linewidth=0.6, alpha=0.35)
                ax.tick_params(which="minor", bottom=False, left=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                for i, ref in enumerate(combined.index):
                    for j, comp in enumerate(combined.columns):
                        text = combined.loc[ref, comp]
                        pval = p_pivot.loc[ref, comp]
                        is_sig = pd.notna(pval) and pval < p_threshold
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            fontsize=plt.rcParams["font.size"],
                            fontweight="bold" if is_sig else "normal",
                            color="black",
                        )

                plt.tight_layout()
                save_path = subdir / f"pivot_delta_pvalue_{stem}.png"
                save_plot_png_pdf(fig, save_path, dpi=300)
                plt.close(fig)
def p_to_stars_base_comparison(p: float) -> str:
    """Return significance stars for raw p-values."""
    if not np.isfinite(p):
        return ""
    if p < 0.005:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def get_base_comparison_pvalue(
    pairwise_df: pd.DataFrame | None,
    baseline_model: str,
    comparator_model: str,
    condition: str,
    explanation_type: str,
    metric: str,
) -> float:
    """
    Get p-value for comparator_model versus baseline_model
    for the same condition, explanation type, and metric.
    """
    if pairwise_df is None or pairwise_df.empty:
        return np.nan

    row = pairwise_df[
        (pairwise_df["reference_model"] == baseline_model)
        & (pairwise_df["comparator_model"] == comparator_model)
        & (pairwise_df["condition"] == condition)
        & (pairwise_df["explanation_type"] == explanation_type)
        & (pairwise_df["metric"] == metric)
    ]

    if row.empty:
        return np.nan

    return float(row["pvalue"].iloc[0])

def plot_boxplots_per_metric(
    df: pd.DataFrame,
    output_dir: Path,
    model_order: list[str],
    metrics_to_plot: list[str] | None = None,
    pairwise_df: pd.DataFrame | None = None,
    baseline_model: str | None = None,
) -> None:
    """
    For each metric, create two stacked grouped-boxplot panels.
    Row 1: node explanations. Row 2: edge explanations.
    Within each row, models are grouped on the x-axis and coloured boxes show
    non-seizure versus seizure windows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline_model is None:
        baseline_model = model_order[0]

    plot_df = df[df["Condition"].isin(CONDITION_ORDER)].copy()
    if metrics_to_plot is None:
        metrics_to_plot = sorted(plot_df["Metric"].dropna().unique())

    for metric in metrics_to_plot:
        metric_df = plot_df[plot_df["Metric"] == metric].copy()
        if metric_df.empty:
            continue

        metric_title, metric_ylabel = metric_title_and_ylabel(metric)
        y_scale, shared_ylim = _shared_metric_y_limits(metric_df, metric, model_order)
        star_y = _star_y_from_ylim(metric, shared_ylim)

        n_rows = len(EXPLANATION_TYPE_ORDER)
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=XAI_GRID_FIGSIZE,
            squeeze=False,
            sharex=True,
            sharey=False,
        )
        axes_flat = axes.ravel()

        model_positions = np.arange(len(model_order), dtype=float)
        condition_offsets = {
            "Non-seizure": -0.16,
            "Seizure": 0.16,
        }
        box_width = 0.26

        legend_handles = [
            Patch(
                facecolor=CONDITION_COLORS[condition],
                edgecolor=CONDITION_COLORS[condition],
                alpha=0.28,
                label=display_condition(condition),
            )
            for condition in CONDITION_ORDER
        ]

        for r, explanation_type in enumerate(EXPLANATION_TYPE_ORDER):
            ax = axes_flat[r]
            data: list[np.ndarray] = []
            positions: list[float] = []
            position_conditions: list[str] = []
            data_exists: dict[tuple[str, str], bool] = {}

            for model_idx, model in enumerate(model_order):
                sub = metric_df[
                    (metric_df["Model"] == model) & (metric_df["ExplanationType"] == explanation_type)
                ]
                for condition in CONDITION_ORDER:
                    vals = sub.loc[sub["Condition"] == condition, "Value"].to_numpy(dtype=float)
                    vals = _prepare_plot_values(metric, vals)
                    has_data = len(vals) > 0
                    data_exists[(model, condition)] = has_data
                    if has_data:
                        data.append(vals)
                        positions.append(model_positions[model_idx] + condition_offsets[condition])
                        position_conditions.append(condition)

            if data:
                box = ax.boxplot(
                    data,
                    positions=positions,
                    widths=box_width,
                    showmeans=False,
                    showfliers=SHOWFLIERS,
                    patch_artist=True,
                )
                for patch, condition in zip(box["boxes"], position_conditions):
                    color = CONDITION_COLORS.get(condition, UNKNOWN_COLOR)
                    patch.set_facecolor(color)
                    patch.set_alpha(0.40)
                    patch.set_edgecolor(color)
                    patch.set_linewidth(0.8)
                for median, condition in zip(box["medians"],position_conditions):
                    color = CONDITION_COLORS.get(condition, UNKNOWN_COLOR)
                    median.set_color(color)
                    median.set_linewidth(0.9)
                for key in ("whiskers", "caps"):
                    for artist in box[key]:
                        artist.set_color("#444444")
                        artist.set_linewidth(0.7)

                if pairwise_df is not None:
                    for model_idx, model in enumerate(model_order):
                        if model == baseline_model:
                            continue
                        for condition in CONDITION_ORDER:
                            if not data_exists.get((model, condition), False):
                                continue
                            pval = get_base_comparison_pvalue(
                                pairwise_df=pairwise_df,
                                baseline_model=baseline_model,
                                comparator_model=model,
                                condition=condition,
                                explanation_type=explanation_type,
                                metric=metric,
                            )
                            stars = p_to_stars_base_comparison(pval)
                            if stars:
                                ax.text(
                                    model_positions[model_idx] + condition_offsets[condition],
                                    star_y,
                                    stars,
                                    ha="center",
                                    va="center",
                                    fontsize=XAI_STAR_SIZE,
                                    fontweight="bold",
                                    color="black",
                                )
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

            ax.set_yscale(y_scale)
            ax.set_ylim(shared_ylim)
            if metric == OUTPUTCOMPLETENESS_METRIC_NAME:
                ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            ax.set_title(display_explanation_type(explanation_type), fontsize=XAI_MODEL_TITLE_SIZE, fontweight="normal")
            ax.set_ylabel(metric_ylabel, fontsize=plt.rcParams["axes.labelsize"])
            ax.grid(alpha=0.25, axis="y", linewidth=0.4)
            ax.set_xlim(-0.55, len(model_order) - 0.45)
            ax.set_xticks(model_positions)
            ax.set_xticklabels(
                [display_model(model) for model in model_order],
                fontsize=plt.rcParams["xtick.labelsize"],
                rotation=18,
                ha="right",
            )
            left = ax.get_position().x0
            right = ax.get_position().x1
            center_x = 0.5 * (left + right)
        
        fig.suptitle(metric_title, x=center_x+0.025, y=1, fontsize=plt.rcParams["figure.titlesize"], fontweight="normal")
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(center_x+0.025, 0.95),
            ncol=len(legend_handles),
            frameon=False,
            handlelength=1.2,
            columnspacing=1.0,
        )
        if pairwise_df is not None and baseline_model is not None and not PANEL_READY_EXPORT:
            fig.text(
                0.5,
                0.905,
                (
                    f"Stars compare each model with {display_model(baseline_model)} "
                    r"within the same condition: * $p<0.05$, ** $p<0.005$."
                ),
                ha="center",
                va="center",
                fontsize=plt.rcParams["axes.labelsize"],
            )
            layout_top = 0.82
        else:
            layout_top = 0.95
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = output_dir / f"boxplot_{safe_name(metric)}.png"
        save_plot_png_pdf(fig, save_path, dpi=250)
        plt.close(fig)

def plot_global_rma_barplot(
    df: pd.DataFrame,
    output_dir: Path,
    model_order: list[str],
    metric_name: str = RMA_METRIC_NAME,
) -> None:
    """
    Plot global RMA as a grouped bar plot across models.

    RMA is a global scalar, not a per-window distribution.
    Therefore, this function only plots descriptive values and pairwise deltas.
    No statistical test is performed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rma_df = df[
        (df["Condition"] == "Global")
        & (df["Metric"] == metric_name)
        & (df["ExplanationType"].isin(EXPLANATION_TYPE_ORDER))
    ].copy()

    if rma_df.empty:
        print(f"[SKIP] No global RMA values found for metric: {metric_name}")
        return

    # Average in case duplicate rows exist for the same model/type.
    rma_summary = (
        rma_df.groupby(["Model", "ExplanationType"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "RMA"})
    )

    rma_summary["Model"] = pd.Categorical(
        rma_summary["Model"],
        categories=model_order,
        ordered=True,
    )
    rma_summary["ExplanationType"] = pd.Categorical(
        rma_summary["ExplanationType"],
        categories=EXPLANATION_TYPE_ORDER,
        ordered=True,
    )
    rma_summary = rma_summary.sort_values(["Model", "ExplanationType"])

    rma_summary.to_csv(output_dir / "xai_global_rma_values.csv", index=False)

    pivot = rma_summary.pivot(
        index="Model",
        columns="ExplanationType",
        values="RMA",
    ).reindex(index=model_order, columns=EXPLANATION_TYPE_ORDER)

    pivot.to_csv(output_dir / "xai_global_rma_summary.csv")

    # Descriptive deltas versus baseline model.
    baseline_model = model_order[0]
    delta_rows = []
    for explanation_type in EXPLANATION_TYPE_ORDER:
        if explanation_type not in pivot.columns:
            continue

        baseline_value = pivot.loc[baseline_model, explanation_type]

        for model in model_order:
            value = pivot.loc[model, explanation_type] if model in pivot.index else np.nan
            delta = value - baseline_value if np.isfinite(value) and np.isfinite(baseline_value) else np.nan

            delta_rows.append(
                {
                    "baseline_model": baseline_model,
                    "comparator_model": model,
                    "explanation_type": explanation_type,
                    "baseline_rma": baseline_value,
                    "comparator_rma": value,
                    "delta_vs_baseline": delta,
                    "pvalue": np.nan,
                    "note": "Global RMA is one scalar per model/explainer; no inferential test performed.",
                }
            )

    pd.DataFrame(delta_rows).to_csv(
        output_dir / "xai_global_rma_deltas_vs_baseline.csv",
        index=False,
    )

    # Plot grouped bar plot.
    x = np.arange(len(model_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=XAI_RMA_FIGSIZE)

    for i, explanation_type in enumerate(EXPLANATION_TYPE_ORDER):
        if explanation_type not in pivot.columns:
            continue

        values = pivot[explanation_type].to_numpy(dtype=float)
        offset = (i - (len(EXPLANATION_TYPE_ORDER) - 1) / 2) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            color=EXPLANATION_TYPE_COLORS.get(explanation_type, UNKNOWN_COLOR),
            label=display_explanation_type(explanation_type),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([display_model(model) for model in model_order], rotation=20, ha="right")
    ax.set_ylabel("RMA")
    ax.set_title("Coherence", fontweight="normal")
    ax.set_ylim(bottom=0,top=1.05)
    ax.grid(alpha=0.25, axis="y", linewidth=0.4)
    ax.legend(title="Explanation")

    # ax.text(
    #     0.5,
    #     -0.25,
    #     "RMA is reported descriptively only; no statistical test was performed because each bar is one global scalar.",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="top",
    #     fontsize=9,
    # )

    fig.tight_layout()
    save_path = output_dir / f"barplot_{safe_name(metric_name)}.png"
    save_plot_png_pdf(fig, save_path, dpi=300)
    plt.close(fig)
# -------------------------
# Main
# -------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_dir = OUT_DIR / "CSVfiles"
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_values = []
    for model_name, log_id in LOGS.items():
        all_values.append(load_xai_results(model_name, log_id))

    values_df = pd.concat(all_values, ignore_index=True)
    values_df.to_csv(csv_dir / "xai_all_values_long.csv", index=False)

    summary_df = make_summary_table(values_df)
    summary_df.to_csv(csv_dir / "xai_summary_median_iqr.csv", index=False)
    print(f"\nSaved summary table to: {csv_dir / 'xai_summary_median_iqr.csv'}")

    pairwise_df = make_pairwise_model_tests(values_df, MODEL_ORDER)
    pairwise_path = csv_dir / "xai_pairwise_between_models.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"Saved pairwise model tests to: {pairwise_path}")

    condition_df = make_condition_tests(values_df, MODEL_ORDER)
    condition_path = csv_dir / "xai_seizure_vs_nonseizure_within_models.csv"
    condition_df.to_csv(condition_path, index=False)
    print(f"Saved seizure-vs-non-seizure tests to: {condition_path}")

    plot_delta_pvalue_pivot_tables(
        all_results_df=pairwise_df,
        output_dir=OUT_DIR / "PivotTables",
        metrics_to_plot=METRICS_TO_PLOT,
        model_order=MODEL_ORDER,
        p_threshold=P_THRESHOLD,
    )

    plot_boxplots_per_metric(
        df=values_df,
        output_dir=OUT_DIR / "Boxplots",
        model_order=MODEL_ORDER,
        metrics_to_plot=METRICS_TO_PLOT,
        pairwise_df=pairwise_df,
        baseline_model=BASELINE_MODEL,
    )

    plot_global_rma_barplot(
        df=values_df,
        output_dir=OUT_DIR / "GlobalRMA",
        model_order=MODEL_ORDER,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")

    print("\nSummary preview:")
    print(summary_df.head(20).to_string(index=False))

    if not pairwise_df.empty:
        print("\nPairwise test preview:")
        cols = [
            "condition",
            "explanation_type",
            "metric",
            "reference_model",
            "comparator_model",
            "n_pairs",
            "median_diff_paired",
            "pvalue",
            "qvalue_bh",
            "method_used",
        ]
        print(pairwise_df[cols].head(20).to_string(index=False))

    print(f"\nDone. Outputs are in: {OUT_DIR}")


def _main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--robustness-sweep-summary",
        action="store_true",
        help="Create compact XAI robustness-loss sweep summary figures.",
    )
    parser.add_argument(
        "--robustness-results-root",
        default="./ArchiveModelsRobLossSweep",
        help="Root directory with robustness-loss sweep XAI results.",
    )
    parser.add_argument(
        "--robustness-out-dir",
        default="./Report/figures/XAI_metrics/robustness_sweep",
        help="Output directory for robustness-loss sweep XAI summary figures.",
    )
    args, _unknown = parser.parse_known_args()

    if args.robustness_sweep_summary:
        from Plot_xai_RLS import plot_robustness_sweep_summary

        plot_robustness_sweep_summary(
            results_root=args.robustness_results_root,
            out_dir=args.robustness_out_dir,
        )
        return

    main()


if __name__ == "__main__":
    _main_cli()
