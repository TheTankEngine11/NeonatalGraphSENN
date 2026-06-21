from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Thesis layout: A4 width minus inner/outer/binding margins in main.tex.
THESIS_TEXT_WIDTH_IN = 15.0 / 2.54
THESIS_HALF_WIDTH_IN = 0.48 * THESIS_TEXT_WIDTH_IN
RLS_PANEL_FIGSIZE = (THESIS_HALF_WIDTH_IN, 2.05)
RLS_SUMMARY_FIGSIZE = (THESIS_TEXT_WIDTH_IN, 6.15)
NON_SEIZURE_COLOR = "#56B4E9"  # blue
SEIZURE_COLOR = "#E69F00"     # orange
GLOBAL_COLOR = "#333333"       # muted green
NODE_COLOR = "#009E73" 
EDGE_COLOR = "#CC79A7"
UNKNOWN_COLOR = "#9e9e9e"      # neutral grey\


HIGH_REG_SHADE = "0.90"


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
        "lines.markersize": 2.4,
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

try:
    from Statistics_XAI import (
        _extract_numeric_values,
        _normalise_condition_name,
        display_explainer,
        ensure_sample_index,
        infer_explanation_type,
        rows_from_split_json,
        safe_name,
    )
except Exception as exc:  # pragma: no cover - fallback for standalone use.
    warnings.warn(
        f"Could not import Statistics_XAI helpers ({exc}). Using local fallbacks.",
        RuntimeWarning,
    )

    def safe_name(s: Any) -> str:
        s = str(s)
        s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
        return s.strip("_") or "unnamed"

    def _extract_numeric_values(obj: Any) -> list[float]:
        out: list[float] = []

        def rec(x: Any) -> None:
            if isinstance(x, dict):
                for v in x.values():
                    rec(v)
            elif isinstance(x, (list, tuple, set)):
                for v in x:
                    rec(v)
            elif isinstance(x, np.ndarray):
                for v in x.ravel():
                    rec(v.item() if hasattr(v, "item") else v)
            elif isinstance(x, (str, bytes)) or x is None:
                return
            elif isinstance(x, (bool, np.bool_)):
                out.append(float(bool(x)))
            elif isinstance(x, (int, float, np.integer, np.floating)):
                v = float(x)
                if np.isfinite(v):
                    out.append(v)

        rec(obj)
        return out

    def _normalise_condition_name(name: Any) -> str:
        s = str(name).strip().replace("_", "-").lower()
        if s in {"non-seizure", "non seizure", "nonseizure", "no seizure", "no-seizure"}:
            return "Non-seizure"
        if s in {"seizure", "seizures", "yes seizure", "yes-seizure"}:
            return "Seizure"
        if s == "global":
            return "Global"
        return str(name)

    def infer_explanation_type(explainer: Any) -> str:
        key = str(explainer).upper()
        if "EDGE" in key:
            return "Edge"
        if any(token in key for token in ("NODE", "IG", "FOCUS")):
            return "Node"
        return "Unknown"

    def display_explainer(explainer: Any) -> str:
        return str(explainer)

    def ensure_sample_index(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "SampleIdx" not in df.columns:
            df["SampleIdx"] = df.groupby(["Condition", "Explainer", "Metric"], sort=False).cumcount()
        return df

    def rows_from_split_json(json_path: Path) -> list[dict[str, Any]]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows: list[dict[str, Any]] = []
        for split_key, condition in [("non_seizure", "Non-seizure"), ("seizure", "Seizure")]:
            metrics = data.get(split_key, {}).get("metrics", {})
            for explainer, expl_metrics in metrics.items():
                for metric, values in expl_metrics.items():
                    for sample_idx, v in enumerate(_extract_numeric_values(values)):
                        rows.append(
                            {
                                "Condition": condition,
                                "Explainer": str(explainer),
                                "Metric": str(metric),
                                "Value": float(v),
                                "SampleIdx": int(sample_idx),
                            }
                        )
        global_rma = data.get("global_rma", {})
        if isinstance(global_rma, dict):
            metric = "Coherence_RelevanceMassAccuracy_sample_global"
            for explainer, block in global_rma.items():
                if isinstance(block, dict) and block.get(metric) is not None:
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


FLAT_CSV_NAME = "custom_metrics_flat_split.csv"
SPLIT_JSON_NAME = "custom_metrics_results_split.json"
PER_CONDITION_JSON_NAME = "custom_metrics_results.json"

LAMBDA_LABELS = [
    "0",
    "1e-8",
    "3e-8",
    "1e-7",
    "3e-7",
    "1e-6",
    "3e-6",
    "1e-5",
    "3e-5",
    "1e-4",
    "3e-4",
    "1e-3",
    "3e-3",
    "1e-2",
    "3e-2",
    "1e-1",
    "3e-1",
    "1",
]
LAMBDA_VALUES = np.array([float(x) for x in LAMBDA_LABELS], dtype=float)
LAMBDA_TO_LABEL = {float(v): label for v, label in zip(LAMBDA_VALUES, LAMBDA_LABELS)}
LAMBDA_ORDER = {label: idx for idx, label in enumerate(LAMBDA_LABELS)}
HIGH_REG_START_LABEL = "3e-2"
PERFORMANCE_METRIC = "AUPRC"
PANEL_READY_EXPORT = True

CONDITION_ORDER = ["Non-seizure", "Seizure", "Global"]
EXPLANATION_TYPE_ORDER = ["Node", "Edge", "Unknown"]
CONDITION_COLORS = {
    "Non-seizure": NON_SEIZURE_COLOR,
    "Seizure": SEIZURE_COLOR,
    "Global": GLOBAL_COLOR,
    "Unknown": UNKNOWN_COLOR,
}
EXPLANATION_LINESTYLES = {
    "Node": "-",
    "Edge": "--",
    "Unknown": ":",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    contains: str
    filename_stem: str
    title: str
    ylabel: str
    higher_is_better: bool
    yaxis: tuple[float | None, float | None] | None = None
    use_abs: bool = False
    global_metric: bool = False


METRIC_SPECS = [
    MetricSpec(
        key="continuity",
        contains="Continuity_RelativeInputStability",
        filename_stem="robustness_sweep_continuity",
        title="Continuity",
        ylabel="RIS",
        higher_is_better=False,
        yaxis=(0.0, None),
    ),
    MetricSpec(
        key="correctness_abs_spearman",
        contains="Correctness_ParamRandomisation_corr_spearman",
        filename_stem="robustness_sweep_correctness_abs_spearman",
        title="Correctness",
        ylabel=r"MPRT Spearman $\rho$",
        higher_is_better=False,
        yaxis=(-1.0, 1.0),
        use_abs=False,
    ),
    MetricSpec(
        key="output_completeness",
        contains="OutputCompleteness_TargetEvidenceDeletion_IROF_AOC",
        filename_stem="robustness_sweep_output_completeness",
        title="Output completeness",
        ylabel="IROF AOC",
        higher_is_better=True,
        yaxis=(None, 1.0),
    ),
    MetricSpec(
        key="topk_deletion",
        contains="TopKDeletion_drop",
        filename_stem="robustness_sweep_topk_deletion",
        title="Top-k deletion",
        ylabel="Top-k drop",
        higher_is_better=True,
        yaxis=(-1.0, 1.0),
    ),
    MetricSpec(
        key="coherence",
        contains="Coherence_RelevanceMassAccuracy_sample_global",
        filename_stem="robustness_sweep_coherence",
        title="Coherence",
        ylabel="RMA",
        higher_is_better=True,
        yaxis=(0.4, 0.8),
        global_metric=True,
    ),
]

CAPTION = (
    "Classification AUPRC and XAI robustness metrics across robustness-loss weights. "
    "AUPRC is shown as mean plus/minus SD across folds. XAI metrics are shown as "
    "median with interquartile range. The shaded region starts at "
    f"lambda_rob={HIGH_REG_START_LABEL}."
)

_PREFIXED_LAMBDA_RE = re.compile(
    r"(?:lambda(?:_rob)?|rob(?:ustness)?[_-]?loss|robloss)"
    r"\s*(?:=|_|-)?\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
    flags=re.IGNORECASE,
)
_EXACT_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


def _warn(warnings_out: list[str], message: str) -> None:
    warnings_out.append(message)
    print(f"[WARN] {message}")


def _lambda_label(value: float, warnings_out: list[str] | None = None) -> str:
    if not np.isfinite(value):
        return "unknown"
    diffs = np.abs(LAMBDA_VALUES - value)
    idx = int(np.argmin(diffs))
    tol = max(1e-15, abs(float(LAMBDA_VALUES[idx])) * 1e-9)
    if diffs[idx] <= tol:
        return LAMBDA_LABELS[idx]
    label = f"{value:g}"
    if warnings_out is not None:
        _warn(warnings_out, f"Lambda value {label} is outside the fixed thesis order; appending it after known lambdas.")
    return label


def parse_lambda_rob(path_or_name: str | Path) -> float | None:
    """
    Parse robustness loss weights from common path/name variants.

    Supported examples include lambda=1e-4, lambda_1e-4, lambda-1e-4,
    rob_loss_1e-4, robloss1e-4, 0.0001, base, and lambda=0.
    """
    path = Path(path_or_name)
    candidates = [str(path_or_name)]
    candidates.extend(reversed(path.parts))

    for text in candidates:
        lower = text.lower()
        if re.search(r"(?:^|[_\-/\\])base(?:$|[_\-/\\])", lower):
            return 0.0
        match = _PREFIXED_LAMBDA_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    for text in reversed(path.parts):
        cleaned = text.strip().strip("()[]{}")
        if _EXACT_NUMERIC_RE.match(cleaned):
            try:
                return float(cleaned)
            except ValueError:
                continue
    return None


def _canonical_col(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _pick_column(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {_canonical_col(col): col for col in columns}
    for alias in aliases:
        col = lookup.get(_canonical_col(alias))
        if col is not None:
            return col
    return None


def _normalise_long_dataframe(df: pd.DataFrame, source: Path, warnings_out: list[str]) -> pd.DataFrame:
    columns = list(df.columns)
    condition_col = _pick_column(columns, ["Condition", "window_class", "class", "label", "target_label"])
    explainer_col = _pick_column(columns, ["Explainer", "explainer", "explainer_name", "method"])
    explanation_type_col = _pick_column(columns, ["ExplanationType", "explanation_type", "explanation type"])
    metric_col = _pick_column(columns, ["Metric", "metric", "metric_name", "metric name"])
    value_col = _pick_column(columns, ["Value", "value", "metric_value", "score", "result"])
    sample_col = _pick_column(columns, ["SampleIdx", "sample_idx", "sample_index", "WindowIdx", "window_idx", "Index", "idx"])

    missing = []
    if metric_col is None:
        missing.append("Metric")
    if value_col is None:
        missing.append("Value")
    if missing:
        _warn(warnings_out, f"Skipping {source}: missing long-format columns {missing}.")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Condition"] = df[condition_col] if condition_col is not None else "Global"
    if explainer_col is not None:
        out["Explainer"] = df[explainer_col]
    elif explanation_type_col is not None:
        out["Explainer"] = df[explanation_type_col]
    else:
        out["Explainer"] = "Unknown"
    out["Metric"] = df[metric_col]
    out["Value"] = pd.to_numeric(df[value_col], errors="coerce")
    if sample_col is not None:
        out["SampleIdx"] = df[sample_col]

    out = out[np.isfinite(out["Value"])].copy()
    if out.empty:
        _warn(warnings_out, f"Skipping {source}: no finite metric values after normalisation.")
        return out

    out["Condition"] = out["Condition"].map(_normalise_condition_name)
    out = ensure_sample_index(out)
    return out


def _condition_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        normalised = _normalise_condition_name(part)
        if normalised in {"Non-seizure", "Seizure", "Global"}:
            return normalised
    return "Global"


def _rows_from_condition_json(json_path: Path, warnings_out: list[str]) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if any(key in data for key in ("non_seizure", "seizure", "global_rma")):
        return rows_from_split_json(json_path)

    condition = _condition_from_path(json_path.parent)
    metrics = data.get("metrics", data)
    if not isinstance(metrics, dict):
        _warn(warnings_out, f"Skipping {json_path}: JSON metric block is not a dictionary.")
        return []

    rows: list[dict[str, Any]] = []
    for explainer, expl_metrics in metrics.items():
        if explainer in {"meta", "error"} or not isinstance(expl_metrics, dict):
            continue
        for metric, values in expl_metrics.items():
            for sample_idx, value in enumerate(_extract_numeric_values(values)):
                rows.append(
                    {
                        "Condition": condition,
                        "Explainer": str(explainer),
                        "Metric": str(metric),
                        "Value": float(value),
                        "SampleIdx": int(sample_idx),
                    }
                )
    return rows


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_single_result_file(path: Path, warnings_out: list[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _normalise_long_dataframe(pd.read_csv(path), path, warnings_out)
    if suffix == ".json":
        rows = _rows_from_condition_json(path, warnings_out)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    if suffix == ".parquet":
        return _normalise_long_dataframe(pd.read_parquet(path), path, warnings_out)
    if suffix in {".pkl", ".pickle"}:
        obj = _load_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return _normalise_long_dataframe(obj, path, warnings_out)
        _warn(warnings_out, f"Skipping {path}: pickle did not contain a pandas DataFrame.")
        return pd.DataFrame()
    _warn(warnings_out, f"Skipping {path}: unsupported file type.")
    return pd.DataFrame()


def _load_result_dir(metric_dir: Path, warnings_out: list[str]) -> pd.DataFrame:
    flat_csv = metric_dir / FLAT_CSV_NAME
    split_json = metric_dir / SPLIT_JSON_NAME

    if flat_csv.exists():
        df = _normalise_long_dataframe(pd.read_csv(flat_csv), flat_csv, warnings_out)
        if split_json.exists():
            global_rows = [row for row in rows_from_split_json(split_json) if row.get("Condition") == "Global"]
            if global_rows:
                df = pd.concat([df, pd.DataFrame(global_rows)], ignore_index=True)
        return df

    if split_json.exists():
        rows = rows_from_split_json(split_json)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    generic_files = [
        *metric_dir.glob("*.parquet"),
        *metric_dir.glob("*.pkl"),
        *metric_dir.glob("*.pickle"),
        *metric_dir.glob("*.csv"),
        *metric_dir.glob("*.json"),
    ]
    frames = [_load_single_result_file(path, warnings_out) for path in generic_files]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_key(metric_dir: Path) -> Path:
    parts = metric_dir.parts
    for idx, part in enumerate(parts):
        if part.startswith("Results_"):
            return Path(*parts[: idx + 1])
    if metric_dir.name in {"XAI_metrics", "Explainability_metrics"}:
        return metric_dir.parent
    return metric_dir


def _metric_dir_rank(metric_dir: Path) -> tuple[int, str]:
    name = metric_dir.name
    if name == "XAI_metrics":
        rank = 0
    elif name == "Explainability_metrics":
        rank = 1
    else:
        rank = 2
    return rank, str(metric_dir)


def discover_result_dirs(results_root: str | Path, warnings_out: list[str]) -> list[Path]:
    root = Path(results_root)
    preferred_dirs = {
        path.parent for path in root.rglob(FLAT_CSV_NAME)
    } | {
        path.parent for path in root.rglob(SPLIT_JSON_NAME)
    }

    if not preferred_dirs:
        suffixes = {".csv", ".json", ".parquet", ".pkl", ".pickle"}
        generic_files = [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in suffixes
            and ("metric" in p.name.lower() or any(part in {"XAI_metrics", "Explainability_metrics"} for part in p.parts))
        ]
        preferred_dirs = {p.parent for p in generic_files}

    grouped: dict[Path, list[Path]] = {}
    for metric_dir in preferred_dirs:
        grouped.setdefault(_run_key(metric_dir), []).append(metric_dir)

    selected = []
    for dirs in grouped.values():
        selected.append(sorted(dirs, key=_metric_dir_rank)[0])
    def sort_key(path: Path) -> tuple[bool, float, str]:
        lambda_value = parse_lambda_rob(path)
        return lambda_value is None, lambda_value if lambda_value is not None else math.inf, str(path)

    selected = sorted(selected, key=sort_key)

    if not selected:
        _warn(warnings_out, f"No XAI result files found under {root}.")
    elif len(preferred_dirs) != len(selected):
        print(f"[INFO] Deduplicated {len(preferred_dirs)} metric directories to {len(selected)} result runs.")
    return selected


def load_robustness_sweep_xai_results(results_root: str | Path = "./ArchiveModelsRobLossSweep") -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    result_dirs = discover_result_dirs(results_root, warnings_out)
    frames: list[pd.DataFrame] = []

    for metric_dir in result_dirs:
        lambda_value = parse_lambda_rob(metric_dir)
        if lambda_value is None:
            _warn(warnings_out, f"Skipping {metric_dir}: could not infer lambda_rob from path.")
            continue

        df = _load_result_dir(metric_dir, warnings_out)
        if df.empty:
            _warn(warnings_out, f"Skipping {metric_dir}: no usable XAI values found.")
            continue

        label = _lambda_label(lambda_value, warnings_out)
        df = df.copy()
        df["lambda_rob_value"] = float(lambda_value)
        df["lambda_rob"] = label
        df["lambda_order"] = LAMBDA_ORDER.get(label, len(LAMBDA_ORDER))
        df["source_dir"] = str(metric_dir)
        df["Condition"] = df["Condition"].map(_normalise_condition_name)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df[np.isfinite(df["Value"])].copy()
        df["Metric"] = df["Metric"].astype(str)
        df["Explainer"] = df["Explainer"].astype(str)
        df["ExplanationType"] = df["Explainer"].map(infer_explanation_type)
        df["ExplainerDisplay"] = df["Explainer"].map(display_explainer)
        frames.append(df)
        print(f"[OK] Loaded {len(df):>6} XAI values for lambda={label:<5} from {metric_dir}")

    if not frames:
        return pd.DataFrame(), warnings_out

    values_df = pd.concat(frames, ignore_index=True)
    values_df["ExplanationType"] = values_df["ExplanationType"].fillna("Unknown")
    values_df["Condition"] = values_df["Condition"].fillna("Unknown")
    return values_df, warnings_out


def _find_column_case_insensitive(df: pd.DataFrame, requested: str) -> str | None:
    requested_lower = requested.lower()
    for col in df.columns:
        if str(col).lower() == requested_lower:
            return str(col)
    return None


def discover_performance_files(results_root: str | Path, warnings_out: list[str]) -> list[Path]:
    root = Path(results_root)
    files = sorted({p for p in root.rglob("per_fold_metrics_recompute_F2.csv") if p.is_file()})

    if not files:
        files = sorted(
            {p for p in root.rglob("per_fold_metrics*.csv") if p.is_file()}
            | {p for p in root.rglob("*_per_fold.csv") if p.is_file()}
        )

    if not files:
        fallback_root = Path("./Results_sweep_summary_RobLoss/per_run")
        if fallback_root.exists():
            files = sorted({p for p in fallback_root.rglob("*_per_fold.csv") if p.is_file()})

    if not files:
        _warn(warnings_out, f"No per-fold classification performance files found under {root}.")

    return files


def load_robustness_sweep_performance_results(
    results_root: str | Path = "./ArchiveModelsRobLossSweep",
    metric: str = PERFORMANCE_METRIC,
) -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    frames: list[pd.DataFrame] = []

    for csv_path in discover_performance_files(results_root, warnings_out):
        lambda_value = parse_lambda_rob(csv_path)
        if lambda_value is None:
            _warn(warnings_out, f"Skipping {csv_path}: could not infer lambda_rob from path.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            _warn(warnings_out, f"Skipping {csv_path}: could not read CSV ({exc}).")
            continue

        metric_col = _find_column_case_insensitive(df, metric)
        if metric_col is None:
            _warn(warnings_out, f"Skipping {csv_path}: missing column '{metric}'.")
            continue

        fold_col = _find_column_case_insensitive(df, "fold")
        value = pd.to_numeric(df[metric_col], errors="coerce")
        sub = pd.DataFrame({"value": value})
        sub = sub[np.isfinite(sub["value"])].copy()
        if sub.empty:
            _warn(warnings_out, f"Skipping {csv_path}: no finite {metric} values.")
            continue

        if fold_col is not None:
            sub["fold"] = df.loc[sub.index, fold_col].to_numpy()
        else:
            sub["fold"] = np.arange(len(sub))

        label = _lambda_label(lambda_value, warnings_out)
        sub["lambda_rob_value"] = float(lambda_value)
        sub["lambda_rob"] = label
        sub["lambda_order"] = LAMBDA_ORDER.get(label, len(LAMBDA_ORDER))
        sub["metric"] = metric.upper()
        sub["source_file"] = str(csv_path)
        frames.append(sub)
        print(f"[OK] Loaded {len(sub):>3} {metric.upper()} fold values for lambda={label:<5} from {csv_path}")

    if not frames:
        return pd.DataFrame(), warnings_out

    return pd.concat(frames, ignore_index=True), warnings_out


def _std(values: pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan


def _q(values: pd.Series, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _metric_stats(values_df: pd.DataFrame, specs: list[MetricSpec], warnings_out: list[str]) -> pd.DataFrame:
    summary_frames: list[pd.DataFrame] = []

    for spec in specs:
        sub = values_df[values_df["Metric"].str.contains(re.escape(spec.contains), case=False, na=False)].copy()
        if sub.empty:
            _warn(warnings_out, f"Missing metric matching '{spec.contains}'.")
            continue

        if spec.global_metric and (sub["Condition"] == "Global").any():
            sub = sub[sub["Condition"] == "Global"].copy()

        sub["plot_value"] = sub["Value"].abs() if spec.use_abs else sub["Value"]
        sub["metric_key"] = spec.key
        sub["source_metric_name"] = sub["Metric"]
        if spec.use_abs:
            sub["metric_name"] = "Correctness_ParamRandomisation_abs_corr_spearman"
        else:
            sub["metric_name"] = sub["Metric"]
        sub["window_class"] = sub["Condition"]
        sub["label"] = sub["window_class"]
        sub["explanation_type"] = sub["ExplanationType"]

        group_cols = [
            "metric_key",
            "metric_name",
            "source_metric_name",
            "lambda_rob",
            "lambda_rob_value",
            "lambda_order",
            "explanation_type",
            "window_class",
            "label",
        ]
        stats = (
            sub.groupby(group_cols, dropna=False)["plot_value"]
            .agg(
                n="count",
                median="median",
                q1=lambda s: _q(s, 0.25),
                q3=lambda s: _q(s, 0.75),
                mean="mean",
                std=_std,
            )
            .reset_index()
        )
        stats["metric_title"] = spec.title
        stats["ylabel"] = spec.ylabel
        stats["higher_is_better"] = spec.higher_is_better
        summary_frames.append(stats)

    if not summary_frames:
        return pd.DataFrame()

    out = pd.concat(summary_frames, ignore_index=True)
    metric_order = {spec.key: idx for idx, spec in enumerate(specs)}
    out["_metric_order"] = out["metric_key"].map(metric_order).fillna(len(metric_order)).astype(int)
    out["_explanation_order"] = out["explanation_type"].map({k: i for i, k in enumerate(EXPLANATION_TYPE_ORDER)}).fillna(99)
    out["_condition_order"] = out["window_class"].map({k: i for i, k in enumerate(CONDITION_ORDER)}).fillna(99)
    out = out.sort_values(["_metric_order", "lambda_order", "_explanation_order", "_condition_order"]).drop(
        columns=["_metric_order", "_explanation_order", "_condition_order"]
    )
    return out


def _performance_stats(performance_df: pd.DataFrame, metric: str, warnings_out: list[str]) -> pd.DataFrame:
    if performance_df.empty:
        _warn(warnings_out, f"No {metric.upper()} performance values were loaded.")
        return pd.DataFrame()

    group_cols = ["metric", "lambda_rob", "lambda_rob_value", "lambda_order"]
    stats = (
        performance_df.groupby(group_cols, dropna=False)["value"]
        .agg(
            n="count",
            mean="mean",
            std=_std,
            median="median",
            q1=lambda s: _q(s, 0.25),
            q3=lambda s: _q(s, 0.75),
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values("lambda_order")
    )
    return stats


def _available_lambda_labels(stats_df: pd.DataFrame) -> list[str]:
    if stats_df.empty or "lambda_rob" not in stats_df.columns:
        return LAMBDA_LABELS

    extras = [
        label
        for label in stats_df["lambda_rob"].dropna().astype(str).unique()
        if label not in LAMBDA_ORDER
    ]
    extras = sorted(extras, key=lambda label: float(label) if _EXACT_NUMERIC_RE.match(label) else math.inf)
    return LAMBDA_LABELS + extras


def _available_lambda_labels_from_frames(*frames: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for frame in frames:
        if frame.empty or "lambda_rob" not in frame.columns:
            continue
        labels.extend(frame["lambda_rob"].dropna().astype(str).tolist())

    extras = sorted(
        {label for label in labels if label not in LAMBDA_ORDER},
        key=lambda label: float(label) if _EXACT_NUMERIC_RE.match(label) else math.inf,
    )
    return LAMBDA_LABELS + extras


def _ordered_series_data(series_df: pd.DataFrame, lambda_labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in lambda_labels:
        row = series_df[series_df["lambda_rob"] == label]
        if row.empty:
            rows.append({"lambda_rob": label, "median": np.nan, "q1": np.nan, "q3": np.nan, "n": 0})
        else:
            rows.append(row.iloc[0].to_dict())
    return pd.DataFrame(rows)


def _ordered_performance_data(performance_stats_df: pd.DataFrame, lambda_labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in lambda_labels:
        row = performance_stats_df[performance_stats_df["lambda_rob"] == label]
        if row.empty:
            rows.append({"lambda_rob": label, "mean": np.nan, "std": np.nan, "n": 0})
        else:
            rows.append(row.iloc[0].to_dict())
    return pd.DataFrame(rows)


def _lambda_axis_values(lambda_labels: list[str]) -> np.ndarray:
    values: list[float] = []
    for label in lambda_labels:
        try:
            values.append(float(label))
            continue
        except ValueError:
            parsed = parse_lambda_rob(label)
            values.append(float(parsed) if parsed is not None else np.nan)
    return np.asarray(values, dtype=float)


def _add_common_axis_format(ax: plt.Axes, lambda_labels: list[str], annotate_high_reg: bool = True) -> None:
    x = _lambda_axis_values(lambda_labels)
    finite_mask = np.isfinite(x)
    positive = x[finite_mask & (x > 0)]

    if len(positive) > 0:
        linthresh = float(np.min(positive) / 3.0)
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.45, base=10)
        ax.set_xlim(-linthresh, float(np.max(positive) * 1.18))
    elif finite_mask.any():
        ax.set_xlim(float(np.min(x[finite_mask]) - 0.5), float(np.max(x[finite_mask]) + 0.5))

    tick_values = x[finite_mask]
    tick_labels = [label for label, keep in zip(lambda_labels, finite_mask) if keep]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels, rotation=65, ha="right")
    ax.set_xlabel(r"Robustness loss weight $\lambda_{\mathrm{rob}}$")
    ax.tick_params(axis="x", pad=1.2)
    ax.grid(alpha=0.25, axis="y", linewidth=0.4)
    ax.grid(alpha=0.10, axis="x", linewidth=0.35)

    if HIGH_REG_START_LABEL in lambda_labels:
        start_idx = lambda_labels.index(HIGH_REG_START_LABEL)
        start_value = x[start_idx]
        if np.isfinite(start_value):
            ax.axvspan(start_value, ax.get_xlim()[1], color=HIGH_REG_SHADE, alpha=0.75, zorder=0)
            if annotate_high_reg and not PANEL_READY_EXPORT:
                ax.text(
                    start_value,
                    0.98,
                    "Degraded performance",
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="top",
                    fontsize=plt.rcParams["legend.fontsize"],
                    color="0.35",
                )


def _apply_metric_yaxis(ax: plt.Axes, spec: MetricSpec) -> None:
    if spec.yaxis is None:
        return

    ymin, ymax = spec.yaxis
    ax.set_ylim(bottom=ymin, top=ymax)


def _series_style(row: pd.Series, global_metric: bool) -> tuple[str, str, str]:
    condition = str(row.get("window_class", "Unknown"))
    explanation_type = str(row.get("explanation_type", "Unknown"))
    if global_metric:
        color = {"Node": NODE_COLOR, "Edge": EDGE_COLOR, "Unknown": UNKNOWN_COLOR}.get(
            explanation_type,
            UNKNOWN_COLOR,
        )
        linestyle = EXPLANATION_LINESTYLES.get(explanation_type, "-")
        label = "_nolegend_" if explanation_type == "Node" else explanation_type
    else:
        color = CONDITION_COLORS.get(condition, "#666666")
        linestyle = EXPLANATION_LINESTYLES.get(explanation_type, "-")
        label = condition
    return color, linestyle, label


def _compact_legend(ax: plt.Axes, outside: bool = False) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen or label.startswith("_"):
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if not unique_handles:
        return
    if outside:
        ax.legend(
            unique_handles,
            unique_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            title="Series",
            handlelength=1.4,
            borderpad=0.2,
            labelspacing=0.25,
            handletextpad=0.4,
        )
    else:
        ax.legend(
            unique_handles,
            unique_labels,
            loc="upper left",
            frameon=False,
            handlelength=1.4,
            borderpad=0.2,
            labelspacing=0.25,
            handletextpad=0.4,
        )


def _plot_metric_on_axis(
    ax: plt.Axes,
    stats_df: pd.DataFrame,
    spec: MetricSpec,
    lambda_labels: list[str],
    show_title: bool = True,
    outside_legend: bool = True,
    annotate_high_reg: bool = True,
) -> bool:
    metric_stats = stats_df[stats_df["metric_key"] == spec.key].copy()
    if metric_stats.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    if spec.global_metric:
        series_cols = ["explanation_type"]
    else:
        series_cols = ["window_class", "explanation_type"]

    x = _lambda_axis_values(lambda_labels)
    plotted = False
    for _, series_key_df in metric_stats.groupby(series_cols, dropna=False, sort=False):
        series_key_df = series_key_df.sort_values("lambda_order")
        ordered = _ordered_series_data(series_key_df, lambda_labels)
        y = ordered["median"].to_numpy(dtype=float)
        q1 = ordered["q1"].to_numpy(dtype=float)
        q3 = ordered["q3"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if not mask.any():
            continue

        style_row = series_key_df.iloc[0]
        color, linestyle, label = _series_style(style_row, spec.global_metric)
        ax.plot(
            x[mask],
            y[mask],
            marker="o",
            markersize=2.4,
            linewidth=1.1,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        band_mask = np.isfinite(q1) & np.isfinite(q3) & np.isfinite(x)
        if band_mask.any():
            ax.fill_between(x[band_mask], q1[band_mask], q3[band_mask], color=color, alpha=0.15, linewidth=0)
        plotted = True

    _add_common_axis_format(ax, lambda_labels, annotate_high_reg=annotate_high_reg)
    _apply_metric_yaxis(ax, spec)
    ax.set_ylabel(spec.ylabel)
    if show_title:
        ax.set_title(spec.title)
    _compact_legend(ax, outside=outside_legend)
    return plotted


def _plot_performance_on_axis(
    ax: plt.Axes,
    performance_stats_df: pd.DataFrame,
    lambda_labels: list[str],
    show_title: bool = True,
    annotate_high_reg: bool = True,
) -> bool:
    if performance_stats_df.empty:
        ax.text(0.5, 0.5, "No performance data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    ordered = _ordered_performance_data(performance_stats_df, lambda_labels)
    x = _lambda_axis_values(lambda_labels)
    y = ordered["mean"].to_numpy(dtype=float)
    std = ordered["std"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if not mask.any():
        ax.text(0.5, 0.5, "No performance data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    ax.plot(
        x[mask],
        y[mask],
        marker="o",
        markersize=2.5,
        linewidth=1.15,
        color=GLOBAL_COLOR,
        label=PERFORMANCE_METRIC.upper(),
    )
    band_mask = np.isfinite(y) & np.isfinite(std) & np.isfinite(x)
    if band_mask.any():
        lower = np.clip(y[band_mask] - std[band_mask], 0.0, 1.0)
        upper = np.clip(y[band_mask] + std[band_mask], 0.0, 1.0)
        ax.fill_between(x[band_mask], lower, upper, color=GLOBAL_COLOR, alpha=0.15, linewidth=0)

    _add_common_axis_format(ax, lambda_labels, annotate_high_reg=annotate_high_reg)
    ax.set_ylabel(PERFORMANCE_METRIC.upper())
    ax.set_ylim(0.0, 1.0)
    if show_title:
        ax.set_title("Classification performance")
    return True


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, generated_files: list[Path]) -> None:
    for suffix, kwargs in [
        (".png", {"dpi": 300}),
        (".pdf", {}),
    ]:
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        generated_files.append(path)
        print(f"[OK] Saved {path}")


def _plot_individual_metrics(stats_df: pd.DataFrame, out_dir: Path, lambda_labels: list[str]) -> list[Path]:
    generated_files: list[Path] = []
    for spec in METRIC_SPECS:
        fig, ax = plt.subplots(figsize=RLS_PANEL_FIGSIZE)
        plotted = _plot_metric_on_axis(
            ax=ax,
            stats_df=stats_df,
            spec=spec,
            lambda_labels=lambda_labels,
            show_title=True,
            outside_legend=False,
            annotate_high_reg=False,
        )
        if plotted:
            fig.tight_layout()
            _save_figure(fig, out_dir, spec.filename_stem, generated_files)
        else:
            print(f"[SKIP] No data available for individual figure: {spec.title}")
        plt.close(fig)
    return generated_files


def _plot_individual_performance(
    performance_stats_df: pd.DataFrame,
    out_dir: Path,
    lambda_labels: list[str],
) -> list[Path]:
    generated_files: list[Path] = []
    fig, ax = plt.subplots(figsize=RLS_PANEL_FIGSIZE)
    plotted = _plot_performance_on_axis(
        ax=ax,
        performance_stats_df=performance_stats_df,
        lambda_labels=lambda_labels,
        show_title=True,
        annotate_high_reg=False,
    )
    if plotted:
        fig.tight_layout()
        _save_figure(fig, out_dir, "robustness_sweep_classification_auprc", generated_files)
    else:
        print("[SKIP] No data available for classification performance figure.")
    plt.close(fig)
    return generated_files


def _plot_summary(
    stats_df: pd.DataFrame,
    performance_stats_df: pd.DataFrame,
    out_dir: Path,
    lambda_labels: list[str],
) -> list[Path]:
    generated_files: list[Path] = []
    fig, axes = plt.subplots(3, 2, figsize=RLS_SUMMARY_FIGSIZE, squeeze=False)
    axes_flat = axes.ravel()

    axis_idx = 0
    if not performance_stats_df.empty:
        _plot_performance_on_axis(
            ax=axes_flat[axis_idx],
            performance_stats_df=performance_stats_df,
            lambda_labels=lambda_labels,
            show_title=True,
            annotate_high_reg=True,
        )
        axis_idx += 1

    for spec in METRIC_SPECS:
        if axis_idx >= len(axes_flat):
            break
        _plot_metric_on_axis(
            ax=axes_flat[axis_idx],
            stats_df=stats_df,
            spec=spec,
            lambda_labels=lambda_labels,
            show_title=True,
            outside_legend=False,
            annotate_high_reg=(axis_idx == 0),
        )
        axis_idx += 1

    for ax in axes_flat[axis_idx:]:
        ax.axis("off")

    if not PANEL_READY_EXPORT:
        fig.suptitle("Robustness loss sweep", fontsize=plt.rcParams["figure.titlesize"], fontweight="normal")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()
    _save_figure(fig, out_dir, "robustness_sweep_xai_summary", generated_files)
    plt.close(fig)
    return generated_files


def plot_robustness_sweep_summary(
    results_root: str | Path = "./ArchiveModelsRobLossSweep",
    out_dir: str | Path = "./Report/figures/XAI_metrics/robustness_sweep",
) -> tuple[pd.DataFrame, list[Path], list[str]]:
    """
    Create compact thesis-ready XAI robustness sweep summary plots.

    Returns the summary statistics table, generated file paths, and warnings.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    values_df, warnings_out = load_robustness_sweep_xai_results(results_root=results_root)
    performance_df, performance_warnings = load_robustness_sweep_performance_results(
        results_root=results_root,
        metric=PERFORMANCE_METRIC,
    )
    warnings_out.extend(performance_warnings)

    stats_df = _metric_stats(values_df, METRIC_SPECS, warnings_out) if not values_df.empty else pd.DataFrame()
    performance_stats_df = (
        _performance_stats(performance_df, PERFORMANCE_METRIC, warnings_out)
        if not performance_df.empty
        else pd.DataFrame()
    )

    if stats_df.empty and performance_stats_df.empty:
        _warn(warnings_out, "No XAI or classification performance values were loaded; no figures were created.")
        return pd.DataFrame(), [], warnings_out

    lambda_labels = _available_lambda_labels_from_frames(stats_df, performance_stats_df)
    csv_cols = [
        "lambda_rob",
        "lambda_rob_value",
        "metric_name",
        "source_metric_name",
        "metric_key",
        "explanation_type",
        "window_class",
        "label",
        "n",
        "median",
        "q1",
        "q3",
        "mean",
        "std",
        "higher_is_better",
    ]
    generated_files: list[Path] = []
    return_df = pd.DataFrame()

    if not stats_df.empty:
        csv_path = out_path / "robustness_sweep_xai_summary_stats.csv"
        stats_df[csv_cols].to_csv(csv_path, index=False)
        generated_files.append(csv_path)
        return_df = stats_df[csv_cols]
        print(f"[OK] Saved {csv_path}")

    if not performance_stats_df.empty:
        performance_csv_cols = [
            "metric",
            "lambda_rob",
            "lambda_rob_value",
            "lambda_order",
            "n",
            "mean",
            "std",
            "median",
            "q1",
            "q3",
            "min",
            "max",
        ]
        performance_csv_path = out_path / "robustness_sweep_classification_auprc_stats.csv"
        performance_stats_df[performance_csv_cols].to_csv(performance_csv_path, index=False)
        generated_files.append(performance_csv_path)
        if return_df.empty:
            return_df = performance_stats_df[performance_csv_cols]
        print(f"[OK] Saved {performance_csv_path}")

    generated_files.extend(_plot_summary(stats_df, performance_stats_df, out_path, lambda_labels))
    if not performance_stats_df.empty:
        generated_files.extend(_plot_individual_performance(performance_stats_df, out_path, lambda_labels))
    if not stats_df.empty:
        generated_files.extend(_plot_individual_metrics(stats_df, out_path, lambda_labels))

    print("\nCaption:")
    print(CAPTION)
    return return_df, generated_files, warnings_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot robustness-loss sweep XAI metric summaries.")
    parser.add_argument("--results-root", default="./ArchiveModelsRobLossSweep", help="Root directory with sweep results.")
    parser.add_argument(
        "--out-dir",
        default="./Results_statistics/XAI/RLS/LinePlots",
        help="Directory where figures and summary CSV should be written.",
    )
    args = parser.parse_args()

    stats_df, generated_files, warnings_out = plot_robustness_sweep_summary(
        results_root=args.results_root,
        out_dir=args.out_dir,
    )

    print("\nGenerated files:")
    for path in generated_files:
        print(path)

    if warnings_out:
        print("\nWarnings:")
        for message in warnings_out:
            print(f"- {message}")
    else:
        print("\nWarnings: none")

    if not stats_df.empty:
        print("\nSummary stats preview:")
        print(stats_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
