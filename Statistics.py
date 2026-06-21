from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os
# =========================
# EDIT THESE SETTINGS FOR CORRECT DIRECTORIES (GLOBAL)
# Set to be analysed models in main()
# =========================
BASE_DIR = Path("./Results_Performance/")

# Indicate which column has the fold numbers.
FOLD_COL = "fold"

CSV_NAME = "per_fold_metrics_recompute_F2.csv"
# =========================


def load_csv(log_id: str) -> pd.DataFrame:
    csv_path = BASE_DIR / f"Results_{log_id}" / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(df.dtypes)
    print(f"Loaded: {csv_path}")
    return df


def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, fold_col: str | None):
    if fold_col is None:
        if len(df1) != len(df2):
            raise ValueError(
                f"Row counts differ ({len(df1)} vs {len(df2)}). "
                f"Set FOLD_COL to align by a shared fold identifier."
            )
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        return df1, df2

    if fold_col not in df1.columns or fold_col not in df2.columns:
        raise ValueError(f"FOLD_COL='{fold_col}' not found in both CSVs.")

    common_cols = sorted(set(df1.columns).intersection(df2.columns))
    merged = pd.merge(
        df1[common_cols],
        df2[common_cols],
        on=fold_col,
        suffixes=("_1", "_2"),
        how="inner",
    ).sort_values(fold_col)

    if len(merged) == 0:
        raise ValueError(f"No overlapping rows found using FOLD_COL='{fold_col}'.")

    df1_aligned = pd.DataFrame()
    df2_aligned = pd.DataFrame()

    for col in common_cols:
        if col == fold_col:
            df1_aligned[col] = merged[col]
            df2_aligned[col] = merged[col]
        else:
            df1_aligned[col] = merged[f"{col}_1"]
            df2_aligned[col] = merged[f"{col}_2"]

    return df1_aligned.reset_index(drop=True), df2_aligned.reset_index(drop=True)


def compare_metric(x: np.ndarray, y: np.ndarray):
    mask = ~(pd.isna(x) | pd.isna(y))
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)

    n_total = len(x)
    diffs = y - x
    n_nonzero = int(np.sum(diffs != 0))

    if n_total == 0:
        return {
            "n_pairs": 0,
            "n_nonzero": 0,
            "mean_1": np.nan,
            "mean_2": np.nan,
            "mean_diff": np.nan,
            "statistic": np.nan,
            "pvalue": np.nan,
            "method_used": "no_data",
            "note": "No valid paired values",
        }

    if n_nonzero == 0:
        return {
            "n_pairs": n_total,
            "n_nonzero": 0,
            "mean_1": float(np.mean(x)),
            "mean_2": float(np.mean(y)),
            "mean_diff": float(np.mean(diffs)),
            "statistic": 0.0,
            "pvalue": 1.0,
            "method_used": "degenerate",
            "note": "All paired differences are zero",
        }

    try:
        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
        method_used = "exact"
        note = ""
    except Exception as e:
        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
        method_used = "auto"
        note = f"Exact unavailable: {e}"

    return {
        "n_pairs": n_total,
        "n_nonzero": n_nonzero,
        "mean_1": float(np.mean(x)),
        "mean_2": float(np.mean(y)),
        "mean_diff": float(np.mean(diffs)),
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "method_used": method_used,
        "note": note,
    }

def plot_delta_pvalue_pivot_tables(
    all_results_df: pd.DataFrame,
    output_dir: Path,
    metrics_to_plot: list[str] | None = None,
    model_order: list[str]| None=None,
    delta_col: str = "mean_diff",
    pvalue_col: str = "pvalue",
    reference_col: str = "reference_model",
    comparator_col: str = "comparator_model",
    metric_col: str = "metric",
    p_threshold: float = 0.05,
    delta_fmt: str = "{:+.3f}",
    p_fmt: str = "{:.3g}",
) -> None:
    """
    Create pivot tables with the differneces and p values

    Saves, per metric:
        - pivot_delta_<metric>.csv
        - pivot_pvalue_<metric>.csv
        - pivot_delta_pvalue_<metric>.csv
        - pivot_delta_pvalue_<metric>.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_to_plot is None:
        metrics_to_plot = sorted(all_results_df[metric_col].dropna().unique())

    for metric in metrics_to_plot:
        metric_df = all_results_df[all_results_df[metric_col] == metric].copy()

        if metric_df.empty:
            print(f"[SKIP] No data for metric: {metric}")
            continue

        delta_pivot = metric_df.pivot(
            index=reference_col,
            columns=comparator_col,
            values=delta_col,
        )
        # .sort_index(axis=0).sort_index(axis=1)

        p_pivot = metric_df.pivot(
            index=reference_col,
            columns=comparator_col,
            values=pvalue_col,
        )
        # .reindex(index=delta_pivot.index, columns=delta_pivot.columns)

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
                    combined.loc[ref, comp] = f"{delta_fmt.format(delta)} ({p_fmt.format(pval)})"

        # Save CSVs
        safe_metric = str(metric).replace("/", "_").replace(" ", "_")
        delta_pivot.to_csv(output_dir / f"pivot_delta_{safe_metric}.csv")
        p_pivot.to_csv(output_dir / f"pivot_pvalue_{safe_metric}.csv")
        combined.to_csv(output_dir / f"pivot_delta_pvalue_{safe_metric}.csv")

        # Plot white-background table
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

        ax.set_xticklabels(combined.columns, rotation=45, ha="right")
        ax.set_yticklabels(combined.index)

        ax.set_xlabel("Comparator model")
        ax.set_ylabel("Reference model")
        ax.set_title(f"{metric}: Δ mean difference with Wilcoxon p-values")

        # Table-like grid
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", linewidth=0.6, alpha=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Remove plot spines for cleaner table look
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
                    fontsize=10,
                    fontweight="bold" if is_sig else "normal",
                    color="black",
                )

        plt.tight_layout()

        save_path = output_dir / f"pivot_delta_pvalue_{safe_metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"[OK] Saved pivot table for {metric}: {save_path}")


def main():
    #dictionary of to be calculated classification performance differneces; dict{"save_name": "model_log_path"}
    # LOG_base = {
    #     "STGAT": "491483_MTbase_LR2e-2_WD1e-3"
    # }

    # LOG_compare = {
    #     "SENN_IC": "492093_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-3",
    #     "SENN_FC_thetax": "486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0",
    #     "SENN_FC_thetah": "486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0",
    #     "LogisticConcepts": "486189_MTLogisticConcepts_LR2e-2_WD1e-4",
    # }
    LOG_base = {
        "STGAT": "491483_MTbase_LR2e-2_WD1e-3",
        "SENN IC": "492092_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-4",
        "SENN FC theta(x)": "486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0",
        "SENN FC theta(h)": "486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0",
        "Logistic Regression": "486189_MTLogisticConcepts_LR2e-2_WD1e-4",
        }
    LOG_compare = {
        "STGAT": "491483_MTbase_LR2e-2_WD1e-3",
        "SENN IC": "492092_MTSENNrawx_LR2e-3_WD1e-3_robloss3e-4",
        "SENN FC theta(x)": "486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0",
        "SENN FC theta(h)": "486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0",
        "Logistic Regression": "486189_MTLogisticConcepts_LR2e-2_WD1e-4",
        }

    all_pairwise_results = []

    # LOG_compare = ["492093_MTSENNrawx_LR2e-3_WD1e-3_robloss1e-3", "486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0" , "486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0" ,"486189_MTLogisticConcepts_LR2e-2_WD1e-4"] # SENN-IC
    # LOG2 = "486167_MTSENNfixed_LR2e-3_WD1e-5_robloss0.0" # SENN-FC theta x
    # LOG2 = "486176_MTSENNfixed_concepttheta_LR2e-3_WD1e-5_robloss0.0" # SENN-fc theta h
    # LOG2 = "486189_MTLogisticConcepts_LR2e-2_WD1e-4" # Logistic regression

    #Reverse for nice order in tables
    model_order = [
    "STGAT",
    "SENN IC",
    "SENN FC theta(x)",
    "SENN FC theta(h)",
    "Logistic Regression",
    ]
    for name1 in LOG_base:
        for name2 in LOG_compare:
            
            LOG1,LOG2 = LOG_base[name1],LOG_compare[name2]

            df1 = load_csv(LOG1)
            df2 = load_csv(LOG2)

            df1, df2 = align_dataframes(df1, df2, FOLD_COL)

            shared_cols = [c for c in df1.columns if c in df2.columns]

            numeric_metrics = []
            for col in shared_cols:
                if col == FOLD_COL:
                    continue
                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    numeric_metrics.append(col)

            if not numeric_metrics:
                raise ValueError("No shared numeric metric columns found.")

            results = []
            for metric in numeric_metrics:
                out = compare_metric(df1[metric].to_numpy(), df2[metric].to_numpy())
                out["metric"] = metric
                results.append(out)

            results_df = pd.DataFrame(results)[
                [
                    "metric",
                    "n_pairs",
                    "n_nonzero",
                    "mean_1",
                    "mean_2",
                    "mean_diff",
                    "statistic",
                    "pvalue",
                    "method_used",
                    "note",
                ]
            ].sort_values("pvalue", na_position="last")

            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            pd.set_option("display.float_format", lambda x: f"{x:.6g}")

            print("\n==============================")
            print(f"Comparing Results_{LOG1} vs Results_{LOG2}")
            print(f"Base dir   : {BASE_DIR}")
            print(f"CSV        : {CSV_NAME}")
            print(f"Fold column: {FOLD_COL}")
            print("==============================\n")

            print(results_df.to_string(index=False))

            out_path ="./Results_statistics"
            os.makedirs(out_path,exist_ok=True)
            out_pathcsv ="./Results_statistics/CSVfiles"
            os.makedirs(out_pathcsv,exist_ok=True)
            results_df.to_csv(os.path.join(out_pathcsv,f"wilcoxon_all_metrics_{name1}_vs_{name2}.csv"), index=False)
            print(f"\nSaved results to: {out_path}")


            #### Pivot table ####
            results_df["reference_model"] = name1
            results_df["comparator_model"] = name2
            results_df["reference_log"] = LOG1
            results_df["comparator_log"] = LOG2

            all_pairwise_results.append(results_df)

            # end loop log 2
        #end loop log 1

    all_results_df = pd.concat(all_pairwise_results, ignore_index=True)
    all_results_path = os.path.join(out_path, "wilcoxon_all_pairwise_results.csv")
    all_results_df.to_csv(all_results_path, index=False)

    plot_delta_pvalue_pivot_tables(
        all_results_df=all_results_df,
        output_dir=os.path.join(out_path, "PivotTables"),
        metrics_to_plot=[
            "AUROC",
            "AUPRC",
            "AUPRG",
            "F2",
            "kappa",
            "precision",
            "recall",
            "precision_gain",
            "recall_gain",
        ],
        model_order=model_order,
        p_threshold=0.05,
    )

    print(f"\nSaved combined pairwise results to: {all_results_path}")


if __name__ == "__main__":
    main()