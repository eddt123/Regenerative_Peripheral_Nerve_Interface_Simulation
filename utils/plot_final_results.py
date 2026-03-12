#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process benchmark run CSV logs and produce:
1) final_selectivity_vs_dimension.png  (mean ± 95% CI across runs)
2) final_selectivity_boxplot_by_dimension.png (distribution per N, per optimiser)
3) final_runs_extracted.csv            (one row per run file)
4) final_summary_by_optimizer_and_N.csv

Usage:
  python process_benchmark_results.py --data_dir path/to/data/benchmark_paper_activation_function

Optional:
  python process_benchmark_results.py --data_dir ... --out_dir ... --by_target
"""

import os
import glob
import math
import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_CSV_NAME = "optimizer_summary.csv"  # if present, we will prefer it


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _tuple_target(row, decimals=6):
    """Round target to avoid float uniqueness issues."""
    tx = round(_safe_float(row.get("target_x", np.nan)), decimals)
    ty = round(_safe_float(row.get("target_y", np.nan)), decimals)
    tz = round(_safe_float(row.get("target_z", np.nan)), decimals)
    return (tx, ty, tz)


def _infer_best_from_run_df(df: pd.DataFrame) -> float:
    """
    Prefer best_so_far; fallback to step_best_selectivity.
    """
    if "best_so_far" in df.columns:
        s = pd.to_numeric(df["best_so_far"], errors="coerce")
        if s.notna().any():
            return float(s.max())
    if "step_best_selectivity" in df.columns:
        s = pd.to_numeric(df["step_best_selectivity"], errors="coerce")
        if s.notna().any():
            return float(s.max())
    return float("nan")


def _read_single_run_csv(path: str) -> dict | None:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
        return None

    if df.empty:
        return None

    # Extract meta (prefer first row; these are constant per file in your logger) :contentReference[oaicite:1]{index=1}
    first = df.iloc[0].to_dict()

    optimizer = str(first.get("optimizer", "UNKNOWN"))
    N = first.get("N", None)
    if pd.isna(N) or N is None:
        # fallback to n_rows*n_per_row
        n_rows = first.get("n_rows", None)
        n_per_row = first.get("n_per_row", None)
        if (n_rows is not None) and (n_per_row is not None):
            try:
                N = int(n_rows) * int(n_per_row)
            except Exception:
                N = np.nan

    best = _infer_best_from_run_df(df)

    # Keep useful identifiers if present
    repeat = first.get("repeat", np.nan)
    eval_seed = first.get("eval_seed", np.nan)
    algo_seed = first.get("algo_seed", np.nan)
    grid = None
    if ("n_rows" in first) and ("n_per_row" in first):
        try:
            grid = f"{int(first['n_rows'])}x{int(first['n_per_row'])}"
        except Exception:
            grid = None

    target = _tuple_target(first)

    return {
        "run_file": os.path.basename(path),
        "run_path": os.path.abspath(path),
        "optimizer": optimizer,
        "N": int(N) if (N is not None and not pd.isna(N)) else np.nan,
        "grid": grid,
        "repeat": repeat,
        "eval_seed": eval_seed,
        "algo_seed": algo_seed,
        "target_x": target[0],
        "target_y": target[1],
        "target_z": target[2],
        "best": best,
    }


def load_runs_from_directory(data_dir: str) -> pd.DataFrame:
    """
    Strategy:
      1) If optimizer_summary.csv exists, load it (fast, already per-run).
      2) Otherwise, parse individual run CSV logs (*.csv excluding known summary/stat files).
    """
    data_dir = os.path.abspath(data_dir)
    summary_path = os.path.join(data_dir, SUMMARY_CSV_NAME)

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        # Standardize minimal columns
        needed = {"optimizer", "N", "best"}
        if not needed.issubset(df.columns):
            raise ValueError(
                f"Found {SUMMARY_CSV_NAME} but it doesn't contain required columns {needed}. "
                f"Columns present: {list(df.columns)}"
            )
        # Add target columns if missing
        for c in ["target_x", "target_y", "target_z", "repeat"]:
            if c not in df.columns:
                df[c] = np.nan
        df["run_file"] = df.get("tag", pd.Series(["(from_optimizer_summary)"] * len(df)))
        return df[["run_file", "optimizer", "N", "best", "repeat", "target_x", "target_y", "target_z"]].copy()

    # Else crawl per-run CSV logs
    all_csv = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # Exclude typical analysis outputs (but keep everything else)
    exclude = {
        "statistics_summary.csv",
        "pairwise_tests.csv",
        "landscape_characterization.csv",
        "final_runs_extracted.csv",
        "final_summary_by_optimizer_and_N.csv",
    }
    run_csvs = [p for p in all_csv if os.path.basename(p) not in exclude]

    rows = []
    for p in run_csvs:
        row = _read_single_run_csv(p)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No usable run CSVs found in {data_dir}. "
            f"Expected per-run logs like <TAG>.csv containing 'best_so_far' etc."
        )
    return pd.DataFrame(rows)


def summarize_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Group by optimiser and N, compute mean/std and 95% CI.
    """
    df = df_runs.copy()
    df = df[pd.to_numeric(df["best"], errors="coerce").notna()]
    df["best"] = pd.to_numeric(df["best"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce").astype(int)

    def q25(x): return float(np.percentile(x, 25))
    def q75(x): return float(np.percentile(x, 75))

    g = df.groupby(["optimizer", "N"], as_index=False).agg(
        n_runs=("best", "count"),
        mean_best=("best", "mean"),
        std_best=("best", "std"),
        median_best=("best", "median"),
        q25_best=("best", q25),
        q75_best=("best", q75),
        max_best=("best", "max"),
        min_best=("best", "min"),
    )

    # 95% CI with normal approx (fine for n>=~5; if n small, still a useful visual)
    ci_half = []
    for _, r in g.iterrows():
        n = int(r["n_runs"])
        sd = float(r["std_best"]) if not pd.isna(r["std_best"]) else 0.0
        half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
        ci_half.append(half)
    g["ci95_halfwidth"] = ci_half
    g["ci95_low"] = g["mean_best"] - g["ci95_halfwidth"]
    g["ci95_high"] = g["mean_best"] + g["ci95_halfwidth"]
    return g.sort_values(["optimizer", "N"]).reset_index(drop=True)


def plot_final_vs_dimension(df_summary: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(8.5, 5.0))

    for opt, grp in df_summary.groupby("optimizer"):
        grp = grp.sort_values("N")
        x = grp["N"].to_numpy()
        y = grp["mean_best"].to_numpy()
        ylo = grp["ci95_low"].to_numpy()
        yhi = grp["ci95_high"].to_numpy()

        plt.plot(x, y, marker="o", linewidth=2, label=opt)
        plt.fill_between(x, ylo, yhi, alpha=0.2)

    plt.xlabel("Dimensions (N)")
    plt.ylabel("Final best selectivity (mean ± 95% CI)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_boxplot_by_dimension(df_runs: pd.DataFrame, out_path: str, title: str):
    """
    For each N, show boxplots per optimiser.
    """
    df = df_runs.copy()
    df = df[pd.to_numeric(df["best"], errors="coerce").notna()]
    df["best"] = pd.to_numeric(df["best"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce").astype(int)

    Ns = sorted(df["N"].unique())
    opts = sorted(df["optimizer"].unique())

    # Layout: one subplot per N
    fig, axes = plt.subplots(1, len(Ns), figsize=(max(8, 3.2 * len(Ns)), 4.6), squeeze=False)
    axes = axes[0]

    for ax, N in zip(axes, Ns):
        dN = df[df["N"] == N]
        data = []
        labels = []
        for opt in opts:
            vals = dN[dN["optimizer"] == opt]["best"].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(opt)

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_alpha(0.6)

        ax.set_title(f"N={N}")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Final best selectivity")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing run CSVs (or optimizer_summary.csv).")
    ap.add_argument("--out_dir", default=None, help="Output folder (default: data_dir).")
    ap.add_argument("--by_target", action="store_true",
                    help="Also make separate selectivity-vs-dimension plots per target.")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else data_dir
    os.makedirs(out_dir, exist_ok=True)

    df_runs = load_runs_from_directory(data_dir)
    # Standardize
    for col in ["target_x", "target_y", "target_z"]:
        if col not in df_runs.columns:
            df_runs[col] = np.nan

    # Save extracted runs
    runs_path = os.path.join(out_dir, "final_runs_extracted.csv")
    df_runs.to_csv(runs_path, index=False)

    # Summary across all targets
    df_summary = summarize_runs(df_runs)
    summary_path = os.path.join(out_dir, "final_summary_by_optimizer_and_N.csv")
    df_summary.to_csv(summary_path, index=False)

    plot_final_vs_dimension(
        df_summary,
        out_path=os.path.join(out_dir, "final_selectivity_vs_dimension.png"),
        title="Final best selectivity vs dimension (aggregated over targets/repeats)",
    )

    plot_boxplot_by_dimension(
        df_runs,
        out_path=os.path.join(out_dir, "final_selectivity_boxplot_by_dimension.png"),
        title="Final best selectivity distributions by dimension",
    )

    # Optional: per-target plots
    if args.by_target:
        # Make a target id string for grouping
        df_runs = df_runs.copy()
        df_runs["target_id"] = df_runs.apply(
            lambda r: f"x{r['target_x']:.6f}_y{r['target_y']:.6f}_z{r['target_z']:.6f}", axis=1
        )
        for tid, grp in df_runs.groupby("target_id"):
            df_s = summarize_runs(grp)
            plot_final_vs_dimension(
                df_s,
                out_path=os.path.join(out_dir, f"final_selectivity_vs_dimension__{tid}.png"),
                title=f"Final best selectivity vs dimension (target {tid})",
            )

    print("Done.")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Wrote:  {runs_path}")
    print(f"  Wrote:  {summary_path}")
    print(f"  Plot:   {os.path.join(out_dir, 'final_selectivity_vs_dimension.png')}")
    print(f"  Plot:   {os.path.join(out_dir, 'final_selectivity_boxplot_by_dimension.png')}")


if __name__ == "__main__":
    main()
