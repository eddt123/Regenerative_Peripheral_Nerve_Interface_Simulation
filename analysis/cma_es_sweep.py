#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimizer Result Analysis for RPNI Simulation
=============================================
Reads all *_progress.csv files from data/cma_es_sweep/,
computes convergence metrics, determines the best optimizer,
and performs statistical significance testing (robust to empty cases).
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================
# CONFIGURATION
# ==============================================================
BASE_DIR = os.path.join( "data", "cma_es_sweep")
SAVE_PREFIX = "analysis"
ALPHA = 0.05  # significance threshold

# ==============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def load_all_csvs(base_dir):
    files = glob.glob(os.path.join(base_dir, "**", "*_progress.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No *_progress.csv found in {base_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
            df["__file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    print(f"[LOAD] Loaded {len(dfs)} CSVs from {base_dir}")
    return pd.concat(dfs, ignore_index=True)

def normalize_schema(df):
    # Ensure columns exist
    for col in [
        "optimizer", "sigma0", "popsize", "repeat", "iteration",
        "iteration_selectivity", "best_so_far", "target_x", "target_y", "target_z",
        "acq_func", "kappa", "xi", "n_init", "phase"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Cast numeric
    for c in [
        "sigma0", "popsize", "repeat", "iteration",
        "iteration_selectivity", "best_so_far",
        "target_x", "target_y", "target_z",
        "kappa", "xi", "n_init"
    ]:
        df[c] = df[c].apply(safe_float)

    # Phase default
    df["phase"] = df["phase"].fillna("optimization")

    # Build run key
    df["run_key"] = df.apply(lambda r: (
        r["optimizer"], r["sigma0"], r["popsize"],
        r["acq_func"], r["kappa"], r["xi"], r["n_init"],
        (r["target_x"], r["target_y"], r["target_z"]), r["repeat"]
    ), axis=1)

    return df

def compute_per_run_metrics(df):
    rows = []
    for key, g in df.groupby("run_key"):
        g = g[g["phase"] == "optimization"]
        if g.empty:
            continue
        g = g.sort_values("iteration")
        y = g["best_so_far"].fillna(g["iteration_selectivity"])
        if y.isna().all():
            continue
        y = y.values
        x = g["iteration"].values
        if len(x) < 2:
            continue

        auc = np.trapz(y, x)
        span = max(x.max() - x.min(), 1)
        auc_norm = auc / span
        best_sel = float(np.max(y))

        rows.append({
            "optimizer": g["optimizer"].iloc[0],
            "sigma0": g["sigma0"].iloc[0],
            "popsize": g["popsize"].iloc[0],
            "acq_func": g["acq_func"].iloc[0],
            "kappa": g["kappa"].iloc[0],
            "xi": g["xi"].iloc[0],
            "n_init": g["n_init"].iloc[0],
            "repeat": g["repeat"].iloc[0],
            "target": (g["target_x"].iloc[0], g["target_y"].iloc[0], g["target_z"].iloc[0]),
            "auc_norm": auc_norm,
            "best_selectivity": best_sel,
        })
    return pd.DataFrame(rows)

def pick_best_hyperparams(per_run):
    # Stable hyperparam signature per optimizer
    def sig(r):
        opt = str(r["optimizer"])
        if "CMAES_MONTAGE" in opt:
            return f"montage_s{r['sigma0']}_p{r['popsize']}"
        elif "CMAES" in opt:  # includes sweep run's optimization phase
            return f"CMAES_s{r['sigma0']}_p{r['popsize']}"
        elif opt == "BO":
            return f"BO_{r['acq_func']}_k{r['kappa']}_x{r['xi']}_n{r['n_init']}"
        else:
            return opt

    per_run = per_run.copy()
    per_run["hyp_sig"] = per_run.apply(sig, axis=1)

    agg = (
        per_run.groupby(["optimizer", "hyp_sig"])
        .agg(
            mean_best=("best_selectivity", "mean"),
            std_best=("best_selectivity", "std"),
            mean_auc=("auc_norm", "mean"),
            std_auc=("auc_norm", "std"),
            n=("auc_norm", "count"),
        )
        .reset_index()
    )

    # Choose best per optimizer by best selectivity then AUC
    best = (
        agg.sort_values(["optimizer", "mean_best", "mean_auc"], ascending=[True, False, False])
        .groupby("optimizer")
        .head(1)
        .reset_index(drop=True)
    )
    return agg, best, per_run

def pairwise_stats(per_run, best_table, alpha=0.05):
    """
    Compare optimizers using only their best hyperparams.
    Returns empty table (with columns) if fewer than 2 optimizers or too few samples.
    """
    # Collect samples for each optimizer at its best hyperparams
    sample_blocks = {}
    for _, row in best_table.iterrows():
        opt = row["optimizer"]
        hyp = row["hyp_sig"]
        block = per_run[(per_run["optimizer"] == opt) & (per_run["hyp_sig"] == hyp)][
            ["auc_norm", "best_selectivity"]
        ].copy()
        block["opt"] = opt
        sample_blocks[opt] = block

    opts = list(sample_blocks.keys())
    results = []

    for i in range(len(opts)):
        for j in range(i + 1, len(opts)):
            a, b = opts[i], opts[j]
            da = sample_blocks[a]
            db = sample_blocks[b]
            # Need at least two repeats per group to have meaningful variance
            if len(da) < 2 or len(db) < 2:
                continue
            for metric in ["auc_norm", "best_selectivity"]:
                try:
                    t_res = stats.ttest_ind(da[metric], db[metric], equal_var=False, nan_policy="omit")
                    u_res = stats.mannwhitneyu(da[metric], db[metric], alternative="two-sided")
                    results.append({
                        "metric": metric,
                        "opt1": a,
                        "opt2": b,
                        "t_p": float(t_res.pvalue),
                        "u_p": float(u_res.pvalue),
                        "n1": int(len(da)),
                        "n2": int(len(db)),
                        "mean1": float(np.nanmean(da[metric])),
                        "mean2": float(np.nanmean(db[metric])),
                        "std1": float(np.nanstd(da[metric], ddof=1)),
                        "std2": float(np.nanstd(db[metric], ddof=1)),
                    })
                except Exception:
                    # Robust: skip pathological cases
                    continue

    # If no pairwise results, return an empty frame with expected columns
    if not results:
        cols = ["metric", "opt1", "opt2", "t_p", "u_p", "t_p_holm", "u_p_holm",
                "t_p_sig", "u_p_sig", "n1", "n2", "mean1", "mean2", "std1", "std2"]
        return pd.DataFrame(columns=cols)

    res = pd.DataFrame(results)

    # Holm-Bonferroni correction
    def holm_adjust(pvals):
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        adj = np.zeros_like(pvals)
        for rank, idx in enumerate(order):
            adj[idx] = min((m - rank) * pvals[idx], 1.0)
        return adj

    res["t_p_holm"] = holm_adjust(res["t_p"].values)
    res["u_p_holm"] = holm_adjust(res["u_p"].values)
    res["t_p_sig"] = res["t_p_holm"] < alpha
    res["u_p_sig"] = res["u_p_holm"] < alpha
    return res

# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print(f"[LOAD] Reading logs from {BASE_DIR}")
    raw = load_all_csvs(BASE_DIR)
    df = normalize_schema(raw)

    print("[PROCESS] Computing per-run metrics...")
    per_run = compute_per_run_metrics(df)
    per_run_path = os.path.join(BASE_DIR, f"{SAVE_PREFIX}_per_run_metrics.csv")
    per_run.to_csv(per_run_path, index=False)
    print(f"Saved → {per_run_path}")

    print("[AGGREGATE] Summarizing hyperparameters...")
    agg, best, per_run = pick_best_hyperparams(per_run)
    agg_path = os.path.join(BASE_DIR, f"{SAVE_PREFIX}_hyperparam_summary.csv")
    best_path = os.path.join(BASE_DIR, f"{SAVE_PREFIX}_best_hyperparams.csv")
    agg.to_csv(agg_path, index=False)
    best.to_csv(best_path, index=False)

    print("[STATS] Running pairwise significance tests...")
    stats_df = pairwise_stats(per_run, best, alpha=ALPHA)
    stats_path = os.path.join(BASE_DIR, f"{SAVE_PREFIX}_stats_pairwise.csv")
    stats_df.to_csv(stats_path, index=False)

    print("\n===== BEST HYPERPARAMETERS PER OPTIMIZER =====")
    if len(best):
        print(best.to_string(index=False))
    else:
        print("No optimizers with sufficient runs to summarize.")

    print("\n===== PAIRWISE SIGNIFICANCE (Holm-adjusted) =====")
    if len(stats_df):
        print(stats_df[["metric", "opt1", "opt2", "n1", "n2",
                        "t_p_holm", "t_p_sig", "u_p_holm", "u_p_sig"]].to_string(index=False))
    else:
        print("Insufficient distinct optimizers or repeats for pairwise testing (no comparisons possible).")

    print(f"\nAll summaries saved in: {BASE_DIR}")
