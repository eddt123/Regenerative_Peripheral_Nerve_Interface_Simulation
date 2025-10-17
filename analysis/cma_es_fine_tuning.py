#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter Performance Analysis
===================================
Reads all optimizer run CSVs (CMA-ES, BO, etc.) in a folder,
computes convergence metrics (AUC, best selectivity),
aggregates by hyperparameter set, and tests statistical significance.
"""

import os
import pandas as pd
import numpy as np
import ast
from scipy.stats import ttest_ind, f_oneway
from glob import glob

# ==============================================================
# CONFIG
# ==============================================================
FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "cma_es_fine_tuning")
FOLDER = os.path.abspath(FOLDER)

OUTPUT_SUMMARY = os.path.join(FOLDER, "hyperparameter_summary.csv")

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================
def compute_auc(iterations, selectivity):
    """Compute area under the convergence curve using trapezoidal rule."""
    return np.trapz(selectivity, iterations)

def parse_currents_column(df):
    """Convert stringified currents column to numeric arrays."""
    if "currents" in df.columns:
        df["currents"] = df["currents"].apply(lambda s: np.array(ast.literal_eval(s)) if isinstance(s, str) else np.nan)
    return df

# ==============================================================
# LOAD AND COMBINE ALL FILES
# ==============================================================
all_files = glob(os.path.join(FOLDER, "*_progress.csv"))
records = []

for f in all_files:
    try:
        df = pd.read_csv(f)
        df = parse_currents_column(df)
        if "iteration_selectivity" not in df.columns:
            continue

        tag = os.path.basename(f).replace("_progress.csv", "")
        optimizer = df["optimizer"].iloc[0] if "optimizer" in df.columns else "Unknown"

        # compute metrics
        auc = compute_auc(df["iteration"], df["iteration_selectivity"])
        best_sel = df["iteration_selectivity"].max()

        # extract hyperparameters from first row (consistent within file)
        params = {col: df[col].iloc[0] for col in df.columns if col not in [
            "iteration", "iteration_selectivity", "best_so_far",
            "iteration_best_found", "currents"
        ]}

        params.update({
            "optimizer": optimizer,
            "file": os.path.basename(f),
            "auc": auc,
            "best_selectivity": best_sel,
        })
        records.append(params)
    except Exception as e:
        print(f"Error reading {f}: {e}")

runs = pd.DataFrame(records)
print(f"Loaded {len(runs)} runs")

# ==============================================================
# AGGREGATE ACROSS REPEATS
# ==============================================================
# Define columns that define a "hyperparameter set" (exclude repeat, file, etc.)
hyper_cols = [c for c in runs.columns if c not in ["file", "auc", "best_selectivity", "repeat"]]

grouped = runs.groupby(hyper_cols).agg(
    mean_auc=("auc", "mean"),
    std_auc=("auc", "std"),
    mean_best=("best_selectivity", "mean"),
    std_best=("best_selectivity", "std"),
    n=("auc", "count")
).reset_index()

# ==============================================================
# STATISTICAL TESTING
# ==============================================================
# Compare within each optimizer type
results = []
for opt in grouped["optimizer"].unique():
    subset = runs[runs["optimizer"] == opt]
    groups = [g["auc"].values for _, g in subset.groupby(
        [c for c in subset.columns if c not in ["file", "auc", "best_selectivity", "repeat", "optimizer"]]
    ) if len(g) > 1]
    if len(groups) > 1:
        F, p = f_oneway(*groups)
        results.append({"optimizer": opt, "anova_F": F, "anova_p": p})
anova_df = pd.DataFrame(results)

# ==============================================================
# RANK & OUTPUT
# ==============================================================
grouped["rank_auc"] = grouped["mean_auc"].rank(ascending=False)
grouped["rank_best"] = grouped["mean_best"].rank(ascending=False)

best_overall = grouped.sort_values("mean_auc", ascending=False).head(10)
print("\n===== TOP 10 HYPERPARAMETER CONFIGURATIONS =====")
print(best_overall.to_string(index=False))

grouped.to_csv(OUTPUT_SUMMARY, index=False)
anova_df.to_csv(os.path.join(FOLDER, "anova_results.csv"), index=False)

print(f"\nSummary saved to: {OUTPUT_SUMMARY}")
print(f"ANOVA results saved to: {os.path.join(FOLDER, 'anova_results.csv')}")
