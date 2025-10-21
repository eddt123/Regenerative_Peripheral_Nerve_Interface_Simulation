#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust CMA-ES Montage Analysis Script
=====================================
Handles both summary and progress CSVs (like CMAES_VAR_s2.0e-04_p12_r0.csv).
Automatically detects structure, extracts stats, and ranks montage dimensions,
sigmas, and populations by selectivity and consistency.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem  # standard error of mean

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cma_es_montage_final")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[Setup] Analysing folder: {RESULTS_DIR}")

# ---------------------------------------------------------------
# HELPER: Robust CSV reader
# ---------------------------------------------------------------
def read_csv_safe(path):
    """
    Reads a CSV and skips over non-numeric rows or malformed headers.
    Automatically detects if file is a summary (one row per run) or progress (many iterations).
    """
    try:
        df = pd.read_csv(path)
        # Drop empty rows
        df.dropna(how="all", inplace=True)

        # Handle string headers accidentally included in data
        for col in df.columns:
            if isinstance(df[col].iloc[0], str) and df[col].iloc[0].strip() == col:
                df = df.drop(index=0).reset_index(drop=True)

        # Try to convert numeric columns
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass

        return df
    except Exception as e:
        print(f"[warn] Failed to read {os.path.basename(path)}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------
# STEP 1 — Load all CSVs
# ---------------------------------------------------------------
all_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv") and "CMAES_VAR" in f]
print(f"[Data] Found {len(all_files)} candidate CSV files.")

records = []

for f in all_files:
    path = os.path.join(RESULTS_DIR, f)
    df = read_csv_safe(path)
    if df.empty:
        continue

    # --- Handle missing or malformed numeric data ---
    numeric_cols = [c for c in ["best_so_far", "best_reward", "iteration_selectivity"] if c in df.columns]
    if not numeric_cols:
        print(f"[skip] {f} has no selectivity data.")
        continue

    col = numeric_cols[0]
    y = pd.to_numeric(df[col], errors="coerce").dropna().values
    if len(y) == 0:
        continue

    x = np.arange(len(y))
    auc = np.trapz(y, x) / max(len(y), 1)
    best_sel = float(np.max(y))
    thresh = 0.9 * best_sel
    speed = np.argmax(y >= thresh) + 1 if np.any(y >= thresh) else len(y)

    # Extract metadata robustly
    def safe_get(k, default=None):
        if k in df.columns:
            try:
                return float(df[k].iloc[0])
            except Exception:
                return default
        return default

    records.append({
        "file": f,
        "montage_dim": int(safe_get("montage_dim", 0)),
        "sigma0": safe_get("sigma0", np.nan),
        "popsize": int(safe_get("popsize", np.nan)),
        "AUC": auc,
        "best_selectivity": best_sel,
        "speed_to90": speed,
        "iterations": len(y),
    })

if not records:
    raise RuntimeError("No valid numeric results parsed. Check CSV formatting.")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "parsed_summary.csv"), index=False)
print(f"[Parsed] {len(df)} valid runs extracted.")


# ---------------------------------------------------------------
# STEP 2 — Aggregate statistics
# ---------------------------------------------------------------
grouped = (
    df.groupby(["montage_dim", "sigma0", "popsize"])
    .agg({
        "AUC": ["mean", "std", sem],
        "best_selectivity": ["mean", "std", sem],
        "speed_to90": ["mean", "std", sem],
        "iterations": "mean"
    })
)

grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
grouped.reset_index(inplace=True)

# Composite performance metric
grouped["score"] = (
    0.6 * grouped["AUC_mean"] +
    0.4 * grouped["best_selectivity_mean"] -
    0.01 * grouped["speed_to90_mean"]
)

grouped.sort_values("score", ascending=False, inplace=True)
grouped.to_csv(os.path.join(OUTPUT_DIR, "ranked_montage_summary.csv"), index=False)

# ---------------------------------------------------------------
# STEP 3 — Print ranked results
# ---------------------------------------------------------------
print("\n================ CMA-ES MONTAGE PERFORMANCE SUMMARY ================")
print(f"Analysed {len(df)} runs from {len(all_files)} files\n")

best = grouped.iloc[0]
print("Best configuration:")
print(f"  • Montage dimension : {int(best['montage_dim'])}")
print(f"  • Sigma0            : {best['sigma0']*1e3:.3f} mA")
print(f"  • Population size   : {int(best['popsize'])}")
print(f"  • AUC (mean ± sd)   : {best['AUC_mean']:.4f} ± {best['AUC_std']:.4f}")
print(f"  • Selectivity (±sd) : {best['best_selectivity_mean']:.4f} ± {best['best_selectivity_std']:.4f}")
print(f"  • Speed to 90%      : {best['speed_to90_mean']:.1f} ± {best['speed_to90_std']:.1f}")
print(f"  • Composite score   : {best['score']:.4f}")
print("-------------------------------------------------------------------\n")

print("Top 5 configurations:\n")
print(grouped.head(5).to_string(index=False))
print("-------------------------------------------------------------------")

# ---------------------------------------------------------------
# STEP 4 — Visualise
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
for s in sorted(grouped["sigma0"].unique()):
    sub = grouped[grouped["sigma0"] == s]
    plt.errorbar(sub["montage_dim"], sub["AUC_mean"], yerr=sub["AUC_std"], capsize=3,
                 marker='o', label=f"σ={s*1e3:.3f} mA")
plt.xlabel("Montage Dimension")
plt.ylabel("Mean AUC")
plt.title("CMA-ES Montage AUC vs Dimension")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "auc_vs_dimension.png"), dpi=180)
plt.close()

plt.figure(figsize=(7,5))
plt.scatter(grouped["AUC_mean"], grouped["best_selectivity_mean"],
            c=grouped["speed_to90_mean"], cmap="plasma", s=80)
plt.colorbar(label="Speed to 90% best")
plt.xlabel("Mean AUC")
plt.ylabel("Mean Final Selectivity")
plt.title("Trade-off: Efficiency vs Final Quality")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tradeoff_auc_selectivity.png"), dpi=180)
plt.close()

print(f"\n[Done] Analysis complete. Results saved to {OUTPUT_DIR}")
print("===================================================================")
