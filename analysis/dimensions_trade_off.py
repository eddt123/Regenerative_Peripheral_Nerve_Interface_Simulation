#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMA-ES vs PSO: Trade-off between speed and final selectivity
===========================================================

Loads all CMAES_* and PSO_* CSVs in DATA_DIR and computes:

Per target:
    S_best(target) = max best_so_far over all runs

Per run:
    - final_rel  = max(best_rel)
    - time_to_95 = first eval where best_rel >= 0.95 * S_best
    - auc_rel    = normalised AUC of best_rel vs evals

Then aggregates per (optimizer, N) and plots:
    (A) Final_rel vs N
    (B) Median time_to_95 vs N (log scale)
    (C) (optional) AUC_rel vs N
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

# Folder containing all your CMAES_* and PSO_* CSVs
DATA_DIR = r"C:\Users\eddyt\Desktop\GitHub\Regenerative_Peripheral_Nerve_Interface_Simulation\data\benchmark_long"  # <-- change this

# Only compare these optimizers (adjust if needed)
OPTIMIZERS_TO_PLOT = ["CMAES", "PSO"]

# Threshold relative to target-best selectivity
REL_THRESH = 0.95


# ============================================================
# 1. Load all history rows
# ============================================================

all_rows = []

for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Skipping {path} (read error: {e})")
        continue

    required_cols = {
        "optimizer", "N", "repeat",
        "target_x", "target_y", "target_z",
        "evals_so_far", "best_so_far"
    }
    if not required_cols.issubset(df.columns):
        print(f"Skipping {path} (missing columns)")
        continue

    df = df[list(required_cols)].copy()
    df["file"] = os.path.basename(path)
    all_rows.append(df)

if not all_rows:
    raise RuntimeError("No valid CSVs found in DATA_DIR – check path and files.")

hist = pd.concat(all_rows, ignore_index=True)

# Focus only on optimizers of interest
hist = hist[hist["optimizer"].isin(OPTIMIZERS_TO_PLOT)].copy()

target_cols = ["target_x", "target_y", "target_z"]

# ============================================================
# 2. Best selectivity per target across ALL runs
# ============================================================

target_best = (
    hist.groupby(target_cols)["best_so_far"]
        .max()
        .rename("target_best")
        .reset_index()
)

hist = hist.merge(target_best, on=target_cols, how="left")
hist["best_rel"] = hist["best_so_far"] / hist["target_best"]

# ============================================================
# 3. Collapse to per-run metrics
# ============================================================

run_cols = ["file", "optimizer", "N", "repeat"] + target_cols

run_stats = []

for keys, sub in hist.groupby(run_cols):
    file, opt, N, rep, tx, ty, tz = keys
    sub = sub.sort_values("evals_so_far")

    evals = sub["evals_so_far"].values.astype(float)
    best_rel = sub["best_rel"].values.astype(float)

    # Final relative selectivity for this run (0–1)
    final_rel = best_rel.max()

    # Time to reach REL_THRESH of target-best selectivity
    mask = best_rel >= REL_THRESH
    if mask.any():
        time_to_95 = evals[mask][0]
        reached_95 = True
    else:
        time_to_95 = evals[-1]  # or np.nan if you prefer
        reached_95 = False

    # Normalised AUC of best_rel vs evals (0–1)
    # Use trapezoidal rule and normalise by (max_evals * 1)
    if len(evals) > 1:
        auc = np.trapz(best_rel, evals)
        max_T = evals[-1]
        auc_rel = auc / max_T
    else:
        auc_rel = final_rel  # degenerate case

    run_stats.append({
        "file": file,
        "optimizer": opt,
        "N": int(N),
        "repeat": int(rep),
        "target_x": tx, "target_y": ty, "target_z": tz,
        "final_rel": final_rel,
        "time_to_95": time_to_95,
        "reached_95": reached_95,
        "auc_rel": auc_rel,
    })

run_stats = pd.DataFrame(run_stats)

print("Per-run stats (head):")
print(run_stats.head())

# ============================================================
# 4. Dimension-level summaries
# ============================================================

# (A) Final relative selectivity vs N
summary_final = (
    run_stats
    .groupby(["optimizer", "N"])
    .agg(
        mean_final_rel=("final_rel", "mean"),
        q25=("final_rel", lambda x: np.quantile(x, 0.25)),
        q75=("final_rel", lambda x: np.quantile(x, 0.75)),
    )
    .reset_index()
)

# (B) Speed: median time_to_95 among runs that reach the threshold
summary_speed = (
    run_stats[run_stats["reached_95"]]
    .groupby(["optimizer", "N"])
    .agg(
        median_time_95=("time_to_95", "median"),
        q25=("time_to_95", lambda x: np.quantile(x, 0.25)),
        q75=("time_to_95", lambda x: np.quantile(x, 0.75)),
    )
    .reset_index()
)

# Also compute fraction of runs reaching 95% at each (optimizer, N)
frac_reached = (
    run_stats
    .groupby(["optimizer", "N"])["reached_95"]
    .mean()
    .rename("frac_reached_95")
    .reset_index()
)

summary_speed = summary_speed.merge(frac_reached, on=["optimizer", "N"], how="left")

# (C) AUC vs N (optional)
summary_auc = (
    run_stats
    .groupby(["optimizer", "N"])
    .agg(
        mean_auc_rel=("auc_rel", "mean"),
        auc_q25=("auc_rel", lambda x: np.quantile(x, 0.25)),
        auc_q75=("auc_rel", lambda x: np.quantile(x, 0.75)),
    )
    .reset_index()
)

print("\nFinal-rel summary:")
print(summary_final)

print("\nSpeed summary:")
print(summary_speed)

# ============================================================
# 5. Plotting
# ============================================================

# --- Figure 1: Final_rel vs N AND Median time_to_95 vs N ---

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1, ax2 = axes

colors = {"CMAES": "C0", "PSO": "C1"}

# Plot A: final_rel vs N
for opt in OPTIMIZERS_TO_PLOT:
    sub = summary_final[summary_final["optimizer"] == opt].sort_values("N")
    if sub.empty:
        continue

    ax1.plot(
        sub["N"],
        sub["mean_final_rel"],
        "-o",
        label=opt,
        color=colors.get(opt, None),
    )

    # IQR band
    ax1.fill_between(
        sub["N"],
        sub["q25"],
        sub["q75"],
        alpha=0.2,
        color=colors.get(opt, None),
    )

ax1.set_xlabel("Number of electrodes (N)")
ax1.set_ylabel("Mean final selectivity / best at that target")
ax1.set_ylim(0.9, 1.01)
ax1.grid(True, alpha=0.3)
ax1.legend(title="Optimizer")

# Plot B: median time_to_95 vs N (log y-axis)
for opt in OPTIMIZERS_TO_PLOT:
    sub = summary_speed[summary_speed["optimizer"] == opt].sort_values("N")
    if sub.empty:
        continue

    ax2.plot(
        sub["N"],
        sub["median_time_95"],
        "-o",
        label=f"{opt}",
        color=colors.get(opt, None),
    )

    ax2.fill_between(
        sub["N"],
        sub["q25"],
        sub["q75"],
        alpha=0.2,
        color=colors.get(opt, None),
    )

    # annotate with fraction of runs that reached 95%
    for _, row in sub.iterrows():
        frac = row["frac_reached_95"]
        ax2.text(
            row["N"],
            row["median_time_95"],
            f"{frac*100:.0f}%",
            fontsize=8,
            ha="center",
            va="bottom",
        )

ax2.set_xlabel("Number of electrodes (N)")
ax2.set_ylabel(f"Median evals to reach {int(REL_THRESH*100)}% of target-best")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)
ax2.legend(title="Optimizer")

fig.tight_layout()
plt.show()

# --- Figure 2 (optional): AUC vs N ---

fig2, ax3 = plt.subplots(figsize=(6, 4))

for opt in OPTIMIZERS_TO_PLOT:
    sub = summary_auc[summary_auc["optimizer"] == opt].sort_values("N")
    if sub.empty:
        continue

    ax3.plot(
        sub["N"],
        sub["mean_auc_rel"],
        "-o",
        label=opt,
        color=colors.get(opt, None),
    )

    ax3.fill_between(
        sub["N"],
        sub["auc_q25"],
        sub["auc_q75"],
        alpha=0.2,
        color=colors.get(opt, None),
    )

ax3.set_xlabel("Number of electrodes (N)")
ax3.set_ylabel("Mean normalised AUC of best_rel")
ax3.set_ylim(0.8, 1.01)
ax3.grid(True, alpha=0.3)
ax3.legend(title="Optimizer")

fig2.tight_layout()
plt.show()
