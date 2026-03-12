#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Tripolar Bayesian Optimization Benchmark
Features:
  - Search Space: Tripolar triplets (i, j, k) -> i=+I, j=-I/2, k=-I/2
  - Model: 10D Gaussian Process (Pos_i, Pos_j, Pos_k, Amplitude)
  - Analysis: Feature Importance (ARD) and Cross-Target Comparison plots.
"""

import os
import time
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

# --- Simulator import ---
try:
    from utils.run_selectivity_simulation import run_selectivity_simulation
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from run_selectivity_simulation import run_selectivity_simulation  # type: ignore

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "tripolar_bo")
RUNS_DIR   = os.path.join(OUTPUT_DIR, "runs")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
PLOTS_PER_RUN_DIR = os.path.join(PLOTS_DIR, "per_run")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_PER_RUN_DIR, exist_ok=True)

# Geometry / Tissue
RADIUS, HEIGHT, SIGMA_T = 0.01, 0.04, 0.08
GRID = (4, 3) # 12 electrodes

# Target points to compare (Meters)
TARGET_POINTS = [
    (0.0,    0.0,    0.0),      # centre of cylinder
    (0.002,  0.0035, 0.010),    # upper, off-centre (≈4.0 mm radially)
    (0.004,   0.002,  0.015),   # upper, ≈4.5 mm radius
    (0.005, 0.000, 0.000)
]

# Optimization Settings
I_MAX = 1e-3
AMP_LEVELS = np.linspace(0.05e-3, I_MAX, 25)
N_TRIALS = 200
N_INIT_RANDOM = 15
UCB_K = 2.0
REPEATS = 3
SEED_BASE = 42

# Labels for the 10D feature vector
FEATURE_LABELS = ['Src_x','Src_y','Src_z','Ret1_x','Ret1_y','Ret1_z','Ret2_x','Ret2_y','Ret2_z','Amp']

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def build_tripolar_candidates(grid, amp_levels):
    """
    Creates the finite pool of (triplet, amplitude) combinations.
    Tripolar: i = source, j & k = shared returns.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    
    # Generate electrode positions in mm
    z_vals = np.linspace(-HEIGHT/2, HEIGHT/2, n_rows)
    thetas = np.linspace(0.0, 2*np.pi, n_per_row, endpoint=False)
    pos_mm = []
    for z in z_vals:
        for th in thetas:
            pos_mm.append([RADIUS * math.cos(th) * 1e3, RADIUS * math.sin(th) * 1e3, z * 1e3])
    pos_mm = np.asarray(pos_mm)

    # Unique permutations of 3 electrodes
    triplets = list(itertools.permutations(range(N), 3))
    
    X_feat, meta, currents_list = [], [], []
    for (i, j, k) in triplets:
        for amp in amp_levels:
            cur = np.zeros(N)
            cur[i] = +amp
            cur[j] = -amp / 2.0
            cur[k] = -amp / 2.0
            currents_list.append(cur)
            # 10D Feature: [Xi, Yi, Zi, Xj, Yj, Zj, Xk, Yk, Zk, Amp]
            X_feat.append(np.concatenate([pos_mm[i], pos_mm[j], pos_mm[k], [amp * 1e3]]))
            meta.append((i, j, k, amp))
            
    return np.asarray(X_feat), meta, currents_list

def init_gp():
    """Initializes a GP with Matérn kernel and ARD (individual length-scales)."""
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(10), nu=2.5) + 
              WhiteKernel(noise_level=1e-5))
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=2
    )

def get_importance(gp):
    """Calculates feature importance as the inverse of the learned length-scales."""
    # Access length scales from the optimized kernel
    l_scales = gp.kernel_.k1.k2.length_scale
    return 1.0 / l_scales

# =============================================================================
# PLOTTING
# =============================================================================

def plot_per_run(best_curve, importance, tag):
    # Plot 1: Optimization Progress
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(best_curve)+1), best_curve, color='blue', lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far Selectivity")
    plt.title(f"BO Progress: {tag}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_PER_RUN_DIR, f"{tag}_progress.png"))
    plt.close()

    # Plot 2: Per-Run Feature Importance
    plt.figure(figsize=(8, 5))
    plt.bar(FEATURE_LABELS, importance, color='teal')
    plt.ylabel("Relevance (1/Length-scale)")
    plt.title(f"Feature Importance: {tag}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PER_RUN_DIR, f"{tag}_importance.png"))
    plt.close()

def plot_cross_target_comparison(summary_dict, out_path):
    """Plots grouped bars showing how importance shifts per target point."""
    plt.figure(figsize=(14, 7))
    x = np.arange(len(FEATURE_LABELS))
    width = 0.8 / len(summary_dict)
    
    for i, (t_idx, imp_vec) in enumerate(summary_dict.items()):
        plt.bar(x + (i * width), imp_vec, width, label=f"Target {t_idx}")

    plt.xlabel("Stimulation Features")
    plt.ylabel("Learned Relevance")
    plt.title("Cross-Target Comparison: What drives selectivity at different locations?")
    plt.xticks(x + width/2, FEATURE_LABELS, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"Generating Tripolar Search Space for {GRID} Grid...")
    X_all, meta, currents_list = build_tripolar_candidates(GRID, AMP_LEVELS)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    target_relevance_summary = {}

    for t_idx, tp in enumerate(TARGET_POINTS):
        all_rep_importance = []
        
        for rep in range(REPEATS):
            tag = f"Target{t_idx}_Rep{rep}"
            rng = np.random.default_rng(SEED_BASE + rep + (t_idx * 100))
            
            # Setup BO
            remaining = list(range(len(currents_list)))
            init_indices = rng.choice(remaining, N_INIT_RANDOM, replace=False).tolist()
            X_obs, y_obs, best_curve = [], [], []
            gp = init_gp()
            
            print(f"--- Starting {tag} ---")
            
            for step in range(1, N_TRIALS + 1):
                if step <= N_INIT_RANDOM:
                    idx = init_indices[step-1]
                else:
                    # Fit and find next point using UCB
                    gp.fit(np.array(X_obs), np.array(y_obs))
                    X_rem = X_scaled[remaining]
                    mu, std = gp.predict(X_rem, return_std=True)
                    idx = remaining[np.argmax(mu + UCB_K * std)]
                
                # Simulation Call
                res = run_selectivity_simulation(
                    n_rows=GRID[0], n_per_row=GRID[1], currents=currents_list[idx],
                    target_point=tp, radius=RADIUS, height=HEIGHT, sigma=SIGMA_T
                )
                sel = res["selectivity"]
                
                # Update records
                X_obs.append(X_scaled[idx])
                y_obs.append(sel)
                remaining.remove(idx)
                best_curve.append(max(y_obs))

            # Analysis for this run
            importance = get_importance(gp)
            all_rep_importance.append(importance)
            plot_per_run(best_curve, importance, tag)

            # Save per-run CSV
            pd.DataFrame(X_obs, columns=FEATURE_LABELS).assign(selectivity=y_obs).to_csv(
                os.path.join(RUNS_DIR, f"{tag}_data.csv"), index=False
            )

        # Average results for this target point
        target_relevance_summary[t_idx] = np.mean(all_rep_importance, axis=0)

    # Final Comparative Analysis
    plot_cross_target_comparison(target_relevance_summary, os.path.join(PLOTS_DIR, "final_target_comparison.png"))
    print(f"\nBenchmark Complete. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()