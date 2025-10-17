#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive CMA-ES vs Bayesian Optimisation Benchmark
=========================================================================
Runs both optimisers (CMA-ES, BO) on your RPNI selectivity simulation
in the full 12-electrode space. Logs currents, target points, and performance.
"""

import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from cma import CMAEvolutionStrategy
from skopt import Optimizer as SkOptimizer
from utils.run_selectivity_simulation import run_selectivity_simulation

# ==============================================================
# CONFIGURATION
# ==============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "cma_es_montage_fine_tune")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ELECTRODES = 12
RANGE = 1e-3
N_ITER = 100
REPEATS = 10

# --- CMA-ES hyperparameters ---
SIGMAS = [0.3e-3]
POPSIZES = [16]

# --- BO hyperparameters ---
ACQ_FUNCS = ["EI", "LCB", "PI"]
KAPPAS = [0.1, 0.5, 1.0, 2.0, 3.0]
XIS = [0.001, 0.005, 0.01, 0.05]
N_INIT_POINTS = [5, 10]

# Simulation geometry
radius, height, sigma = 0.01, 0.04, 0.25

# ==============================================================
# FIXED TARGET POINTS
# ==============================================================
TARGET_POINTS = [
    #(0.0020, -0.00346, -0.0200) #validated example

    # ----- Central region -----
    (0.0, 0.0, 0.0),                    # center of RPNI

    (0.002, 0.0035, 0.010),             # upper right quadrant
    #(-0.002, -0.0035, -0.010),          # lower left quadrant


    # ----- Off-center mid-depth nerves -----
    #(0.003, 0.002, 0.015),              # proximal oblique
    (-0.003, -0.002, -0.015),           # distal oblique
]
#z_levels = np.linspace(-height/3, height/3, 3)
#r_levels = [0.0, 0.004, 0.0075]
#angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
#for z in z_levels:
#    for r in r_levels:
#        for th in angles[:2]:
#            TARGET_POINTS.append((r*np.cos(th), r*np.sin(th), z))
#print(f"[Setup] Using {len(TARGET_POINTS)} fixed target points.")

# ==============================================================
# OBJECTIVE FUNCTION
# ==============================================================
def eval_selectivity(currents, target_point):
    currents = np.asarray(currents)
    currents -= np.mean(currents)
    res = run_selectivity_simulation(
        n_rows=4, n_per_row=3,
        currents=currents,
        target_point=target_point,
        radius=radius, height=height, sigma=sigma,
        n_off_samples=1200, metric="activation",
        grounded_boundary=True, R_outer=0.10,
    )
    return -float(res["selectivity"])  # minimiser

# ==============================================================
# PLOTTER
# ==============================================================
# ==============================================================
# PLOTTER (with target point annotation)
# ==============================================================
def save_live_plot(iters, vals, bests, tag, target_point):
    plt.figure(figsize=(6, 4))
    plt.plot(iters, vals, 'o-', color='gray', lw=1.5, label='Iteration Selectivity')
    plt.plot(iters, bests, '-', color='blue', lw=2.0, label='Best-so-far')
    plt.xlabel("Iteration")
    plt.ylabel("Selectivity")

    # Annotate which target point is being optimised
    tx, ty, tz = target_point
    title_str = (
        f"{tag}\n"
        f"Target point: x={tx*1e3:.2f} mm, y={ty*1e3:.2f} mm, z={tz*1e3:.2f} mm"
    )

    plt.title(title_str, fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fname = f"{tag}_x{tx*1e3:.1f}_y{ty*1e3:.1f}_z{tz*1e3:.1f}_progress.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
    plt.close()

# ==============================================================
# CMA-ES FULL
# ==============================================================
def run_cma_montage_variable_dim(sigma0, popsize, repeat, target_point, dims_to_test):
    """
    Extended CMA-ES that tries multiple montage dimensionalities and montage types.

    Each montage type (bipolar, tripolar, quadrupolar, random composite) is
    generated, and for each type a CMA-ES optimisation is run in several
    reduced spaces (e.g. [2, 4, 6, 8, 12] montage bases).

    Parameters
    ----------
    sigma0 : float
        CMA-ES initial step size.
    popsize : int
        CMA-ES population size.
    repeat : int
        Repeat index for reproducibility.
    target_point : tuple(float, float, float)
        Target neural point in cylinder (x, y, z).
    dims_to_test : list[int]
        Montage-space dimensionalities to test (e.g. [2, 4, 6, 8, 12]).
    """
    import numpy as np, pandas as pd
    from tqdm import tqdm
    from cma import CMAEvolutionStrategy

    tag_base = f"CMAES_MONTAGEVAR_s{sigma0:.1e}_p{popsize}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag_base}_progress.csv")
    header_written = False

    # -------------------------------------------------------------
    # 1. DEFINE MONTAGE TYPES
    # -------------------------------------------------------------
    def build_montages(n_electrodes=12):
        montages = []

        # --- Bipolar (+1, -1) ---
        for i in range(n_electrodes):
            pair = np.zeros(n_electrodes)
            pair[i] = +1
            pair[(i + 1) % n_electrodes] = -1
            montages.append(pair)

        # --- Tripolar (+1, -2, +1) ---
        for i in range(n_electrodes):
            trip = np.zeros(n_electrodes)
            trip[i] = +1
            trip[(i + 1) % n_electrodes] = -2
            trip[(i + 2) % n_electrodes] = +1
            montages.append(trip)

        # --- Quadrupolar (+1, -1, -1, +1) ---
        for i in range(n_electrodes):
            quad = np.zeros(n_electrodes)
            quad[i] = +1
            quad[(i + 1) % n_electrodes] = -1
            quad[(i + 2) % n_electrodes] = -1
            quad[(i + 3) % n_electrodes] = +1
            montages.append(quad)

        # --- Random composite patterns ---
        rng = np.random.default_rng(42)
        for _ in range(n_electrodes):
            rand_pattern = rng.choice([-1, 0, 1], size=n_electrodes, p=[0.4, 0.2, 0.4])
            if np.sum(np.abs(rand_pattern)) > 0:
                rand_pattern = rand_pattern / np.sum(np.abs(rand_pattern))
            montages.append(rand_pattern)

        return np.array(montages)

    montages = build_montages()
    n_templates = len(montages)

    # -------------------------------------------------------------
    # 2. LOOP OVER DIMENSIONS
    # -------------------------------------------------------------
    best_global = {"best_reward": -np.inf, "montage_dim": None, "montage_type": None, "currents": None}

    for dim in dims_to_test:
        # Choose first `dim` montages
        templates = montages[:dim]
        tag = f"{tag_base}_dim{dim}"
        print(f"\n[{tag}] Optimising montage space of dimension {dim}")

        # ---------------------------------------------------------
        # 3. CMA-ES setup
        # ---------------------------------------------------------
        es = CMAEvolutionStrategy(np.zeros(dim), sigma0, {"popsize": popsize, "verb_disp": 0})
        best_so_far, best_iter = -np.inf, 0
        iter_vals, best_vals = [], []

        for it in tqdm(range(N_ITER), desc=tag, leave=False):
            weights_list = es.ask()
            Y, combos = [], []

            for w in weights_list:
                # Combine montages linearly
                currents = np.sum([w[j] * templates[j] for j in range(dim)], axis=0)
                currents = np.clip(currents, -RANGE, RANGE)
                obj = eval_selectivity(currents, target_point)  # negative selectivity
                Y.append(obj)
                combos.append(currents)

            es.tell(weights_list, Y)

            iter_best = -np.min(Y)
            iter_best_idx = int(np.argmin(Y))
            iter_best_currents = combos[iter_best_idx]

            iter_vals.append(iter_best)
            if iter_best > best_so_far:
                best_so_far, best_iter = iter_best, it + 1
            best_vals.append(best_so_far)

            # Live plot
            save_live_plot(np.arange(1, len(iter_vals)+1), iter_vals, best_vals, tag, target_point)

            # Logging
            pd.DataFrame([{
                "optimizer": "CMAES_MONTAGEVAR",
                "sigma0": sigma0,
                "popsize": popsize,
                "repeat": repeat,
                "iteration": it + 1,
                "iteration_selectivity": float(iter_best),
                "best_so_far": float(best_so_far),
                "iteration_best_found": int(best_iter),
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                "montage_dim": dim,
                "weights": weights_list[iter_best_idx].tolist(),
                "currents": iter_best_currents.tolist()
            }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
            header_written = True

        print(f"[{tag}] Finished. Best selectivity = {best_so_far:.4f}")

        # Track global best montage configuration
        if best_so_far > best_global["best_reward"]:
            best_global.update({
                "best_reward": best_so_far,
                "montage_dim": dim,
                "currents": iter_best_currents.tolist()
            })

    print(f"\n[Summary] Target {target_point} best config: "
          f"{best_global['montage_dim']}D with selectivity {best_global['best_reward']:.4f}")

    return best_global


# ==============================================================
# MAIN BENCHMARK
# ==============================================================
if __name__ == "__main__":
    start = time.time()
    all_runs = []

    print("\n=== PHASE 4: Variable-Dimension Montage CMA-ES ===")
    dims_to_test = [2, 4, 6, 8, 12]

    for sigma0, popsize in product(SIGMAS, POPSIZES):
        for r, tp in product(range(REPEATS), TARGET_POINTS):
            best_cfg = run_cma_montage_variable_dim(sigma0, popsize, r, tp, dims_to_test)
            best_cfg.update({
                "sigma0": sigma0,
                "popsize": popsize,
                "repeat": r,
                "target_point": tp,
                "optimizer": "CMAES_MONTAGEVAR"
            })
            all_runs.append(best_cfg)

    df = pd.DataFrame(all_runs)
    summary_path = os.path.join(OUTPUT_DIR, "montage_variable_summary.csv")
    df.to_csv(summary_path, index=False)

    print(f"\nAll runs complete. Summary saved to {summary_path}")



