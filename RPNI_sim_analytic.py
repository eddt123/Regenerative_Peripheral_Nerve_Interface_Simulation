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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "cma_es_sweep_fine_tune")
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
def run_cma_experiment(sigma0, popsize, repeat, target_point):
    tag = f"CMAES_s{sigma0:.1e}_p{popsize}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}_progress.csv")
    header_written = False
    x0 = np.random.uniform(-RANGE, RANGE, N_ELECTRODES)
    x0 -= np.mean(x0)
    es = CMAEvolutionStrategy(x0, sigma0, {"popsize": popsize, "verb_disp": 0})
    best_so_far, best_iter = -np.inf, 0
    iter_vals, best_vals = [], []

    for it in tqdm(range(N_ITER), desc=tag, leave=False):
        X = [np.clip(x, -RANGE, RANGE) for x in es.ask()]
        Y = [eval_selectivity(x, target_point) for x in X]
        es.tell(X, Y)
        iter_best = -np.min(Y)
        iter_best_idx = int(np.argmin(Y))
        iter_best_currents = X[iter_best_idx]

        iter_vals.append(iter_best)
        if iter_best > best_so_far:
            best_so_far, best_iter = iter_best, it + 1
        best_vals.append(best_so_far)

        save_live_plot(np.arange(1, len(iter_vals)+1), iter_vals, best_vals, tag, target_point)


        pd.DataFrame([{
            "optimizer": "CMAES",
            "sigma0": sigma0,
            "popsize": popsize,
            "repeat": repeat,
            "iteration": it + 1,
            "iteration_selectivity": iter_best,
            "best_so_far": best_so_far,
            "iteration_best_found": best_iter,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
            "currents": iter_best_currents.tolist()
        }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
        header_written = True

    return {
        "optimizer": "CMAES",
        "tag": tag,
        "best": best_so_far,
        "iter": best_iter,
        "sigma0": sigma0,
        "popsize": popsize,
        "repeat": repeat,
        "target_point": target_point,
    }

def run_cma_sweep_experiment(sigma0, popsize, repeat, target_point):
    """
    CMA-ES optimisation with an initial electrode sweep phase.
    The sweep stimulates each electrode individually (± amplitude) to find a
    physics-informed starting point for CMA-ES, improving convergence stability.
    """

    tag = f"CMAES_s{sigma0:.1e}_p{popsize}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}_progress.csv")
    header_written = False
    sweep_amplitude = 0.5e-3  # 0.5 mA single-electrode sweep amplitude
    best_so_far, best_iter = -np.inf, 0
    iter_vals, best_vals = [], []

    # ==============================================================
    # 1. ELECTRODE SWEEP INITIALISATION
    # ==============================================================
    sweep_results = []
    print(f"\n[{tag}] Starting electrode sweep initialisation...")

    for i in range(N_ELECTRODES):
        for sign in [+1, -1]:
            currents = np.zeros(N_ELECTRODES)
            currents[i] = sign * sweep_amplitude
            val = -eval_selectivity(currents, target_point)  # negative (minimiser)
            sweep_results.append((currents, val))
            pd.DataFrame([{
                "optimizer": "CMAES_SWEEP",
                "sigma0": sigma0,
                "popsize": popsize,
                "repeat": repeat,
                "phase": "sweep",
                "electrode": i,
                "sign": sign,
                "iteration": len(sweep_results),
                "iteration_selectivity": -val,
                "best_so_far": np.nan,
                "iteration_best_found": np.nan,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                "currents": currents.tolist()
            }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
            header_written = True

    best_sweep_currents, best_sweep_val = max(sweep_results, key=lambda x: x[1])
    x0 = best_sweep_currents.copy()
    print(f"[{tag}] Sweep done. Best electrode pattern selectivity = {-best_sweep_val:.3f}")

    # ==============================================================
    # 2. CMA-ES INITIALISATION (starts from best sweep pattern)
    # ==============================================================
    es = CMAEvolutionStrategy(x0, sigma0, {"popsize": popsize, "verb_disp": 0})

    # ==============================================================
    # 3. CMA-ES MAIN OPTIMISATION LOOP
    # ==============================================================
    for it in tqdm(range(N_ITER-N_ELECTRODES), desc=tag, leave=False):
        X = [np.clip(x, -RANGE, RANGE) for x in es.ask()]
        Y = [eval_selectivity(x, target_point) for x in X]
        es.tell(X, Y)

        iter_best = -np.min(Y)
        iter_best_idx = int(np.argmin(Y))
        iter_best_currents = X[iter_best_idx]
        iter_vals.append(iter_best)

        if iter_best > best_so_far:
            best_so_far, best_iter = iter_best, it + 1
        best_vals.append(best_so_far)

        # Plot progress
        save_live_plot(np.arange(1, len(iter_vals) + 1), iter_vals, best_vals, tag, target_point)

        # Log to CSV
        pd.DataFrame([{
            "optimizer": "CMAES",
            "sigma0": sigma0,
            "popsize": popsize,
            "repeat": repeat,
            "phase": "optimization",
            "iteration": it + 1,
            "iteration_selectivity": iter_best,
            "best_so_far": best_so_far,
            "iteration_best_found": best_iter,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
            "currents": iter_best_currents.tolist()
        }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
        header_written = True

    return {
        "optimizer": "CMAES",
        "tag": tag,
        "best": best_so_far,
        "iter": best_iter,
        "sigma0": sigma0,
        "popsize": popsize,
        "repeat": repeat,
        "target_point": target_point,
    }


def run_cma_montage_reduced(sigma0, popsize, repeat, target_point):
    """
    CMA-ES optimisation in a reduced montage space:
      1) Sweep all bipolar/tripolar montages
      2) Keep top-K most selective
      3) Optimise montage weights via CMA-ES (objective = negative selectivity)
    """
    import numpy as np, cma, pandas as pd
    from tqdm import tqdm

    tag = f"CMAES_MONTAGE_s{sigma0:.1e}_p{popsize}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}_progress.csv")
    header_written = False

    # ---------- 1. Build montage templates ----------
    montages = []
    for i in range(N_ELECTRODES):
        # Bipolar (+1, -1)
        pair = np.zeros(N_ELECTRODES); pair[i] = +1; pair[(i + 1) % N_ELECTRODES] = -1
        montages.append(pair)
        # Tripolar (+1, -2, +1)
        trip = np.zeros(N_ELECTRODES); trip[i] = +1; trip[(i + 1) % N_ELECTRODES] = -2; trip[(i + 2) % N_ELECTRODES] = +1
        montages.append(trip)
    montages = np.array(montages)

    # ---------- 2. Initial sweep (for ranking templates only) ----------
    sweep_amplitudes = (0.5, 1.0, 2.0)  # mA
    sweep_results = []
    print(f"[{tag}] Sweeping {len(montages)} montages × {len(sweep_amplitudes)} amplitudes...")

    for mid, pattern in enumerate(montages):
        for amp_mA in sweep_amplitudes:
            currents = amp_mA * 1e-3 * pattern
            # objective for CMA is NEGATIVE selectivity:
            obj = eval_selectivity(np.clip(currents, -RANGE, RANGE), target_point)   # obj = -S
            sel = -obj  # positive selectivity for logging/ranking
            sweep_results.append((mid, amp_mA, sel))
            pd.DataFrame([{
                "optimizer": "CMAES_MONTAGE",
                "phase": "sweep",
                "montage_id": mid,
                "amplitude_mA": amp_mA,
                "selectivity": float(sel),
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
            header_written = True

    # Rank by positive selectivity
    sweep_results.sort(key=lambda x: x[2], reverse=True)
    top_k = 6
    top_templates = [montages[idx] for idx, _, _ in sweep_results[:top_k]]
    print(f"[{tag}] Selected top-{top_k} montages for optimisation.")

    # ---------- 3. CMA-ES on montage weights (objective = negative selectivity) ----------
    dim = len(top_templates)
    x0 = np.zeros(dim)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": popsize, "verb_disp": 0})

    best_so_far, best_iter = -np.inf, 0
    iter_vals, best_vals = [], []

    for it in tqdm(range(N_ITER), desc=tag, leave=False):
        weights_list = es.ask()
        Y = []  # values to MINIMIZE (keep as negative selectivity)
        combos = []

        for w in weights_list:
            # Combine montages linearly and clip to range
            currents = np.sum([w[j] * top_templates[j] for j in range(dim)], axis=0)
            currents = np.clip(currents, -RANGE, RANGE)
            obj = eval_selectivity(currents, target_point)  # NEGATIVE selectivity
            Y.append(obj)
            combos.append(currents)

        es.tell(weights_list, Y)

        # Convert back to positive selectivity for reporting
        iter_best = -np.min(Y)
        iter_best_idx = int(np.argmin(Y))
        iter_best_currents = combos[iter_best_idx]

        iter_vals.append(iter_best)
        if iter_best > best_so_far:
            best_so_far, best_iter = iter_best, it + 1
        best_vals.append(best_so_far)

        save_live_plot(np.arange(1, len(iter_vals)+1), iter_vals, best_vals, tag, target_point)

        pd.DataFrame([{
            "optimizer": "CMAES_MONTAGE",
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
            "weights": weights_list[iter_best_idx].tolist(),
            "currents": iter_best_currents.tolist()
        }]).to_csv(csv_path, mode="a", index=False, header=False)

    return {
        "optimizer": "CMAES_MONTAGE",
        "tag": tag,
        "best": float(best_so_far),
        "iter": int(best_iter),
        "sigma0": sigma0,
        "popsize": popsize,
        "repeat": repeat,
        "target_point": target_point,
    }





# ==============================================================
# BO FULL
# ==============================================================
def run_bo_experiment(acq_func, kappa, xi, n_init, repeat, target_point):
    tag = f"BO_{acq_func}_k{kappa}_x{xi}_n{n_init}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}_progress.csv")
    header_written = False
    eps = 1e-9

    space = [(-RANGE, RANGE)] * N_ELECTRODES
    opt = SkOptimizer(
        space,
        base_estimator="GP",
        acq_func=acq_func,
        acq_func_kwargs={"kappa": kappa, "xi": xi},
        random_state=repeat,
    )

    best_so_far, best_iter = -np.inf, 0
    iter_vals, best_vals = [], []

    for it in tqdm(range(N_ITER), desc=tag, leave=False):
        x = np.clip(opt.ask(), -RANGE + eps, RANGE - eps)
        y = eval_selectivity(x, target_point)
        try:
            opt.tell(list(x), float(y))
        except ValueError:
            x = np.clip(x, -RANGE + 1e-6, RANGE - 1e-6)
            opt.tell(list(x), float(y))

        iter_sel = -y
        iter_vals.append(iter_sel)
        if iter_sel > best_so_far:
            best_so_far, best_iter = iter_sel, it + 1
        best_vals.append(best_so_far)

        save_live_plot(np.arange(1, len(iter_vals)+1), iter_vals, best_vals, tag, target_point)


        pd.DataFrame([{
            "optimizer": "BO",
            "acq_func": acq_func,
            "kappa": kappa,
            "xi": xi,
            "n_init": n_init,
            "repeat": repeat,
            "iteration": it + 1,
            "iteration_selectivity": iter_sel,
            "best_so_far": best_so_far,
            "iteration_best_found": best_iter,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
            "currents": x.tolist()
        }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
        header_written = True

    return {
        "optimizer": "BO",
        "tag": tag,
        "best": best_so_far,
        "iter": best_iter,
        "acq_func": acq_func,
        "kappa": kappa,
        "xi": xi,
        "n_init": n_init,
        "repeat": repeat,
        "target_point": target_point,
    }

# ==============================================================
# MAIN BENCHMARK
# ==============================================================
if __name__ == "__main__":
    start = time.time()
    all_runs = []


    #print("\n=== PHASE 1: Sweep CMA-ES ===")
    #for sigma0, popsize in product(SIGMAS, POPSIZES):
    #    for r, tp in product(range(REPEATS), TARGET_POINTS):
    #        all_runs.append(run_cma_sweep_experiment(sigma0, popsize, r, tp))

   # print("\n=== PHASE 2: CMA-ES ===")
   # for sigma0, popsize in product(SIGMAS, POPSIZES):
    #    for r, tp in product(range(REPEATS), TARGET_POINTS):
     #       all_runs.append(run_cma_experiment(sigma0, popsize, r, tp))

    print("\n=== PHASE 3: Montage-Reduced CMA-ES ===")
    for sigma0, popsize in product(SIGMAS, POPSIZES):
        for r, tp in product(range(REPEATS), TARGET_POINTS):
            all_runs.append(run_cma_montage_reduced(sigma0, popsize, r, tp))

    # print("\n=== PHASE 2: Full-space Bayesian Optimisation ===")
    # for acq, kappa, xi, n_init in product(ACQ_FUNCS, KAPPAS, XIS, N_INIT_POINTS):
    #     for r, tp in product(range(REPEATS), TARGET_POINTS):
    #         all_runs.append(run_bo_experiment(acq, kappa, xi, n_init, r, tp))

    df = pd.DataFrame(all_runs)
    summary_path = os.path.join(OUTPUT_DIR, "optimizer_summary.csv")
    df.to_csv(summary_path, index=False)


