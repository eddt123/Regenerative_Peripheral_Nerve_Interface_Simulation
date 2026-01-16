#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # prevent the crash 

import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from scipy import stats

# --- Simulator import ---
try:
    from utils.run_selectivity_simulation import run_selectivity_simulation
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from run_selectivity_simulation import run_selectivity_simulation  # type: ignore

# --- Optimizers ---
from cma import CMAEvolutionStrategy
from skopt import Optimizer as SkOptimizer

# ======================================================================
# CONFIG
# ======================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "benchmark_paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tissue / geometry
RADIUS   = 0.01   # m
HEIGHT   = 0.04   # m
SIGMA_T  = 0.08   # S/m Nerve Rahman et al 2023

# Electrode layouts (N = n_rows * n_per_row)
ELECTRODE_GRIDS = [
    (4, 2),   # N=8
    #(2,2),
    #(3,2),
    #(3,3),
    (4, 3),   # N=12
    (4, 4),   # N=16
    (5,5),
    #(6,6)
]

# Currents
RANGE     = 1e-3         # ±1 mA bounds
ZERO_SUM  = False         # no longer used as this is set in the opt functions

# Targets (in metres)
TARGET_POINTS = [
    # --- Original 3 ---
    (0.0,    0.0,    0.0),      # centre of cylinder
    (0.002,  0.0035, 0.010),    # upper, off-centre (≈4.0 mm radially)
    #(-0.003, -0.002, -0.015),   # lower, off-centre (≈3.6 mm radially)

    # --- Mid-plane ring (z = 0), different quadrants & radii ---
    (0.005,   0.0,    0.0),     # +x direction, 5 mm radius
    #(0.0,    -0.005,  0.0),     # -y direction, 5 mm radius
    #(-0.004,  0.004,  0.0),     # quadrant II, ≈5.7 mm radius
    #(0.0035,  0.006,  0.0),     # quadrant I, ≈6.9 mm radius

    # --- Upper half (z > 0), varied angles/radii ---
    (0.004,   0.002,  0.015),   # upper, ≈4.5 mm radius
    #(0.0,     0.006,  0.012),   # upper, +y, 6 mm radius
    #(-0.005,  0.0,    0.018),   # upper, -x, 5 mm radius

    # --- Lower half (z < 0), varied angles/radii ---
    #(0.0025, -0.0045, -0.012),  # lower, ≈5.1 mm radius
    #(-0.006, -0.003,  -0.018),  # lower, ≈6.7 mm radius
]


# Budget: DIMENSION-SCALED (key fix #1)
EVALS_PER_DIM = 200      # 200*N evaluations per run
REPEATS = 3             
SEED_BASE = 42

# CMA-ES: DIMENSION-SCALED population
CMA_SIGMA0 = 0.3e-3
def cma_popsize(N):
    """Standard CMA-ES population size: 4 + floor(3*ln(N))"""
    return 4 + int(3 * np.log(N))

# BO: SEQUENTIAL
BO_ACQ = "EI"
BO_KAPPA = 1.96  # ~95% confidence for LCB
BO_XI = 0.01
def bo_n_initial(N):
    """Initial random points for BO: max of 2*N or 10"""
    return max(2*N, 10)

# PSO: DIMENSION-SCALED population (key fix #4)
def pso_popsize(N):
    """Common PSO population: 10 + 2*sqrt(N)"""
    return 10 + int(2 * np.sqrt(N))

PSO_W = 0.729    # Clerc's constriction coefficient
PSO_C1 = 1.49445
PSO_C2 = 1.49445
PSO_VCLAMP = 0.5 * RANGE

# ======================================================================
# EXTRA OPTIMIZERS: DE, CEM, SA (dimension-scaled, fair budgets)
# ======================================================================

# ---- Differential Evolution (DE) ----
def de_population_size(N):
    """SciPy convention: population size = pop_factor * N (candidate vectors)."""
    return max(10 * N, 40)     # classic 10*N with a small floor for very small N

DE_F  = 0.8                    # differential weight
DE_CR = 0.9                    # crossover rate

# ---- Cross-Entropy Method (CEM) ----
def cem_popsize(N):
    """Samples per iteration."""
    return max(8 * N, 64)

CEM_ELITE_FRAC = 0.2           # top fraction kept
CEM_ALPHA      = 0.7           # smoothing (0..1)
CEM_SIGMA0     = 0.35e-3       # initial per-dimension std
CEM_SIGMA_MIN  = 1e-6          # floor to avoid collapse

# ---- Simulated Annealing (SA) ----
SA_ALPHA        = 0.985        # geometric cooling factor per step
SA_STEP_SIGMA   = 0.25 * RANGE # proposal step std (per-dim) before projection
SA_WARMUP_SAMPLES = 40         # to set T0 fairly (deducted from budget)


# ======================================================================
# PROJECTION: exact zero-sum + box
# ======================================================================
def project_zero_sum_box(x, lo, hi, tol=1e-12, max_iter=100):
    """Bisection on Lagrange multiplier for exact projection."""
    x = np.asarray(x, dtype=float)
    lo = float(lo); hi = float(hi)
    
    if hi <= lo + 1e-18:
        y = np.clip(x, lo, hi)
        y -= np.mean(y)
        return np.clip(y, lo, hi)

    lam_lo = np.min(x - hi)
    lam_hi = np.max(x - lo)

    def f(lam):
        return np.clip(x - lam, lo, hi).sum()

    f_lo = f(lam_lo)
    f_hi = f(lam_hi)
    
    if abs(f_lo) < tol:
        return np.clip(x - lam_lo, lo, hi)
    if abs(f_hi) < tol:
        return np.clip(x - lam_hi, lo, hi)

    if f_lo < 0 and f_hi < 0:
        lam_hi = lam_lo + 1e-12
        f_hi = f(lam_hi)
    if f_lo > 0 and f_hi > 0:
        lam_lo = lam_hi - 1e-12
        f_lo = f(lam_lo)

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        f_mid = f(lam_mid)
        if abs(f_mid) < tol or (lam_hi - lam_lo) < tol:
            return np.clip(x - lam_mid, lo, hi)
        if f_mid > 0:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    lam_mid = 0.5 * (lam_lo + lam_hi)
    return np.clip(x - lam_mid, lo, hi)

def project_currents(x, lo=-RANGE, hi=RANGE, zero_sum=ZERO_SUM):
    """Projection wrapper."""
    x = np.asarray(x, dtype=float)
    if zero_sum:
        return project_zero_sum_box(x, lo, hi)
    else:
        return np.clip(x, lo, hi)

# ======================================================================
# EVALUATION
# ======================================================================
def eval_selectivity(x, target_point, grid, rng_seed=None):
    n_rows, n_per_row = grid
    x_orig = np.array(x, dtype=float)
    x_proj = project_currents(x_orig, -RANGE, RANGE, ZERO_SUM)
    
    res = run_selectivity_simulation(
        n_rows=n_rows, n_per_row=n_per_row,
        currents=x_proj,
        target_point=target_point,
        radius=RADIUS, height=HEIGHT, sigma=SIGMA_T,
        n_off_samples=1200, metric="activation",
        grounded_boundary=True, R_outer=0.10,
        rng=(SEED_BASE if rng_seed is None else rng_seed),
    )
    return -float(res["selectivity"]), x_orig, x_proj


def eval_selectivity_grounded(x, target_point, grid, rng_seed=None):
    """
    Evaluate selectivity with grounded outer boundary, NO zero-sum constraint.
    Only simple box clipping to [-RANGE, RANGE].
    Returns:
        y   : NEGATIVE selectivity (for minimizers)
        x_c: clipped currents actually used (N-dim)
    """
    n_rows, n_per_row = grid
    x_c = np.asarray(x, dtype=float)
    x_c = np.clip(x_c, -RANGE, RANGE)

    res = run_selectivity_simulation(
        n_rows=n_rows,
        n_per_row=n_per_row,
        currents=x_c,
        target_point=target_point,
        radius=RADIUS,
        height=HEIGHT,
        sigma=SIGMA_T,
        n_off_samples=1200,
        metric="activation",
        grounded_boundary=True,
        R_outer=0.10,
        rng=(SEED_BASE if rng_seed is None else rng_seed),
    )
    # OPTIMIZERS MINIMISE: objective = -selectivity
    return -float(res["selectivity"]), x_c


# ======================================================================
# UTILITIES
# ======================================================================

def make_tag(prefix, grid, repeat, tp):
    n_rows, n_per_row = grid
    tx, ty, tz = tp
    return (f"{prefix}_N{n_rows*n_per_row}_grid{n_rows}x{n_per_row}_r{repeat}"
            f"_x{tx*1e3:.1f}_y{ty*1e3:.1f}_z{tz*1e3:.1f}")

def save_progress_plot(xs, vals, bests, tag, target_point):
    plt.figure(figsize=(7, 4))
    plt.plot(xs, vals, 'o-', alpha=0.6, ms=3, lw=1, label='Iteration best')
    plt.plot(xs, bests, '-', lw=2, label='Best-so-far')
    plt.xlabel("Function evaluations")
    plt.ylabel("Selectivity")
    tx, ty, tz = target_point
    plt.title(f"{tag}\nTarget: ({tx*1e3:.1f}, {ty*1e3:.1f}, {tz*1e3:.1f}) mm", fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_progress.png"), dpi=150)
    plt.close()

def log_step(csv_path, data_dict, header_written):
    """Helper to log one step to CSV."""
    pd.DataFrame([data_dict]).to_csv(
        csv_path, mode="a", index=False, header=not header_written
    )

# ======================================================================
# LANDSCAPE CHARACTERISATION
# ======================================================================
def characterize_landscape(grid, target_point, n_samples=200, seed=999):
    """
    Sample random feasible points to estimate:
    - Mean and variance of objective
    - Correlation length (estimate)
    - Approximate multimodality
    """
    N = grid[0] * grid[1]
    rng = np.random.default_rng(seed)
    
    values = []
    points = []
    for _ in range(n_samples):
        x = rng.uniform(-RANGE, RANGE, N)
        y, _, _ = eval_selectivity(x, target_point, grid, rng_seed=seed)
        values.append(-y)  # Convert back to selectivity (positive)
        points.append(x)
    
    values = np.array(values)
    points = np.array(points)
    
    # Basic statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    median_val = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    
    # Estimate local optima count (crude: count peaks in sorted samples)
    sorted_vals = np.sort(values)
    diffs = np.diff(sorted_vals)
    large_gaps = np.sum(diffs > std_val)  # Rough multimodality indicator
    
    # Estimate correlation (distance vs value similarity)
    # Sample 50 pairs
    n_pairs = min(50, n_samples // 2)
    correlations = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_samples, 2, replace=False)
        dist = np.linalg.norm(points[i] - points[j])
        val_diff = abs(values[i] - values[j])
        if dist > 1e-9:
            correlations.append((dist, val_diff))
    
    if correlations:
        dists, val_diffs = zip(*correlations)
        # Negative correlation suggests smooth landscape
        corr_coef = np.corrcoef(dists, val_diffs)[0, 1] if len(dists) > 1 else 0.0
    else:
        corr_coef = 0.0
    
    return {
        "N": N,
        "grid": grid,
        "target": target_point,
        "mean_selectivity": mean_val,
        "std_selectivity": std_val,
        "median_selectivity": median_val,
        "q25_selectivity": q25,
        "q75_selectivity": q75,
        "approx_multimodality": large_gaps,
        "distance_value_correlation": corr_coef,
        "n_samples": n_samples,
    }

# ======================================================================
# OPTIMISATION MODELS
# ======================================================================


# ---------------------------------------
# Baseline PSO (grounded, no sweep)
# ---------------------------------------
def run_pso_grounded(
    grid, repeat, target_point, eval_budget,
    eval_seed: int,
):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    # Optimizer randomness (separate from eval_seed)
    algo_seed = SEED_BASE + 3000 * repeat + 97 * N
    rng = np.random.default_rng(algo_seed)

    tag = make_tag("PSO", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    popsize = pso_popsize(N)
    iters = max(1, eval_budget // popsize)

    # init swarm
    X = np.clip(rng.uniform(-RANGE, RANGE, size=(popsize, N)), -RANGE, RANGE)
    V = np.zeros_like(X)

    # eval initial
    pbest_pos = X.copy()
    pbest_val = np.empty(popsize, dtype=float)

    evals_so_far = 0
    for i in range(popsize):
        if evals_so_far >= eval_budget:
            break
        y, x_used = eval_selectivity_grounded(pbest_pos[i], target_point, grid, rng_seed=eval_seed)
        evals_so_far += 1
        pbest_val[i] = y
        pbest_pos[i] = x_used
        X[i] = x_used

    g_idx = int(np.argmin(pbest_val[:max(1, min(popsize, evals_so_far))]))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    best_so_far = -gbest_val
    best_at_eval = evals_so_far

    xs_axis, step_vals, best_vals = [evals_so_far], [best_so_far], [best_so_far]
    header_written = False

    log_step(csv_path, {
        "optimizer": "PSO_GROUNDED",
        "stage": 1,
        "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
        "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": evals_so_far,
        "step_best_selectivity": best_so_far,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
        "currents_gen_best": gbest_pos.tolist(),
        "currents_best_so_far": gbest_pos.tolist(),
        "eval_seed": eval_seed,
        "algo_seed": algo_seed,
    }, header_written)
    header_written = True

    # main loop
    for it in tqdm(range(1, iters), desc=f"{tag}_pso", leave=False):
        if evals_so_far >= eval_budget:
            break

        r1 = rng.random(size=(popsize, N))
        r2 = rng.random(size=(popsize, N))
        V = PSO_W * V + PSO_C1 * r1 * (pbest_pos - X) + PSO_C2 * r2 * (gbest_pos - X)  # main velocity equation
        if PSO_VCLAMP is not None:
            V = np.clip(V, -PSO_VCLAMP, PSO_VCLAMP)

        X = np.clip(X + V, -RANGE, RANGE)

        it_best_sel = -np.inf
        it_best_curr = None

        for i in range(popsize):
            if evals_so_far >= eval_budget:
                break

            y, x_used = eval_selectivity_grounded(X[i], target_point, grid, rng_seed=eval_seed)
            evals_so_far += 1
            X[i] = x_used
            sel = -y

            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = x_used

            if sel > it_best_sel:
                it_best_sel = sel
                it_best_curr = x_used.copy()

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                gbest_pos = x_used.copy()

        # update global best from pbests (consistent PSO)
        g_idx = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_val = pbest_val[g_idx]

        xs_axis.append(evals_so_far)
        step_vals.append(it_best_sel if np.isfinite(it_best_sel) else best_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "PSO_GROUNDED",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": (it_best_sel if np.isfinite(it_best_sel) else best_so_far),
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "currents_gen_best": (it_best_curr.tolist() if it_best_curr is not None else None),
            "currents_best_so_far": gbest_pos.tolist(),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "PSO_GROUNDED",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "popsize": int(popsize),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
    }



# ---------------------------------------
# Baseline CMA-ES (grounded, no sweep)
# ---------------------------------------
def run_cma_grounded(
    grid, repeat, target_point, eval_budget,
    eval_seed: int,
    sigma0: float = CMA_SIGMA0,
):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    # Optimizer randomness (separate from eval_seed)
    algo_seed = SEED_BASE + 1000 * repeat + 131 * N
    rng = np.random.default_rng(algo_seed)

    tag = make_tag("CMA_GROUNDED", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    popsize = max(4, cma_popsize(N))  # safety floor

    x0 = rng.uniform(-RANGE, RANGE, size=N)
    es = CMAEvolutionStrategy(
        x0, sigma0,
        {
            "popsize": popsize,
            "verb_disp": 0,
            "seed": algo_seed,
            "bounds": [-RANGE, RANGE],
            "CMA_mirrors": 0,  # optional robustness
        }
    )

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = None

    xs_axis, step_vals, best_vals = [], [], []
    header_written = False
    step_index = 0

    # only run full CMA generations
    while (evals_so_far + popsize) <= eval_budget:
        step_index += 1
        X_ask = es.ask()

        X_eval, Y = [], []
        gen_best = -np.inf
        gen_best_curr = None

        for x in X_ask:
            x = np.clip(np.asarray(x, float), -RANGE, RANGE)
            y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=eval_seed)
            evals_so_far += 1

            X_eval.append(x_used)
            Y.append(y)

            sel = -y
            if sel > gen_best:
                gen_best = sel
                gen_best_curr = x_used.copy()

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_used.copy()

        
        es.tell(X_eval, Y)

        xs_axis.append(evals_so_far)
        step_vals.append(gen_best if np.isfinite(gen_best) else best_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "CMA_GROUNDED",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "sigma0": sigma0, "popsize": popsize,
            "repeat": repeat,
            "step_index": step_index,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": (gen_best if np.isfinite(gen_best) else best_so_far),
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "currents_gen_best": (gen_best_curr.tolist() if gen_best_curr is not None else None),
            "currents_best_so_far": (best_currents.tolist() if best_currents is not None else None),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)
        header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "CMA_GROUNDED",
        "tag": tag,
        "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "popsize": int(popsize),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
    }



# -------------------------------------------------------
# Fair MS: sweep -> PSO (same sweep fraction as CMA)
# -------------------------------------------------------
# -----------------------------
# Helper: EXHAUSTIVE pair-sweep stage (grounded, no zero-sum)
# -----------------------------
def run_pair_sweep_grounded(
    grid, repeat, target_point, eval_budget,
    eval_seed: int,
    I0=0.8e-3,
    tag_prefix="PAIRSWEEP",
    csv_path=None,
    header_written=False,
    assert_within_budget=True,
):
    """
    Exhaustive sweep over ALL unordered electrode pairs (i < j)
    using dipolar pattern: +I0 at i, -I0 at j, others 0.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs_total = len(all_pairs)

    if assert_within_budget and n_pairs_total > eval_budget:
        raise ValueError(
            f"Pair sweep needs {n_pairs_total} evals, but eval_budget={eval_budget}. "
            f"Increase budget or disable assert_within_budget."
        )

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, step_vals, best_vals = [], [], []
    pair_results = []

    for idx, (i, j) in enumerate(tqdm(all_pairs, desc=f"{tag_prefix}_stage0_pairsweep", leave=False)):
        if evals_so_far >= eval_budget:
            break

        currents = np.zeros(N, dtype=float)
        currents[i] = +I0
        currents[j] = -I0

        y, x_used = eval_selectivity_grounded(currents, target_point, grid, rng_seed=eval_seed)
        sel = -y
        evals_so_far += 1

        pair_results.append((sel, x_used.copy()))

        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()

        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)

        # sparse logging
        if csv_path is not None and (idx == 0 or idx == n_pairs_total - 1 or ((idx + 1) % 10 == 0)):
            row = {
                "optimizer": tag_prefix,
                "stage": 0,
                "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                "repeat": repeat,
                "step_index": idx + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
                "pair_i": i, "pair_j": j,
                "currents_gen_best": x_used.tolist(),         # “suggested” this step
                "currents_best_so_far": best_currents.tolist(),
                "I0": I0,
                "eval_seed": eval_seed,
            }
            log_step(csv_path, row, header_written)
            header_written = True

    return (pair_results, evals_so_far, best_so_far, best_at_eval, best_currents,
            xs_axis, step_vals, best_vals, header_written)



# -------------------------------------------------------
# Fair MS: EXHAUSTIVE sweep -> PSO
# -------------------------------------------------------
def run_ms_sweep_then_pso_fair(
    grid, repeat, target_point, eval_budget,
    eval_seed: int,
    I0=0.8e-3,
    top_k_pairs=3,
    assert_sweep_within_budget=True,
):
    """
    Stage 0: exhaustive pair sweep over ALL unordered pairs (i<j), +/-I0.
    Stage 1: PSO warm-started with top-K pair patterns as initial particles,
             remainder random.
    FAIRNESS: ALL objective evals use eval_seed.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    # Optimizer randomness only
    algo_seed = SEED_BASE + 9100 * repeat + 97 * N
    rng = np.random.default_rng(algo_seed)

    tag = make_tag("MS_SWEEP_PSO_FAIR", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # Stage 0: exhaustive sweep (FAIR: eval_seed)
    (pair_results, evals_so_far, best_so_far, best_at_eval, best_currents,
     xs_axis, step_vals, best_vals, header_written) = run_pair_sweep_grounded(
        grid, repeat, target_point, eval_budget,
        eval_seed=eval_seed,
        I0=I0,
        tag_prefix="SWEEP_PSO",
        csv_path=csv_path,
        header_written=False,
        assert_within_budget=assert_sweep_within_budget,
    )

    if evals_so_far >= eval_budget or len(pair_results) == 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "SWEEP_PSO",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "eval_seed": int(eval_seed),
            "algo_seed": int(algo_seed),
        }

    remaining = eval_budget - evals_so_far
    popsize = pso_popsize(N)

    # Need at least one full swarm evaluation to be consistent & simple
    if remaining < popsize:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "SWEEP_PSO",
            "tag": tag,
            "best": float(best_so_far),
            "best_found_at_eval": int(best_at_eval),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "eval_seed": int(eval_seed),
            "algo_seed": int(algo_seed),
        }

    # Full swarm iterations only (ignore leftover < popsize)
    iters = remaining // popsize

    # Warm-start: top-K pair vectors
    pair_results.sort(key=lambda t: t[0], reverse=True)
    K = min(top_k_pairs, popsize, len(pair_results))

    X = np.zeros((popsize, N), dtype=float)
    for k in range(K):
        X[k] = np.clip(pair_results[k][1], -RANGE, RANGE)
    for i in range(K, popsize):
        X[i] = rng.uniform(-RANGE, RANGE, N)
    X = np.clip(X, -RANGE, RANGE)

    V = np.zeros_like(X)
    pbest_pos = X.copy()
    pbest_val = np.empty(popsize, dtype=float)

    # Iteration 0: evaluate whole swarm
    for i in range(popsize):
        y, x_used = eval_selectivity_grounded(pbest_pos[i], target_point, grid, rng_seed=eval_seed)
        pbest_val[i] = y
        pbest_pos[i] = x_used
        X[i] = x_used

    evals_so_far += popsize

    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    swarm_best = -gbest_val
    if swarm_best > best_so_far:
        best_so_far = swarm_best
        best_at_eval = evals_so_far
        best_currents = gbest_pos.copy()

    xs_axis.append(evals_so_far)
    step_vals.append(swarm_best)
    best_vals.append(best_so_far)

    log_step(csv_path, {
        "optimizer": "SWEEP_PSO",
        "stage": 1,
        "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
        "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": evals_so_far,
        "step_best_selectivity": swarm_best,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
        "I0": I0, "top_k_pairs": top_k_pairs,
        "currents": best_currents.tolist(),
        "eval_seed": eval_seed,
        "algo_seed": algo_seed,
    }, header_written)
    header_written = True

    # Remaining full iterations
    for it in tqdm(range(1, iters), desc=f"{tag}_stage1_pso", leave=False):
        # velocity + position update
        r1 = rng.random(size=(popsize, N))
        r2 = rng.random(size=(popsize, N))
        V = PSO_W * V + PSO_C1 * r1 * (pbest_pos - X) + PSO_C2 * r2 * (gbest_pos - X)
        if PSO_VCLAMP is not None:
            V = np.clip(V, -PSO_VCLAMP, PSO_VCLAMP)

        X = np.clip(X + V, -RANGE, RANGE)

        it_best_sel = -np.inf
        it_best_curr = None

        # evaluate full swarm (FAIR: eval_seed)
        for i in range(popsize):
            y, x_used = eval_selectivity_grounded(X[i], target_point, grid, rng_seed=eval_seed)
            X[i] = x_used
            evals_so_far += 1
            sel = -y

            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = x_used

            if sel > it_best_sel:
                it_best_sel = sel
                it_best_curr = x_used.copy()

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_used.copy()

        g_idx = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_val = pbest_val[g_idx]

        xs_axis.append(evals_so_far)
        step_vals.append(it_best_sel if np.isfinite(it_best_sel) else best_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "SWEEP_PSO",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": (it_best_sel if np.isfinite(it_best_sel) else best_so_far),
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "I0": I0, "top_k_pairs": top_k_pairs,
            "currents": (it_best_curr.tolist() if it_best_curr is not None else None),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "SWEEP_PSO",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "popsize": int(popsize),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
    }



# -------------------------------------------------------
# Fair MS: EXHAUSTIVE sweep -> CMA
# -------------------------------------------------------
def run_ms_sweep_then_cma_fair(
    grid, repeat, target_point, eval_budget,
    eval_seed: int,
    I0=0.8e-3,
    top_k_pairs=3,
    assert_sweep_within_budget=True,
):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    # Optimizer randomness only
    algo_seed = SEED_BASE + 7000 * repeat + 157 * N
    rng = np.random.default_rng(algo_seed)

    tag = make_tag("SWEEP_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # Stage 0: exhaustive sweep (FAIR: eval_seed)
    (pair_results, evals_so_far, best_so_far, best_at_eval, best_currents,
     xs_axis, step_vals, best_vals, header_written) = run_pair_sweep_grounded(
        grid, repeat, target_point, eval_budget,
        eval_seed=eval_seed,
        I0=I0,
        tag_prefix="SWEEP_CMA",
        csv_path=csv_path,
        header_written=False,
        assert_within_budget=assert_sweep_within_budget,
    )

    if evals_so_far >= eval_budget or len(pair_results) == 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "SWEEP_CMA",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "eval_seed": int(eval_seed),
            "algo_seed": int(algo_seed),
        }

    # Warm-start from top-K
    pair_results.sort(key=lambda t: t[0], reverse=True)
    k = min(top_k_pairs, len(pair_results))
    X_top = np.stack([c for (_, c) in pair_results[:k]], axis=0)

    x0 = np.clip(np.mean(X_top, axis=0), -RANGE, RANGE)
    sigma_est = np.std(X_top, axis=0).mean() if k > 1 else 0.3 * RANGE
    sigma0 = max(1e-6, min(0.5 * RANGE, max(0.1 * RANGE, sigma_est)))

    popsize = max(4, cma_popsize(N))  # safety floor

    remaining = eval_budget - evals_so_far
    # Need at least one full CMA generation to avoid partial tell()
    if remaining < popsize:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "SWEEP_CMA",
            "tag": tag,
            "best": float(best_so_far),
            "best_found_at_eval": int(best_at_eval),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "eval_seed": int(eval_seed),
            "algo_seed": int(algo_seed),
        }

    es = CMAEvolutionStrategy(
        x0, sigma0,
        {
            "popsize": popsize,
            "verb_disp": 0,
            "seed": algo_seed + 1,
            "bounds": [-RANGE, RANGE],
            "CMA_mirrors": 0,  # optional robustness
        }
    )

    step_index = 0
    # IMPORTANT: only run full CMA generations
    while (evals_so_far + popsize) <= eval_budget:
        step_index += 1
        X_ask = es.ask()

        X_eval, Y = [], []
        gen_best = -np.inf
        gen_best_curr = None

        for x in X_ask:
            x = np.clip(np.asarray(x, float), -RANGE, RANGE)

            # FAIR: same eval_seed as sweep
            y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=eval_seed)
            evals_so_far += 1

            X_eval.append(x_used)
            Y.append(y)

            sel = -y
            if sel > gen_best:
                gen_best = sel
                gen_best_curr = x_used.copy()

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_used.copy()

        es.tell(X_eval, Y)

        xs_axis.append(evals_so_far)
        step_vals.append(gen_best if np.isfinite(gen_best) else best_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "SWEEP_CMA",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "repeat": repeat,
            "step_index": step_index,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": (gen_best if np.isfinite(gen_best) else best_so_far),
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "I0": I0, "top_k_pairs": top_k_pairs,
            "sigma0": sigma0, "popsize": popsize,
            "currents_gen_best": (gen_best_curr.tolist() if gen_best_curr is not None else None),
            "currents_best_so_far": (best_currents.tolist() if best_currents is not None else None),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)
        header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "SWEEP_CMA",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
    }





# ======================================================================
# STATISTICAL ANALYSIS (key fix #7)
# ======================================================================
def compute_statistics(df_summary):
    """Compute confidence intervals and statistical tests."""
    results = []
    
    for (N, tp), grp in df_summary.groupby(['N', 'target_point']):
        opts = grp['optimizer'].unique()
        
        for opt in opts:
            opt_data = grp[grp['optimizer'] == opt]['best'].values
            
            mean = np.mean(opt_data)
            std = np.std(opt_data, ddof=1)
            median = np.median(opt_data)
            
            # 95% confidence interval (t-distribution)
            n = len(opt_data)
            sem = std / np.sqrt(n)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            
            results.append({
                'N': N,
                'target': tp,
                'optimizer': opt,
                'mean': mean,
                'std': std,
                'median': median,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'n_repeats': n,
            })
    
    return pd.DataFrame(results)

def pairwise_tests(df_summary):
    """Mann-Whitney U tests between all optimizer pairs."""
    results = []
    
    for (N, tp), grp in df_summary.groupby(['N', 'target_point']):
        opts = grp['optimizer'].unique()
        
        for i, opt1 in enumerate(opts):
            for opt2 in opts[i+1:]:
                data1 = grp[grp['optimizer'] == opt1]['best'].values
                data2 = grp[grp['optimizer'] == opt2]['best'].values
                
                statistic, pvalue = stats.mannwhitneyu(
                    data1, data2, alternative='two-sided'
                )
                
                # Effect size (rank-biserial correlation)
                n1, n2 = len(data1), len(data2)
                r = 1 - (2*statistic) / (n1 * n2)
                
                results.append({
                    'N': N,
                    'target': tp,
                    'optimizer_1': opt1,
                    'optimizer_2': opt2,
                    'statistic': statistic,
                    'p_value': pvalue,
                    'effect_size': r,
                    'significant_0.05': pvalue < 0.05,
                })
    
    return pd.DataFrame(results)




# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    start = time.time()
    
    print("\n=== FIXED Fair Benchmark: Dimension-scaled, statistically powered ===\n")
    
    # ------------------------------------------------------------------
    # Landscape Characterisation
    # ------------------------------------------------------------------
    # print("Characterizing optimization landscapes...")
    # landscape_results = []
    # for grid in ELECTRODE_GRIDS:
    #     for tp in TARGET_POINTS:
    #         print(f"  Landscape for N={grid[0]*grid[1]}, target={tp}")
    #         char = characterize_landscape(grid, tp, n_samples=200)
    #         landscape_results.append(char)
    
    # df_landscape = pd.DataFrame(landscape_results)
    # df_landscape.to_csv(os.path.join(OUTPUT_DIR, "landscape_characterization.csv"), index=False)
    # print(f"Landscape characterization saved.\n")
    
    # ------------------------------------------------------------------
    # Main Optimization Runs
    # ------------------------------------------------------------------
    print("Running optimization benchmarks...")
    summaries = []
    
    # for grid, repeat, tp in product(ELECTRODE_GRIDS, range(REPEATS), TARGET_POINTS):
    #     N = grid[0] * grid[1]
    #     budget = EVALS_PER_DIM * N
        
        
    #     print(f"\nN={N}, repeat={repeat+1}/{REPEATS}, target={tp}")
    #     print(f"  Budget: {budget} evaluations")
        
        
    #     # --- Random Search ---
    #     # summaries.append(run_random_search(grid, repeat, tp, budget))
        
        
    #     # --- Bayesian Optimization ---
    #     # summaries.append(run_bo(grid, repeat, tp, budget))
        
    #     # --- Particle Swarm Optimization ---
    #     # summaries.append(run_pso(grid, repeat, tp, budget))
        
    #     # --- Differential Evolution ---
    #     #summaries.append(run_de(grid, repeat, tp, budget))

    #     # --- Cross-Entropy Method ---
    #     #summaries.append(run_cem(grid, repeat, tp, budget))

    #     #summaries.append(run_sa(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_hierarchical_cma(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_pairs_sweep_then_cma(grid, repeat, tp, budget))
    #     #summaries.append(run_cma(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_adaptive_cma(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_sweep_then_bo(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_sweep_then_pso(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_sweep_then_tr_pso(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_sweep_sep_cma(grid, repeat,tp,budget))
    #     #summaries.append(run_ms_sweep_memetic(grid, repeat,tp,budget))
    #     #summaries.append(run_ms_sweep_progressive_cma(grid, repeat,tp,budget))
    #     #summaries.append(run_ms_shaped_cma(grid, repeat,tp,budget))

    #     # === PROGRESSIVE SEARCH SPACE VARIANTS ===
    #     #summaries.append(run_prog_cma_linear_50(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_exp_slow(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_step_curriculum(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_sigmoid(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_delayed(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_trust_funnel(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_soft_penalty(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_aggressive_start(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_stagnation(grid, repeat, tp, budget))
    #     #summaries.append(run_prog_cma_late_bloomer(grid, repeat, tp, budget))

    #     #summaries.append(run_ms_sweep_surrogate_cma(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_sweep_restart_cma(grid, repeat, tp, budget))
    #     #summaries.append(run_ms_sweep_cma_lbfgs(grid, repeat, tp, budget))

       
    #     # Final behcmarking PSO + CMA-ES with sweep 
    #     # ----------------------------
    #     # BASELINES (no sweep)
    #     # ----------------------------
    #     summaries.append(run_pso_grounded(grid, repeat, tp, budget))
    #     summaries.append(run_cma_grounded(grid, repeat, tp, budget))

    #     # ----------------------------
    #     # SWEEP + OPTIMIZER (fair sweep)
    #     # ----------------------------
    #     summaries.append(run_ms_sweep_then_pso_fair(
    #         grid, repeat, tp, budget,
    #         I0=0.8e-3,
    #         top_k_pairs=20,
    #         assert_sweep_within_budget=True
    #     ))

    #     summaries.append(run_ms_sweep_then_cma_fair(
    #         grid, repeat, tp, budget,
    #         I0=0.8e-3,
    #         top_k_pairs=20,
    #         assert_sweep_within_budget=True
    #     ))

    for grid in ELECTRODE_GRIDS:
        N = grid[0] * grid[1]
        for repeat in range(REPEATS):
            for t_idx, tp in enumerate(TARGET_POINTS):
                budget = EVALS_PER_DIM * N
                eval_seed = SEED_BASE + 1_000_000*repeat + 10_000*N + t_idx

                summaries.append(run_ms_sweep_then_cma_fair(grid, repeat, tp, budget, eval_seed=eval_seed))
                summaries.append(run_ms_sweep_then_pso_fair(grid, repeat, tp, budget, eval_seed=eval_seed))
                summaries.append(run_pso_grounded(grid, repeat, tp, budget, eval_seed=eval_seed))
                summaries.append(run_cma_grounded(grid, repeat, tp, budget, eval_seed=eval_seed))
                






    
    # ------------------------------------------------------------------
    # Save Raw Results
    # ------------------------------------------------------------------
    df = pd.DataFrame(summaries)
    if df.empty:
        raise RuntimeError("No runs were executed: 'summaries' is empty. Did you comment out all optimizers?")

    df.to_csv(os.path.join(OUTPUT_DIR, "optimizer_summary.csv"), index=False)

    
    # ------------------------------------------------------------------
    # Statistical Analysis
    # ------------------------------------------------------------------
    print("\n=== Computing Statistics ===")
    df_stats = compute_statistics(df)
    df_stats.to_csv(os.path.join(OUTPUT_DIR, "statistics_summary.csv"), index=False)
    
    print("\n=== Running Pairwise Statistical Tests ===")
    df_tests = pairwise_tests(df)
    df_tests.to_csv(os.path.join(OUTPUT_DIR, "pairwise_tests.csv"), index=False)
    
    # ------------------------------------------------------------------
    # Summary Plots
    # ------------------------------------------------------------------
    print("\n=== Generating Summary Plots ===")
    
    # 1. Performance by dimension (boxplots)
    fig, axes = plt.subplots(1, len(ELECTRODE_GRIDS), figsize=(15, 4))
    if len(ELECTRODE_GRIDS) == 1:
        axes = [axes]
    
    for idx, grid in enumerate(ELECTRODE_GRIDS):
        N = grid[0] * grid[1]
        data = df[df['N'] == N]
        
        optimizers = sorted(data['optimizer'].unique())

        colors = ['gray', 'blue', 'green', 'red']

        # Include added optimizers if present
        extra_opts   = ['DE', 'CEM', 'SA']
        extra_colors = ['purple', 'orange', 'black']
        present = data['optimizer'].unique().tolist()
        to_add = [o for o in extra_opts if o in present]
        optimizers = optimizers + to_add
        colors     = colors + [extra_colors[extra_opts.index(o)] for o in to_add]

        positions, labels = [], []
        for i, opt in enumerate(optimizers):
            opt_data = data[data['optimizer'] == opt]['best'].values
            if len(opt_data) > 0:
                positions.append(opt_data)
                labels.append(opt)
        
        bp = axes[idx].boxplot(positions, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[idx].set_title(f'N={N} ({grid[0]}×{grid[1]})')
        axes[idx].set_ylabel('Selectivity' if idx == 0 else '')
        axes[idx].grid(alpha=0.3, axis='y')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "performance_by_dimension.png"), dpi=200)
    plt.close()
    
    # 2. Convergence curves (average over repeats)
    for grid in ELECTRODE_GRIDS:
        N = grid[0] * grid[1]
        fig, axes = plt.subplots(1, len(TARGET_POINTS), figsize=(15, 4))
        if len(TARGET_POINTS) == 1:
            axes = [axes]
        
        for t_idx, tp in enumerate(TARGET_POINTS):
            for opt in sorted(df['optimizer'].unique()):
                prefix = opt
                all_curves = []
                for rep in range(REPEATS):
                    tag = make_tag(prefix, grid, rep, tp)
                    csv_file = os.path.join(OUTPUT_DIR, f"{tag}.csv")
                    if os.path.exists(csv_file):
                        try:
                            df_run = pd.read_csv(csv_file)
                            all_curves.append(df_run[['evals_so_far', 'best_so_far']].values)
                        except Exception:
                            pass

                if all_curves:
                    max_evals = max(c[-1, 0] for c in all_curves)
                    eval_points = np.linspace(0, max_evals, 100)
                    
                    interpolated = [np.interp(eval_points, c[:, 0], c[:, 1]) for c in all_curves]
                    mean_curve = np.mean(interpolated, axis=0)
                    std_curve = np.std(interpolated, axis=0)
                                        
                    color_map = {
                        "PSO_GROUNDED": "red",
                        "CMA_GROUNDED": "blue",
                        "MS_SWEEP_PSO_FAIR": "orange",
                        "MS_SWEEP_CMA_FAIR": "green",
                    }

                    
                    axes[t_idx].plot(
                        eval_points, mean_curve, label=opt,
                        color=color_map.get(opt, None), lw=2
                    )
                    axes[t_idx].fill_between(
                        eval_points,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2,
                        color=color_map.get(opt, None)
                    )

            
            axes[t_idx].set_xlabel('Function evaluations')
            axes[t_idx].set_ylabel('Selectivity' if t_idx == 0 else '')
            axes[t_idx].set_title(f'Target: ({tp[0]*1e3:.1f}, {tp[1]*1e3:.1f}, {tp[2]*1e3:.1f}) mm')
            axes[t_idx].legend()
            axes[t_idx].grid(alpha=0.3)
        
        plt.suptitle(f'Average Convergence Curves (N={N}, {REPEATS} repeats)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"convergence_N{N}.png"), dpi=200)
        plt.close()
    
    # 3. Statistical significance heatmap
    for N in [g[0]*g[1] for g in ELECTRODE_GRIDS]:
        test_data = df_tests[df_tests['N'] == N]
        if len(test_data) == 0:
            continue
        
        optimizers = sorted(df[df['N'] == N]['optimizer'].unique())
        n_opts = len(optimizers)
        
        pval_matrix = np.ones((n_opts, n_opts))
        for _, row in test_data.iterrows():
            i = optimizers.index(row['optimizer_1'])
            j = optimizers.index(row['optimizer_2'])
            pval_matrix[i, j] = row['p_value']
            pval_matrix[j, i] = row['p_value']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pval_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        
        ax.set_xticks(np.arange(n_opts))
        ax.set_yticks(np.arange(n_opts))
        ax.set_xticklabels(optimizers)
        ax.set_yticklabels(optimizers)
        
        for i in range(n_opts):
            for j in range(n_opts):
                if i != j:
                    pval = pval_matrix[i, j]
                    text = f'{pval:.3f}'
                    if pval < 0.001:
                        text = '***'
                    elif pval < 0.01:
                        text = '**'
                    elif pval < 0.05:
                        text = '*'
                    ax.text(j, i, text, ha="center", va="center",
                           color="black" if pval > 0.05 else "white",
                           fontweight='bold' if pval < 0.05 else 'normal')
        
        ax.set_title(f'Pairwise Significance (Mann-Whitney U, N={N})\n* p<0.05, ** p<0.01, *** p<0.001')
        plt.colorbar(im, ax=ax, label='p-value')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"significance_heatmap_N{N}.png"), dpi=200)
        plt.close()
    
    # ------------------------------------------------------------------
    # Print Statistical Summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY (Mean ± 95% CI)")
    print("="*70)
    for N in sorted(df_stats['N'].unique()):
        print(f"\n--- N={N} dimensions ---")
        subset = df_stats[df_stats['N'] == N]
        for opt in sorted(subset['optimizer'].unique()):
            opt_data = subset[subset['optimizer'] == opt]
            if len(opt_data) > 0:
                mean = opt_data['mean'].mean()
                ci_width = (opt_data['ci_upper'] - opt_data['ci_lower']).mean() / 2
                print(f"  {opt:10s}: {mean:.4f} ± {ci_width:.4f}")
    
    print("\n" + "="*70)
    print("SIGNIFICANT DIFFERENCES (p < 0.05)")
    print("="*70)
    sig_tests = df_tests[df_tests['significant_0.05']]
    for N in sorted(sig_tests['N'].unique()):
        print(f"\n--- N={N} dimensions ---")
        subset = sig_tests[sig_tests['N'] == N]
        for _, row in subset.iterrows():
            print(f"  {row['optimizer_1']:10s} vs {row['optimizer_2']:10s}: "
                  f"p={row['p_value']:.4f}, effect={row['effect_size']:+.3f}")
    
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"COMPLETED in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # ------------------------------------------------------------------
    # Final Summary Table
    # ------------------------------------------------------------------
    print("\nFINAL PERFORMANCE TABLE")
    print("-" * 90)
    print(f"{'Optimizer':<12} {'N':<4} {'Mean':<10} {'Median':<10} {'Std':<10} {'Best':<10} {'Worst':<10}")
    print("-" * 90)
    
    final_opts = sorted(df['optimizer'].unique())

    for N in sorted(df['N'].unique()):
        for opt in final_opts:
            opt_data = df[(df['N'] == N) & (df['optimizer'] == opt)]['best'].values
            if len(opt_data) > 0:
                print(f"{opt:<20} {N:<4} {np.mean(opt_data):<10.4f} "
                    f"{np.median(opt_data):<10.4f} {np.std(opt_data):<10.4f} "
                    f"{np.max(opt_data):<10.4f} {np.min(opt_data):<10.4f}")
        print("-" * 90)

    
    print("\nKey metrics saved:")
    print(f"  - optimizer_summary.csv: Raw results for all runs")
    print(f"  - statistics_summary.csv: Mean, median, CI for each condition")
    print(f"  - pairwise_tests.csv: Statistical significance tests")
    print(f"  - landscape_characterization.csv: Problem difficulty metrics")
    print(f"  - Individual run CSVs and plots for detailed analysis")
    print(f"\nAll files in: {OUTPUT_DIR}\n")
