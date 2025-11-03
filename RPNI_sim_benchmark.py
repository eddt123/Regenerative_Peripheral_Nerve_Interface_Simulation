#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FIXED Fair Benchmark: CMA-ES vs BO vs PSO vs Random Search
===========================================================
Fixes:
1. Dimension-scaled budgets (200*N evaluations per run)
2. Sequential BO (not batch) with proper initialization
3. Correct projection handling (tell original points to maintain model consistency)
4. Dimension-scaled population sizes following standard recommendations
5. Random search baseline
6. 20 repeats for statistical power
7. Statistical analysis with confidence intervals and significance tests
8. Landscape characterization (sampling variance, local optima estimates)
9. All algorithms get equal evaluation budgets

Requires: cma, scikit-optimize, numpy, pandas, matplotlib, scipy, tqdm
"""

import os, time, math
import numpy as np
import pandas as pd
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "benchmark_extra")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tissue / geometry
RADIUS   = 0.01   # m
HEIGHT   = 0.04   # m
SIGMA_T  = 0.25   # S/m

# Electrode layouts (N = n_rows * n_per_row)
ELECTRODE_GRIDS = [
    #(4, 2),   # N=8
    (2,2),
    (3,2),
    (3,3),
    (4, 3),   # N=12
    (4, 4),   # N=16
    (5,5),
]

# Currents
RANGE     = 1e-3         # ±1 mA bounds
ZERO_SUM  = True         # enforce sum(currents)=0

# Targets
TARGET_POINTS = [
    (0.0, 0.0, 0.0),
    (0.002, 0.0035, 0.010),
    (-0.003, -0.002, -0.015),
]

# Budget: DIMENSION-SCALED (key fix #1)
EVALS_PER_DIM = 200      # 200*N evaluations per run
REPEATS = 20             # Increased from 5 (key fix #5)
SEED_BASE = 42

# CMA-ES: DIMENSION-SCALED population (key fix #4)
CMA_SIGMA0 = 0.3e-3
def cma_popsize(N):
    """Standard CMA-ES population size: 4 + floor(3*ln(N))"""
    return 4 + int(3 * np.log(N))

# BO: SEQUENTIAL (key fix #2)
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
    """
    KEY FIX #3: Returns (NEGATIVE selectivity, original_x, projected_x).
    This allows optimizers to be told the original point while evaluating projected.
    """
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
# LANDSCAPE CHARACTERIZATION (key fix #8)
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
# RANDOM SEARCH BASELINE (key fix #5)
# ======================================================================
def run_random_search(grid, repeat, target_point, eval_budget):
    """Pure random search baseline."""
    N = grid[0] * grid[1]
    seed = SEED_BASE + 10000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("Random", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")
    
    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, step_vals, best_vals = [], [], []
    
    header_written = False
    for i in tqdm(range(eval_budget), desc=tag, leave=False):
        x = rng.uniform(-RANGE, RANGE, N)
        y, x_orig, x_proj = eval_selectivity(x, target_point, grid, rng_seed=seed)
        
        sel = -y
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = i + 1
        
        xs_axis.append(i + 1)
        step_vals.append(sel)
        best_vals.append(best_so_far)
        
        # Log every 10th evaluation to reduce file size
        if (i + 1) % 10 == 0 or i == 0 or i == eval_budget - 1:
            log_step(csv_path, {
                "optimizer": "Random",
                "n_rows": grid[0], "n_per_row": grid[1], "N": N,
                "repeat": repeat,
                "eval_index": i + 1,
                "evals_so_far": i + 1,
                "current_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }, header_written)
            header_written = True
    
    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    
    return {
        "optimizer": "Random", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": eval_budget
    }

# ======================================================================
# CMA-ES (dimension-scaled, correct projection handling)
# ======================================================================
def run_cma(grid, repeat, target_point, eval_budget):
    """
    CMA-ES (dimension-scaled) with correct projection handling.
    Evaluates at projected feasible points and tells the optimizer
    those same evaluated points to maintain model consistency.
    """
    N = grid[0] * grid[1]
    popsize = cma_popsize(N)
    seed = SEED_BASE + 1000*repeat + N
    tag = make_tag("CMAES", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # Budget accounting
    iters = max(1, eval_budget // popsize)
    used_budget = iters * popsize

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-RANGE, RANGE, N)
    x0 = project_currents(x0, -RANGE, RANGE, ZERO_SUM)

    es = CMAEvolutionStrategy(
        x0, CMA_SIGMA0,
        {"popsize": popsize, "verb_disp": 0, "seed": seed, "bounds": [-RANGE, RANGE]}
    )

    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, iter_vals, best_vals = [], [], []
    evals_so_far = 0
    header_written = False

    for it in tqdm(range(iters), desc=tag, leave=False):
        X_ask = es.ask()

        # Evaluate at projected feasible points
        X_eval, Y = [], []
        for x_ask in X_ask:
            y, x_orig, x_proj = eval_selectivity(x_ask, target_point, grid, rng_seed=seed)
            X_eval.append(x_proj)
            Y.append(y)

        # Tell CMA the *evaluated* (projected) points
        es.tell(X_eval, Y)

        evals_so_far += len(Y)
        k = int(np.argmin(Y))
        step_best = -Y[k]

        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        xs_axis.append(evals_so_far)
        iter_vals.append(step_best)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "CMAES",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "sigma0": CMA_SIGMA0, "popsize": popsize,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": step_best,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)
        header_written = True

    save_progress_plot(xs_axis, iter_vals, best_vals, tag, target_point)

    return {
        "optimizer": "CMAES", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": popsize
    }


# ======================================================================
# BO (SEQUENTIAL, proper initialization, correct projection)
# ======================================================================
def run_bo(grid, repeat, target_point, eval_budget):
    """
    Sequential Bayesian Optimization (skopt) with correct projection.
    Evaluates at projected feasible points and tells the optimizer
    the same evaluated points.
    """
    N = grid[0] * grid[1]
    n_init = bo_n_initial(N)
    seed = SEED_BASE + 2000*repeat + N
    tag = make_tag(f"BO_{BO_ACQ}", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    opt = SkOptimizer(
        [(-RANGE, RANGE)] * N,
        base_estimator="GP",
        acq_func=BO_ACQ,
        acq_func_kwargs={"kappa": BO_KAPPA, "xi": BO_XI},
        n_initial_points=n_init,
        random_state=seed
    )

    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    for i in tqdm(range(eval_budget), desc=tag, leave=False):
        # Ask one point at a time (sequential BO)
        x_ask = np.array(opt.ask(), dtype=float)

        # Evaluate at projected feasible point
        y, x_orig, x_proj = eval_selectivity(x_ask, target_point, grid, rng_seed=seed)

        # Tell the optimizer the *evaluated* point
        opt.tell(x_proj.tolist(), float(y))

        sel = -y
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = i + 1

        xs_axis.append(i + 1)
        step_vals.append(sel)
        best_vals.append(best_so_far)

        # Log every 10th evaluation
        if (i + 1) % 10 == 0 or i == 0 or i == eval_budget - 1:
            log_step(csv_path, {
                "optimizer": "BO",
                "n_rows": grid[0], "n_per_row": grid[1], "N": N,
                "acq_func": BO_ACQ, "kappa": BO_KAPPA, "xi": BO_XI,
                "n_initial": n_init,
                "repeat": repeat,
                "eval_index": i + 1,
                "evals_so_far": i + 1,
                "current_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }, header_written)
            header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "BO", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": eval_budget, "n_initial": n_init
    }


# ======================================================================
# PSO (dimension-scaled, correct projection)
# ======================================================================
def run_pso(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    popsize = pso_popsize(N)
    seed = SEED_BASE + 3000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("PSO", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")
    
    # Budget accounting
    iters = max(1, eval_budget // popsize)
    used_budget = iters * popsize
    
    # Init swarm in feasible space
    X = np.vstack([
        project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM)
        for _ in range(popsize)
    ])
    V = np.zeros_like(X)
    
    # Evaluate initial population
    pbest_pos = X.copy()
    pbest_val = np.empty(popsize, dtype=float)
    for i in range(popsize):
        y, _, _ = eval_selectivity(pbest_pos[i], target_point, grid, rng_seed=seed)
        pbest_val[i] = y
    
    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = pbest_val[g_idx]
    
    best_so_far = -gbest_val
    best_at_eval = popsize
    xs_axis = [popsize]
    step_vals = [best_so_far]
    best_vals = [best_so_far]
    evals_so_far = popsize
    
    header_written = False
    log_step(csv_path, {
        "optimizer": "PSO",
        "n_rows": grid[0], "n_per_row": grid[1], "N": N,
        "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": evals_so_far,
        "step_best_selectivity": best_so_far,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0],
        "target_y": target_point[1],
        "target_z": target_point[2],
    }, header_written)
    header_written = True
    
    # Main loop
    for it in tqdm(range(1, iters), desc=tag, leave=False):
        r1 = rng.random(size=(popsize, N))
        r2 = rng.random(size=(popsize, N))
        
        V = PSO_W*V + PSO_C1*r1*(pbest_pos - X) + PSO_C2*r2*(gbest_pos - X)
        if PSO_VCLAMP is not None:
            V = np.clip(V, -PSO_VCLAMP, PSO_VCLAMP)
        
        X_new = X + V
        # Project to feasible space
        X = np.vstack([
            project_currents(x, -RANGE, RANGE, ZERO_SUM) for x in X_new
        ])
        
        it_best_sel = -np.inf
        for i in range(popsize):
            y, _, x_proj = eval_selectivity(X[i], target_point, grid, rng_seed=seed)
            X[i] = x_proj  # Ensure consistency
            
            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = x_proj
            
            sel = -y
            if sel > it_best_sel:
                it_best_sel = sel
        
        g_idx = int(np.argmin(pbest_val))
        if pbest_val[g_idx] < gbest_val:
            gbest_val = pbest_val[g_idx]
            gbest_pos = pbest_pos[g_idx].copy()
        
        evals_so_far += popsize
        if it_best_sel > best_so_far:
            best_so_far = it_best_sel
            best_at_eval = evals_so_far
        
        xs_axis.append(evals_so_far)
        step_vals.append(it_best_sel)
        best_vals.append(best_so_far)
        
        log_step(csv_path, {
            "optimizer": "PSO",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": it_best_sel,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)
    
    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    
    return {
        "optimizer": "PSO", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": popsize
    }

# ======================================================================
# Differential Evolution (custom DE/rand/1/bin, correct projection)
# ======================================================================
def run_de(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    seed = SEED_BASE + 4000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("DE", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    NP = de_population_size(N)          # number of candidate vectors in the population
    iters = max(1, eval_budget // NP)   # each generation evaluates ~NP candidates
    used_budget = iters * NP

    # Init feasible population
    X = np.vstack([
        project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM)
        for _ in range(NP)
    ])
    # Evaluate
    F_vals = np.empty(NP, dtype=float)
    for i in range(NP):
        y, _, _ = eval_selectivity(X[i], target_point, grid, rng_seed=seed)
        F_vals[i] = y  # minimize y

    best_idx = int(np.argmin(F_vals))
    best_val = F_vals[best_idx]
    best_vec = X[best_idx].copy()

    best_so_far = -best_val
    best_at_eval = NP

    xs_axis  = [NP]
    step_vals = [best_so_far]
    best_vals = [best_so_far]
    header_written = False

    log_step(csv_path, {
        "optimizer": "DE",
        "n_rows": grid[0], "n_per_row": grid[1], "N": N,
        "popsize": NP, "F": DE_F, "CR": DE_CR,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": NP,
        "step_best_selectivity": best_so_far,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0],
        "target_y": target_point[1],
        "target_z": target_point[2],
    }, header_written)
    header_written = True

    # Main generations
    for gen in tqdm(range(1, iters), desc=tag, leave=False):
        X_new = X.copy()
        F_new = F_vals.copy()

        for i in range(NP):
            # choose 3 distinct indices != i
            idxs = rng.choice([j for j in range(NP) if j != i], size=3, replace=False)
            r1, r2, r3 = idxs
            # mutation
            V = X[r1] + DE_F * (X[r2] - X[r3])
            # binomial crossover
            cross = rng.random(N) < DE_CR
            if not np.any(cross):
                cross[rng.integers(0, N)] = True
            U = np.where(cross, V, X[i])
            # project & evaluate
            U = project_currents(U, -RANGE, RANGE, ZERO_SUM)
            y, _, Uproj = eval_selectivity(U, target_point, grid, rng_seed=seed)
            # selection
            if y < F_vals[i]:
                X_new[i] = Uproj
                F_new[i] = y

        X, F_vals = X_new, F_new

        # Best of this generation
        b_idx = int(np.argmin(F_vals))
        if F_vals[b_idx] < best_val:
            best_val = F_vals[b_idx]
            best_vec = X[b_idx].copy()

        evals_so_far = (gen+1) * NP
        step_best = -float(np.min(F_vals))
        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        xs_axis.append(evals_so_far)
        step_vals.append(step_best)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "DE",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "popsize": NP, "F": DE_F, "CR": DE_CR,
            "repeat": repeat,
            "step_index": gen + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": step_best,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "DE", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": NP
    }


# ======================================================================
# Cross-Entropy Method (CEM) with projection
# ======================================================================
def run_cem(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    seed = SEED_BASE + 5000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("CEM", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    K = cem_popsize(N)
    iters = max(1, eval_budget // K)
    used_budget = iters * K

    m = np.zeros(N, dtype=float)           # start at 0 mA
    s = np.full(N, CEM_SIGMA0, dtype=float)

    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False
    evals_so_far = 0

    for it in tqdm(range(iters), desc=tag, leave=False):
        # Sample & project
        X = m + s * rng.standard_normal(size=(K, N))
        X = np.vstack([project_currents(x, -RANGE, RANGE, ZERO_SUM) for x in X])

        Y = np.empty(K, dtype=float)
        for i in range(K):
            y, _, _ = eval_selectivity(X[i], target_point, grid, rng_seed=seed)
            Y[i] = y
        evals_so_far += K

        # Rank by MIN(y) -> MAX(selectivity)
        idx = np.argsort(Y)
        elite_k = max(1, int(np.ceil(CEM_ELITE_FRAC * K)))
        elites = X[idx[:elite_k]]

        # Update parameters with smoothing
        new_m = elites.mean(axis=0)
        new_s = elites.std(axis=0)
        m = (1 - CEM_ALPHA) * m + CEM_ALPHA * new_m
        s = (1 - CEM_ALPHA) * s + CEM_ALPHA * np.maximum(new_s, CEM_SIGMA_MIN)

        # Clip to bounds (center & spread)
        m = np.clip(m, -RANGE, RANGE)
        s = np.clip(s, CEM_SIGMA_MIN, RANGE)

        step_best = -float(np.min(Y))
        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        xs_axis.append(evals_so_far)
        step_vals.append(step_best)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "CEM",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "popsize": K, "elite_frac": CEM_ELITE_FRAC, "alpha": CEM_ALPHA,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": step_best,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)
        header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "CEM", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": K
    }


# ======================================================================
# Simulated Annealing (projected random-walk, fair budget)
# ======================================================================
def run_sa(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    seed = SEED_BASE + 6000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("SA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # Warm-up to set T0 by objective spread (deduct from budget)
    warm = min(SA_WARMUP_SAMPLES, max(0, eval_budget // 10))
    temps = []
    if warm > 0:
        vals = []
        for _ in range(warm):
            x = rng.uniform(-RANGE, RANGE, N)
            x = project_currents(x, -RANGE, RANGE, ZERO_SUM)
            y, _, _ = eval_selectivity(x, target_point, grid, rng_seed=seed)
            vals.append(float(y))
        vals = np.array(vals)
        T0 = np.std(vals) if np.std(vals) > 1e-12 else 1.0
    else:
        T0 = 1.0

    remaining = max(1, eval_budget - warm)

    # Start from best warm-up or random feasible
    if warm > 0:
        best_idx = int(np.argmin(vals))
        x = project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM) if best_idx < 0 else None
        x = project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM) if x is None else x
    else:
        x = project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM)

    y, _, _ = eval_selectivity(x, target_point, grid, rng_seed=seed)

    x_best = x.copy()
    y_best = y

    best_so_far = -float(y_best)
    best_at_eval = warm + 1

    xs_axis  = [warm + 1]
    step_vals = [best_so_far]
    best_vals = [best_so_far]
    header_written = False

    T = T0
    for k in tqdm(range(1, remaining), desc=tag, leave=False):
        # Gaussian proposal + projection
        prop = x + SA_STEP_SIGMA * rng.standard_normal(N)
        prop = project_currents(prop, -RANGE, RANGE, ZERO_SUM)
        y_prop, _, _ = eval_selectivity(prop, target_point, grid, rng_seed=seed)

        # Metropolis acceptance (minimize y)
        dy = y_prop - y
        if dy < 0 or rng.random() < np.exp(-dy / max(T, 1e-12)):
            x, y = prop, y_prop

        # Track global best
        if y < y_best:
            x_best, y_best = x.copy(), y

        # Update temp
        T *= SA_ALPHA

        evals_so_far = warm + 1 + k
        step_best = -float(y_best)
        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        # Log sparsely to keep files small
        if k == 1 or k == remaining-1 or k % max(10, N) == 0:
            xs_axis.append(evals_so_far)
            step_vals.append(-float(y))
            best_vals.append(best_so_far)
            log_step(csv_path, {
                "optimizer": "SA",
                "n_rows": grid[0], "n_per_row": grid[1], "N": N,
                "alpha": SA_ALPHA, "step_sigma": SA_STEP_SIGMA, "T0": T0,
                "repeat": repeat,
                "step_index": k + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": -float(y),
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }, header_written)
            header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "SA", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": warm + remaining
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
    print("Characterizing optimization landscapes...")
    landscape_results = []
    for grid in ELECTRODE_GRIDS:
        for tp in TARGET_POINTS:
            print(f"  Landscape for N={grid[0]*grid[1]}, target={tp}")
            char = characterize_landscape(grid, tp, n_samples=200)
            landscape_results.append(char)
    
    df_landscape = pd.DataFrame(landscape_results)
    df_landscape.to_csv(os.path.join(OUTPUT_DIR, "landscape_characterization.csv"), index=False)
    print(f"Landscape characterization saved.\n")
    
    # ------------------------------------------------------------------
    # Main Optimization Runs
    # ------------------------------------------------------------------
    print("Running optimization benchmarks...")
    summaries = []
    
    for grid, repeat, tp in product(ELECTRODE_GRIDS, range(REPEATS), TARGET_POINTS):
        N = grid[0] * grid[1]
        budget = EVALS_PER_DIM * N
        
        print(f"\nN={N}, repeat={repeat+1}/{REPEATS}, target={tp}")
        print(f"  Budget: {budget} evaluations")
        
        # -----------------------------
        # Optimizers (commented = skip)
        # -----------------------------
        
        # --- Random Search ---
        # summaries.append(run_random_search(grid, repeat, tp, budget))
        
        # --- CMA-ES ---
        # summaries.append(run_cma(grid, repeat, tp, budget))
        
        # --- Bayesian Optimization ---
        # summaries.append(run_bo(grid, repeat, tp, budget))
        
        # --- Particle Swarm Optimization ---
        # summaries.append(run_pso(grid, repeat, tp, budget))
        
        # --- Differential Evolution ---
        summaries.append(run_de(grid, repeat, tp, budget))

        # --- Cross-Entropy Method ---
        summaries.append(run_cem(grid, repeat, tp, budget))

        # --- Simulated Annealing ---
        summaries.append(run_sa(grid, repeat, tp, budget))
    
    # ------------------------------------------------------------------
    # Save Raw Results
    # ------------------------------------------------------------------
    df = pd.DataFrame(summaries)
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
        
        optimizers = ['Random', 'CMAES', 'BO', 'PSO']
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
            for opt in ['Random', 'CMAES', 'BO', 'PSO', 'DE', 'CEM', 'SA']:
                prefix = f"BO_{BO_ACQ}" if opt == 'BO' else opt
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
                        'Random':'gray', 'CMAES':'blue', 'BO':'green', 'PSO':'red',
                        'DE':'purple', 'CEM':'orange', 'SA':'black'
                    }
                    if opt in color_map:
                        axes[t_idx].plot(eval_points, mean_curve, label=opt,
                                         color=color_map[opt], lw=2)
                        axes[t_idx].fill_between(eval_points,
                                                 mean_curve - std_curve,
                                                 mean_curve + std_curve,
                                                 alpha=0.2, color=color_map[opt])
            
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
        for opt in ['Random', 'CMAES', 'BO', 'PSO', 'DE', 'CEM', 'SA']:
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
    
    for N in sorted(df['N'].unique()):
        for opt in ['Random', 'CMAES', 'BO', 'PSO', 'DE', 'CEM', 'SA']:
            opt_data = df[(df['N'] == N) & (df['optimizer'] == opt)]['best'].values
            if len(opt_data) > 0:
                print(f"{opt:<12} {N:<4} {np.mean(opt_data):<10.4f} "
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
