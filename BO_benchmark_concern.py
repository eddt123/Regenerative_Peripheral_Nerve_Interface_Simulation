#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BENCHMARK: Addressing Concern 1 — Sweep+Optimizer vs Bayesian Optimisation Restarts
====================================================================================

PURPOSE
-------
A PI raised the concern that warm-starting an optimiser with a pair-sweep may be
"cheating" and that simple BO restarts (cold-start) would be a fairer baseline.

This script runs a head-to-head comparison under IDENTICAL BUDGETS and
IDENTICAL EVAL SEEDS so every method sees the same objective landscape.

METHODS COMPARED
----------------
  1. BO_SINGLE    — Single-run BO (vanilla baseline, no tricks)
  2. BO_RESTART   — BO with R independent cold restarts; each restart
                    gets budget/R evaluations, best-of-R reported.
                    This is the PI's suggested alternative.
  3. SWEEP_BO     — Exhaustive dipolar pair sweep → BO warm-started with
                    top-K sweep results seeded into the GP prior.
                    This is the method under scrutiny.
  4. SWEEP_PSO    — Sweep → PSO (reference from main benchmark).

FAIRNESS GUARANTEES
-------------------
  • Same total evaluation budget       = EVALS_PER_DIM × N
  • Same eval_seed per condition       → deterministic objective landscape
  • Independent algo_seed per method   → independent stochastic choices
  • Sweep evaluations counted against total budget (no free evaluations)
  • Wall-clock time recorded per run   → honest speed comparison
  • 60-second timeout per run          → prevents BO from hanging at high N
    (partial results recorded if timeout triggers)

SCIENTIFIC VALIDITY NOTES
-------------------------
  • The GP in BO has O(n³) fitting cost in the number of observations.
    At high N with large budgets, BO may hit the timeout before exhausting
    its budget.  This is a REAL disadvantage of BO, not an artefact — it
    reflects the practical wall-clock cost a user would experience.
  • We report BOTH selectivity-at-budget-completion AND
    selectivity-per-second to capture the full trade-off.
  • Mann-Whitney U tests are used (non-parametric, appropriate for small n).
    With REPEATS=3, statistical power is limited; results should be
    interpreted alongside effect sizes, not p-values alone.

OUTPUTS (saved to data/bayes_opt_concern/)
------------------------------------------
  • optimizer_summary.csv            — one row per run
  • statistics_summary.csv           — mean, CI, median per condition
  • pairwise_tests.csv               — Mann-Whitney U between all pairs
  • convergence_N{N}.png             — mean convergence curves per grid
  • boxplot_comparison.png           — selectivity distributions
  • time_comparison.png              — wall-clock time distributions
  • pareto_selectivity_vs_time.png   — selectivity vs speed trade-off
  • significance_heatmap_N{N}.png    — pairwise p-value heatmaps
  • final_report.png                 — single-figure summary for the paper
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from scipy import stats

# ── Simulator import ─────────────────────────────────────────────────
try:
    from utils.run_selectivity_simulation import run_selectivity_simulation
except Exception:
    sys.path.append(os.path.dirname(__file__))
    try:
        from run_selectivity_simulation import run_selectivity_simulation
    except Exception:
        for _p in [os.path.join(os.path.dirname(__file__), "utils"),
                   os.path.join(os.path.dirname(__file__), "..")]:
            sys.path.insert(0, _p)
        from run_selectivity_simulation import run_selectivity_simulation

# ── Optimisers ────────────────────────────────────────────────────────
from skopt import Optimizer as SkOptimizer

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================
# CONFIG
# ======================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "bayes_opt_concern")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tissue / geometry (identical to main benchmark)
RADIUS  = 0.01   # m
HEIGHT  = 0.04   # m
SIGMA_T = 0.08   # S/m

# Electrode grids
ELECTRODE_GRIDS = [
    (4, 2),   # N=8   (28 pairs)
    (4, 3),   # N=12  (66 pairs)
    (4, 4),   # N=16  (120 pairs)
    (5, 5),   # N=25  (300 pairs)
]

# Current bounds
RANGE = 1e-3  # ±1 mA

# Target points
TARGET_POINTS = [
    (0.0,   0.0,    0.0),
    (0.002, 0.0035, 0.010),
    (0.004, 0.002,  0.015),
    (0.005, 0.000,  0.000),
]

# Budget and repeats
EVALS_PER_DIM = 200
REPEATS       = 3
SEED_BASE     = 42

# Wall-clock timeout per optimizer run (seconds)
TIMEOUT_SECONDS = 60.0

# ── BO-specific ───────────────────────────────────────────────────────
BO_ACQ = "EI"
BO_XI  = 0.01

def bo_n_initial(N):
    return max(2 * N, 10)

# ── BO restart config ─────────────────────────────────────────────────
BO_N_RESTARTS = 5

# ── Sweep config ──────────────────────────────────────────────────────
SWEEP_I0 = 0.8e-3

# ── PSO params ────────────────────────────────────────────────────────
def pso_popsize(N):
    return 10 + int(2 * np.sqrt(N))

PSO_W      = 0.729
PSO_C1     = 1.49445
PSO_C2     = 1.49445
PSO_VCLAMP = 0.5 * RANGE


# ======================================================================
# EVALUATION
# ======================================================================
def eval_selectivity_grounded(x, target_point, grid, rng_seed=None):
    """
    Returns (neg_selectivity, clipped_currents).
    neg_selectivity is NEGATIVE so minimisers work; negate for reporting.
    """
    n_rows, n_per_row = grid
    x_c = np.clip(np.asarray(x, dtype=float), -RANGE, RANGE)

    res = run_selectivity_simulation(
        n_rows=n_rows, n_per_row=n_per_row,
        currents=x_c,
        target_point=target_point,
        radius=RADIUS, height=HEIGHT, sigma=SIGMA_T,
        n_off_samples=1200,
        E_th=1, k=0.5,
        metric="activation",
        grounded_boundary=True,
        use_activating_function=False,
        R_outer=0.10,
        rng=(SEED_BASE if rng_seed is None else rng_seed),
    )
    return -float(res["selectivity"]), x_c


# ======================================================================
# UTILITIES
# ======================================================================
def make_tag(prefix, grid, repeat, tp):
    n_rows, n_per_row = grid
    tx, ty, tz = tp
    return (f"{prefix}_N{n_rows*n_per_row}_grid{n_rows}x{n_per_row}_r{repeat}"
            f"_x{tx*1e3:.1f}_y{ty*1e3:.1f}_z{tz*1e3:.1f}")


def log_step(csv_path, data_dict, header_written):
    pd.DataFrame([data_dict]).to_csv(
        csv_path, mode="a", index=False, header=not header_written)


def timed_out(t_start):
    """Check if wall-clock timeout has been exceeded."""
    return (time.time() - t_start) >= TIMEOUT_SECONDS


# ======================================================================
# PAIR SWEEP (shared stage-0 helper)
# ======================================================================
def run_pair_sweep_grounded(
    grid, repeat, target_point, eval_budget, eval_seed,
    I0=SWEEP_I0, tag_prefix="SWEEP", csv_path=None,
    header_written=False, t_start=None,
):
    """
    Exhaustive sweep over all N(N-1)/2 unordered pairs.
    Respects both eval_budget and wall-clock timeout.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    evals_so_far  = 0
    best_so_far   = -np.inf
    best_at_eval  = 0
    best_at_time  = 0.0
    best_currents = np.zeros(N)

    xs_axis, step_vals, best_vals = [], [], []
    pair_results = []

    for idx, (i, j) in enumerate(all_pairs):
        if evals_so_far >= eval_budget:
            break
        if t_start is not None and timed_out(t_start):
            break

        currents    = np.zeros(N)
        currents[i] = +I0
        currents[j] = -I0

        y, x_used    = eval_selectivity_grounded(currents, target_point,
                                                 grid, rng_seed=eval_seed)
        sel          = -y
        evals_so_far += 1

        pair_results.append((sel, x_used.copy()))

        if sel > best_so_far:
            best_so_far   = sel
            best_at_eval  = evals_so_far
            best_at_time  = time.time() - t_start if t_start else 0.0
            best_currents = x_used.copy()

        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)

        if csv_path and (idx == 0 or idx == len(all_pairs) - 1
                         or (idx + 1) % 10 == 0):
            log_step(csv_path, {
                "optimizer": tag_prefix, "stage": 0,
                "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                "repeat": repeat, "step_index": idx + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "wall_time_s": time.time() - t_start if t_start else 0.0,
                "target_x": target_point[0], "target_y": target_point[1],
                "target_z": target_point[2],
                "eval_seed": eval_seed,
            }, header_written)
            header_written = True

    return (pair_results, evals_so_far, best_so_far, best_at_eval,
            best_at_time, best_currents,
            xs_axis, step_vals, best_vals, header_written)


# ======================================================================
# 1. BO_SINGLE — vanilla baseline
# ======================================================================
def run_bo_single(grid, repeat, target_point, eval_budget, eval_seed):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    algo_seed = SEED_BASE + 5000 * repeat + 71 * N
    tag       = make_tag("BO_SINGLE", grid, repeat, target_point)
    csv_path  = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    dims = [(-RANGE, RANGE)] * N
    opt  = SkOptimizer(dims, base_estimator="GP", acq_func=BO_ACQ,
                       acq_func_kwargs={"xi": BO_XI},
                       random_state=algo_seed)

    t_start      = time.time()
    evals_so_far = 0
    best_so_far  = -np.inf
    best_at_eval = 0
    best_at_time = 0.0
    timed_out_flag = False

    xs_axis, step_vals, best_vals, time_axis = [], [], [], []
    header_written = False

    for i in range(eval_budget):
        if timed_out(t_start):
            timed_out_flag = True
            break

        x_suggest = opt.ask()
        y, x_used = eval_selectivity_grounded(x_suggest, target_point,
                                              grid, rng_seed=eval_seed)
        opt.tell(x_used.tolist(), y)
        evals_so_far += 1
        sel = -y
        elapsed = time.time() - t_start

        if sel > best_so_far:
            best_so_far  = sel
            best_at_eval = evals_so_far
            best_at_time = elapsed

        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)
        time_axis.append(elapsed)

        if i == 0 or i == eval_budget - 1 or (i + 1) % 20 == 0:
            log_step(csv_path, {
                "optimizer": "BO_SINGLE", "stage": 1,
                "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                "repeat": repeat, "step_index": i + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "wall_time_s": elapsed,
                "target_x": target_point[0], "target_y": target_point[1],
                "target_z": target_point[2],
                "eval_seed": eval_seed, "algo_seed": algo_seed,
            }, header_written)
            header_written = True

    wall_time = time.time() - t_start

    return {
        "optimizer": "BO_SINGLE", "tag": tag,
        "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
        "best_found_at_eval": int(best_at_eval),
        "best_found_at_time": float(best_at_time),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "budget": int(eval_budget),
        "budget_exhausted": evals_so_far >= eval_budget,
        "timed_out": timed_out_flag,
        "wall_time_s": float(wall_time),
        "evals_per_second": evals_so_far / max(wall_time, 1e-9),
        "eval_seed": int(eval_seed), "algo_seed": int(algo_seed),
    }


# ======================================================================
# 2. BO_RESTART
# ======================================================================
def run_bo_restarts(grid, repeat, target_point, eval_budget, eval_seed,
                    n_restarts=BO_N_RESTARTS):
    """
    Split budget evenly across n_restarts independent BO runs.
    Each restart: fresh GP, random initialisation, independent algo seed.
    Report best-of-all-restarts.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    tag      = make_tag("BO_RESTART", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    budget_per_restart = eval_budget // n_restarts
    leftover           = eval_budget - budget_per_restart * n_restarts

    t_start        = time.time()
    evals_so_far   = 0
    global_best    = -np.inf
    global_best_eval = 0
    global_best_time = 0.0
    timed_out_flag = False

    xs_axis, step_vals, best_vals, time_axis = [], [], [], []
    header_written = False

    for r in range(n_restarts):
        if timed_out(t_start):
            timed_out_flag = True
            break

        algo_seed   = SEED_BASE + 6000 * repeat + 113 * N + 7 * r
        this_budget = budget_per_restart + (1 if r < leftover else 0)

        dims = [(-RANGE, RANGE)] * N
        opt  = SkOptimizer(dims, base_estimator="GP", acq_func=BO_ACQ,
                           acq_func_kwargs={"xi": BO_XI},
                           random_state=algo_seed)

        for i in range(this_budget):
            if timed_out(t_start):
                timed_out_flag = True
                break

            x_suggest = opt.ask()
            y, x_used = eval_selectivity_grounded(x_suggest, target_point,
                                                  grid, rng_seed=eval_seed)
            opt.tell(x_used.tolist(), y)
            evals_so_far += 1
            sel     = -y
            elapsed = time.time() - t_start

            if sel > global_best:
                global_best      = sel
                global_best_eval = evals_so_far
                global_best_time = elapsed

            xs_axis.append(evals_so_far)
            step_vals.append(sel)
            best_vals.append(global_best)
            time_axis.append(elapsed)

            if i == 0 or i == this_budget - 1 or (i + 1) % 20 == 0:
                log_step(csv_path, {
                    "optimizer": "BO_RESTART", "stage": 1,
                    "restart": r, "n_restarts": n_restarts,
                    "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                    "repeat": repeat, "step_index": evals_so_far,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": sel,
                    "best_so_far": global_best,
                    "best_found_at_eval": global_best_eval,
                    "wall_time_s": elapsed,
                    "target_x": target_point[0], "target_y": target_point[1],
                    "target_z": target_point[2],
                    "eval_seed": eval_seed, "algo_seed": algo_seed,
                }, header_written)
                header_written = True

        if timed_out_flag:
            break

    wall_time = time.time() - t_start

    return {
        "optimizer": "BO_RESTART", "tag": tag,
        "best": float(global_best if np.isfinite(global_best) else 0.0),
        "best_found_at_eval": int(global_best_eval),
        "best_found_at_time": float(global_best_time),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "budget": int(eval_budget),
        "budget_exhausted": evals_so_far >= eval_budget,
        "timed_out": timed_out_flag,
        "wall_time_s": float(wall_time),
        "evals_per_second": evals_so_far / max(wall_time, 1e-9),
        "n_restarts": n_restarts,
        "eval_seed": int(eval_seed),
    }


# ======================================================================
# 3. SWEEP_BO — method under scrutiny
# ======================================================================
def run_sweep_then_bo(grid, repeat, target_point, eval_budget, eval_seed,
                      I0=SWEEP_I0, top_k_pairs=5):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    algo_seed = SEED_BASE + 8000 * repeat + 137 * N
    tag       = make_tag("SWEEP_BO", grid, repeat, target_point)
    csv_path  = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    t_start = time.time()

    # ── Stage 0: exhaustive pair sweep ────────────────────────────────
    (pair_results, evals_so_far, best_so_far, best_at_eval,
     best_at_time, best_currents,
     xs_axis, step_vals, best_vals, header_written) = run_pair_sweep_grounded(
        grid, repeat, target_point, eval_budget, eval_seed,
        I0=I0, tag_prefix="SWEEP_BO", csv_path=csv_path,
        header_written=False, t_start=t_start,
    )

    timed_out_flag = timed_out(t_start)

    if evals_so_far >= eval_budget or not pair_results or timed_out_flag:
        wall_time = time.time() - t_start
        return {
            "optimizer": "SWEEP_BO", "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "best_found_at_time": float(best_at_time),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "budget": int(eval_budget),
            "budget_exhausted": evals_so_far >= eval_budget,
            "timed_out": timed_out_flag,
            "wall_time_s": float(wall_time),
            "evals_per_second": evals_so_far / max(wall_time, 1e-9),
            "eval_seed": int(eval_seed), "algo_seed": int(algo_seed),
        }

    remaining = eval_budget - evals_so_far

    # ── Stage 1: BO warm-started with sweep knowledge ────────────────
    dims = [(-RANGE, RANGE)] * N
    opt  = SkOptimizer(dims, base_estimator="GP", acq_func=BO_ACQ,
                       acq_func_kwargs={"xi": BO_XI},
                       random_state=algo_seed)

    # Warm-start: tell GP about top-K sweep results
    pair_results.sort(key=lambda t: t[0], reverse=True)
    K = min(top_k_pairs, len(pair_results))
    for k in range(K):
        sel_k, x_k = pair_results[k]
        opt.tell(x_k.tolist(), -sel_k)  # BO minimises

    for i in range(remaining):
        if timed_out(t_start):
            timed_out_flag = True
            break

        x_suggest = opt.ask()
        y, x_used = eval_selectivity_grounded(x_suggest, target_point,
                                              grid, rng_seed=eval_seed)
        opt.tell(x_used.tolist(), y)
        evals_so_far += 1
        sel     = -y
        elapsed = time.time() - t_start

        if sel > best_so_far:
            best_so_far  = sel
            best_at_eval = evals_so_far
            best_at_time = elapsed

        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)

        if i == 0 or i == remaining - 1 or (i + 1) % 20 == 0:
            log_step(csv_path, {
                "optimizer": "SWEEP_BO", "stage": 1,
                "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                "repeat": repeat, "step_index": evals_so_far,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "wall_time_s": elapsed,
                "target_x": target_point[0], "target_y": target_point[1],
                "target_z": target_point[2],
                "eval_seed": eval_seed, "algo_seed": algo_seed,
            }, header_written)
            header_written = True

    wall_time = time.time() - t_start

    return {
        "optimizer": "SWEEP_BO", "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "best_found_at_time": float(best_at_time),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "budget": int(eval_budget),
        "budget_exhausted": evals_so_far >= eval_budget,
        "timed_out": timed_out_flag,
        "wall_time_s": float(wall_time),
        "evals_per_second": evals_so_far / max(wall_time, 1e-9),
        "top_k_pairs": K,
        "eval_seed": int(eval_seed), "algo_seed": int(algo_seed),
    }


# ======================================================================
# 4. SWEEP_PSO — reference from main benchmark
# ======================================================================
def run_sweep_then_pso(grid, repeat, target_point, eval_budget, eval_seed,
                       I0=SWEEP_I0, top_k_pairs=5):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    algo_seed = SEED_BASE + 9100 * repeat + 97 * N
    rng       = np.random.default_rng(algo_seed)
    tag       = make_tag("SWEEP_PSO", grid, repeat, target_point)
    csv_path  = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    t_start = time.time()

    (pair_results, evals_so_far, best_so_far, best_at_eval,
     best_at_time, best_currents,
     xs_axis, step_vals, best_vals, header_written) = run_pair_sweep_grounded(
        grid, repeat, target_point, eval_budget, eval_seed,
        I0=I0, tag_prefix="SWEEP_PSO", csv_path=csv_path,
        header_written=False, t_start=t_start,
    )

    timed_out_flag = timed_out(t_start)

    if evals_so_far >= eval_budget or not pair_results or timed_out_flag:
        wall_time = time.time() - t_start
        return {
            "optimizer": "SWEEP_PSO", "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "best_found_at_time": float(best_at_time),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "budget": int(eval_budget),
            "budget_exhausted": evals_so_far >= eval_budget,
            "timed_out": timed_out_flag,
            "wall_time_s": float(wall_time),
            "evals_per_second": evals_so_far / max(wall_time, 1e-9),
            "eval_seed": int(eval_seed), "algo_seed": int(algo_seed),
        }

    remaining = eval_budget - evals_so_far
    popsize   = pso_popsize(N)

    if remaining < popsize:
        wall_time = time.time() - t_start
        return {
            "optimizer": "SWEEP_PSO", "tag": tag,
            "best": float(best_so_far),
            "best_found_at_eval": int(best_at_eval),
            "best_found_at_time": float(best_at_time),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "budget": int(eval_budget),
            "budget_exhausted": False,
            "timed_out": False,
            "wall_time_s": float(wall_time),
            "evals_per_second": evals_so_far / max(wall_time, 1e-9),
            "eval_seed": int(eval_seed), "algo_seed": int(algo_seed),
        }

    iters = remaining // popsize

    # Warm-start: top-K pairs as initial particles
    pair_results.sort(key=lambda t: t[0], reverse=True)
    K = min(top_k_pairs, popsize, len(pair_results))

    X = np.zeros((popsize, N))
    for k in range(K):
        X[k] = np.clip(pair_results[k][1], -RANGE, RANGE)
    for i in range(K, popsize):
        X[i] = rng.uniform(-RANGE, RANGE, N)
    X = np.clip(X, -RANGE, RANGE)

    V = np.zeros_like(X)
    pbest_pos = X.copy()
    pbest_val = np.empty(popsize)

    # Evaluate initial swarm
    for i in range(popsize):
        if timed_out(t_start):
            timed_out_flag = True
            break
        y, x_used = eval_selectivity_grounded(pbest_pos[i], target_point,
                                              grid, rng_seed=eval_seed)
        pbest_val[i] = y
        pbest_pos[i] = x_used
        X[i] = x_used
    evals_so_far += popsize

    g_idx     = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    swarm_best = -pbest_val[g_idx]

    if swarm_best > best_so_far:
        best_so_far  = swarm_best
        best_at_eval = evals_so_far
        best_at_time = time.time() - t_start

    xs_axis.append(evals_so_far)
    step_vals.append(swarm_best)
    best_vals.append(best_so_far)

    # Main PSO loop
    for it in range(1, iters):
        if timed_out(t_start):
            timed_out_flag = True
            break

        r1 = rng.random((popsize, N))
        r2 = rng.random((popsize, N))
        V  = (PSO_W * V
              + PSO_C1 * r1 * (pbest_pos - X)
              + PSO_C2 * r2 * (gbest_pos - X))
        V = np.clip(V, -PSO_VCLAMP, PSO_VCLAMP)
        X = np.clip(X + V, -RANGE, RANGE)

        it_best_sel = -np.inf
        for i in range(popsize):
            if timed_out(t_start):
                timed_out_flag = True
                break
            y, x_used = eval_selectivity_grounded(X[i], target_point, grid,
                                                  rng_seed=eval_seed)
            X[i] = x_used
            evals_so_far += 1
            sel = -y

            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = x_used
            if sel > it_best_sel:
                it_best_sel = sel
            if sel > best_so_far:
                best_so_far  = sel
                best_at_eval = evals_so_far
                best_at_time = time.time() - t_start

        g_idx     = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[g_idx].copy()

        xs_axis.append(evals_so_far)
        step_vals.append(it_best_sel if np.isfinite(it_best_sel) else best_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "SWEEP_PSO", "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "repeat": repeat, "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": it_best_sel if np.isfinite(it_best_sel) else best_so_far,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "wall_time_s": time.time() - t_start,
            "target_x": target_point[0], "target_y": target_point[1],
            "target_z": target_point[2],
            "eval_seed": eval_seed, "algo_seed": algo_seed,
        }, header_written)
        header_written = True

        if timed_out_flag:
            break

    wall_time = time.time() - t_start

    return {
        "optimizer": "SWEEP_PSO", "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "best_found_at_time": float(best_at_time),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "budget": int(eval_budget),
        "budget_exhausted": evals_so_far >= eval_budget,
        "timed_out": timed_out_flag,
        "wall_time_s": float(wall_time),
        "evals_per_second": evals_so_far / max(wall_time, 1e-9),
        "eval_seed": int(eval_seed), "algo_seed": int(algo_seed),
    }


# ======================================================================
# STATISTICAL ANALYSIS
# ======================================================================
def compute_statistics(df_summary):
    results = []
    for (N, tp), grp in df_summary.groupby(["N", "target_point"]):
        for opt in grp["optimizer"].unique():
            vals  = grp[grp["optimizer"] == opt]["best"].values
            times = grp[grp["optimizer"] == opt]["wall_time_s"].values
            evals = grp[grp["optimizer"] == opt]["used_evals"].values
            n     = len(vals)
            mean  = np.mean(vals)
            std   = np.std(vals, ddof=1) if n > 1 else 0.0
            sem   = std / np.sqrt(n) if n > 1 else 0.0
            ci    = (stats.t.interval(0.95, max(n - 1, 1), loc=mean, scale=sem)
                     if n > 1 else (mean, mean))
            results.append({
                "N": N, "target": tp, "optimizer": opt,
                "mean_selectivity": mean,
                "std_selectivity": std,
                "median_selectivity": np.median(vals),
                "ci_lower": ci[0], "ci_upper": ci[1],
                "mean_wall_time_s": np.mean(times),
                "mean_evals_used": np.mean(evals),
                "mean_evals_per_sec": np.mean(evals / np.maximum(times, 1e-9)),
                "n_repeats": n,
                "n_timed_out": int(grp[grp["optimizer"] == opt]["timed_out"].sum()),
            })
    return pd.DataFrame(results)


def pairwise_tests(df_summary):
    results = []
    for (N, tp), grp in df_summary.groupby(["N", "target_point"]):
        opts = sorted(grp["optimizer"].unique())
        for i, opt1 in enumerate(opts):
            for opt2 in opts[i + 1:]:
                d1 = grp[grp["optimizer"] == opt1]["best"].values
                d2 = grp[grp["optimizer"] == opt2]["best"].values
                if len(d1) < 2 or len(d2) < 2:
                    continue
                stat, pval = stats.mannwhitneyu(d1, d2, alternative="two-sided")
                n1, n2 = len(d1), len(d2)
                r = 1 - (2 * stat) / (n1 * n2)
                results.append({
                    "N": N, "target": tp,
                    "optimizer_1": opt1, "optimizer_2": opt2,
                    "U_statistic": stat, "p_value": pval,
                    "effect_size_r": r,
                    "significant_0.05": pval < 0.05,
                    "mean_1": np.mean(d1), "mean_2": np.mean(d2),
                    "better": opt1 if np.mean(d1) > np.mean(d2) else opt2,
                })
    return pd.DataFrame(results)


# ======================================================================
# PLOTTING
# ======================================================================
COLOR_MAP = {
    "BO_SINGLE":  "#888888",
    "BO_RESTART": "#8B008B",
    "SWEEP_BO":   "#FF8C00",
    "SWEEP_PSO":  "#DC143C",
}
LABEL_MAP = {
    "BO_SINGLE":  "BO (single)",
    "BO_RESTART": "BO (5 restarts)",
    "SWEEP_BO":   "Sweep → BO",
    "SWEEP_PSO":  "Sweep → PSO",
}
OPT_ORDER = ["BO_SINGLE", "BO_RESTART", "SWEEP_BO", "SWEEP_PSO"]


def ordered_opts(present):
    return [o for o in OPT_ORDER if o in present]


def plot_convergence(df, electrode_grids, target_points, repeats):
    for grid in electrode_grids:
        N    = grid[0] * grid[1]
        n_tp = len(target_points)
        fig, axes = plt.subplots(1, n_tp, figsize=(5 * n_tp, 4))
        if n_tp == 1:
            axes = [axes]

        for t_idx, tp in enumerate(target_points):
            for opt in ordered_opts(df["optimizer"].unique()):
                all_curves = []
                for rep in range(repeats):
                    tag = make_tag(opt, grid, rep, tp)
                    csv_file = os.path.join(OUTPUT_DIR, f"{tag}.csv")
                    if os.path.exists(csv_file):
                        try:
                            df_run = pd.read_csv(csv_file)
                            if "evals_so_far" in df_run.columns and "best_so_far" in df_run.columns:
                                all_curves.append(
                                    df_run[["evals_so_far", "best_so_far"]].dropna().values)
                        except Exception:
                            pass

                if all_curves:
                    max_ev = max(c[-1, 0] for c in all_curves if len(c) > 0)
                    ev_pts = np.linspace(0, max_ev, 200)
                    interp = [np.interp(ev_pts, c[:, 0], c[:, 1])
                              for c in all_curves if len(c) > 0]
                    if interp:
                        mean_c = np.mean(interp, axis=0)
                        std_c  = np.std(interp, axis=0)
                        color  = COLOR_MAP.get(opt, "blue")
                        label  = LABEL_MAP.get(opt, opt)

                        axes[t_idx].plot(ev_pts, mean_c, label=label,
                                         color=color, lw=2)
                        axes[t_idx].fill_between(ev_pts,
                                                  mean_c - std_c,
                                                  mean_c + std_c,
                                                  alpha=0.15, color=color)

            axes[t_idx].set_xlabel("Function evaluations")
            axes[t_idx].set_ylabel("Selectivity" if t_idx == 0 else "")
            axes[t_idx].set_title(
                f"({tp[0]*1e3:.1f}, {tp[1]*1e3:.1f}, {tp[2]*1e3:.1f}) mm",
                fontsize=9)
            axes[t_idx].legend(fontsize=7)
            axes[t_idx].grid(alpha=0.3)

        plt.suptitle(f"Convergence — N={N}  ({repeats} repeats, mean ± std)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"convergence_N{N}.png"), dpi=200)
        plt.close()


def plot_final_report(df, df_stats, df_tests, electrode_grids):
    """
    Single multi-panel figure for the paper:
      Row 1: Boxplot of selectivity per grid size
      Row 2: Wall-clock time per grid size
      Row 3: Pareto — selectivity vs wall-clock time (scatter)
    """
    n_grids = len(electrode_grids)

    fig = plt.figure(figsize=(5 * n_grids, 14))
    gs  = gridspec.GridSpec(3, n_grids, hspace=0.35, wspace=0.3)

    for col, grid in enumerate(electrode_grids):
        N    = grid[0] * grid[1]
        data = df[df["N"] == N]
        opts = ordered_opts(data["optimizer"].unique())

        # ── Row 1: Selectivity boxplot ──
        ax1 = fig.add_subplot(gs[0, col])
        box_data = [data[data["optimizer"] == o]["best"].values for o in opts]
        labels   = [LABEL_MAP.get(o, o) for o in opts]
        colors   = [COLOR_MAP.get(o, "blue") for o in opts]

        bp = ax1.boxplot(box_data, labels=labels, patch_artist=True,
                         widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax1.set_title(f"N={N} ({grid[0]}×{grid[1]})", fontweight="bold")
        ax1.set_ylabel("Selectivity" if col == 0 else "")
        ax1.grid(alpha=0.3, axis="y")
        ax1.tick_params(axis="x", rotation=30, labelsize=7)

        # Annotate significance stars between SWEEP_BO and BO_RESTART
        if not df_tests.empty:
            pair = df_tests[(df_tests["N"] == N) &
                            (((df_tests["optimizer_1"] == "SWEEP_BO") &
                              (df_tests["optimizer_2"] == "BO_RESTART")) |
                             ((df_tests["optimizer_1"] == "BO_RESTART") &
                              (df_tests["optimizer_2"] == "SWEEP_BO")))]
            if not pair.empty:
                p = pair.iloc[0]["p_value"]
                star = "***" if p < 0.001 else "**" if p < 0.01 \
                       else "*" if p < 0.05 else "n.s."
                ymax = ax1.get_ylim()[1]
                # Find positions of the two bars
                idx_bo = opts.index("BO_RESTART") + 1 if "BO_RESTART" in opts else None
                idx_sw = opts.index("SWEEP_BO") + 1 if "SWEEP_BO" in opts else None
                if idx_bo and idx_sw:
                    ax1.annotate(star,
                                 xy=((idx_bo + idx_sw) / 2, ymax * 0.97),
                                 ha="center", fontsize=10, fontweight="bold")

        # ── Row 2: Wall-clock time boxplot ──
        ax2 = fig.add_subplot(gs[1, col])
        time_data = [data[data["optimizer"] == o]["wall_time_s"].values
                     for o in opts]
        bp2 = ax2.boxplot(time_data, labels=labels, patch_artist=True,
                          widths=0.6)
        for patch, c in zip(bp2["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax2.set_ylabel("Wall-clock time (s)" if col == 0 else "")
        ax2.axhline(TIMEOUT_SECONDS, ls="--", color="red", alpha=0.5,
                     label=f"Timeout ({TIMEOUT_SECONDS:.0f}s)")
        ax2.grid(alpha=0.3, axis="y")
        ax2.tick_params(axis="x", rotation=30, labelsize=7)
        if col == 0:
            ax2.legend(fontsize=7)

        # ── Row 3: Pareto scatter ──
        ax3 = fig.add_subplot(gs[2, col])
        for o in opts:
            od = data[data["optimizer"] == o]
            ax3.scatter(od["wall_time_s"], od["best"],
                        color=COLOR_MAP.get(o, "blue"),
                        label=LABEL_MAP.get(o, o),
                        s=50, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax3.set_xlabel("Wall-clock time (s)")
        ax3.set_ylabel("Selectivity" if col == 0 else "")
        ax3.axvline(TIMEOUT_SECONDS, ls="--", color="red", alpha=0.3)
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=6, loc="lower right")

    fig.suptitle("Concern 1: Sweep+Optimiser vs BO Restarts — Fair Comparison",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_report.png"),
                dpi=250, bbox_inches="tight")
    plt.close()


def plot_significance_heatmaps(df, df_tests, electrode_grids):
    for grid in electrode_grids:
        N = grid[0] * grid[1]
        test_data = df_tests[df_tests["N"] == N]
        if test_data.empty:
            continue

        optimizers = ordered_opts(df[df["N"] == N]["optimizer"].unique())
        n_opts = len(optimizers)
        pval_matrix = np.ones((n_opts, n_opts))

        for _, row in test_data.iterrows():
            o1, o2 = row["optimizer_1"], row["optimizer_2"]
            if o1 in optimizers and o2 in optimizers:
                i = optimizers.index(o1)
                j = optimizers.index(o2)
                pval_matrix[i, j] = row["p_value"]
                pval_matrix[j, i] = row["p_value"]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(pval_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1)
        labels = [LABEL_MAP.get(o, o) for o in optimizers]
        ax.set_xticks(np.arange(n_opts))
        ax.set_yticks(np.arange(n_opts))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(n_opts):
            for j in range(n_opts):
                if i != j:
                    p = pval_matrix[i, j]
                    txt = ("***" if p < 0.001 else "**" if p < 0.01
                           else "*" if p < 0.05 else f"{p:.3f}")
                    ax.text(j, i, txt, ha="center", va="center",
                            color="white" if p < 0.05 else "black",
                            fontweight="bold" if p < 0.05 else "normal",
                            fontsize=9)

        ax.set_title(f"Pairwise Mann-Whitney U (N={N})\n"
                     f"* p<0.05  ** p<0.01  *** p<0.001")
        plt.colorbar(im, ax=ax, label="p-value")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,
                                 f"significance_heatmap_N{N}.png"), dpi=200)
        plt.close()


def plot_efficiency(df, electrode_grids):
    """
    Bar chart: selectivity achieved per second of wall-clock time.
    This is the key trade-off metric.
    """
    fig, axes = plt.subplots(1, len(electrode_grids),
                             figsize=(5 * len(electrode_grids), 4.5))
    if len(electrode_grids) == 1:
        axes = [axes]

    for col, grid in enumerate(electrode_grids):
        N    = grid[0] * grid[1]
        data = df[df["N"] == N]
        opts = ordered_opts(data["optimizer"].unique())

        means, stds, colors, labels = [], [], [], []
        for o in opts:
            od = data[data["optimizer"] == o]
            efficiency = od["best"] / od["wall_time_s"].clip(lower=0.01)
            means.append(efficiency.mean())
            stds.append(efficiency.std())
            colors.append(COLOR_MAP.get(o, "blue"))
            labels.append(LABEL_MAP.get(o, o))

        x_pos = np.arange(len(opts))
        axes[col].bar(x_pos, means, yerr=stds, color=colors, alpha=0.7,
                      capsize=4, edgecolor="black", linewidth=0.5)
        axes[col].set_xticks(x_pos)
        axes[col].set_xticklabels(labels, rotation=30, fontsize=7)
        axes[col].set_title(f"N={N}")
        axes[col].set_ylabel("Selectivity / second" if col == 0 else "")
        axes[col].grid(alpha=0.3, axis="y")

    plt.suptitle("Efficiency: Selectivity per Second of Wall-Clock Time")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "efficiency_comparison.png"), dpi=200)
    plt.close()


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    t0_global = time.time()

    print("\n" + "=" * 74)
    print("  CONCERN-1 BENCHMARK — SCIENTIFICALLY FAIR COMPARISON")
    print("  Sweep+Optimiser  vs  BO Cold Restarts  vs  Vanilla BO")
    print(f"  Timeout per run: {TIMEOUT_SECONDS:.0f}s | Repeats: {REPEATS}")
    print("=" * 74 + "\n")

    summaries = []

    for grid in ELECTRODE_GRIDS:
        N = grid[0] * grid[1]
        n_pairs = N * (N - 1) // 2

        for repeat in range(REPEATS):
            for t_idx, tp in enumerate(TARGET_POINTS):
                budget    = EVALS_PER_DIM * N
                eval_seed = SEED_BASE + 1_000_000 * repeat + 10_000 * N + t_idx

                print(f"\n── N={N}  pairs={n_pairs}  repeat={repeat+1}/{REPEATS}  "
                      f"target=({tp[0]*1e3:.1f},{tp[1]*1e3:.1f},{tp[2]*1e3:.1f})mm  "
                      f"budget={budget} ──")

                # 1. Vanilla BO
                print(f"  [1/4] BO_SINGLE ...", end=" ", flush=True)
                res = run_bo_single(grid, repeat, tp, budget, eval_seed)
                summaries.append(res)
                print(f"sel={res['best']:.4f}  evals={res['used_evals']}  "
                      f"time={res['wall_time_s']:.1f}s  "
                      f"{'TIMEOUT' if res['timed_out'] else 'ok'}")

                # 2. BO with cold restarts (PI's suggestion)
                print(f"  [2/4] BO_RESTART ...", end=" ", flush=True)
                res = run_bo_restarts(grid, repeat, tp, budget, eval_seed)
                summaries.append(res)
                print(f"sel={res['best']:.4f}  evals={res['used_evals']}  "
                      f"time={res['wall_time_s']:.1f}s  "
                      f"{'TIMEOUT' if res['timed_out'] else 'ok'}")

                # 3. Sweep → BO (method under scrutiny)
                print(f"  [3/4] SWEEP_BO ...", end=" ", flush=True)
                res = run_sweep_then_bo(grid, repeat, tp, budget, eval_seed)
                summaries.append(res)
                print(f"sel={res['best']:.4f}  evals={res['used_evals']}  "
                      f"time={res['wall_time_s']:.1f}s  "
                      f"{'TIMEOUT' if res['timed_out'] else 'ok'}")

                # 4. Sweep → PSO (reference)
                print(f"  [4/4] SWEEP_PSO ...", end=" ", flush=True)
                res = run_sweep_then_pso(grid, repeat, tp, budget, eval_seed)
                summaries.append(res)
                print(f"sel={res['best']:.4f}  evals={res['used_evals']}  "
                      f"time={res['wall_time_s']:.1f}s  "
                      f"{'TIMEOUT' if res['timed_out'] else 'ok'}")

    # ── Save raw results ─────────────────────────────────────────────
    df = pd.DataFrame(summaries)
    if df.empty:
        raise RuntimeError("No runs completed.")
    df.to_csv(os.path.join(OUTPUT_DIR, "optimizer_summary.csv"), index=False)

    # ── Statistics ────────────────────────────────────────────────────
    print("\n=== Computing statistics ===")
    df_stats = compute_statistics(df)
    df_stats.to_csv(os.path.join(OUTPUT_DIR, "statistics_summary.csv"),
                    index=False)

    print("=== Pairwise tests ===")
    df_tests = pairwise_tests(df)
    df_tests.to_csv(os.path.join(OUTPUT_DIR, "pairwise_tests.csv"),
                    index=False)

    # ── Plots ─────────────────────────────────────────────────────────
    print("=== Generating plots ===")
    plot_convergence(df, ELECTRODE_GRIDS, TARGET_POINTS, REPEATS)
    plot_significance_heatmaps(df, df_tests, ELECTRODE_GRIDS)
    plot_efficiency(df, ELECTRODE_GRIDS)
    plot_final_report(df, df_stats, df_tests, ELECTRODE_GRIDS)

    # ── Console report ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n{'Optimizer':<16} {'N':<4} {'Mean Sel':<10} {'Median':<10} "
          f"{'Std':<8} {'Mean Time':<10} {'Evals Used':<11} {'Timeouts':<9}")
    print("-" * 90)
    for N in sorted(df["N"].unique()):
        for opt in ordered_opts(df["optimizer"].unique()):
            sub = df[(df["N"] == N) & (df["optimizer"] == opt)]
            if sub.empty:
                continue
            vals  = sub["best"].values
            times = sub["wall_time_s"].values
            evals = sub["used_evals"].values
            to    = sub["timed_out"].sum()
            print(f"{LABEL_MAP.get(opt, opt):<16} {N:<4} "
                  f"{np.mean(vals):<10.4f} {np.median(vals):<10.4f} "
                  f"{np.std(vals):<8.4f} {np.mean(times):<10.1f} "
                  f"{np.mean(evals):<11.0f} {to:<9d}")
        print("-" * 90)

    # ── Key comparison: SWEEP_BO vs BO_RESTART ────────────────────────
    print("\n" + "=" * 90)
    print("KEY COMPARISON:  Sweep→BO  vs  BO with Restarts")
    print("(This directly addresses the PI's concern)")
    print("=" * 90)

    for N in sorted(df["N"].unique()):
        sw = df[(df["N"] == N) & (df["optimizer"] == "SWEEP_BO")]["best"].values
        br = df[(df["N"] == N) & (df["optimizer"] == "BO_RESTART")]["best"].values

        if len(sw) < 2 or len(br) < 2:
            print(f"\n  N={N}: insufficient data for comparison")
            continue

        stat, pval = stats.mannwhitneyu(sw, br, alternative="two-sided")
        n1, n2 = len(sw), len(br)
        r = 1 - (2 * stat) / (n1 * n2)
        winner = "SWEEP_BO" if np.mean(sw) > np.mean(br) else "BO_RESTART"

        sw_time = df[(df["N"] == N) & (df["optimizer"] == "SWEEP_BO")]["wall_time_s"].values
        br_time = df[(df["N"] == N) & (df["optimizer"] == "BO_RESTART")]["wall_time_s"].values

        print(f"\n  N={N}:")
        print(f"    Sweep→BO      mean={np.mean(sw):.4f} ± {np.std(sw):.4f}  "
              f"time={np.mean(sw_time):.1f}s")
        print(f"    BO (restarts) mean={np.mean(br):.4f} ± {np.std(br):.4f}  "
              f"time={np.mean(br_time):.1f}s")
        print(f"    Mann-Whitney U: p={pval:.4f}  effect_size_r={r:+.3f}")
        print(f"    Winner: {LABEL_MAP.get(winner, winner)}")
        if pval < 0.05:
            print(f"    → Statistically significant at α=0.05")
        else:
            print(f"    → NOT significant at α=0.05 "
                  f"(note: {REPEATS} repeats limits power)")

    # ── Timeout report ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TIMEOUT REPORT")
    print("=" * 90)
    for opt in ordered_opts(df["optimizer"].unique()):
        sub = df[df["optimizer"] == opt]
        n_to = sub["timed_out"].sum()
        n_total = len(sub)
        pct = 100 * n_to / n_total if n_total else 0
        avg_evals = sub["used_evals"].mean()
        avg_budget = sub["budget"].mean()
        print(f"  {LABEL_MAP.get(opt, opt):<16}  "
              f"timeouts={n_to}/{n_total} ({pct:.0f}%)  "
              f"avg_evals={avg_evals:.0f}/{avg_budget:.0f}")

    # ── Scientific validity statement ─────────────────────────────────
    print("\n" + "=" * 90)
    print("SCIENTIFIC VALIDITY NOTES")
    print("=" * 90)
    print(f"""
  1. BUDGET FAIRNESS: All methods receive the same total evaluation budget
     ({EVALS_PER_DIM}×N). Sweep evaluations are counted against this budget.
  2. LANDSCAPE IDENTITY: All methods within a condition share the same
     eval_seed, so the objective function is deterministic and identical.
  3. TIMEOUT: {TIMEOUT_SECONDS:.0f}s wall-clock limit per run. BO methods may time
     out at high N due to O(n³) GP fitting — this is a genuine disadvantage
     of BO, not an artefact.  Partial results are recorded honestly.
  4. STATISTICAL POWER: With {REPEATS} repeats, Mann-Whitney U has limited
     power.  Effect sizes (rank-biserial r) are reported alongside p-values.
     Interpret with caution; increase REPEATS for publication.
  5. INDEPENDENCE: Each method uses a separate algo_seed for its stochastic
     decisions.  Only the objective landscape (eval_seed) is shared.
""")

    elapsed = time.time() - t0_global
    print(f"Total benchmark time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR}\n")