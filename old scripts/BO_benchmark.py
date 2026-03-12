#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BO-only single-target benchmark (derived from your final_benchmark.py style)

Runs (single target point):
  1) BO_GROUNDED     : BO over FULL N-dim currents (box clipped)
  2) SWEEP_BO        : Pair sweep at fixed I0, then BO warm-start (still uses anti-repeat in BO stage)
  3) BO_BIPOLAR      : BO over bipolar patterns only (pair + amplitude)
  4) BO_TRIPOLAR     : BO over tripolar patterns only (tripole + amplitude)

Key fix vs "stuck":
  - Hard no-duplicate evaluation guard (quantised signature + retry/jitter/random fallback)
  - Epsilon-random exploration
  - Acquisition optimizer set to "sampling" with many candidates (API-compatible across skopt versions)
  - Logs currents_this_step + step_selectivity so you can verify exploration
"""

import os
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Simulator import (same pattern as your script) ---
try:
    from utils.run_selectivity_simulation import run_selectivity_simulation
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from run_selectivity_simulation import run_selectivity_simulation  # type: ignore

# --- Bayesian Optimizer ---
from skopt import Optimizer as SkOptimizer
from skopt.space import Integer, Real


# =============================================================================
# CONFIG
# =============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "benchmark_bo_single_target")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tissue / geometry
RADIUS  = 0.01   # m
HEIGHT  = 0.04   # m
SIGMA_T = 0.08   # S/m

# Electrode layouts (N = n_rows * n_per_row)
ELECTRODE_GRIDS = [
    (4, 3),  # N=12
]

# Single target point (edit)
TARGET_POINT = (0.002, 0.0035, 0.010)  # meters

# Current bounds
RANGE = 1e-3  # +/- 1 mA

# Budgets / repeats
EVALS_PER_DIM = 200
REPEATS = 3
SEED_BASE = 42

# BO settings
BO_ACQ   = "EI"    # "EI" or "LCB"
BO_XI    = 0.05    # EI exploration knob (higher => more exploration)
BO_KAPPA = 5.0     # LCB exploration knob (higher => more exploration)

def bo_n_initial(dim: int) -> int:
    return max(10, 2 * dim)

# Acquisition optimisation mode
ACQ_OPTIMIZER = "sampling"   # "sampling" is robust in high-d, and avoids local optimizer repeats
ACQ_N_POINTS  = 15000        # number of acquisition candidates if sampling is supported

# Sweep settings (only for SWEEP_BO)
SWEEP_I0 = 0.8e-3
SWEEP_TOP_K = 20

# --- Anti-repeat / exploration controls ---
CURR_QUANT_A      = 1e-7   # quantisation for currents (A) used in signature
AMP_QUANT_A       = 1e-7   # quantisation for amplitudes (A) in constrained spaces
MAX_DUP_TRIES     = 25     # how many tries to avoid duplicates before forcing random
EPS_RANDOM        = 0.15   # fraction of iterations to sample random even if BO behaves
JITTER_SIGMA      = 0.05 * RANGE  # jitter scale used when BO proposes duplicates


# =============================================================================
# EVALUATION (grounded, box clip)
# =============================================================================
def eval_selectivity_grounded(x, target_point, grid, rng_seed=None):
    """
    Evaluate selectivity with grounded outer boundary, NO zero-sum constraint.
    Only box clipping to [-RANGE, RANGE].

    Returns:
        y   : NEGATIVE selectivity (for minimizers)
        x_c : clipped currents actually used (N-dim)
    """
    x_c = np.asarray(x, dtype=float)
    x_c = np.clip(x_c, -RANGE, RANGE)

    res = run_selectivity_simulation(
        n_rows=grid[0],
        n_per_row=grid[1],
        currents=x_c,
        target_point=target_point,
        radius=RADIUS,
        height=HEIGHT,
        sigma=SIGMA_T,
        n_off_samples=1200,
        E_th=1, k=0.5,
        metric="activation",
        grounded_boundary=True,
        use_activating_function=False,
        R_outer=0.10,
        rng=(SEED_BASE if rng_seed is None else rng_seed),
    )
    return -float(res["selectivity"]), x_c


# =============================================================================
# SIGNATURE / NO-DUPLICATE HELPERS
# =============================================================================
def quantize_currents(x: np.ndarray) -> np.ndarray:
    """Clip then quantize to fixed resolution for stable signature checks."""
    x = np.clip(np.asarray(x, dtype=float), -RANGE, RANGE)
    return np.round(x / CURR_QUANT_A) * CURR_QUANT_A

def currents_signature(xq: np.ndarray):
    """Signature as integer bins (stable + hashable)."""
    bins = np.round(np.asarray(xq) / CURR_QUANT_A).astype(np.int64)
    return tuple(bins.tolist())

def quantize_amp(a: float) -> float:
    a = float(np.clip(a, 0.0, RANGE))
    return float(np.round(a / AMP_QUANT_A) * AMP_QUANT_A)

def sample_random_currents(N: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-RANGE, RANGE, size=N)

def jitter_currents(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float) + rng.normal(0.0, JITTER_SIGMA, size=x.shape), -RANGE, RANGE)

def choose_unique_currents_from_bo(opt: SkOptimizer, N: int, seen_sigs: set, rng: np.random.Generator):
    """
    For full-space BO:
      - with prob EPS_RANDOM -> random
      - else -> opt.ask()
    Enforce uniqueness on quantized signature; if duplicate, retry, jitter, then force random.
    Returns:
        xq, sig, was_forced_random, dup_tries
    """
    was_forced_random = False
    dup_tries = 0

    for t in range(MAX_DUP_TRIES):
        dup_tries = t

        if rng.random() < EPS_RANDOM:
            x = sample_random_currents(N, rng)
            was_forced_random = True
        else:
            x = np.asarray(opt.ask(), dtype=float)

        xq = quantize_currents(x)
        sig = currents_signature(xq)
        if sig not in seen_sigs:
            return xq, sig, was_forced_random, dup_tries

        # jitter attempt
        x = jitter_currents(x, rng)
        xq = quantize_currents(x)
        sig = currents_signature(xq)
        if sig not in seen_sigs:
            return xq, sig, True, dup_tries + 1

    # Final fallback: force random until unique
    while True:
        x = sample_random_currents(N, rng)
        xq = quantize_currents(x)
        sig = currents_signature(xq)
        if sig not in seen_sigs:
            return xq, sig, True, MAX_DUP_TRIES


def choose_unique_idx_amp(opt: SkOptimizer, n_choices: int, seen_sigs: set, rng: np.random.Generator):
    """
    For bipolar/tripolar:
      decision vars are (idx, amp).
    Enforce uniqueness on (idx, amp_quant_bin).
    Returns:
        idx, amp, sig, was_forced_random, dup_tries
    """
    was_forced_random = False
    dup_tries = 0

    for t in range(MAX_DUP_TRIES):
        dup_tries = t

        if rng.random() < EPS_RANDOM:
            idx = int(rng.integers(0, n_choices))
            amp = quantize_amp(rng.uniform(0.0, RANGE))
            was_forced_random = True
        else:
            idx, amp = opt.ask()
            idx = int(np.clip(int(idx), 0, n_choices - 1))
            amp = quantize_amp(float(amp))

        sig = (idx, int(round(amp / AMP_QUANT_A)))
        if sig not in seen_sigs:
            return idx, amp, sig, was_forced_random, dup_tries

        # jitter amp
        amp2 = quantize_amp(amp + float(rng.normal(0.0, 0.05 * RANGE)))
        sig2 = (idx, int(round(amp2 / AMP_QUANT_A)))
        if sig2 not in seen_sigs:
            return idx, amp2, sig2, True, dup_tries + 1

    # Final fallback
    while True:
        idx = int(rng.integers(0, n_choices))
        amp = quantize_amp(rng.uniform(0.0, RANGE))
        sig = (idx, int(round(amp / AMP_QUANT_A)))
        if sig not in seen_sigs:
            return idx, amp, sig, True, MAX_DUP_TRIES


# =============================================================================
# skopt construction (API-compatible across versions)
# =============================================================================
def make_skopt_optimizer(
    dimensions,
    random_state: int,
    n_initial_points: int,
):
    """
    Create a skopt.Optimizer in a way that:
      - never passes n_points into __init__ (that causes the crash)
      - uses acq_optimizer_kwargs={'n_points': ...} when supported
      - falls back gracefully if some kwargs aren't supported in local skopt
    """
    acq_kwargs = {"xi": BO_XI} if BO_ACQ.upper() == "EI" else {"kappa": BO_KAPPA}

    base_kwargs = dict(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func=BO_ACQ,
        acq_func_kwargs=acq_kwargs,
        random_state=random_state,
        n_initial_points=n_initial_points,
    )

    # Try to enable sampling optimiser with many candidates if supported
    if ACQ_OPTIMIZER:
        base_kwargs["acq_optimizer"] = ACQ_OPTIMIZER
    if ACQ_OPTIMIZER == "sampling":
        base_kwargs["acq_optimizer_kwargs"] = {"n_points": ACQ_N_POINTS}

    # Progressive fallback: strip kwargs if TypeError
    for drop_keys in [
        [],  # first attempt: everything
        ["acq_optimizer_kwargs"],  # some versions don't support this
        ["acq_optimizer_kwargs", "acq_optimizer"],  # some versions don't support acq_optimizer
        ["acq_func_kwargs"],  # very old versions (unlikely)
    ]:
        try_kwargs = dict(base_kwargs)
        for k in drop_keys:
            try_kwargs.pop(k, None)
        try:
            return SkOptimizer(**try_kwargs)
        except TypeError:
            continue

    # If everything failed, raise the last TypeError
    return SkOptimizer(**base_kwargs)


# =============================================================================
# UTILITIES (logging + plotting)
# =============================================================================
def make_tag(prefix, grid, repeat, tp):
    n_rows, n_per_row = grid
    tx, ty, tz = tp
    return (f"{prefix}_N{n_rows*n_per_row}_grid{n_rows}x{n_per_row}_r{repeat}"
            f"_x{tx*1e3:.1f}_y{ty*1e3:.1f}_z{tz*1e3:.1f}")

def save_progress_plot(xs, bests, tag, target_point):
    plt.figure(figsize=(7, 4))
    plt.plot(xs, bests, "-", lw=2)
    plt.xlabel("Function evaluations")
    plt.ylabel("Selectivity (best-so-far)")
    tx, ty, tz = target_point
    plt.title(f"{tag}\nTarget: ({tx*1e3:.1f}, {ty*1e3:.1f}, {tz*1e3:.1f}) mm", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_progress.png"), dpi=150)
    plt.close()

def log_step(csv_path, row_dict, header_written):
    pd.DataFrame([row_dict]).to_csv(csv_path, mode="a", index=False, header=not header_written)


# =============================================================================
# SWEEP stage (returns ALL evaluated signatures to avoid re-eval in BO warm start)
# =============================================================================
def run_pair_sweep_grounded(
    grid, target_point, eval_budget, eval_seed: int,
    I0=SWEEP_I0, top_k=SWEEP_TOP_K,
):
    """
    Sweep all unordered pairs (i<j) with [+I0 at i, -I0 at j].
    If budget < total pairs, evaluate a deterministic random subset of size eval_budget.

    Returns:
      init_X (top-K current vectors), init_y (objectives for those),
      best_so_far, best_at_eval, best_currents,
      used_evals, sweep_time_s,
      sweep_seen_sigs (signatures of ALL evaluated sweep currents)
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs = len(all_pairs)

    rng = np.random.default_rng(eval_seed + 777)
    if n_pairs > eval_budget:
        idx = rng.choice(n_pairs, size=eval_budget, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    t0 = time.perf_counter()

    results = []  # (sel, currents_vec, y)
    sweep_seen_sigs = set()

    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    used_evals = 0
    for (i, j) in pairs:
        currents = np.zeros(N, dtype=float)
        currents[i] = +I0
        currents[j] = -I0
        currents = quantize_currents(currents)

        sig = currents_signature(currents)
        sweep_seen_sigs.add(sig)

        y, x_used = eval_selectivity_grounded(currents, target_point, grid, rng_seed=eval_seed)
        sel = -y
        used_evals += 1

        results.append((sel, x_used.copy(), float(y)))

        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = used_evals
            best_currents = x_used.copy()

    sweep_time_s = time.perf_counter() - t0

    results.sort(key=lambda t: t[0], reverse=True)
    k = min(top_k, len(results))
    init_X = [results[r][1].tolist() for r in range(k)]
    init_y = [results[r][2] for r in range(k)]  # objective (neg selectivity)

    return (
        init_X, init_y,
        float(best_so_far), int(best_at_eval), best_currents,
        int(used_evals), float(sweep_time_s),
        sweep_seen_sigs
    )


# =============================================================================
# 1) BO over FULL N-dim currents (no sweep)
# =============================================================================
def run_bo_grounded_single_target(grid, repeat, target_point, eval_budget, eval_seed: int):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    algo_seed = SEED_BASE + 5000 * repeat + 23 * N
    tag = make_tag("BO", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    opt = make_skopt_optimizer(
        dimensions=[Real(-RANGE, RANGE)] * N,
        random_state=algo_seed,
        n_initial_points=bo_n_initial(N),
    )

    rng = np.random.default_rng(algo_seed + 111)
    seen = set()

    t0 = time.perf_counter()

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, best_vals = [], []
    header_written = False

    for step in tqdm(range(eval_budget), desc=f"{tag}_bo", leave=False):
        xq, sig, was_rand, dup_tries = choose_unique_currents_from_bo(opt, N, seen, rng)
        seen.add(sig)

        y, x_used = eval_selectivity_grounded(xq, target_point, grid, rng_seed=eval_seed)
        sel = -y

        # Tell BO the point that was actually evaluated (already quantized/clipped)
        opt.tell(xq.tolist(), float(y))

        evals_so_far += 1

        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()

        xs_axis.append(evals_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "BO_GROUNDED",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "repeat": repeat,
            "step_index": step + 1,
            "evals_so_far": evals_so_far,
            "step_selectivity": sel,
            "step_best_selectivity": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "currents_this_step": x_used.tolist(),
            "currents_gen_best": best_currents.tolist(),
            "was_forced_random": bool(was_rand),
            "dup_tries": int(dup_tries),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)
        header_written = True

    total_time_s = time.perf_counter() - t0
    save_progress_plot(xs_axis, best_vals, tag, target_point)

    return {
        "optimizer": "BO_GROUNDED",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": int(N),
        "grid": grid,
        "repeat": int(repeat),
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
        "time_total_s": float(total_time_s),
        "time_sweep_s": 0.0,
        "time_bo_s": float(total_time_s),
        "time_per_eval_s": float(total_time_s / max(1, evals_so_far)),
    }


# =============================================================================
# 2) SWEEP -> BO warm-start (still anti-repeat)
# =============================================================================
def run_sweep_then_bo_single_target(grid, repeat, target_point, eval_budget, eval_seed: int):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    algo_seed = SEED_BASE + 8000 * repeat + 29 * N
    tag = make_tag("SWEEP_BO", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    t_total0 = time.perf_counter()

    # --- Stage 0: sweep ---
    (init_X, init_y,
     best_so_far, best_at_eval, best_currents,
     used_sweep_evals, sweep_time_s,
     sweep_seen_sigs) = run_pair_sweep_grounded(
        grid, target_point, eval_budget=eval_budget, eval_seed=eval_seed,
        I0=SWEEP_I0, top_k=SWEEP_TOP_K
    )

    evals_so_far = used_sweep_evals
    header_written = False

    # Log sweep summary row
    log_step(csv_path, {
        "optimizer": "SWEEP_BO",
        "stage": 0,
        "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": evals_so_far,
        "step_selectivity": best_so_far,
        "step_best_selectivity": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
        "I0": SWEEP_I0,
        "top_k_pairs": SWEEP_TOP_K,
        "currents_this_step": best_currents.tolist(),
        "currents_gen_best": best_currents.tolist(),
        "was_forced_random": False,
        "dup_tries": 0,
        "eval_seed": eval_seed,
        "algo_seed": algo_seed,
        "sweep_used_evals": used_sweep_evals,
        "sweep_time_s": sweep_time_s,
    }, header_written)
    header_written = True

    xs_axis = [evals_so_far]
    best_vals = [best_so_far]

    # If sweep consumed all budget
    if evals_so_far >= eval_budget:
        total_time_s = time.perf_counter() - t_total0
        save_progress_plot(xs_axis, best_vals, tag, target_point)
        return {
            "optimizer": "SWEEP_BO",
            "tag": tag,
            "best": float(best_so_far),
            "best_found_at_eval": int(best_at_eval),
            "N": int(N),
            "grid": grid,
            "repeat": int(repeat),
            "target_point": target_point,
            "used_evals": int(evals_so_far),
            "eval_seed": int(eval_seed),
            "algo_seed": int(algo_seed),
            "time_total_s": float(total_time_s),
            "time_sweep_s": float(sweep_time_s),
            "time_bo_s": 0.0,
            "time_per_eval_s": float(total_time_s / max(1, evals_so_far)),
        }

    remaining = eval_budget - evals_so_far

    # --- Stage 1: BO warm-start ---
    # Reduce random initial points by how many points we preloaded
    n_init = max(0, bo_n_initial(N) - len(init_X))
    opt = make_skopt_optimizer(
        dimensions=[Real(-RANGE, RANGE)] * N,
        random_state=algo_seed,
        n_initial_points=n_init,
    )

    # Seed BO with top-K sweep points
    for x0, y0 in zip(init_X, init_y):
        opt.tell(list(map(float, x0)), float(y0))

    # Seen set starts with ALL sweep-evaluated signatures (so BO won't re-evaluate them)
    rng = np.random.default_rng(algo_seed + 222)
    seen = set(sweep_seen_sigs)

    t_bo0 = time.perf_counter()

    for step in tqdm(range(remaining), desc=f"{tag}_bo", leave=False):
        xq, sig, was_rand, dup_tries = choose_unique_currents_from_bo(opt, N, seen, rng)
        seen.add(sig)

        y, x_used = eval_selectivity_grounded(xq, target_point, grid, rng_seed=eval_seed)
        sel = -y

        opt.tell(xq.tolist(), float(y))

        evals_so_far += 1

        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()

        xs_axis.append(evals_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "SWEEP_BO",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "repeat": repeat,
            "step_index": step + 1,
            "evals_so_far": evals_so_far,
            "step_selectivity": sel,
            "step_best_selectivity": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "currents_this_step": x_used.tolist(),
            "currents_gen_best": best_currents.tolist(),
            "was_forced_random": bool(was_rand),
            "dup_tries": int(dup_tries),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)

    bo_time_s = time.perf_counter() - t_bo0
    total_time_s = time.perf_counter() - t_total0

    save_progress_plot(xs_axis, best_vals, tag, target_point)

    return {
        "optimizer": "SWEEP_BO",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": int(N),
        "grid": grid,
        "repeat": int(repeat),
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
        "time_total_s": float(total_time_s),
        "time_sweep_s": float(sweep_time_s),
        "time_bo_s": float(bo_time_s),
        "time_per_eval_s": float(total_time_s / max(1, evals_so_far)),
    }


# =============================================================================
# 3) BO constrained to BIPOLAR patterns only (no sweep)
# =============================================================================
def build_ordered_pairs(N: int):
    return [(i, j) for i in range(N) for j in range(N) if i != j]

def run_bo_bipolar_single_target(grid, repeat, target_point, eval_budget, eval_seed: int):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    pairs = build_ordered_pairs(N)
    n_pairs = len(pairs)

    algo_seed = SEED_BASE + 12000 * repeat + 31 * N
    tag = make_tag("BO_BIPOLAR", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    opt = make_skopt_optimizer(
        dimensions=[Integer(0, n_pairs - 1), Real(0.0, RANGE)],
        random_state=algo_seed,
        n_initial_points=bo_n_initial(2),
    )

    rng = np.random.default_rng(algo_seed + 999)
    seen = set()

    t0 = time.perf_counter()

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, best_vals = [], []
    header_written = False

    for step in tqdm(range(eval_budget), desc=f"{tag}_bo", leave=False):
        pair_idx, amp, sig, was_rand, dup_tries = choose_unique_idx_amp(opt, n_pairs, seen, rng)
        seen.add(sig)

        i, j = pairs[pair_idx]
        currents = np.zeros(N, dtype=float)
        currents[i] = +amp
        currents[j] = -amp
        currents = quantize_currents(currents)

        y, x_used = eval_selectivity_grounded(currents, target_point, grid, rng_seed=eval_seed)
        sel = -y

        opt.tell([pair_idx, amp], float(y))

        evals_so_far += 1

        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()

        xs_axis.append(evals_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "BO_BIPOLAR",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "repeat": repeat,
            "step_index": step + 1,
            "evals_so_far": evals_so_far,
            "step_selectivity": sel,
            "step_best_selectivity": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "bipolar_pair_idx": int(pair_idx),
            "bipolar_i": int(i),
            "bipolar_j": int(j),
            "bipolar_amp": float(amp),
            "currents_this_step": x_used.tolist(),
            "currents_gen_best": best_currents.tolist(),
            "was_forced_random": bool(was_rand),
            "dup_tries": int(dup_tries),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)
        header_written = True

    total_time_s = time.perf_counter() - t0
    save_progress_plot(xs_axis, best_vals, tag, target_point)

    return {
        "optimizer": "BO_BIPOLAR",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": int(N),
        "grid": grid,
        "repeat": int(repeat),
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
        "time_total_s": float(total_time_s),
        "time_sweep_s": 0.0,
        "time_bo_s": float(total_time_s),
        "time_per_eval_s": float(total_time_s / max(1, evals_so_far)),
    }


# =============================================================================
# 4) BO constrained to TRIPOLAR patterns only (no sweep)
# =============================================================================
def build_tripolar_patterns(N: int):
    """
    Tripolar: choose cathode c and two anodes (a1<a2) from remaining electrodes.
    Currents: cathode = -I, anodes = +I/2 each (net 0).
    """
    patterns = []
    for c in range(N):
        others = [k for k in range(N) if k != c]
        for idx1 in range(len(others)):
            for idx2 in range(idx1 + 1, len(others)):
                a1 = others[idx1]
                a2 = others[idx2]
                patterns.append((c, a1, a2))
    return patterns

def run_bo_tripolar_single_target(grid, repeat, target_point, eval_budget, eval_seed: int):
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    tripoles = build_tripolar_patterns(N)
    n_trip = len(tripoles)

    algo_seed = SEED_BASE + 16000 * repeat + 37 * N
    tag = make_tag("BO_TRIPOLAR", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    opt = make_skopt_optimizer(
        dimensions=[Integer(0, n_trip - 1), Real(0.0, RANGE)],
        random_state=algo_seed,
        n_initial_points=bo_n_initial(2),
    )

    rng = np.random.default_rng(algo_seed + 1234)
    seen = set()

    t0 = time.perf_counter()

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, best_vals = [], []
    header_written = False

    for step in tqdm(range(eval_budget), desc=f"{tag}_bo", leave=False):
        trip_idx, amp, sig, was_rand, dup_tries = choose_unique_idx_amp(opt, n_trip, seen, rng)
        seen.add(sig)

        c, a1, a2 = tripoles[trip_idx]
        currents = np.zeros(N, dtype=float)
        currents[c]  = -amp
        currents[a1] = +0.5 * amp
        currents[a2] = +0.5 * amp
        currents = quantize_currents(currents)

        y, x_used = eval_selectivity_grounded(currents, target_point, grid, rng_seed=eval_seed)
        sel = -y

        opt.tell([trip_idx, amp], float(y))

        evals_so_far += 1

        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()

        xs_axis.append(evals_so_far)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "BO_TRIPOLAR",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "repeat": repeat,
            "step_index": step + 1,
            "evals_so_far": evals_so_far,
            "step_selectivity": sel,
            "step_best_selectivity": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "tripolar_idx": int(trip_idx),
            "tripolar_cathode": int(c),
            "tripolar_anode1": int(a1),
            "tripolar_anode2": int(a2),
            "tripolar_amp": float(amp),
            "currents_this_step": x_used.tolist(),
            "currents_gen_best": best_currents.tolist(),
            "was_forced_random": bool(was_rand),
            "dup_tries": int(dup_tries),
            "eval_seed": eval_seed,
            "algo_seed": algo_seed,
        }, header_written)
        header_written = True

    total_time_s = time.perf_counter() - t0
    save_progress_plot(xs_axis, best_vals, tag, target_point)

    return {
        "optimizer": "BO_TRIPOLAR",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": int(N),
        "grid": grid,
        "repeat": int(repeat),
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
        "time_total_s": float(total_time_s),
        "time_sweep_s": 0.0,
        "time_bo_s": float(total_time_s),
        "time_per_eval_s": float(total_time_s / max(1, evals_so_far)),
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n=== BO single-target benchmark (full vs sweep vs bipolar vs tripolar, anti-repeat enabled) ===\n")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Target point: {TARGET_POINT}")
    print(f"CURR_QUANT_A={CURR_QUANT_A}  AMP_QUANT_A={AMP_QUANT_A}  EPS_RANDOM={EPS_RANDOM}  MAX_DUP_TRIES={MAX_DUP_TRIES}")
    print(f"ACQ_OPTIMIZER={ACQ_OPTIMIZER}  ACQ_N_POINTS={ACQ_N_POINTS}  BO_ACQ={BO_ACQ}  BO_XI={BO_XI}  BO_KAPPA={BO_KAPPA}\n")

    summaries = []

    for grid in ELECTRODE_GRIDS:
        N = grid[0] * grid[1]
        budget = EVALS_PER_DIM * N

        for repeat in range(REPEATS):
            eval_seed = SEED_BASE + 1_000_000 * repeat + 10_000 * N
            print(f"Grid={grid} (N={N}), repeat={repeat+1}/{REPEATS}, budget={budget}, eval_seed={eval_seed}")

            #summaries.append(run_bo_grounded_single_target(grid, repeat, TARGET_POINT, budget, eval_seed=eval_seed))
            summaries.append(run_sweep_then_bo_single_target(grid, repeat, TARGET_POINT, budget, eval_seed=eval_seed))
            summaries.append(run_bo_bipolar_single_target(grid, repeat, TARGET_POINT, budget, eval_seed=eval_seed))
            summaries.append(run_bo_tripolar_single_target(grid, repeat, TARGET_POINT, budget, eval_seed=eval_seed))

    df = pd.DataFrame(summaries)
    out_csv = os.path.join(OUTPUT_DIR, "optimizer_summary.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== Summary (mean over repeats) ===")
    for (grid, opt), grp in df.groupby(["grid", "optimizer"]):
        print(f"{opt:12s} grid={grid}  "
              f"best(mean)={grp['best'].mean():.4f}  "
              f"time_total_s(mean)={grp['time_total_s'].mean():.2f}  "
              f"time_per_eval_s(mean)={grp['time_per_eval_s'].mean():.4f}")

    print(f"\nSaved: {out_csv}\n")
