#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bonizzato-style GP Bayesian Optimization benchmark for your RPNI simulation
(with multiple target points + greedy-random baseline + AUC analysis + per-run plots).

BO (paper-consistent choices):
  - GP surrogate (GaussianProcessRegressor)
  - Kernel: Matérn ν=5/2 (nu fixed at 2.5), ARD length-scales, + WhiteKernel noise
  - Acquisition: UCB: a(x) = μ(x) + k σ(x)
  - Search space reduction: bipolar patterns only:
      candidate = (ordered pair i!=j, amplitude from discrete levels)
  - No repeats: evaluated candidates removed from pool

Baseline:
  - GREEDY_RANDOM: random sampling without replacement on the same candidate pool
    (tracks best-so-far)

Evaluation:
  - Multiple hardcoded target points
  - AUC = area under best-so-far selectivity vs iteration

Outputs:
  data/benchmark_bo_bonizzato_multi/
    runs/                          # per-run CSVs
    plots/
      per_run/                     # per-run progress plots (BO and RAND)
      per_target/                  # per-target mean curves (BO vs RAND over repeats)
      mean_best_curve.png          # overall mean curves
      auc_boxplot.png              # AUC distribution by method
      auc_diff_per_target.png      # paired AUC diffs (BO-RAND) per run
    summary_runs.csv

Usage:
  python bo_bonizzato_style_benchmark_multi.py

If you already have CSVs and just want plots:
  set DO_RUNS=False and REBUILD_PLOTS_FROM_CSV=True
"""

import os
import time
import math
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

# --- Simulator import (same pattern as your repo) ---
try:
    from utils.run_selectivity_simulation import run_selectivity_simulation
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from run_selectivity_simulation import run_selectivity_simulation  # type: ignore


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "benchmark_bo_bonizzato_multi")
RUNS_DIR   = os.path.join(OUTPUT_DIR, "runs")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
PLOTS_PER_RUN_DIR    = os.path.join(PLOTS_DIR, "per_run")
PLOTS_PER_TARGET_DIR = os.path.join(PLOTS_DIR, "per_target")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PLOTS_PER_RUN_DIR, exist_ok=True)
os.makedirs(PLOTS_PER_TARGET_DIR, exist_ok=True)

# Control flags
DO_RUNS = True
REBUILD_PLOTS_FROM_CSV = True   # will regenerate plots from existing CSVs after runs (or without running)

# Geometry / tissue
RADIUS  = 0.01   # m
HEIGHT  = 0.04   # m
SIGMA_T = 0.08   # S/m

# Electrode layout
GRID = (4, 3)  # N=12

# Multiple target points (meters) - hardcoded, not random.
TARGET_POINTS = [
    (0.0,    0.0,    0.0),      # centre of cylinder
    (0.002,  0.0035, 0.010),    # upper, off-centre (≈4.0 mm radially)
    (0.004,   0.002,  0.015),   # upper, ≈4.5 mm radius
    (0.005, 0.000, 0.000)
]

# Stimulation bounds
I_MAX = 1e-3  # A

# Candidate set discretisation (amplitude grid)
N_AMP_LEVELS = 25
AMP_MIN = 0.05e-3
AMP_LEVELS = np.linspace(AMP_MIN, I_MAX, N_AMP_LEVELS)

# Trial budget
N_TRIALS = 200
N_INIT_RANDOM = 15

# UCB exploration parameter (k in paper)
UCB_K = 2.0

# GP fitting controls
MAX_TRAIN_SIZE = 250
GP_N_RESTARTS  = 2
ALPHA_NOISE    = 1e-6

# Repeats
REPEATS = 3

# Determinism / fairness
SEED_BASE = 42


# =============================================================================
# Helpers
# =============================================================================

def layout_electrodes(n_rows: int, n_per_row: int, radius: float, height: float) -> np.ndarray:
    z_vals = np.linspace(-height / 2, height / 2, n_rows)
    thetas = np.linspace(0.0, 2 * np.pi, n_per_row, endpoint=False)
    pos = []
    for z in z_vals:
        for th in thetas:
            pos.append([radius * math.cos(th), radius * math.sin(th), z])
    return np.asarray(pos, dtype=float)

def eval_selectivity_grounded(currents: np.ndarray, target_point, grid, rng_seed: int) -> float:
    currents = np.clip(np.asarray(currents, dtype=float), -I_MAX, I_MAX)
    res = run_selectivity_simulation(
        n_rows=grid[0],
        n_per_row=grid[1],
        currents=currents,
        target_point=target_point,
        radius=RADIUS,
        height=HEIGHT,
        sigma=SIGMA_T,
        n_off_samples=1200,
        E_th=1,
        k=0.5,
        metric="activation",
        grounded_boundary=True,
        use_activating_function=False,
        R_outer=0.10,
        rng=rng_seed,
    )
    return float(res["selectivity"])

def build_bipolar_candidates(grid, amp_levels: np.ndarray):
    """
    Candidate pool:
      - ordered pairs (i,j), i!=j
      - amplitudes in amp_levels
    GP feature vector uses electrode geometry for generalization:
      X = [pos_i(mm), pos_j(mm), amp(mA)]  (7D)
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    pos_m  = layout_electrodes(n_rows, n_per_row, RADIUS, HEIGHT)
    pos_mm = pos_m * 1e3

    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

    X_feat, meta, currents_list = [], [], []
    for (i, j) in pairs:
        for amp in amp_levels:
            cur = np.zeros(N, dtype=float)
            cur[i] = +float(amp)
            cur[j] = -float(amp)
            currents_list.append(cur)
            X_feat.append(np.concatenate([pos_mm[i], pos_mm[j], [float(amp * 1e3)]]))
            meta.append((int(i), int(j), float(amp)))

    return np.asarray(X_feat, dtype=float), meta, currents_list

def compute_auc(best_so_far: np.ndarray) -> float:
    y = np.asarray(best_so_far, dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)
    return float(np.trapz(y, x))

def init_gp() -> GaussianProcessRegressor:
    d = 7
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=ALPHA_NOISE,
        normalize_y=True,
        n_restarts_optimizer=GP_N_RESTARTS,
        random_state=0,
    )

def make_run_tag(method: str, target_idx: int, repeat: int) -> str:
    tx, ty, tz = TARGET_POINTS[target_idx]
    return f"{method}_t{target_idx}_r{repeat}_x{tx*1e3:.1f}_y{ty*1e3:.1f}_z{tz*1e3:.1f}"

def save_run_csv(tag: str, rows: list) -> str:
    df = pd.DataFrame(rows)
    out = os.path.join(RUNS_DIR, f"{tag}.csv")
    df.to_csv(out, index=False)
    return out


# =============================================================================
# Plot helpers
# =============================================================================

def plot_progress_from_curve(best_curve: np.ndarray, title: str, out_path: str):
    x = np.arange(1, len(best_curve) + 1)
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(x, best_curve, lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far selectivity")
    plt.title(title, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_mean_best_curves(curves_by_method: dict, out_path: str):
    plt.figure(figsize=(7.5, 4.5))
    for method, curves in curves_by_method.items():
        if len(curves) == 0:
            continue
        T = min(len(c) for c in curves)
        M = np.vstack([c[:T] for c in curves])
        mean = M.mean(axis=0)
        std  = M.std(axis=0)
        x = np.arange(1, T + 1)
        plt.plot(x, mean, label=method)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far selectivity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_auc_boxplot(df_summary: pd.DataFrame, out_path: str):
    plt.figure(figsize=(6.5, 4.5))
    methods = list(df_summary["method"].unique())
    data = [df_summary.loc[df_summary["method"] == m, "auc"].values for m in methods]
    plt.boxplot(data, labels=methods, showmeans=True)
    plt.ylabel("AUC of best-so-far selectivity")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_auc_diff_per_target(df_summary: pd.DataFrame, out_path: str):
    piv = df_summary.pivot_table(index=["target_idx", "repeat"], columns="method", values="auc", aggfunc="mean")
    if "GP_UCB_BO" not in piv.columns or "GREEDY_RANDOM" not in piv.columns:
        return
    diff = (piv["GP_UCB_BO"] - piv["GREEDY_RANDOM"]).reset_index()
    y = (diff["GP_UCB_BO"] - diff["GREEDY_RANDOM"]).values

    plt.figure(figsize=(7.5, 4.2))
    plt.axhline(0.0, lw=1)
    plt.scatter(np.arange(len(y)), y)
    plt.xticks(
        np.arange(len(y)),
        [f"t{int(r.target_idx)}_r{int(r.repeat)}" for r in diff.itertuples()],
        rotation=45, ha="right"
    )
    plt.ylabel("AUC difference (BO - RAND)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_per_target_mean(df_all_runs: pd.DataFrame, out_dir: str):
    """
    For each target_idx: plot mean best-so-far curve for BO and RAND over repeats.
    Uses CSV data (best_so_far vs iter).
    """
    for target_idx in sorted(df_all_runs["target_idx"].unique()):
        df_t = df_all_runs[df_all_runs["target_idx"] == target_idx].copy()
        plt.figure(figsize=(7.2, 4.2))

        for method in ["GP_UCB_BO", "GREEDY_RANDOM"]:
            df_m = df_t[df_t["method"] == method]
            if df_m.empty:
                continue
            # collect per-repeat curves
            curves = []
            for rep in sorted(df_m["repeat"].unique()):
                s = df_m[df_m["repeat"] == rep].sort_values("iter")["best_so_far"].to_numpy()
                curves.append(s)
            T = min(len(c) for c in curves)
            M = np.vstack([c[:T] for c in curves])
            mean = M.mean(axis=0)
            std  = M.std(axis=0)
            x = np.arange(1, T+1)
            plt.plot(x, mean, label=method)
            plt.fill_between(x, mean-std, mean+std, alpha=0.2)

        tx, ty, tz = TARGET_POINTS[int(target_idx)]
        plt.title(f"Target {target_idx}: ({tx*1e3:.1f}, {ty*1e3:.1f}, {tz*1e3:.1f}) mm", fontsize=9)
        plt.xlabel("Iteration")
        plt.ylabel("Best-so-far selectivity")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"target_{int(target_idx)}_mean_curves.png")
        plt.savefig(out_path, dpi=160)
        plt.close()


# =============================================================================
# Methods
# =============================================================================

def run_greedy_random(meta, currents_list, target_idx, target_point, eval_seed, n_trials, rng):
    remaining = np.arange(len(currents_list))
    rng.shuffle(remaining)
    remaining = remaining[:n_trials]

    best = -np.inf
    best_cur = None
    best_curve = []
    rows = []

    t0 = time.perf_counter()
    for t, idx in enumerate(remaining, start=1):
        sel = eval_selectivity_grounded(currents_list[idx], target_point, GRID, rng_seed=eval_seed)

        if sel > best:
            best = sel
            best_cur = currents_list[idx].copy()
        best_curve.append(best)

        i, j, amp = meta[idx]
        rows.append({
            "method": "GREEDY_RANDOM",
            "target_idx": int(target_idx),
            "repeat": None,  # filled by caller
            "iter": int(t),
            "candidate_idx": int(idx),
            "pair_i": int(i),
            "pair_j": int(j),
            "amp_A": float(amp),
            "selectivity": float(sel),
            "best_so_far": float(best),
            "eval_seed": int(eval_seed),
            "currents_this_step": currents_list[idx].tolist(),
            "currents_best_so_far": (best_cur.tolist() if best_cur is not None else None),
        })
    dt = time.perf_counter() - t0
    return np.asarray(best_curve, dtype=float), rows, dt


def run_gp_ucb_bo(X_scaled, meta, currents_list, target_idx, target_point, eval_seed, n_trials, rng):
    n_cand = len(currents_list)
    remaining = list(range(n_cand))

    init = rng.choice(remaining, size=min(N_INIT_RANDOM, n_trials), replace=False).tolist()

    X_obs, y_obs = [], []
    best = -np.inf
    best_cur = None
    best_curve = []
    rows = []

    gp = init_gp()

    t0 = time.perf_counter()

    def evaluate_idx(t, idx):
        nonlocal best, best_cur
        sel = eval_selectivity_grounded(currents_list[idx], target_point, GRID, rng_seed=eval_seed)

        if sel > best:
            best = sel
            best_cur = currents_list[idx].copy()
        best_curve.append(best)

        i, j, amp = meta[idx]
        rows.append({
            "method": "GP_UCB_BO",
            "target_idx": int(target_idx),
            "repeat": None,  # filled by caller
            "iter": int(t),
            "candidate_idx": int(idx),
            "pair_i": int(i),
            "pair_j": int(j),
            "amp_A": float(amp),
            "selectivity": float(sel),
            "best_so_far": float(best),
            "eval_seed": int(eval_seed),
            "currents_this_step": currents_list[idx].tolist(),
            "currents_best_so_far": (best_cur.tolist() if best_cur is not None else None),
        })

        X_obs.append(X_scaled[idx])
        y_obs.append(sel)

        return sel

    # initial random picks
    t = 0
    for idx in init:
        t += 1
        evaluate_idx(t, idx)
        remaining.remove(idx)
        if t >= n_trials:
            break

    # BO loop
    while t < n_trials and len(remaining) > 0:
        t += 1

        # bound training size
        if len(X_obs) > MAX_TRAIN_SIZE:
            X_train = np.asarray(X_obs[-MAX_TRAIN_SIZE:], dtype=float)
            y_train = np.asarray(y_obs[-MAX_TRAIN_SIZE:], dtype=float)
        else:
            X_train = np.asarray(X_obs, dtype=float)
            y_train = np.asarray(y_obs, dtype=float)

        gp.fit(X_train, y_train)

        X_rem = np.asarray([X_scaled[idx] for idx in remaining], dtype=float)
        mu, std = gp.predict(X_rem, return_std=True)
        std = np.maximum(std, 1e-12)

        acq = mu + float(UCB_K) * std
        pick = int(np.argmax(acq))
        idx = int(remaining[pick])

        evaluate_idx(t, idx)
        remaining.remove(idx)

    dt = time.perf_counter() - t0
    return np.asarray(best_curve, dtype=float), rows, dt


# =============================================================================
# Posthoc plotting from CSVs
# =============================================================================

def load_all_run_csvs() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.csv")))
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    # ensure types
    for c in ["target_idx", "iter"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    return df_all

def rebuild_per_run_plots_from_csv():
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.csv")))
    if not paths:
        print("No run CSVs found to plot.")
        return

    for p in paths:
        df = pd.read_csv(p)
        if "best_so_far" not in df.columns or "iter" not in df.columns:
            continue
        df = df.sort_values("iter")
        best_curve = df["best_so_far"].to_numpy()

        base = os.path.splitext(os.path.basename(p))[0]
        out_png = os.path.join(PLOTS_PER_RUN_DIR, f"{base}_progress.png")

        # add target info if available
        title = base
        if "target_idx" in df.columns:
            try:
                tidx = int(df["target_idx"].iloc[0])
                tx, ty, tz = TARGET_POINTS[tidx]
                title = f"{base}\nTarget: ({tx*1e3:.1f},{ty*1e3:.1f},{tz*1e3:.1f}) mm"
            except Exception:
                pass

        plot_progress_from_curve(best_curve, title, out_png)


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n=== Bonizzato-style BO benchmark: multi-target + greedy-random + AUC + per-run plots ===\n")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"GRID={GRID}, N_TRIALS={N_TRIALS}, N_INIT_RANDOM={N_INIT_RANDOM}, UCB_K={UCB_K}")
    print(f"Targets: {len(TARGET_POINTS)} points, repeats: {REPEATS}\n")

    # Build candidate pool once
    X_all, meta, currents_list = build_bipolar_candidates(GRID, AMP_LEVELS)

    # Scale GP features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    summary_rows = []
    curves_by_method = {"GP_UCB_BO": [], "GREEDY_RANDOM": []}

    if DO_RUNS:
        for target_idx, tp in enumerate(TARGET_POINTS):
            for rep in range(REPEATS):
                eval_seed = SEED_BASE + 10_000 * rep + 1_000 * target_idx + 123

                rng_bo = np.random.default_rng(SEED_BASE + 100_000 * rep + 7_000 * target_idx + 1)
                rng_rd = np.random.default_rng(SEED_BASE + 100_000 * rep + 7_000 * target_idx + 2)

                # ---- BO ----
                tag_bo = make_run_tag("GP_UCB_BO", target_idx, rep)
                best_curve_bo, rows_bo, t_bo = run_gp_ucb_bo(
                    X_scaled, meta, currents_list, target_idx, tp, eval_seed, N_TRIALS, rng_bo
                )
                for r in rows_bo:
                    r["repeat"] = int(rep)
                save_run_csv(tag_bo, rows_bo)

                auc_bo = compute_auc(best_curve_bo)
                curves_by_method["GP_UCB_BO"].append(best_curve_bo)

                # per-run plot
                title = f"{tag_bo}\nUCB_K={UCB_K}  trials={len(best_curve_bo)}"
                plot_progress_from_curve(
                    best_curve_bo, title,
                    os.path.join(PLOTS_PER_RUN_DIR, f"{tag_bo}_progress.png")
                )

                summary_rows.append({
                    "method": "GP_UCB_BO",
                    "target_idx": int(target_idx),
                    "repeat": int(rep),
                    "eval_seed": int(eval_seed),
                    "best_final": float(best_curve_bo[-1]),
                    "auc": float(auc_bo),
                    "time_s": float(t_bo),
                })

                # ---- GREEDY RANDOM ----
                tag_rd = make_run_tag("GREEDY_RANDOM", target_idx, rep)
                best_curve_rd, rows_rd, t_rd = run_greedy_random(
                    meta, currents_list, target_idx, tp, eval_seed, N_TRIALS, rng_rd
                )
                for r in rows_rd:
                    r["repeat"] = int(rep)
                save_run_csv(tag_rd, rows_rd)

                auc_rd = compute_auc(best_curve_rd)
                curves_by_method["GREEDY_RANDOM"].append(best_curve_rd)

                # per-run plot
                title = f"{tag_rd}\ntrials={len(best_curve_rd)}"
                plot_progress_from_curve(
                    best_curve_rd, title,
                    os.path.join(PLOTS_PER_RUN_DIR, f"{tag_rd}_progress.png")
                )

                summary_rows.append({
                    "method": "GREEDY_RANDOM",
                    "target_idx": int(target_idx),
                    "repeat": int(rep),
                    "eval_seed": int(eval_seed),
                    "best_final": float(best_curve_rd[-1]),
                    "auc": float(auc_rd),
                    "time_s": float(t_rd),
                })

                print(f"target={target_idx} rep={rep}: "
                      f"BO best={best_curve_bo[-1]:.4f} auc={auc_bo:.2f} | "
                      f"RAND best={best_curve_rd[-1]:.4f} auc={auc_rd:.2f}")

        df_summary = pd.DataFrame(summary_rows)
        out_summary = os.path.join(OUTPUT_DIR, "summary_runs.csv")
        df_summary.to_csv(out_summary, index=False)
        print("\nSaved summary:", out_summary)

        # Aggregate plots from in-memory curves
        plot_mean_best_curves(curves_by_method, os.path.join(PLOTS_DIR, "mean_best_curve.png"))
        plot_auc_boxplot(df_summary, os.path.join(PLOTS_DIR, "auc_boxplot.png"))
        plot_auc_diff_per_target(df_summary, os.path.join(PLOTS_DIR, "auc_diff_per_target.png"))

        print("Saved aggregate plots in:", PLOTS_DIR)

    # Rebuild plots from CSVs (useful if you already ran and just want plots)
    if REBUILD_PLOTS_FROM_CSV:
        print("\nRebuilding plots from CSVs...")
        rebuild_per_run_plots_from_csv()

        df_all = load_all_run_csvs()
        if not df_all.empty:
            # per-target mean curve plots
            plot_per_target_mean(df_all, PLOTS_PER_TARGET_DIR)

            # rebuild aggregate plots based on summary if available
            summary_path = os.path.join(OUTPUT_DIR, "summary_runs.csv")
            if os.path.exists(summary_path):
                df_summary = pd.read_csv(summary_path)
                plot_auc_boxplot(df_summary, os.path.join(PLOTS_DIR, "auc_boxplot.png"))
                plot_auc_diff_per_target(df_summary, os.path.join(PLOTS_DIR, "auc_diff_per_target.png"))

            print("Rebuilt per-run plots in:", PLOTS_PER_RUN_DIR)
            print("Rebuilt per-target plots in:", PLOTS_PER_TARGET_DIR)
        else:
            print("No run CSVs loaded; skipping per-target rebuild.")

if __name__ == "__main__":
    main()
