#!/usr/bin/env python3
"""
Mapping benchmark (multi-point) using grounded-boundary selectivity simulation.

Goal:
- There are K unknown points (proxy "nerves") inside the cylinder.
- Each stimulation vector x (N electrodes) produces a selectivity vector S(x) in R^K.
- The mapping algorithm should find *separable* solutions for each point with few evaluations.

Default algorithm: Multi-task PSO (particles are assigned to a focus point j).
Fitness for a particle focused on point j:
    fitness(x) = - ( S_j(x) - alpha * max_{i!=j} S_i(x) ) + reg * mean(x^2)
Lower is better (minimizer-friendly).

Outputs (in run folder):
- config.json                  (full configuration + seeds)
- unknown_points.csv           (the "hidden" nerve locations)
- mapping_log.csv              (ONE CSV: stim params + selectivities for ALL points per eval)
- summary.json                 (best found per point + coverage metrics)
- plots:
    - plot_best_per_point.png
    - plot_sensation_over_time.png
    - plot_separability.png

example usage
python mapping_benchmark_pso.py --n_points 6 --particles 36 --iters 60
"""

from __future__ import annotations

import os
import json
import csv
import math
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Import your simulator (provided in run_selectivity_simulation.py)
from utils.run_selectivity_simulation import run_selectivity_simulation


# ----------------------------
# Global simulation constants
# ----------------------------
RANGE = 80e-6          # +/- current bound per electrode (A)
RADIUS = 0.01          # cylinder radius (m)
HEIGHT = 0.04          # cylinder height (m)
SIGMA_T = 0.25         # conductivity (S/m)
SEED_BASE = 1234       # base rng seed


# ----------------------------
# Your requested eval function
# ----------------------------
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
        E_th=1, k=0.5,
        metric="activation",
        grounded_boundary=True,
        use_activating_function=True,
        R_outer=0.10,
        rng=(SEED_BASE if rng_seed is None else rng_seed),
    )
    # OPTIMIZERS MINIMISE: objective = -selectivity
    return -float(res["selectivity"]), x_c


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sample_points_in_cylinder(
    n_points: int,
    radius: float,
    z0: float,
    z1: float,
    rng: np.random.Generator,
    exclude_radius: float = 0.002,
) -> np.ndarray:
    """
    Uniform-in-area sampling in cross-section + uniform z.
    Slightly excludes tiny central ball only to avoid degenerate points if desired.
    """
    pts = []
    while len(pts) < n_points:
        u = rng.random()
        r = radius * math.sqrt(u)
        th = rng.random() * 2 * math.pi
        x, y = r * math.cos(th), r * math.sin(th)
        z = rng.random() * (z1 - z0) + z0
        p = np.array([x, y, z], dtype=float)
        if np.linalg.norm(p) < exclude_radius:
            continue
        pts.append(p)
    return np.vstack(pts)


def selectivity_vector(
    x: np.ndarray,
    points: np.ndarray,
    grid: Tuple[int, int],
    rng_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      S: (K,) selectivity values for each point
      x_c: clipped currents (N,)
    """
    S = np.zeros(len(points), dtype=float)
    x_c_final = None
    for i, p in enumerate(points):
        y, x_c = eval_selectivity_grounded(x, p, grid, rng_seed=rng_seed + i * 100000)
        S[i] = -y
        if x_c_final is None:
            x_c_final = x_c
    return S, x_c_final


def sensation_from_selectivities(S: np.ndarray) -> Tuple[int, float, float]:
    """
    Proxy of "person feedback":
      - location: argmax selectivity index
      - magnitude: max selectivity
      - margin: top1 - top2 (confidence / separability proxy)
    """
    j = int(np.argmax(S))
    s1 = float(S[j])
    if len(S) >= 2:
        s2 = float(np.partition(S, -2)[-2])
    else:
        s2 = 0.0
    return j, s1, (s1 - s2)


@dataclass
class PSOConfig:
    n_particles: int = 36
    n_iters: int = 60
    w: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    vmax_frac: float = 0.25          # vmax = vmax_frac * (ub-lb)
    alpha_sep: float = 0.6           # penalty weight on other points: score = S_j - alpha*max_others
    reg_l2: float = 0.0              # optional current magnitude regularizer
    seed: int = 2026


@dataclass
class BenchmarkConfig:
    n_rows: int = 4
    n_per_row: int = 3
    n_points: int = 6
    point_seed: int = 999
    run_seed: int = 2026
    selectivity_threshold: float = 2.0
    out_root: str = "mapping_runs"


# ----------------------------
# Multi-task PSO mapping
# ----------------------------
def run_mapping_pso(
    bench: BenchmarkConfig,
    pso: PSOConfig,
) -> Dict[str, Any]:
    grid = (bench.n_rows, bench.n_per_row)
    N = bench.n_rows * bench.n_per_row
    K = bench.n_points

    # Unknown points (hidden from algorithm conceptually, but used by simulator)
    rng_pts = np.random.default_rng(bench.point_seed)
    z0, z1 = -HEIGHT / 2, HEIGHT / 2
    points = sample_points_in_cylinder(K, RADIUS, z0, z1, rng_pts)

    # Output folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(bench.out_root, f"mapping_{stamp}_K{K}_N{N}_PSO")
    ensure_dir(out_dir)

    # Save configs + points
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            {"bench": asdict(bench), "pso": asdict(pso), "globals": {
                "RANGE": RANGE, "RADIUS": RADIUS, "HEIGHT": HEIGHT, "SIGMA_T": SIGMA_T, "SEED_BASE": SEED_BASE
            }},
            f, indent=2
        )

    with open(os.path.join(out_dir, "unknown_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point_id", "x_m", "y_m", "z_m"])
        for i, p in enumerate(points):
            w.writerow([i, p[0], p[1], p[2]])

    # Main CSV log (ONE file: stim params + selectivities for all points)
    log_path = os.path.join(out_dir, "mapping_log.csv")
    x_cols = [f"x_{i}" for i in range(N)]
    s_cols = [f"sel_{j}" for j in range(K)]
    header = [
        "eval_id", "iter", "particle", "focus_point",
        "fitness", "score_target_minus_alpha_maxother",
        "sensation_location", "sensation_magnitude", "sensation_margin",
    ] + x_cols + s_cols

    # Initialize PSO
    rng = np.random.default_rng(pso.seed)
    lb, ub = -RANGE, RANGE
    span = ub - lb
    vmax = pso.vmax_frac * span

    X = rng.uniform(lb, ub, size=(pso.n_particles, N))
    V = rng.uniform(-vmax, vmax, size=(pso.n_particles, N))

    focus = np.array([i % K for i in range(pso.n_particles)], dtype=int)

    pbest_X = X.copy()
    pbest_fit = np.full(pso.n_particles, np.inf)

    gbest_X = np.zeros((K, N), dtype=float)
    gbest_fit = np.full(K, np.inf)

    # Track best *selectivity* per point (coverage metric)
    best_sel_per_point = np.full(K, -np.inf)
    best_x_per_point = np.zeros((K, N), dtype=float)

    eval_id = 0

    # Run
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for it in range(pso.n_iters):
            # Evaluate all particles
            for p in range(pso.n_particles):
                j = int(focus[p])

                S, x_c = selectivity_vector(
                    X[p],
                    points,
                    grid=grid,
                    rng_seed=bench.run_seed + it * 10000 + p * 100,
                )

                # separation score for the focused point j
                max_other = float(np.max(np.delete(S, j))) if K > 1 else 0.0
                score = float(S[j] - pso.alpha_sep * max_other)
                l2 = float(np.mean(x_c * x_c))
                fitness = float(-score + pso.reg_l2 * l2)

                # Update pbest (particle-wise)
                if fitness < pbest_fit[p]:
                    pbest_fit[p] = fitness
                    pbest_X[p] = X[p].copy()

                # Update gbest (per-focus-point)
                if fitness < gbest_fit[j]:
                    gbest_fit[j] = fitness
                    gbest_X[j] = X[p].copy()

                # Update best selectivity-per-point archive (coverage)
                if float(S[j]) > best_sel_per_point[j]:
                    best_sel_per_point[j] = float(S[j])
                    best_x_per_point[j] = X[p].copy()

                # "person feedback" proxy
                loc, mag, margin = sensation_from_selectivities(S)

                # Log row: currents + all selectivities
                row = [
                    eval_id, it, p, j,
                    fitness, score,
                    loc, mag, margin,
                ] + list(x_c.tolist()) + list(S.tolist())
                writer.writerow(row)
                eval_id += 1

            # PSO update step (multi-task: each particle uses gbest of its focus)
            r1 = rng.random(size=(pso.n_particles, N))
            r2 = rng.random(size=(pso.n_particles, N))

            G = gbest_X[focus]  # (n_particles, N) each gets its focus best
            V = (
                pso.w * V
                + pso.c1 * r1 * (pbest_X - X)
                + pso.c2 * r2 * (G - X)
            )
            V = np.clip(V, -vmax, vmax)
            X = X + V
            X = np.clip(X, lb, ub)

    # Coverage: did we find a "good" selective pattern per point?
    thr = bench.selectivity_threshold
    covered = (best_sel_per_point >= thr)
    n_covered = int(np.sum(covered))

    # Earliest eval where each point was covered (approximate, via CSV scan)
    cover_eval_id = None
    try:
        # Lightweight scan without pandas: track best-so-far per point from the log file
        best_so_far = np.full(K, -np.inf)
        with open(log_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                eid = int(row["eval_id"])
                for j in range(K):
                    sj = float(row[f"sel_{j}"])
                    if sj > best_so_far[j]:
                        best_so_far[j] = sj
                if np.all(best_so_far >= thr):
                    cover_eval_id = eid
                    break
    except Exception:
        cover_eval_id = None

    summary = {
        "out_dir": out_dir,
        "N_electrodes": N,
        "K_points": K,
        "selectivity_threshold": thr,
        "covered_mask": covered.astype(int).tolist(),
        "n_covered": n_covered,
        "coverage_eval_id_first_time_all_covered": cover_eval_id,
        "best_selectivity_per_point": best_sel_per_point.tolist(),
        "best_x_per_point": best_x_per_point.tolist(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    make_plots(out_dir, log_path, K)

    return summary


def make_plots(out_dir: str, log_path: str, K: int) -> None:
    # Load CSV into numpy arrays (avoid pandas dependency)
    eval_ids = []
    sensation_loc = []
    sensation_mag = []
    sel_mat = []

    with open(log_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            eval_ids.append(int(row["eval_id"]))
            sensation_loc.append(int(row["sensation_location"]))
            sensation_mag.append(float(row["sensation_magnitude"]))
            sel_mat.append([float(row[f"sel_{j}"]) for j in range(K)])

    eval_ids = np.array(eval_ids)
    sensation_loc = np.array(sensation_loc)
    sensation_mag = np.array(sensation_mag)
    sel_mat = np.array(sel_mat)  # (E, K)

    # 1) Best-per-point over time (cumulative max)
    cum_best = np.maximum.accumulate(sel_mat, axis=0)
    plt.figure()
    for j in range(K):
        plt.plot(eval_ids, cum_best[:, j], label=f"point {j}")
    plt.xlabel("evaluation")
    plt.ylabel("best selectivity so far")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_best_per_point.png"), dpi=200)
    plt.close()

    # 2) "Sensation location over time" (argmax), size ~ magnitude
    plt.figure()
    sizes = 10 + 40 * (sensation_mag - sensation_mag.min()) / (1e-9 + (sensation_mag.max() - sensation_mag.min()))
    plt.scatter(eval_ids, sensation_loc, s=sizes, alpha=0.7)
    plt.yticks(range(K))
    plt.xlabel("evaluation")
    plt.ylabel("perceived location (argmax point)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_sensation_over_time.png"), dpi=200)
    plt.close()

    # 3) Separability scatter: top1 vs top2 for each eval
    top1 = np.max(sel_mat, axis=1)
    top2 = np.partition(sel_mat, -2, axis=1)[:, -2] if K >= 2 else np.zeros_like(top1)
    plt.figure()
    plt.scatter(top1, top2, alpha=0.5)
    plt.xlabel("top-1 selectivity")
    plt.ylabel("top-2 selectivity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_separability.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rows", type=int, default=4)
    ap.add_argument("--n_per_row", type=int, default=3)
    ap.add_argument("--n_points", type=int, default=6)
    ap.add_argument("--particles", type=int, default=36)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--alpha_sep", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--out_root", type=str, default="mapping_runs")
    ap.add_argument("--point_seed", type=int, default=999)
    ap.add_argument("--run_seed", type=int, default=2026)
    ap.add_argument("--pso_seed", type=int, default=2026)
    args = ap.parse_args()

    bench = BenchmarkConfig(
        n_rows=args.n_rows,
        n_per_row=args.n_per_row,
        n_points=args.n_points,
        point_seed=args.point_seed,
        run_seed=args.run_seed,
        selectivity_threshold=args.threshold,
        out_root=args.out_root,
    )
    pso = PSOConfig(
        n_particles=args.particles,
        n_iters=args.iters,
        alpha_sep=args.alpha_sep,
        seed=args.pso_seed,
    )

    summary = run_mapping_pso(bench, pso)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
