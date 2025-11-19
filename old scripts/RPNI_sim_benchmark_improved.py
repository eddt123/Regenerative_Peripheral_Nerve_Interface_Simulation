#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMA-ES Multi-Stage vs Full vs Pairs (12 electrodes, grounded boundary)
======================================================================

Hypothesis:
-----------
A multi-stage, adaptive search-space strategy for peripheral nerve
stimulation — starting in a low-dimensional DIPOLAR PAIR space, then
introducing TRIPOLAR structure, and finally allowing FULL 12-D control —
will achieve higher selectivity and/or faster convergence than:

  (a) pairs-only CMA-ES, and
  (b) a single-stage 12-D CMA-ES with the same total evaluation budget.

Key design choices:
-------------------
- 12 electrodes on a cylindrical RPNI (4×3 layout).
- Grounded outer boundary (monopolar / net DC allowed in simulation),
  so we DO NOT enforce a zero-sum constraint on currents.
  Currents are only box-clipped to [-RANGE, RANGE].

- Three CMA-ES conditions:
    1) Pairs-only CMA-ES      ("PAIRS")
    2) Full 12-D CMA-ES       ("FULL")
    3) Multi-stage CMA-ES     ("MULTI"):
         Stage 1: adjacent dipole pairs      (basis B_pairs, low-D)
         Stage 2: coarse tripolar patterns   (basis B_tripolar)
         Stage 3: full 12-D currents         (basis I_12)

- Stage transitions are PLATEAU-BASED:
    For each stage, CMA-ES runs until:
      • at least MIN_GENS_PER_STAGE generations have elapsed, AND
      • best selectivity has not improved by > PLATEAU_EPS over the last
        STALL_GENS generations.
    Then it moves to the next stage, carrying over the best currents as
    a warm-start for the next basis (via least-squares projection).
    If there is no next stage, or if the global evaluation budget is
    exhausted, the run terminates.

- All methods share:
    • SAME total evaluation budget (TOTAL_EVALS)
    • SAME CMA-ES population size rule
    • SAME clipping bounds
    • SAME simulator (run_selectivity_simulation)
    • SAME targets and repeats

Outputs:
--------
Per-condition CSV logs and plots under OUTPUT_DIR:

  - cma_pairs_*.csv / cma_pairs_*.png
  - cma_full_*.csv / cma_full_*.png
  - cma_multi_stage_*.csv / cma_multi_stage_*.png

Plus:
  - optimizer_summary.csv  : final best selectivity per run
  - summary_boxplots.png   : PAIRS vs FULL vs MULTI by target
  - convergence_*.png      : mean ± std convergence per condition

Requires: cma, numpy, pandas, matplotlib, tqdm
"""

from __future__ import annotations
import os, time, math
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from cma import CMAEvolutionStrategy

# ---------------------------------------------------------------------
# Simulator import
# ---------------------------------------------------------------------
try:
    from utils.run_selectivity_simulation import run_selectivity_simulation
except Exception:
    # fallback for flat layout (e.g. when running in a notebook)
    import sys
    sys.path.append(os.path.dirname(__file__))
    from run_selectivity_simulation import run_selectivity_simulation  # type: ignore

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "cma_multistage_12elec")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RADIUS   = 0.01   # m
HEIGHT   = 0.04   # m
SIGMA_T  = 0.25   # S/m
N_ROWS   = 4
N_PER_ROW = 3
N_ELECTRODES = N_ROWS * N_PER_ROW  # 12

RANGE    = 1e-3   # ±1 mA per electrode
N_OFF_SAMPLES = 1200

# Grounded outer boundary: no zero-sum enforced
GROUNDED_BOUNDARY = True

# Targets
TARGET_POINTS: List[Tuple[float, float, float]] = [
    (0.0,   0.0,    0.0),
    (0.002, 0.0035, 0.010),
    (-0.003, -0.002, -0.015),
]

TOTAL_EVALS = 2400   # global budget per run per method
REPEATS     = 20
SEED_BASE   = 42

# Plateau logic (for per-stage stopping)
PLATEAU_EPS          = 1e-3   # minimum improvement in best selectivity to count as progress
STALL_GENS           = 10     # plateau if no such improvement for this many generations
MIN_GENS_PER_STAGE   = 8      # don't plateau too early

# ---------------------------------------------------------------------
# Utilities: evaluation, logging, plotting
# ---------------------------------------------------------------------
def eval_selectivity(currents: np.ndarray,
                     target_point: Tuple[float,float,float],
                     rng_seed: int | None = None) -> float:
    """
    Evaluate selectivity for a 12-electrode current vector.
    - Grounded outer boundary (no zero-sum constraint).
    - Currents clipped to [-RANGE, RANGE].
    Returns positive selectivity (higher is better).
    """
    currents = np.asarray(currents, dtype=float)
    currents = np.clip(currents, -RANGE, RANGE)

    res = run_selectivity_simulation(
        n_rows=N_ROWS,
        n_per_row=N_PER_ROW,
        currents=currents,
        target_point=target_point,
        radius=RADIUS,
        height=HEIGHT,
        sigma=SIGMA_T,
        n_off_samples=N_OFF_SAMPLES,
        metric="activation",
        grounded_boundary=GROUNDED_BOUNDARY,
        R_outer=0.10,
        rng=(SEED_BASE if rng_seed is None else rng_seed),
    )
    return float(res["selectivity"])


def save_progress_plot(xs: List[int],
                       bests: List[float],
                       per_gen_best: List[float],
                       tag: str,
                       target_point: Tuple[float,float,float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(xs, per_gen_best, "o-", ms=3, lw=1, alpha=0.6, label="Generation best")
    plt.plot(xs, bests, "-", lw=2, label="Best-so-far")
    plt.xlabel("Function evaluations")
    plt.ylabel("Selectivity")
    tx, ty, tz = target_point
    plt.title(f"{tag}\nTarget: ({tx*1e3:.1f}, {ty*1e3:.1f}, {tz*1e3:.1f}) mm", fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_progress.png"), dpi=150)
    plt.close()


def log_eval(csv_path: str,
             header_written: bool,
             row: Dict[str, Any]) -> bool:
    pd.DataFrame([row]).to_csv(
        csv_path, mode="a", index=False, header=not header_written
    )
    return True

# ---------------------------------------------------------------------
# Bases: Pairs, Tripolar, Full
# ---------------------------------------------------------------------
def make_pairs_basis(n_elec: int = N_ELECTRODES) -> np.ndarray:
    """
    Adjacent dipole pairs on a ring of electrodes.
    Column k: +1 on electrode k, -1 on electrode (k+1 mod n_elec).
    Gives a n_elec × n_elec matrix (dim = n_elec), but highly structured.
    """
    B = np.zeros((n_elec, n_elec))
    for k in range(n_elec):
        i = k
        j = (k + 1) % n_elec
        B[i, k] =  1.0
        B[j, k] = -1.0
    return B


def make_tripolar_basis(n_elec: int = N_ELECTRODES) -> np.ndarray:
    """
    Coarse tripolar patterns: 4 equally spaced tripoles around the cuff.
    Each column: +1 at center, -0.5 at its two neighbours.
    Dimension = 4 (strictly lower than full 12-D).
    """
    centers = [0, 3, 6, 9]  # 4 tripoles
    cols = []
    for c in centers:
        v = np.zeros(n_elec)
        left  = (c - 1) % n_elec
        right = (c + 1) % n_elec
        v[c]      =  1.0
        v[left]   = -0.5
        v[right]  = -0.5
        cols.append(v)
    B = np.stack(cols, axis=1)  # shape (n_elec, 4)
    return B


def make_full_basis(n_elec: int = N_ELECTRODES) -> np.ndarray:
    """Identity basis: direct control over each electrode."""
    return np.eye(n_elec)


B_PAIRS    = make_pairs_basis()
B_TRIPOLAR = make_tripolar_basis()
B_FULL     = make_full_basis()

# ---------------------------------------------------------------------
# Core CMA-ES runner on a basis
# ---------------------------------------------------------------------
def cma_on_basis(B: np.ndarray,
                 total_evals: int,
                 repeat: int,
                 target_point: Tuple[float,float,float],
                 label: str,
                 stage_id: int,
                 x0: np.ndarray | None = None,
                 sigma0: float | None = None,
                 plateau_eps: float = PLATEAU_EPS,
                 stall_gens: int = STALL_GENS,
                 min_gens: int = MIN_GENS_PER_STAGE,
                 seed_offset: int = 0) -> Dict[str, Any]:
    """
    Run CMA-ES in the parameter space of coefficients 'a' such that:
        currents = B @ a
    where B has shape (12, dim).

    Returns:
      - best_selectivity
      - best_currents
      - used_evals
      - convergence info
    """
    n_elec, dim = B.shape
    assert n_elec == N_ELECTRODES

    seed = SEED_BASE + seed_offset + 1000*repeat + 10*stage_id
    rng  = np.random.default_rng(seed)

    if x0 is None:
        x0 = np.zeros(dim, dtype=float)
    if sigma0 is None:
        sigma0 = 0.3 * RANGE

    # Population size (same rule for all stages and methods)
    popsize = 4 + int(3 * math.log(dim + 1))
    popsize = max(4, popsize)

    max_gens = max(1, total_evals // popsize)
    used_evals = 0

    tag = f"{label}_stage{stage_id}_r{repeat}_x{target_point[0]*1e3:.1f}_y{target_point[1]*1e3:.1f}_z{target_point[2]*1e3:.1f}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    es = CMAEvolutionStrategy(x0, sigma0,
                              {'popsize': popsize,
                               'seed': seed,
                               'verb_disp': 0})

    best_sel = -np.inf
    best_currents = np.zeros(n_elec)
    best_eval_idx = 0

    header_written = False
    eval_axis: List[int]   = []
    best_so_far_axis: List[float] = []
    gen_best_axis: List[float]    = []

    # plateau tracking
    gens_since_improve = 0
    last_improve_sel   = -np.inf

    for gen in tqdm(range(max_gens), desc=tag, leave=False):
        X = es.ask()
        vals = []
        gen_best_sel = -np.inf

        for x in X:
            currents = B @ x
            sel = eval_selectivity(currents, target_point, rng_seed=seed)
            # CMA-ES minimizes -> objective = -selectivity
            vals.append(-sel)
            used_evals += 1

            # update global best
            if sel > best_sel:
                best_sel = sel
                best_currents = currents.copy()
                best_eval_idx = used_evals

            # update per-generation best
            if sel > gen_best_sel:
                gen_best_sel = sel

            # log per-eval (including currents)
            row = {
                "optimizer": label,
                "stage": stage_id,
                "repeat": repeat,
                "eval_index": used_evals,
                "gen_index": gen + 1,
                "selectivity": sel,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }
            for k in range(N_ELECTRODES):
                row[f"I_{k}"] = float(currents[k])
            header_written = log_eval(csv_path, header_written, row)

            if used_evals >= total_evals:
                # We’ll finish this generation here and stop after tell()
                pass

        es.tell(X, vals)

        eval_axis.append(used_evals)
        gen_best_axis.append(gen_best_sel)
        best_so_far_axis.append(best_sel)

        # plateau logic
        if gen == 0:
            last_improve_sel = best_sel
            gens_since_improve = 0
        else:
            if best_sel > last_improve_sel + plateau_eps:
                last_improve_sel = best_sel
                gens_since_improve = 0
            else:
                gens_since_improve += 1

        if gen + 1 >= min_gens and gens_since_improve >= stall_gens:
            # plateau → move to next stage (caller decides)
            break

        if used_evals >= total_evals:
            break

    save_progress_plot(eval_axis, best_so_far_axis, gen_best_axis, tag, target_point)

    return {
        "best_selectivity": float(best_sel),
        "best_currents": best_currents,
        "used_evals": used_evals,
        "popsize": popsize,
        "eval_axis": eval_axis,
        "best_axis": best_so_far_axis,
        "gen_best_axis": gen_best_axis,
        "tag": tag,
    }

# ---------------------------------------------------------------------
# Baseline: PAIRS-only CMA-ES (single-stage, full budget)
# ---------------------------------------------------------------------
def run_pairs(total_evals: int,
              repeat: int,
              target_point: Tuple[float,float,float]) -> Dict[str, Any]:
    res = cma_on_basis(B_PAIRS,
                       total_evals=total_evals,
                       repeat=repeat,
                       target_point=target_point,
                       label="PAIRS",
                       stage_id=1,
                       seed_offset=10)
    return {
        "optimizer": "PAIRS",
        "repeat": repeat,
        "target_point": target_point,
        "best": res["best_selectivity"],
        "used_evals": res["used_evals"],
        "tag": res["tag"],
    }

# ---------------------------------------------------------------------
# Baseline: FULL 12-D CMA-ES (single-stage, full budget)
# ---------------------------------------------------------------------
def run_full(total_evals: int,
             repeat: int,
             target_point: Tuple[float,float,float]) -> Dict[str, Any]:
    res = cma_on_basis(B_FULL,
                       total_evals=total_evals,
                       repeat=repeat,
                       target_point=target_point,
                       label="FULL",
                       stage_id=1,
                       seed_offset=20)
    return {
        "optimizer": "FULL",
        "repeat": repeat,
        "target_point": target_point,
        "best": res["best_selectivity"],
        "used_evals": res["used_evals"],
        "tag": res["tag"],
    }

# ---------------------------------------------------------------------
# Multi-stage CMA-ES: PAIRS → TRIPOLAR → FULL, plateau-based transitions
# ---------------------------------------------------------------------
def project_currents_to_basis(B: np.ndarray,
                              currents: np.ndarray) -> np.ndarray:
    """
    Given basis B (12×d) and a 12-D currents vector, find a least-squares
    coefficient vector a such that B @ a ≈ currents.
    """
    # pseudo-inverse is fine for small d
    pinv = np.linalg.pinv(B)
    a = pinv @ currents
    return a.astype(float)


def run_multistage(total_evals: int,
                   repeat: int,
                   target_point: Tuple[float,float,float]) -> Dict[str, Any]:
    remaining = total_evals
    best_overall_sel = -np.inf
    best_overall_curr = np.zeros(N_ELECTRODES)

    # ---------- Stage 1: PAIRS ----------
    res1 = cma_on_basis(B_PAIRS,
                        total_evals=remaining,
                        repeat=repeat,
                        target_point=target_point,
                        label="MULTI",
                        stage_id=1,
                        seed_offset=30)
    remaining -= res1["used_evals"]
    best_overall_sel = res1["best_selectivity"]
    best_overall_curr = res1["best_currents"].copy()

    if remaining <= 0:
        return {
            "optimizer": "MULTI",
            "repeat": repeat,
            "target_point": target_point,
            "best": best_overall_sel,
            "used_evals": total_evals,
            "tag": res1["tag"],
        }

    # ---------- Stage 2: TRIPOLAR ----------
    x0_stage2 = project_currents_to_basis(B_TRIPOLAR, best_overall_curr)
    res2 = cma_on_basis(B_TRIPOLAR,
                        total_evals=remaining,
                        repeat=repeat,
                        target_point=target_point,
                        label="MULTI",
                        stage_id=2,
                        x0=x0_stage2,
                        seed_offset=40)
    remaining -= res2["used_evals"]
    if res2["best_selectivity"] > best_overall_sel:
        best_overall_sel = res2["best_selectivity"]
        best_overall_curr = res2["best_currents"].copy()

    if remaining <= 0:
        return {
            "optimizer": "MULTI",
            "repeat": repeat,
            "target_point": target_point,
            "best": best_overall_sel,
            "used_evals": total_evals,
            "tag": res2["tag"],
        }

    # ---------- Stage 3: FULL 12-D ----------
    x0_stage3 = project_currents_to_basis(B_FULL, best_overall_curr)
    res3 = cma_on_basis(B_FULL,
                        total_evals=remaining,
                        repeat=repeat,
                        target_point=target_point,
                        label="MULTI",
                        stage_id=3,
                        x0=x0_stage3,
                        seed_offset=50)
    remaining -= res3["used_evals"]
    if res3["best_selectivity"] > best_overall_sel:
        best_overall_sel = res3["best_selectivity"]
        best_overall_curr = res3["best_currents"].copy()

    return {
        "optimizer": "MULTI",
        "repeat": repeat,
        "target_point": target_point,
        "best": best_overall_sel,
        "used_evals": total_evals - remaining,
        "tag": res3["tag"],
    }

# ---------------------------------------------------------------------
# Main: run experiments, summarise, plot
# ---------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()

    print("\n=== CMA-ES Multi-Stage vs Full vs Pairs (12 electrodes, grounded, plateau-based) ===\n")

    summaries: List[Dict[str, Any]] = []

    for tp in TARGET_POINTS:
        for r in range(REPEATS):
            print(f"Target={tp}, repeat={r+1}/{REPEATS}")

            res_pairs = run_pairs(TOTAL_EVALS, r, tp)
            summaries.append(res_pairs)

            res_full  = run_full(TOTAL_EVALS, r, tp)
            summaries.append(res_full)

            res_multi = run_multistage(TOTAL_EVALS, r, tp)
            summaries.append(res_multi)

    df = pd.DataFrame(summaries)
    df.to_csv(os.path.join(OUTPUT_DIR, "optimizer_summary.csv"), index=False)

    # ---------------- Boxplot summary ----------------
    plt.figure(figsize=(8, 5))
    for i, tp in enumerate(TARGET_POINTS):
        subset = df[df["target_point"].apply(tuple) == tp]
        data_pairs = subset[subset["optimizer"] == "PAIRS"]["best"].values
        data_full  = subset[subset["optimizer"] == "FULL"]["best"].values
        data_multi = subset[subset["optimizer"] == "MULTI"]["best"].values

        pos0 = 1 + 4*i
        positions = [pos0, pos0+1, pos0+2]
        plt.boxplot(
            [data_pairs, data_full, data_multi],
            positions=positions,
            widths=0.7,
            labels=["", "", ""],
            patch_artist=True
        )
        # colour coding
        colors = {"PAIRS": "lightgray", "FULL": "lightblue", "MULTI": "lightgreen"}
        for patch, opt in zip(plt.gca().artists[-3:], ["PAIRS", "FULL", "MULTI"]):
            patch.set_facecolor(colors[opt])
            patch.set_alpha(0.7)

        tx, ty, tz = tp
        plt.text(pos0+1, plt.ylim()[1], f"Target {i+1}", ha="center", va="bottom")

    xticks = []
    xlabels = []
    for i in range(len(TARGET_POINTS)):
        base = 1 + 4*i
        xticks.extend([base, base+1, base+2])
        xlabels.extend(["PAIRS", "FULL", "MULTI"])
    plt.xticks(xticks, xlabels, rotation=45)
    plt.ylabel("Best selectivity")
    plt.title("PAIRS vs FULL vs MULTI (per target)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_boxplots.png"), dpi=200)
    plt.close()

    # ---------------- Convergence curves (mean ± std) ----------------
    # We reconstruct convergence by reading the per-stage CSVs.
    methods = ["PAIRS", "FULL", "MULTI"]
    colors  = {"PAIRS":"gray", "FULL":"blue", "MULTI":"green"}

    for tp in TARGET_POINTS:
        fig, ax = plt.subplots(figsize=(7, 4))
        for opt in methods:
            # gather all runs for this target & opt
            curves = []
            for r in range(REPEATS):
                # each method may have multiple stages (for MULTI)
                pattern = f"{opt}_"
                # find CSVs that match this target and repeat
                for fname in os.listdir(OUTPUT_DIR):
                    if not fname.endswith(".csv"):
                        continue
                    if not fname.startswith(pattern):
                        continue
                    if f"_r{r}_" not in fname:
                        continue
                    df_run = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
                    df_run = df_run[df_run["target_x"] == tp[0]]
                    if "eval_index" in df_run.columns and "selectivity" in df_run.columns:
                        curve = df_run[["eval_index","selectivity"]].values
                        curves.append(curve)

            if not curves:
                continue

            max_evals = max(c[-1,0] for c in curves)
            eval_points = np.linspace(1, max_evals, 100)
            interp_curves = [np.interp(eval_points, c[:,0], c[:,1]) for c in curves]
            mean_curve = np.mean(interp_curves, axis=0)
            std_curve  = np.std(interp_curves, axis=0)

            ax.plot(eval_points, mean_curve,
                    label=opt, color=colors[opt], lw=2)
            ax.fill_between(eval_points,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            alpha=0.2, color=colors[opt])

        tx, ty, tz = tp
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Selectivity")
        ax.set_title(f"Convergence (Target: {tx*1e3:.1f}, {ty*1e3:.1f}, {tz*1e3:.1f} mm)")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fname = f"convergence_target_{tp[0]*1e3:.1f}_{tp[1]*1e3:.1f}_{tp[2]*1e3:.1f}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
        plt.close()

    elapsed = time.time() - start
    print("\n" + "="*70)
    print(f"COMPLETED in {elapsed/60:.1f} minutes (wall clock depends on machine)")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")
