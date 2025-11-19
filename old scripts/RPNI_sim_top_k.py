#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive CMA-ES Montage-Space Benchmark
====================================================================
Adds a new two-stage model:
  1) Optimise in the *pair* space with a fixed current (softmax over pairs)
  2) Pick top-K pairs and optimise their amplitudes

Preserves logging (CSV per run) and live plotting conventions.
"""

import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from cma import CMAEvolutionStrategy
from utils.run_selectivity_simulation import run_selectivity_simulation


# ==============================================================
# CONFIGURATION (SWEEPABLE HYPERPARAMETERS)
# ==============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "cma_es_top_k")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Geometry / simulation
N_ELECTRODES = 12
RANGE = 1e-3           # per-electrode current clamp for clipping
radius, height, sigma = 0.01, 0.04, 0.25

# Targets
TARGET_POINTS = [
    (-0.003, -0.002, -0.015),
    (0.0, 0.0, 0.0),
    (0.002, 0.0035, 0.010), 
]
REPEATS = 3

# --- Two-Stage model hyperparameter grids to sweep ---
STAGE1_SIGMAS   = [0.4e-3]
STAGE1_POPSIZES = [12, 16]
STAGE1_MAXITERS = [30, 50]

STAGE2_SIGMAS   = [0.1e-3, 0.3e-3]
STAGE2_POPSIZES = [8, 12]
STAGE2_MAXITERS = [50, 80]

FIXED_AMPS      = [0.6e-3]  # fixed per-pair current used in Stage 1
TOP_KS          = [4, 6, 8]
ADJACENT_ONLY   = [False]     # True: only adjacent bipolar pairs; False: all i<j pairs
CURRENT_LIMITS  = [0.8e-3, 1.0e-3]  # clipping bounds per electrode

# ==============================================================
# OBJECTIVE
# ==============================================================
def eval_selectivity(currents, target_point):
    currents = np.asarray(currents, dtype=float)
    currents -= np.mean(currents)  # enforce net-zero current return
    res = run_selectivity_simulation(
        n_rows=4, n_per_row=3,
        currents=currents,
        target_point=target_point,
        radius=radius, height=height, sigma=sigma,
        n_off_samples=1200, metric="activation",
        grounded_boundary=True, R_outer=0.10,
    )
    return -float(res["selectivity"])   # CMA-ES minimises

# ==============================================================
# LIVE PLOT
# ==============================================================
def save_live_plot(iters, vals, bests, tag, target_point):
    plt.figure(figsize=(5,3.5))
    plt.plot(iters, vals, 'o-', lw=1.2, label='Iter.')
    plt.plot(iters, bests, '-', lw=2, label='Best')
    tx,ty,tz = target_point
    plt.title(f"{tag}\nTarget x={tx*1e3:.1f} y={ty*1e3:.1f} z={tz*1e3:.1f} mm", fontsize=9)
    plt.xlabel("Iteration"); plt.ylabel("Selectivity")
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_{target_point}.png"), dpi=160)
    plt.close()

# ==============================================================
# NEW MODEL: Pairs → Top-K amplitudes (Two-Stage CMA-ES)
# ==============================================================
def run_cma_pairs_then_topk(
    sigma0_stage1,
    popsize_stage1,
    sigma0_stage2,
    popsize_stage2,
    repeat,
    target_point,
    fixed_amp=0.6e-3,
    top_k=6,
    maxiter_stage1=40,
    maxiter_stage2=60,
    adjacent_only=True,
    current_limit=None,
):
    """
    Two-stage CMA-ES:
      (1) Optimise over the *pair space* using a fixed per-pair current (via softmax weights).
      (2) Select the top-K pairs from Stage 1 and optimise their *amplitudes* to maximise selectivity.

    Logging & outputs mirror existing models:
      - Per-iteration CSV appended to OUTPUT_DIR with columns:
        ['sigma0','popsize','repeat','target_x','target_y','target_z',
         'montage_dim','stage','iteration','iteration_selectivity',
         'best_so_far','best_iter','currents', ...]
      - Live plots saved with tags including 'dim'.
    """


    if current_limit is None:
        current_limit = RANGE

    # ---------- Build pair templates ----------
    templates = []
    if adjacent_only:
        for i in range(N_ELECTRODES):
            v = np.zeros(N_ELECTRODES, dtype=float)
            v[i] = 1.0
            v[(i + 1) % N_ELECTRODES] = -1.0
            templates.append(v)
    else:
        for i in range(N_ELECTRODES):
            for j in range(i + 1, N_ELECTRODES):
                v = np.zeros(N_ELECTRODES, dtype=float)
                v[i] = 1.0
                v[j] = -1.0
                templates.append(v)
    templates = np.asarray(templates, dtype=float)
    P = templates.shape[0]

    def softmax(z):
        z = np.asarray(z, dtype=float)
        z = z - np.max(z)
        e = np.exp(z)
        return e / (np.sum(e) + 1e-12)

    # ---------- CSV path & tag ----------
    tag_base = (
        f"CMAES_PAIRTOPK_s1{sigma0_stage1:.1e}_p1{popsize_stage1}"
        f"_s2{sigma0_stage2:.1e}_p2{popsize_stage2}_r{repeat}"
    )
    csv_path = os.path.join(OUTPUT_DIR, f"{tag_base}.csv")
    header_written = False

    # ==========================================================
    # Stage 1: optimise softmax weights over pairs (fixed_amp)
    # ==========================================================
    es1 = CMAEvolutionStrategy(
        np.zeros(P, dtype=float),
        sigma0_stage1,
        {"popsize": int(popsize_stage1), "verb_disp": 0},
    )
    best_so_far = -np.inf
    best_iter = 0
    iter_vals, best_vals = [], []
    z_best = np.zeros(P, dtype=float)

    tag1 = f"{tag_base}_stage1_dim{P}"
    for it in tqdm(range(int(maxiter_stage1)), desc=tag1, leave=False):
        X = es1.ask()
        Y, combos, Zs = [], [], []
        for z in X:
            w = softmax(z)
            currents = fixed_amp * np.dot(w, templates)
            currents = np.clip(currents, -current_limit, current_limit)
            val = eval_selectivity(currents, target_point)  # negative selectivity
            Y.append(val); combos.append(currents); Zs.append(z)

        es1.tell(X, Y)

        sel = -float(np.min(Y))
        idx = int(np.argmin(Y))
        best_c_iter = combos[idx]
        z_star = Zs[idx]

        if sel > best_so_far:
            best_so_far = sel
            best_iter = it + 1
            z_best = np.asarray(z_star, dtype=float)

        iter_vals.append(sel); best_vals.append(best_so_far)
        save_live_plot(np.arange(1, len(iter_vals) + 1), iter_vals, best_vals, tag1, target_point)

        # CSV log (Stage 1)
        pd.DataFrame([{
            "sigma0": float(sigma0_stage1),
            "popsize": int(popsize_stage1),
            "repeat": int(repeat),
            "target_x": float(target_point[0]),
            "target_y": float(target_point[1]),
            "target_z": float(target_point[2]),
            "montage_dim": int(P),
            "stage": "stage1",
            "iteration": it + 1,
            "iteration_selectivity": float(sel),
            "best_so_far": float(best_so_far),
            "best_iter": int(best_iter),
            "currents": best_c_iter.tolist(),
            "fixed_amp": float(fixed_amp),
            "top_k": int(top_k),
            "adjacent_only": bool(adjacent_only),
            "current_limit": float(current_limit),
        }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
        header_written = True

    # Pick Top-K pairs by the softmax weights of the best solution
    w_best = softmax(z_best)
    k = int(min(top_k, P))
    top_idx = np.argsort(w_best)[-k:][::-1]
    templates_k = templates[top_idx]

    # ==========================================================
    # Stage 2: optimise amplitudes for the Top-K pairs
    # ==========================================================
    es2 = CMAEvolutionStrategy(
        np.zeros(k, dtype=float),
        sigma0_stage2,
        {"popsize": int(popsize_stage2), "verb_disp": 0},
    )
    best_so_far2 = -np.inf
    best_iter2 = 0
    iter_vals2, best_vals2 = [], []
    best_c2 = None

    tag2 = f"{tag_base}_stage2_dim{k}"
    for it in tqdm(range(int(maxiter_stage2)), desc=tag2, leave=False):
        X = es2.ask()
        Y, combos = [], []
        for a in X:
            currents = np.dot(a, templates_k)
            currents = np.clip(currents, -current_limit, current_limit)
            val = eval_selectivity(currents, target_point)
            Y.append(val); combos.append(currents)

        es2.tell(X, Y)

        sel = -float(np.min(Y))
        idx = int(np.argmin(Y))
        c_star = combos[idx]

        if sel > best_so_far2:
            best_so_far2 = sel
            best_iter2 = it + 1
            best_c2 = c_star

        iter_vals2.append(sel); best_vals2.append(best_so_far2)
        save_live_plot(np.arange(1, len(iter_vals2) + 1), iter_vals2, best_vals2, tag2, target_point)

        # CSV log (Stage 2)
        pd.DataFrame([{
            "sigma0": float(sigma0_stage2),
            "popsize": int(popsize_stage2),
            "repeat": int(repeat),
            "target_x": float(target_point[0]),
            "target_y": float(target_point[1]),
            "target_z": float(target_point[2]),
            "montage_dim": int(k),
            "stage": "stage2",
            "iteration": it + 1,
            "iteration_selectivity": float(sel),
            "best_so_far": float(best_so_far2),
            "best_iter": int(best_iter2),
            "currents": c_star.tolist(),
            "fixed_amp": float(fixed_amp),
            "selected_pairs": top_idx.tolist(),
            "current_limit": float(current_limit),
        }]).to_csv(csv_path, mode="a", index=False, header=False)

    return {
        "best_reward": float(best_so_far2),
        "montage_dim": int(k),
        "currents": (best_c2.tolist() if best_c2 is not None else None),
        "selected_pairs": top_idx.tolist(),
    }

# ==============================================================
# MAIN: sweep the new Two-Stage model
# ==============================================================
if __name__ == "__main__":
    start = time.time()
    all_runs = []
    print("\n=== CMA-ES Pair→TopK Two-Stage Benchmark ===")

    sweep_space = product(
        STAGE1_SIGMAS, STAGE1_POPSIZES, STAGE1_MAXITERS,
        STAGE2_SIGMAS, STAGE2_POPSIZES, STAGE2_MAXITERS,
        FIXED_AMPS, TOP_KS, ADJACENT_ONLY, CURRENT_LIMITS,
        range(REPEATS), TARGET_POINTS
    )

    for (s1, p1, it1, s2, p2, it2, f_amp, k, adj_only, clip_lim, r, tp) in sweep_space:
        result = run_cma_pairs_then_topk(
            sigma0_stage1=s1,
            popsize_stage1=p1,
            sigma0_stage2=s2,
            popsize_stage2=p2,
            repeat=r,
            target_point=tp,
            fixed_amp=f_amp,
            top_k=k,
            maxiter_stage1=it1,
            maxiter_stage2=it2,
            adjacent_only=adj_only,
            current_limit=clip_lim,
        )
        result.update({
            "sigma0_stage1": s1, "popsize_stage1": p1, "maxiter_stage1": it1,
            "sigma0_stage2": s2, "popsize_stage2": p2, "maxiter_stage2": it2,
            "fixed_amp": f_amp, "top_k": k, "adjacent_only": adj_only,
            "current_limit": clip_lim, "repeat": r,
            "target_point": tp,
        })
        all_runs.append(result)

    df = pd.DataFrame(all_runs)
    summary_path = os.path.join(OUTPUT_DIR, "pairtopk_summary_raw.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nRaw summary saved to {summary_path}")

    # ==========================================================
    # Compute AUC / stats from per-run CSV logs
    # ==========================================================
    print("\nComputing AUC and convergence statistics...")

    auc_records = []
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".csv") and "CMAES_PAIRTOPK" in f:
            path = os.path.join(OUTPUT_DIR, f)
            try:
                d = pd.read_csv(path)
                if "best_so_far" not in d.columns: 
                    continue
                # Aggregate over all iterations (both stages)
                d = d.sort_values(["stage", "iteration"])
                y = d["best_so_far"].values
                x = np.arange(1, len(y) + 1)
                auc = np.trapz(y, x) / max(len(y), 1)
                best_sel = float(np.max(y)) if len(y) else 0.0
                thresh = 0.9 * best_sel
                speed = int(np.argmax(y >= thresh) + 1) if np.any(y >= thresh) else len(y)

                # Read dimensionality of Stage 2 if present; else Stage 1
                mdim = int(d[d["stage"] == "stage2"]["montage_dim"].iloc[0]) if np.any(d["stage"] == "stage2") \
                       else int(d["montage_dim"].iloc[0])

                auc_records.append({
                    "file": f,
                    "montage_dim": mdim,
                    "sigma0": float(d["sigma0"].iloc[0]),
                    "popsize": int(d["popsize"].iloc[0]),
                    "target_x": float(d["target_x"].iloc[0]),
                    "target_y": float(d["target_y"].iloc[0]),
                    "target_z": float(d["target_z"].iloc[0]),
                    "AUC": auc, "best_selectivity": best_sel,
                    "speed_to90": speed, "iterations": len(y)
                })
            except Exception as e:
                print(f"Skip {f}: {e}")

    auc_df = pd.DataFrame(auc_records)
    auc_df.to_csv(os.path.join(OUTPUT_DIR, "pairtopk_auc_details.csv"), index=False)

    # ==========================================================
    # FAIR AGGREGATION (per target → averaged)
    # ==========================================================
    if len(auc_df):
        per_target = (auc_df.groupby(
            ["target_x","target_y","target_z","montage_dim","sigma0","popsize"]
        )[["AUC","best_selectivity","speed_to90"]]
        .mean().reset_index())

        rank_df = (per_target.groupby(["montage_dim","sigma0","popsize"])
                   [["AUC","best_selectivity","speed_to90"]]
                   .agg(["mean","std","count"]).reset_index())
        rank_df.columns = ["montage_dim","sigma0","popsize",
                           "AUC_mean","AUC_std","AUC_n",
                           "Sel_mean","Sel_std","Sel_n",
                           "Speed_mean","Speed_std","Speed_n"]

        rank_df["score"] = (
            rank_df["AUC_mean"]*0.6 + rank_df["Sel_mean"]*0.4 - 0.01*rank_df["Speed_mean"]
        )
        rank_df.sort_values("score", ascending=False, inplace=True)
        rank_path = os.path.join(OUTPUT_DIR, "pairtopk_ranked_summary.csv")
        rank_df.to_csv(rank_path, index=False)

        # Report
        best = rank_df.iloc[0]
        print("\n================== FINAL PERFORMANCE REPORT ==================")
        print(f"Total runs analysed: {len(auc_df)}")
        print(f"Best configuration:")
        print(f"  • Montage dimension : {int(best['montage_dim'])}")
        print(f"  • Sigma0            : {best['sigma0']*1e3:.3f} mA")
        print(f"  • Population size   : {int(best['popsize'])}")
        print(f"  • Mean AUC          : {best['AUC_mean']:.4f} ± {best['AUC_std']:.4f}")
        print(f"  • Mean best select. : {best['Sel_mean']:.4f} ± {best['Sel_std']:.4f}")
        print(f"  • Mean iters→90%    : {best['Speed_mean']:.1f} ± {best['Speed_std']:.1f}")
        print(f"  • Composite score   : {best['score']:.4f}")
        print("---------------------------------------------------------------")
        print("Top 5 configurations by composite score:\n")
        print(rank_df.head(5).to_string(index=False))
        print("---------------------------------------------------------------")

        # Plot overview
        plt.figure(figsize=(7,5))
        for s in rank_df["sigma0"].unique():
            sub = rank_df[rank_df["sigma0"] == s]
            plt.errorbar(sub["montage_dim"], sub["AUC_mean"],
                         yerr=sub["AUC_std"], capsize=3, marker='o',
                         label=f"σ={s*1e3:.2f} mA")
        plt.xlabel("Montage Dimension")
        plt.ylabel("Mean Normalised AUC")
        plt.title("CMA-ES Pair→TopK Performance (AUC ± SD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "pairtopk_ranking_auc_summary.png"), dpi=180)
        plt.close()

    print(f"Runtime: {(time.time()-start)/60:.1f} min")
    print("===============================================================")
