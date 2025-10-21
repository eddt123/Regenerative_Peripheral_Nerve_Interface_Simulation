#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive CMA-ES Montage-Space Benchmark
====================================================================
Explores CMA-ES hyperparameters (sigma0, popsize) and montage
dimensionalities, runs each configuration for several target points
and repeats, and produces a detailed report including AUC, mean best
selectivity, convergence speed, and variance.
"""

import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from cma import CMAEvolutionStrategy
from utils.run_selectivity_simulation import run_selectivity_simulation

# ==============================================================
# CONFIGURATION
# ==============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__),
                          "data", "cma_es_montage_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ELECTRODES = 12
RANGE = 1e-3
N_ITER = 100
REPEATS = 5

SIGMAS   = [0.2e-3, 0.3e-3]
POPSIZES = [12, 16]
DIMS_TO_TEST = [2, 4, 6, 8, 12]

radius, height, sigma = 0.01, 0.04, 0.25

TARGET_POINTS = [
    (0.0, 0.0, 0.0),
    (0.002, 0.0035, 0.010),
    (-0.003, -0.002, -0.015),
]

# ==============================================================
# OBJECTIVE
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
    return -float(res["selectivity"])   # CMA-ES minimises

# ==============================================================
# LIVE PLOT
# ==============================================================
def save_live_plot(iters, vals, bests, tag, target_point):
    plt.figure(figsize=(5,3.5))
    plt.plot(iters, vals, 'o-', color='gray', lw=1.2, label='Iter.')
    plt.plot(iters, bests, '-', color='blue', lw=2, label='Best')
    tx,ty,tz = target_point
    plt.title(f"{tag}\nTarget x={tx*1e3:.1f} y={ty*1e3:.1f} z={tz*1e3:.1f} mm",
              fontsize=9)
    plt.xlabel("Iteration"); plt.ylabel("Selectivity")
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"{tag}_progress.png"),dpi=160)
    plt.close()

# ==============================================================
# CMA-ES Variable-Dimension Montage
# ==============================================================
def run_cma_montage_variable_dim(sigma0, popsize, repeat,
                                 target_point, dims_to_test):
    tag_base = f"CMAES_VAR_s{sigma0:.1e}_p{popsize}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR,f"{tag_base}.csv")
    header_written=False

    # ----- Build montage library -----
    montages=[]
    for i in range(N_ELECTRODES):
        p=np.zeros(N_ELECTRODES);p[i]=1;p[(i+1)%N_ELECTRODES]=-1;montages.append(p)
        t=np.zeros(N_ELECTRODES);t[i]=1;t[(i+1)%N_ELECTRODES]=-2;t[(i+2)%N_ELECTRODES]=1;montages.append(t)
        q=np.zeros(N_ELECTRODES);q[i]=1;q[(i+1)%N_ELECTRODES]=-1;q[(i+2)%N_ELECTRODES]=-1;q[(i+3)%N_ELECTRODES]=1;montages.append(q)
    rng=np.random.default_rng(42)
    for _ in range(N_ELECTRODES):
        rp=rng.choice([-1,0,1],size=N_ELECTRODES,p=[0.4,0.2,0.4])
        if np.sum(np.abs(rp))>0: rp=rp/np.sum(np.abs(rp))
        montages.append(rp)
    montages=np.array(montages)

    best_global={"best_reward":-np.inf,"montage_dim":None,"currents":None}
    # ----------------------------------------------------------
    for dim in dims_to_test:
        templates=montages[:dim]
        tag=f"{tag_base}_dim{dim}"
        print(f"[{tag}] Optimising {dim}-D montage space")
        es=CMAEvolutionStrategy(np.zeros(dim),sigma0,
                                {"popsize":popsize,"verb_disp":0})
        best_so_far=-np.inf;best_iter=0;iter_vals=[];best_vals=[]
        for it in tqdm(range(N_ITER),desc=tag,leave=False):
            X=es.ask();Y=[];combos=[]
            for w in X:
                currents=np.dot(w,templates)
                currents=np.clip(currents,-RANGE,RANGE)
                val=eval_selectivity(currents,target_point)
                Y.append(val);combos.append(currents)
            es.tell(X,Y)
            sel=-np.min(Y);idx=int(np.argmin(Y))
            best_c=combos[idx]
            iter_vals.append(sel)
            if sel>best_so_far:best_so_far, best_iter=sel,it+1
            best_vals.append(best_so_far)
            save_live_plot(np.arange(1,len(iter_vals)+1),
                           iter_vals,best_vals,tag,target_point)
            pd.DataFrame([{
                "sigma0":sigma0,"popsize":popsize,"repeat":repeat,
                "target_x":target_point[0],"target_y":target_point[1],
                "target_z":target_point[2],"montage_dim":dim,
                "iteration":it+1,"iteration_selectivity":sel,
                "best_so_far":best_so_far,"best_iter":best_iter,
                "currents":best_c.tolist()
            }]).to_csv(csv_path,mode="a",index=False,
                       header=not header_written)
            header_written=True
        if best_so_far>best_global["best_reward"]:
            best_global.update({"best_reward":best_so_far,
                                "montage_dim":dim,"currents":best_c.tolist()})
    return best_global


# ==============================================================
# CMA-ES Two-Stage (2D sweep → 12D fine-tune, warm-start version)
# ==============================================================
def run_cma_two_stage(sigma0, popsize, repeat, target_point):
    """
    Two-phase optimisation:
      1) 2D CMA-ES in a reduced montage space (coarse global search)
      2) Switches to full 12D CMA-ES once improvement stagnates for 10 iters
         using a warm-started mean and reduced step-size for stability.

    Returns best configuration and logs iteration history.
    """
    tag = f"CMAES_TWOSTAGE_s{sigma0:.1e}_p{popsize}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")
    header_written = False

    # --- Phase 1: 2D montage basis (opposite pairs for coarse exploration) ---
    montages = np.array([
        [+1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, +1, -1, 0, 0, 0, 0]
    ])
    dim_reduced = montages.shape[0]

    es = CMAEvolutionStrategy(np.zeros(dim_reduced), sigma0, {"popsize": popsize, "verb_disp": 0})

    best_so_far = -np.inf
    best_iter = 0
    iter_vals, best_vals = [], []
    plateau_counter = 0
    PLATEAU_ITERS = 10
    IMPROVEMENT_THRESH = 1e-4
    switched = False

    # ===============================================================
    # Phase 1 — 2D reduced-space search
    # ===============================================================
    for it in tqdm(range(N_ITER), desc=tag, leave=False):
        weights_list = es.ask()
        Y, combos = [], []

        for w in weights_list:
            # combine 2D montages
            currents = np.sum([w[j] * montages[j] for j in range(dim_reduced)], axis=0)
            currents = np.clip(currents, -RANGE, RANGE)
            obj = eval_selectivity(currents, target_point)
            Y.append(obj)
            combos.append(currents)

        es.tell(weights_list, Y)
        iter_best = -np.min(Y)
        iter_best_idx = int(np.argmin(Y))
        iter_best_currents = combos[iter_best_idx]

        iter_vals.append(iter_best)
        if iter_best > best_so_far + IMPROVEMENT_THRESH:
            best_so_far, best_iter = iter_best, it + 1
            plateau_counter = 0
        else:
            plateau_counter += 1
        best_vals.append(best_so_far)

        # Plot progress
        save_live_plot(np.arange(1, len(iter_vals)+1), iter_vals, best_vals, tag, target_point)

        pd.DataFrame([{
            "phase": "reduced2D",
            "sigma0": sigma0, "popsize": popsize, "repeat": repeat,
            "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
            "iteration": it+1, "selectivity": iter_best, "best_so_far": best_so_far,
            "plateau_counter": plateau_counter, "switched": switched,
            "currents": iter_best_currents.tolist()
        }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
        header_written = True

        # ===============================================================
        # Switch condition: plateau detected → fine-tuning phase
        # ===============================================================
        if plateau_counter >= PLATEAU_ITERS and not switched:
            switched = True
            print(f"[{tag}] Plateau detected at iter {it}. Switching to full 12D fine-tune (warm-start).")

            # --- Warm-start setup ---
            mean_2d = es.result.xbest
            sigma_fine = es.sigma * 0.3           # smaller step size for stability
            mean_12d = np.zeros(12)
            # project coarse solution into full space
            mean_12d += np.sum([mean_2d[j] * montages[j] for j in range(dim_reduced)], axis=0)

            # start fine CMA-ES from warm-started mean
            es = CMAEvolutionStrategy(mean_12d, sigma_fine, {"popsize": popsize, "verb_disp": 0})
            plateau_counter = 0
            print(f"[{tag}] Warm-started fine-tune: σ={sigma_fine:.2e}")

        # ===============================================================
        # Phase 2 — Fine-tuning in 12D space
        # ===============================================================
        if switched:
            weights_list = es.ask()
            Y, combos = [], []
            for w in weights_list:
                currents = np.clip(w, -RANGE, RANGE)
                obj = eval_selectivity(currents, target_point)
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

            save_live_plot(np.arange(1, len(iter_vals)+1), iter_vals, best_vals, tag+"_fine", target_point)

            pd.DataFrame([{
                "phase": "fine12D",
                "sigma0": sigma0, "popsize": popsize, "repeat": repeat,
                "target_x": target_point[0], "target_y": target_point[1], "target_z": target_point[2],
                "iteration": it+1, "selectivity": iter_best, "best_so_far": best_so_far,
                "plateau_counter": plateau_counter, "switched": switched,
                "currents": iter_best_currents.tolist()
            }]).to_csv(csv_path, mode="a", index=False, header=False)

    # ===============================================================
    # Return final summary
    # ===============================================================
    return {
        "optimizer": "CMAES_TWOSTAGE",
        "tag": tag,
        "best": float(best_so_far),
        "iter": int(best_iter),
        "sigma0": sigma0,
        "popsize": popsize,
        "repeat": repeat,
        "target_point": target_point
    }



# ==============================================================
# MAIN BENCHMARK
# ==============================================================
if __name__=="__main__":
    start=time.time()
    all_runs=[]
    print("\n=== Variable-Dimension Montage CMA-ES Benchmark ===")
    for sigma0,popsize in product(SIGMAS,POPSIZES):
        for r,tp in product(range(REPEATS),TARGET_POINTS):
            best_cfg=run_cma_montage_variable_dim(
                sigma0,popsize,r,tp,DIMS_TO_TEST)
            best_cfg.update({"sigma0":sigma0,"popsize":popsize,
                             "repeat":r,"target_point":tp})
            all_runs.append(best_cfg)

    df=pd.DataFrame(all_runs)
    summary_path=os.path.join(OUTPUT_DIR,"montage_summary_raw.csv")
    df.to_csv(summary_path,index=False)
    print(f"\nRaw summary saved to {summary_path}")

    # ==========================================================
    # --- Compute AUC and detailed statistics per run ---
    # ==========================================================
    print("\nComputing AUC and convergence statistics...")

    # Gather per-run progress files
    auc_records=[]
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".csv") and "CMAES_VAR" in f and "dim" in f:
            path=os.path.join(OUTPUT_DIR,f)
            try:
                d=pd.read_csv(path)
                if "best_so_far" not in d.columns: continue
                x=np.arange(len(d["best_so_far"]))
                y=d["best_so_far"].values
                auc=np.trapz(y,x)/len(y)
                best_sel=float(np.max(y))
                # speed: first iter reaching 90% of final best
                thresh=0.9*best_sel
                speed=np.argmax(y>=thresh)+1 if np.any(y>=thresh) else len(y)
                auc_records.append({
                    "file":f,
                    "montage_dim":int(d["montage_dim"].iloc[0]),
                    "sigma0":float(d["sigma0"].iloc[0]),
                    "popsize":int(d["popsize"].iloc[0]),
                    "target_x":float(d["target_x"].iloc[0]),
                    "target_y":float(d["target_y"].iloc[0]),
                    "target_z":float(d["target_z"].iloc[0]),
                    "AUC":auc,"best_selectivity":best_sel,
                    "speed_to90":speed,"iterations":len(y)
                })
            except Exception as e:
                print(f"Skip {f}: {e}")

    auc_df=pd.DataFrame(auc_records)
    auc_df.to_csv(os.path.join(OUTPUT_DIR,"auc_details.csv"),index=False)

    

    # ==========================================================
    # FAIR AGGREGATION (per target → averaged)
    # ==========================================================
    per_target=(auc_df.groupby(
        ["target_x","target_y","target_z","montage_dim","sigma0","popsize"]
        )[["AUC","best_selectivity","speed_to90"]]
        .mean().reset_index())

    rank_df=(per_target.groupby(["montage_dim","sigma0","popsize"])
             [["AUC","best_selectivity","speed_to90"]]
             .agg(["mean","std","count"]).reset_index())
    rank_df.columns=["montage_dim","sigma0","popsize",
                     "AUC_mean","AUC_std","AUC_n",
                     "Sel_mean","Sel_std","Sel_n",
                     "Speed_mean","Speed_std","Speed_n"]

    # composite performance score (higher AUC, selectivity; lower speed)
    rank_df["score"]=rank_df["AUC_mean"]*0.6+rank_df["Sel_mean"]*0.4 - 0.01*rank_df["Speed_mean"]
    rank_df.sort_values("score",ascending=False,inplace=True)
    rank_path=os.path.join(OUTPUT_DIR,"ranked_summary.csv")
    rank_df.to_csv(rank_path,index=False)

    # ==========================================================
    # REPORT
    # ==========================================================
    best=rank_df.iloc[0]
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
        sub=rank_df[rank_df["sigma0"]==s]
        plt.errorbar(sub["montage_dim"],sub["AUC_mean"],
                     yerr=sub["AUC_std"],capsize=3,marker='o',
                     label=f"σ={s*1e3:.2f} mA")
    plt.xlabel("Montage Dimension")
    plt.ylabel("Mean Normalised AUC")
    plt.title("CMA-ES Montage Performance (AUC ± SD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"ranking_auc_summary.png"),dpi=180)
    plt.close()

    print(f"Ranking table saved to {rank_path}")
    print(f"Runtime: {(time.time()-start)/60:.1f} min")
    print("===============================================================")
