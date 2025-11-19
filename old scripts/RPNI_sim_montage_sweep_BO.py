#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Bayesian Optimization Montage-Space Benchmark (no skopt)
=====================================================================
- Fixes degeneracy from double-projection and clipping
- Uses smooth projection (zero-sum + isotropic scaling) once, outside the obj
- Normalizes montage templates to a common zero-sum manifold
- Minimal BO built on sklearn GPR with LCB/EI
- Same outputs: per-iteration CSVs, progress plots, AUC summaries
"""

import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from collections import defaultdict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from scipy.stats import norm

from utils.run_selectivity_simulation import run_selectivity_simulation


# ==============================================================
# CONFIGURATION
# ==============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "bo_montage_LCB")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ELECTRODES = 12
RANGE = 1e-3                 # |current| bound per electrode (A)
N_ITER = 100
REPEATS = 5

ACQ_FUNCS = ["LCB"]    # Choose one or both
KAPPAS = [1.0, 2.0, 3.0]          # Stronger exploration for LCB
XIS = [0.1, 0.05]           # Exploration for EI
DIMS_TO_TEST = [2, 4]

# Cylinder + objective config
radius, height, sigma = 0.01, 0.04, 0.25
TARGET_POINTS = [
    (0.0, 0.0, 0.0),
    (0.002, 0.0035, 0.010),
    (-0.003, -0.002, -0.015),
]

# Diagnostics (optional)
LOG_DUPLICATE_CURRENTS = False  # set True to verify degeneracy is gone
ROUND_BUCKETS = 256             # for duplicate binning of currents


# ==============================================================
# OBJECTIVE FUNCTION  (NO zero-sum here; it's done outside)
# ==============================================================
def eval_selectivity(currents, target_point):
    currents = np.asarray(currents, dtype=float)
    # Do NOT subtract mean here; zero-sum is handled once in the caller
    res = run_selectivity_simulation(
        n_rows=4, n_per_row=3,
        currents=currents,
        target_point=target_point,
        radius=radius, height=height, sigma=sigma,
        n_off_samples=1200, metric="activation",
        grounded_boundary=True, R_outer=0.10,
    )
    return -float(res["selectivity"])  # BO minimises


# ==============================================================
# LIVE PLOT
# ==============================================================
def save_live_plot(iters, vals, bests, tag, target_point):
    plt.figure(figsize=(5, 3.5))
    plt.plot(iters, vals, 'o-', lw=1.2, label='Iter.')
    plt.plot(iters, bests, '-', lw=2, label='Best')
    tx, ty, tz = target_point
    plt.title(f"{tag}\nTarget x={tx*1e3:.1f} y={ty*1e3:.1f} z={tz*1e3:.1f} mm", fontsize=9)
    plt.xlabel("Iteration"); plt.ylabel("Selectivity")
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_progress.png"), dpi=160)
    plt.close()


# ==============================================================
# Minimal Bayesian Optimizer (no skopt)
# ==============================================================

class SimpleBO:
    """
    Minimal BO for MINIMIZATION with GP surrogate and LCB/EI acquisitions.
    - Smooth candidate search: random pool (scaled with log(dim)) + acq argmax
    - More robust GP settings to avoid overconfidence on flat regions
    """
    def __init__(self, bounds, acq_func="LCB", kappa=2.0, xi=0.01, random_state=0):
        self.bounds = np.array(bounds, dtype=float)       # (d, 2)
        self.dim = self.bounds.shape[0]
        self.acq_func = acq_func.upper()
        assert self.acq_func in ("LCB", "EI"), "acq_func must be 'LCB' or 'EI'"
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.rng = np.random.default_rng(int(random_state))

        self.X, self.y = [], []

        k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(self.dim), nu=1.5)
        k += WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-2))
        self.gp = GaussianProcessRegressor(
            kernel=k,
            alpha=1e-8,            # modest jitter to reduce overconfidence
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=int(random_state)
        )

        # Larger init design to learn a sensible length-scale in low dims
        self.n_init = max(8, 3 * self.dim)

    def _sample_uniform(self, n):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng.uniform(low, high, size=(n, self.dim))

    @staticmethod
    def _ei_min(y_min, mu, sigma, xi):
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_min - mu - xi
            z = imp / (sigma + 1e-12)
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma < 1e-16] = 0.0
        return ei

    def _score_acq(self, Xcand):
        mu, std = self.gp.predict(Xcand, return_std=True)
        mu = mu.ravel(); std = std.ravel()
        if self.acq_func == "LCB":
            # minimize mu - kappa*std  == maximize -(mu - kappa*std)
            return -(mu - self.kappa * std)
        else:
            y_min = np.min(self.y)
            return self._ei_min(y_min, mu, std, self.xi)

    def ask(self):
        if len(self.X) < self.n_init:
            return self._sample_uniform(1)[0]
        X = np.asarray(self.X); y = np.asarray(self.y)
        self.gp.fit(X, y)
        n_cand = int(4096 * (1.0 + np.log(self.dim + 1)))
        Xcand = self._sample_uniform(n_cand)
        scores = self._score_acq(Xcand)
        return Xcand[int(np.argmax(scores))]

    def tell(self, x, y):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        x = np.minimum(np.maximum(x, self.bounds[:, 0]), self.bounds[:, 1])
        self.X.append(x.ravel()); self.y.append(float(y))


# ==============================================================
# Helpers: build & normalize montage templates
# ==============================================================

def _zero_sum_l1_normalize(v):
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    s = np.sum(np.abs(v))
    return v / s if s > 0 else v

def build_normalized_montages(n_electrodes, seed=42):
    """Returns an array [n_templates, n_electrodes], each zero-sum & L1-normalized."""
    montages = []
    for i in range(n_electrodes):
        # Bipolar (+1, -1)
        p = np.zeros(n_electrodes); p[i] = 1; p[(i + 1) % n_electrodes] = -1
        montages.append(_zero_sum_l1_normalize(p))
        # Tripolar (+1, -2, +1)
        t = np.zeros(n_electrodes); t[i] = 1; t[(i + 1) % n_electrodes] = -2; t[(i + 2) % n_electrodes] = 1
        montages.append(_zero_sum_l1_normalize(t))
        # Quad (+1, -1, -1, +1)
        q = np.zeros(n_electrodes); q[i] = 1; q[(i + 1) % n_electrodes] = -1; q[(i + 2) % n_electrodes] = -1; q[(i + 3) % n_electrodes] = 1
        montages.append(_zero_sum_l1_normalize(q))
    rng = np.random.default_rng(seed)
    for _ in range(n_electrodes):
        rp = rng.choice([-1, 0, 1], size=n_electrodes, p=[0.4, 0.2, 0.4]).astype(float)
        montages.append(_zero_sum_l1_normalize(rp))
    return np.array(montages, dtype=float)


# ==============================================================
# Smooth projection: zero-sum + isotropic scaling (NO elementwise clip)
# ==============================================================

def smooth_project_to_bounds(currents, max_abs=RANGE):
    """Zero-sum once, then isotropically scale if any component exceeds bound."""
    currents = np.asarray(currents, dtype=float)
    currents = currents - currents.mean()
    ma = np.max(np.abs(currents))
    if ma > max_abs and ma > 0:
        currents = currents * (max_abs / ma)
    return currents


# ==============================================================
# BAYESIAN OPTIMIZATION Variable-Dimension Montage
# ==============================================================

def run_bo_montage_variable_dim(acq_func, kappa, xi, repeat,
                                target_point, dims_to_test):
    tag_base = f"BO_VAR_{acq_func}_k{kappa}_xi{xi}_r{repeat}"
    csv_path = os.path.join(OUTPUT_DIR, f"{tag_base}.csv")
    header_written = False

    montages_all = build_normalized_montages(N_ELECTRODES, seed=42)

    # Optional degeneracy diagnostics
    dup_count = defaultdict(int)

    best_global = {"best_reward": -np.inf, "montage_dim": None, "currents": None}

    for dim in dims_to_test:
        templates = montages_all[:dim]     # [dim, N_ELECTRODES]
        tag = f"{tag_base}_dim{dim}"
        print(f"[{tag}] Optimising {dim}-D montage space")

        opt = SimpleBO(bounds=[(-1.0, 1.0)] * dim,
                       acq_func=acq_func, kappa=kappa, xi=xi,
                       random_state=repeat)

        best_so_far = -np.inf
        best_iter = 0
        best_currents = None
        iter_vals, best_vals = [], []

        for it in tqdm(range(N_ITER), desc=tag, leave=False):
            w = np.array(opt.ask(), dtype=float)                 # (dim,)
            # Linear combine templates
            currents = np.dot(w, templates)                      # (N_ELECTRODES,)
            # Smooth, single projection to feasible set
            currents = smooth_project_to_bounds(currents, RANGE)

            # (Optional) degeneracy check
            if LOG_DUPLICATE_CURRENTS:
                key = tuple(np.round(currents / (RANGE / ROUND_BUCKETS)).astype(int))
                dup_count[key] += 1

            val = eval_selectivity(currents, target_point)       # negative selectivity
            opt.tell(w, val)

            sel = -val
            iter_vals.append(sel)
            if sel > best_so_far:
                best_so_far = sel
                best_iter = it + 1
                best_currents = currents.copy()
            best_vals.append(best_so_far)

            # Plot & log
            save_live_plot(np.arange(1, len(iter_vals) + 1), iter_vals, best_vals, tag, target_point)

            pd.DataFrame([{
                "acq_func": acq_func, "kappa": kappa, "xi": xi,
                "repeat": repeat, "target_x": target_point[0],
                "target_y": target_point[1], "target_z": target_point[2],
                "montage_dim": dim, "iteration": it + 1,
                "iteration_selectivity": sel, "best_so_far": best_so_far,
                "best_iter": best_iter, "currents": currents.tolist()
            }]).to_csv(csv_path, mode="a", index=False, header=not header_written)
            header_written = True

        if best_so_far > best_global["best_reward"]:
            best_global.update({
                "best_reward": best_so_far,
                "montage_dim": dim,
                "currents": best_currents.tolist() if best_currents is not None else currents.tolist()
            })

    # Print degeneracy info if enabled
    if LOG_DUPLICATE_CURRENTS:
        # number of unique current patterns vs total
        total = sum(dup_count.values())
        unique = len(dup_count)
        print(f"[Diagnostics] Unique currents: {unique} / total evals across dims: {total}")
    return best_global


# ==============================================================
# MAIN BENCHMARK
# ==============================================================

if __name__ == "__main__":
    start = time.time()
    all_runs = []
    print("\n=== Variable-Dimension Montage Bayesian Optimisation Benchmark ===")
    for acq_func, kappa, xi in product(ACQ_FUNCS, KAPPAS, XIS):
        for r, tp in product(range(REPEATS), TARGET_POINTS):
            best_cfg = run_bo_montage_variable_dim(acq_func, kappa, xi, r, tp, DIMS_TO_TEST)
            best_cfg.update({"acq_func": acq_func, "kappa": kappa, "xi": xi, "repeat": r, "target_point": tp})
            all_runs.append(best_cfg)

    df = pd.DataFrame(all_runs)
    summary_path = os.path.join(OUTPUT_DIR, "bo_montage_summary_raw.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nRaw summary saved to {summary_path}")

    # --- Compute AUC and convergence stats ---
    auc_records = []
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".csv") and f.startswith("BO_VAR_"):
            path = os.path.join(OUTPUT_DIR, f)
            try:
                d = pd.read_csv(path)
                if "best_so_far" not in d.columns:
                    continue
                # group by (dim, repeat, target) to compute per-run curves robustly
                grp_cols = ["montage_dim", "repeat", "target_x", "target_y", "target_z"]
                for _, g in d.groupby(grp_cols):
                    y = g["best_so_far"].values
                    if len(y) == 0:
                        continue
                    x = np.arange(1, len(y) + 1)
                    auc = np.trapz(y, x) / len(y)
                    best_sel = float(np.max(y))
                    thresh = 0.9 * best_sel
                    speed = int(np.argmax(y >= thresh) + 1) if np.any(y >= thresh) else len(y)
                    auc_records.append({
                        "file": f,
                        "montage_dim": int(g["montage_dim"].iloc[0]),
                        "acq_func": str(g["acq_func"].iloc[0]),
                        "kappa": float(g["kappa"].iloc[0]),
                        "xi": float(g["xi"].iloc[0]),
                        "AUC": auc, "best_selectivity": best_sel,
                        "speed_to90": speed, "iterations": len(y)
                    })
            except Exception as e:
                print(f"Skip {f}: {e}")

    auc_df = pd.DataFrame(auc_records)
    auc_df.to_csv(os.path.join(OUTPUT_DIR, "bo_auc_details.csv"), index=False)

    per_target = (auc_df.groupby(["montage_dim", "acq_func", "kappa", "xi"])
                  [["AUC", "best_selectivity", "speed_to90"]].mean().reset_index())

    per_target["score"] = (
        per_target["AUC"] * 0.6
        + per_target["best_selectivity"] * 0.4
        - 0.01 * per_target["speed_to90"]
    )
    per_target.sort_values("score", ascending=False, inplace=True)

    rank_path = os.path.join(OUTPUT_DIR, "bo_ranked_summary.csv")
    per_target.to_csv(rank_path, index=False)

    print("\nTop 5 BO configurations by composite score:\n")
    if len(per_target) > 0:
        print(per_target.head(5).to_string(index=False))
    else:
        print("No results aggregated (check that runs produced per-iteration CSVs).")

    # Plot overview
    if len(per_target) > 0:
        plt.figure(figsize=(7, 5))
        for acq in per_target["acq_func"].unique():
            sub = per_target[per_target["acq_func"] == acq]
            plt.plot(sub["montage_dim"], sub["AUC"], 'o-', label=f"{acq}")
        plt.xlabel("Montage Dimension")
        plt.ylabel("Mean Normalised AUC")
        plt.title("Bayesian Optimisation Montage Performance (AUC)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "bo_auc_summary.png"), dpi=180)
        plt.close()

    print(f"Ranking table saved to {rank_path}")
    print(f"Runtime: {(time.time() - start) / 60:.1f} min")
