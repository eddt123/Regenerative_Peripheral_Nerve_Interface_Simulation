#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytic Cylinder Optimization Simulation
-----------------------------------------
Drop-in replacement for the COMSOL-based script.
Computes potential and activation analytically inside a
homogeneous cylindrical volume conductor with 12 electrodes
(3 rows × 4 columns). Uses same optimizer set and parameters.
"""

from pathlib import Path
import numpy as np

# Import optimizers (same as your repo)
from models.greedy_random_sim import GreedyRandomOptimizer
from models.bo_model_sim import BOSimulation
from models.cmaes_sim import CMAESOptimizer
from models.gradient_descent_sim import GradientDescentSimulator
from models.de_sim import DESimulation
from models.additional_optimisers_sim import (
    TPESimulation,
    RandomForestBOSimulation,
    CEMOptimizer,
    PowellLocalOptimizer,
    BanditOptimizer,
    PSOSimulation,
)

# ---------------------------------------------------------------------
# Analytical simulator replacing COMSOL
# ---------------------------------------------------------------------

R = 1.5e-3          # cylinder radius [m]
SIGMA = 0.3         # conductivity [S/m]
Z_SPACING = 1.0e-3  # distance between electrode rings
H_AF = 0.25e-3      # step for activation fn [m]
ALPHA = 8.0
BETA = 0.5

# Generate 3 rows × 4 electrodes
def electrode_positions(n_rows=3, n_per_row=4, radius=R, dz=Z_SPACING):
    thetas = np.linspace(0, 2*np.pi, n_per_row, endpoint=False)
    zs = np.linspace(-(n_rows-1)/2*dz, (n_rows-1)/2*dz, n_rows)
    pos = []
    for z in zs:
        for theta in thetas:
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            pos.append([x, y, z])
    return np.array(pos)

ELEC_POS = electrode_positions(3, 4)
FOUR_PI_SIG = 4*np.pi*SIGMA


def potential_at(r, currents):
    r = np.atleast_2d(r)
    dif = r[:, None, :] - ELEC_POS[None, :, :]
    dist = np.linalg.norm(dif, axis=-1) + 1e-12
    v = (currents[None, :] / (FOUR_PI_SIG * dist)).sum(axis=1)
    return v


def activation_function(r, u, currents, h=H_AF):
    u = np.asarray(u, float)
    u /= np.linalg.norm(u) + 1e-12
    v_plus = potential_at(r + h*u, currents)
    v0 = potential_at(r, currents)
    v_minus = potential_at(r - h*u, currents)
    return (v_plus - 2*v0 + v_minus) / (h**2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Define fibers
def make_fiber_points():
    rs = [0.5*R, 0.6*R, 0.6*R, 0.9*R, 0.3*R]
    thetas = [0, np.pi/2, np.pi, np.pi/4, 3*np.pi/4]
    zs = [0.0, 0.0, 0.0, 0.0, 0.0]
    pts = np.c_[np.array(rs)*np.cos(thetas),
                np.array(rs)*np.sin(thetas),
                np.array(zs)]
    u = np.array([0, 0, 1])  # fiber direction along z
    idx_map = {20000+i: pts[i] for i in range(len(pts))}
    target_idx = 20000
    off_indices = [k for k in idx_map.keys() if k != target_idx]
    return idx_map, target_idx, off_indices, u


FIBER_IDX_MAP, DEFAULT_TARGET_IDX, DEFAULT_OFF_IDXS, FIBER_DIR = make_fiber_points()


def selectivity_reward(currents, target_idx, off_indices=DEFAULT_OFF_IDXS, alpha=ALPHA, beta=BETA):
    r_t = FIBER_IDX_MAP[target_idx]
    AF_t = activation_function(r_t, FIBER_DIR, currents)
    AF_off = [activation_function(FIBER_IDX_MAP[k], FIBER_DIR, currents) for k in off_indices]
    AF_off = np.array(AF_off)
    return sigmoid(alpha*AF_t) - beta*np.mean(sigmoid(alpha*AF_off))


# Drop-in simulate_selectivity (for all optimizers)
def simulate_selectivity(param_dict, target_idx):
    names = [f"I_d{i}" for i in range(1, 13)]
    currents = np.array([float(param_dict.get(n, 0.0)) for n in names])
    return float(selectivity_reward(currents, target_idx))


# ---------------------------------------------------------------------
# Main optimization runs
# ---------------------------------------------------------------------
def main():
    out_dir = Path("data") / "RPNI_sim_analytic"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_ranges = {f"I_d{i}": (-1.0, 1.0) for i in range(1, 13)}
    target_indices = [DEFAULT_TARGET_IDX]

    # ---------------- Parameter grids (identical to your COMSOL script)
    gr_iters = [5, 10, 20, 40]
    gr_candidates = [40, 20, 10, 5]

    bo_initial = [10, 20, 40]
    kappa_values = [0.5, 1.5, 2.5]

    sigma_values = [0.4, 0.8, 1.2]
    popsize_values = [5, 10, 20]
    cma_iters_values = [40, 20, 10]

    de_popsize_vals = [5, 10, 20]
    de_mutation_vals = [(0.3, 0.9), (0.5, 1.0)]
    de_rec_vals = [0.5, 0.7, 0.9]
    de_maxiter_vals = [10, 20, 40]

    gd_iters = [200]
    lr_values = [0.01, 0.1, 0.5]
    eps_values = [1e-1, 1e-2, 1e-3]

    tpe_trials_vals = [200]
    rf_calls_vals = [200]

    cem_popsize_vals = [20, 50, 100]
    cem_elite_frac_vals = [0.1, 0.2, 0.3]
    cem_iters_vals = [4, 10, 20]
    cem_smooth_vals = [0.7, 0.9]

    powell_restarts_vals = [5, 10, 20]

    bandit_res_vals = [3, 5, 7]
    bandit_pulls_vals = [200]
    bandit_c_vals = [0.5, 1.0, 2.0]

    pso_popsize_vals = [20]
    pso_w_vals = [0.5]
    pso_c1_vals = [1.5]
    pso_c2_vals = [1.5]
    pso_iters_vals = [10]

    # ---------------------------------------------------------------
    for tidx in target_indices:
        # --- BO (LCB)
        # for bo_init in bo_initial:
        #     for kappa in kappa_values:
        #         bo_csv = out_dir / f"bo_i{bo_init}_kappa{kappa}_t{tidx}.csv"
        #         bo_opt = BOSimulation(
        #             current_ranges=current_ranges,
        #             target_idx=tidx,
        #             simulate_fn=simulate_selectivity,
        #             out_csv=bo_csv,
        #             n_iters=200 - bo_init,
        #             candidates_per_iter=gr_candidates,
        #             n_initial_points=bo_init,
        #             acq_func="LCB",
        #             random_state=42,
        #             kappa=kappa,
        #         )
        #         bo_opt.optimize()

        # --- CMA-ES
        for sigma in sigma_values:
            for popsize in popsize_values:
                for cma_iters in cma_iters_values:
                    cma_csv = out_dir / f"cmaes_t{tidx}_sig{sigma}_pop{popsize}_gen{cma_iters}.csv"
                    cma_opt = CMAESOptimizer(
                        current_ranges=current_ranges,
                        target_idx=tidx,
                        simulate_fn=simulate_selectivity,
                        out_csv=cma_csv,
                        sigma=sigma,
                        popsize=popsize,
                        n_iters=cma_iters,
                    )
                    cma_opt.optimize()

        # --- DE
        for pop in de_popsize_vals:
            for mut in de_mutation_vals:
                for rec in de_rec_vals:
                    for gens in de_maxiter_vals:
                        de_csv = out_dir / (
                            f"de_t{tidx}_pop{pop}_mut{mut[0]}-{mut[1]}"
                            f"_rec{rec}_gen{gens}.csv"
                        )
                        de_opt = DESimulation(
                            current_ranges=current_ranges,
                            target_idx=tidx,
                            simulate_fn=simulate_selectivity,
                            out_csv=de_csv,
                            popsize=pop,
                            mutation=mut,
                            recombination=rec,
                            maxiter=gens,
                        )
                        de_opt.optimize()

        # --- Gradient Descent
        for lr in lr_values:
            for eps in eps_values:
                gd_csv = out_dir / f"gd_t{tidx}_lr_{lr}_eps_{eps}.csv"
                gd_opt = GradientDescentSimulator(
                    current_ranges=current_ranges,
                    target_idx=tidx,
                    simulate_fn=simulate_selectivity,
                    out_csv=gd_csv,
                    learning_rate=lr,
                    n_iters=gd_iters,
                    eps=eps,
                )
                gd_opt.optimize()

        # --- TPE
        for trials in tpe_trials_vals:
            tpe_csv = out_dir / f"tpe_{trials}_target_{tidx}.csv"
            tpe_opt = TPESimulation(
                current_ranges=current_ranges,
                target_idx=tidx,
                simulate_fn=simulate_selectivity,
                out_csv=tpe_csv,
                n_trials=trials,
                random_seed=42,
            )
            tpe_opt.optimize()

        # --- RF-BO
        for calls in rf_calls_vals:
            rf_csv = out_dir / f"rfbo_{calls}_target_{tidx}.csv"
            rf_opt = RandomForestBOSimulation(
                current_ranges=current_ranges,
                target_idx=tidx,
                simulate_fn=simulate_selectivity,
                out_csv=rf_csv,
                n_calls=calls,
                random_seed=42,
            )
            rf_opt.optimize()

        # --- CEM
        for pop in cem_popsize_vals:
            for elite in cem_elite_frac_vals:
                for iters in cem_iters_vals:
                    for smooth in cem_smooth_vals:
                        cem_csv = out_dir / f"cem_pop{pop}_elite{elite}_it{iters}_s{smooth}_t{tidx}.csv"
                        cem_opt = CEMOptimizer(
                            current_ranges=current_ranges,
                            target_idx=tidx,
                            simulate_fn=simulate_selectivity,
                            out_csv=cem_csv,
                            popsize=pop,
                            elite_frac=elite,
                            n_iters=iters,
                            smoothing=smooth,
                            random_seed=42,
                        )
                        cem_opt.optimize()

        # --- Powell
        for rest in powell_restarts_vals:
            pow_csv = out_dir / f"powell_{rest}restarts_t{tidx}.csv"
            pow_opt = PowellLocalOptimizer(
                current_ranges=current_ranges,
                target_idx=tidx,
                simulate_fn=simulate_selectivity,
                out_csv=pow_csv,
                n_restarts=rest,
                random_seed=42,
            )
            pow_opt.optimize()

        # --- Bandit (UCB)
        for res in bandit_res_vals:
            for pulls in bandit_pulls_vals:
                for c in bandit_c_vals:
                    ban_csv = out_dir / f"bandit_res{res}_pulls{pulls}_c{c}_t{tidx}.csv"
                    ban_opt = BanditOptimizer(
                        current_ranges=current_ranges,
                        target_idx=tidx,
                        simulate_fn=simulate_selectivity,
                        out_csv=ban_csv,
                        resolution=res,
                        total_pulls=pulls,
                        c=c,
                    )
                    ban_opt.optimize()

        # --- PSO
        for popsize in pso_popsize_vals:
            for w in pso_w_vals:
                for c1 in pso_c1_vals:
                    for c2 in pso_c2_vals:
                        for nit in pso_iters_vals:
                            pso_csv = out_dir / f"pso_pop{popsize}_w{w}_c1{c1}_c2{c2}_it{nit}_t{tidx}.csv"
                            pso_opt = PSOSimulation(
                                current_ranges=current_ranges,
                                target_idx=tidx,
                                simulate_fn=simulate_selectivity,
                                out_csv=pso_csv,
                                popsize=popsize,
                                w=w,
                                c1=c1,
                                c2=c2,
                                n_iters=nit,
                                random_seed=42,
                            )
                            pso_opt.optimize()

    print("All analytic optimization runs complete.")


if __name__ == "__main__":
    main()
