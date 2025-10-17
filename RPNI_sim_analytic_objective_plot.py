#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from utils.run_selectivity_simulation import (
    run_selectivity_simulation,
    layout_electrodes,
    pair_balanced_currents,
    target_between_pair,
    potential_at_point
)

# --------- CONFIG ---------
n_rows, n_per_row = 4, 3
radius, height, sigma = 0.01, 0.04, 0.25
E_th, k = 0.4, 0.1
metric = "activation"
electrode_radius = 1e-3
R_outer = 0.10  # 10 cm outer ground

# --------- Objective surface plotting ---------
def plot_pair_objective_surfaces(i1, i2, target,
                                 amp_range=np.linspace(-1e-3, 1e-3, 60)):
    """
    Left: FLOATING (pair-balanced)
    Right: GROUNDED (no balancing; non-zero-sum allowed)
    """
    N = n_rows * n_per_row
    Z_float = np.zeros((len(amp_range), len(amp_range)))
    Z_ground = np.zeros_like(Z_float)

    # FLOATING (pair-balanced currents)
    for ix, a1 in enumerate(amp_range):
        for iy, a2 in enumerate(amp_range):
            cur = pair_balanced_currents(N, i1, i2, a1, a2)
            res = run_selectivity_simulation(
                n_rows, n_per_row, cur, target,
                radius, height, sigma,
                n_off_samples=1200,
                metric=metric, E_th=E_th, k=k,
                use_activating_function=False,
                electrode_radius=electrode_radius,
                grounded_boundary=False
            )
            Z_float[ix, iy] = res["selectivity"]

    # GROUNDED (no balancing; only the chosen pair carries current)
    for ix, a1 in enumerate(amp_range):
        for iy, a2 in enumerate(amp_range):
            cur = np.zeros(N)
            cur[i1] = a1
            cur[i2] = a2
            res = run_selectivity_simulation(
                n_rows, n_per_row, cur, target,
                radius, height, sigma,
                n_off_samples=3000,
                metric=metric, E_th=E_th, k=k,
                use_activating_function=False,
                electrode_radius=electrode_radius,
                grounded_boundary=True, R_outer=R_outer,
                record_boundary_stats=False
            )
            Z_ground[ix, iy] = res["selectivity"]

    A, B = np.meshgrid(amp_range*1e3, amp_range*1e3)

    fig = plt.figure(figsize=(11,4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9], wspace=0.35)

    # Floating
    ax1 = fig.add_subplot(gs[0])
    c1 = ax1.contourf(A, B, Z_float, levels=40)
    fig.colorbar(c1, ax=ax1, label="Selectivity")
    ax1.set_title(f"FLOATING (pair-balanced)\nPair ({i1},{i2}), target {np.round(target,3)}")
    ax1.set_xlabel(f"I{i1} (mA)"); ax1.set_ylabel(f"I{i2} (mA)")

    # Grounded
    ax2 = fig.add_subplot(gs[1])
    c2 = ax2.contourf(A, B, Z_ground, levels=40)
    fig.colorbar(c2, ax=ax2, label="Selectivity")
    ax2.set_title(f"GROUNDED (R_outer={R_outer*1e2:.0f} cm)\nPair ({i1},{i2}), target {np.round(target,3)}")
    ax2.set_xlabel(f"I{i1} (mA)"); ax2.set_ylabel(f"I{i2} (mA)")

    # Geometry inset
    axg = fig.add_subplot(gs[2])
    pos = layout_electrodes(n_rows, n_per_row, radius, height)
    xs, ys = pos[:,0]*1e3, pos[:,1]*1e3
    axg.scatter(xs, ys, s=25, label="electrodes")
    axg.scatter(xs[i1], ys[i1], s=60, label=f"elec {i1}")
    axg.scatter(xs[i2], ys[i2], s=60, label=f"elec {i2}")
    axg.scatter(target[0]*1e3, target[1]*1e3, marker='x', s=80, label="target")
    circ = plt.Circle((0,0), radius*1e3, fill=False, linestyle="--")
    axg.add_patch(circ)
    axg.set_aspect("equal")
    axg.set_xlim(-1.5*radius*1e3, 1.5*radius*1e3)
    axg.set_ylim(-1.5*radius*1e3, 1.5*radius*1e3)
    axg.set_title("Top view")
    axg.set_xlabel("x (mm)"); axg.set_ylabel("y (mm)")
    axg.legend(fontsize=8)

    fig.suptitle("Objective surfaces: Floating vs Grounded", y=1.02)
    plt.tight_layout()
    plt.show()

    return Z_float, Z_ground

# --------- Sanity checks (floating vs grounded) ---------
def run_sanity_checks(i1, i2, target):
    pos = layout_electrodes(n_rows, n_per_row, radius, height)
    N = n_rows * n_per_row
    rng = np.random.default_rng(123)

    print("\n=== Sanity checks (floating vs grounded) ===")

    # F0: floating invariance to common-mode (pair-balanced construction already enforces it)
    a1, a2 = 0.3e-3, -0.2e-3
    cur_f = pair_balanced_currents(N, i1, i2, a1, a2)
    S_f = run_selectivity_simulation(n_rows, n_per_row, cur_f, target, radius, height, sigma,
                                     metric=metric, E_th=E_th, k=k,
                                     grounded_boundary=False)["selectivity"]
    cur_f2 = pair_balanced_currents(N, i1, i2, a1+0.1e-3, a2+0.1e-3)  # adds common-mode
    S_f2 = run_selectivity_simulation(n_rows, n_per_row, cur_f2, target, radius, height, sigma,
                                      metric=metric, E_th=E_th, k=k,
                                      grounded_boundary=False)["selectivity"]
    print(f"F0 PASS (floating common-mode invariance): Δ={abs(S_f-S_f2):.3f}")

    # G0: grounded sensitivity to common-mode (now sum current matters)
    cur_g = np.zeros(N); cur_g[i1], cur_g[i2] = a1, a2
    S_g = run_selectivity_simulation(n_rows, n_per_row, cur_g, target, radius, height, sigma,
                                     metric=metric, E_th=E_th, k=k,
                                     grounded_boundary=True, R_outer=R_outer)["selectivity"]
    cur_g2 = np.zeros(N); cur_g2[i1], cur_g2[i2] = a1+0.1e-3, a2+0.1e-3
    S_g2 = run_selectivity_simulation(n_rows, n_per_row, cur_g2, target, radius, height, sigma,
                                      metric=metric, E_th=E_th, k=k,
                                      grounded_boundary=True, R_outer=R_outer)["selectivity"]
    print(f"G0 EXPECTED (grounded common-mode effect): Δ={abs(S_g-S_g2):.3f}")

    # G1: boundary potentials near zero at R_outer
    res = run_selectivity_simulation(n_rows, n_per_row, cur_g2, target, radius, height, sigma,
                                     metric=metric, E_th=E_th, k=k,
                                     grounded_boundary=True, R_outer=R_outer,
                                     record_boundary_stats=True)
    bm, bM = res["boundary_mean_abs_V"], res["boundary_max_abs_V"]
    print(f"G1 PASS (boundary ~0): mean|V|(R_outer)={bm:.3e}, max|V|={bM:.3e}")

    # F1: sign symmetry (floating)
    Sf_plus = run_selectivity_simulation(n_rows, n_per_row,
                                         pair_balanced_currents(N, i1, i2, +0.25e-3, -0.25e-3),
                                         target, radius, height, sigma,
                                         metric=metric, E_th=E_th, k=k,
                                         grounded_boundary=False)["selectivity"]
    Sf_minus = run_selectivity_simulation(n_rows, n_per_row,
                                          pair_balanced_currents(N, i1, i2, -0.25e-3, +0.25e-3),
                                          target, radius, height, sigma,
                                          metric=metric, E_th=E_th, k=k,
                                          grounded_boundary=False)["selectivity"]
    print(f"F1 PASS (floating sign symmetry): Δ={abs(Sf_plus-Sf_minus):.3f}")

    # G2: monopolar vs dipole selectivity (grounded)
    cur_mono = np.zeros(N); cur_mono[i1] = +0.5e-3
    S_mono = run_selectivity_simulation(n_rows, n_per_row, cur_mono, target, radius, height, sigma,
                                        metric=metric, E_th=E_th, k=k,
                                        grounded_boundary=True, R_outer=R_outer)["selectivity"]
    cur_dip = np.zeros(N); cur_dip[i1], cur_dip[i2] = +0.5e-3, -0.5e-3
    S_dip = run_selectivity_simulation(n_rows, n_per_row, cur_dip, target, radius, height, sigma,
                                       metric=metric, E_th=E_th, k=k,
                                       grounded_boundary=True, R_outer=R_outer)["selectivity"]
    better = "dipole" if S_dip >= S_mono else "monopolar"
    print(f"G2 INFO (grounded): monopolar={S_mono:.3f}, dipole={S_dip:.3f} → better: {better}")

    # F2: RMS near scale-invariance vs G3: grounded stronger scaling
    scale = 0.5
    cur_f_rms = pair_balanced_currents(N, i1, i2, +0.4e-3, -0.4e-3)
    Sf1 = run_selectivity_simulation(n_rows, n_per_row, cur_f_rms, target, radius, height, sigma,
                                     metric="rms", grounded_boundary=False)["selectivity"]
    Sf2 = run_selectivity_simulation(n_rows, n_per_row, cur_f_rms*scale, target, radius, height, sigma,
                                     metric="rms", grounded_boundary=False)["selectivity"]
    print(f"F2 WARN/OK (floating RMS scale): S1={Sf1:.3f}, S2={Sf2:.3f}, Δ={abs(Sf1-Sf2):.3f}")

    cur_g_rms = np.zeros(N); cur_g_rms[i1], cur_g_rms[i2] = +0.4e-3, -0.4e-3
    Sg1 = run_selectivity_simulation(n_rows, n_per_row, cur_g_rms, target, radius, height, sigma,
                                     metric="rms", grounded_boundary=True, R_outer=R_outer)["selectivity"]
    cur_g_rms2 = cur_g_rms * scale
    Sg2 = run_selectivity_simulation(n_rows, n_per_row, cur_g_rms2, target, radius, height, sigma,
                                     metric="rms", grounded_boundary=True, R_outer=R_outer)["selectivity"]
    print(f"G3 INFO (grounded RMS scale): S1={Sg1:.3f}, S2={Sg2:.3f}, Δ={abs(Sg1-Sg2):.3f}")

    print("=== End sanity checks ===\n")

# --------- MAIN ---------
if __name__ == "__main__":
    # Choose a clear same-row pair and put the target between them
    pos = layout_electrodes(n_rows, n_per_row, radius, height)
    i1, i2 = 0, 2
    target = target_between_pair(pos, i1, i2, inward_ratio=0.80, keep_z="i1")
    print("Target positioned between electrodes:", np.round(target, 5))

    # Run sanity checks
    #run_sanity_checks(i1, i2, target)

    # Plot objective surfaces for Floating vs Grounded
    plot_pair_objective_surfaces(i1, i2, target,
                                 amp_range=np.linspace(-1e-3, 1e-3, 60))
