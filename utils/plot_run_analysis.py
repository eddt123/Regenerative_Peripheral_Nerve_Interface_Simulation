#!/usr/bin/env python3
"""
Visualise a single optimiser run CSV.

Plots:
  1. Electrode currents (best-so-far) vs evaluation count
  2. Final electric-field magnitude in the xy-plane at z = target_z,
     with the cylinder boundary, electrode positions, and target point overlaid

Hardcoded input:
  data/benchmark_paper/PSO_GROUNDED_N25_grid5x5_r2_x5.0_y0.0_z0.0.csv
  target point (x5.0 → 0.005 m, y0.0 → 0.000 m, z0.0 → 0.000 m)
"""

import ast
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.colorbar import ColorbarBase

# ── paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data", "benchmark_paper",
    "PSO_GROUNDED_N25_grid5x5_r2_x5.0_y0.0_z0.0.csv",
)
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark_paper")

# ── geometry (must match run_selectivity_simulation defaults) ─────────────────
RADIUS   = 0.01    # cuff radius, m
HEIGHT   = 0.04    # axial span, m
SIGMA    = 0.25    # conductivity, S/m
ELEC_R   = 0.001   # electrode radius, m

# ── add utils to path so we can import the sim helpers ───────────────────────
sys.path.insert(0, _HERE)
from run_selectivity_simulation import layout_electrodes, electric_field_at_point


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_currents(cell):
    """Parse a stringified Python list of floats from a CSV cell."""
    if isinstance(cell, list):
        return np.asarray(cell, dtype=float)
    return np.asarray(ast.literal_eval(str(cell)), dtype=float)


def load_run(csv_path):
    df = pd.read_csv(csv_path)
    # keep one row per step (step_index ascending)
    df = df.sort_values("step_index").reset_index(drop=True)
    return df


def compute_field_grid(currents, sources_pos, z_slice=0.0,
                       res=120, pad=1.2):
    """
    Return (xx, yy, E_mag) for a 2-D grid in the xy-plane at z = z_slice.
    Grid spans ±RADIUS*pad.
    """
    lim = RADIUS * pad
    xs = np.linspace(-lim, lim, res)
    ys = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(xs, ys)

    E_mag = np.zeros_like(xx)
    for i in range(res):
        for j in range(res):
            r = np.array([xx[i, j], yy[i, j], z_slice])
            E = electric_field_at_point(r, sources_pos, currents, SIGMA, ELEC_R)
            E_mag[i, j] = np.linalg.norm(E)

    return xx, yy, E_mag


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── load data ────────────────────────────────────────────────────────────
    df = load_run(CSV_PATH)

    n_rows    = int(df["n_rows"].iloc[0])
    n_per_row = int(df["n_per_row"].iloc[0])
    N         = n_rows * n_per_row
    target_x  = float(df["target_x"].iloc[0])
    target_y  = float(df["target_y"].iloc[0])
    target_z  = float(df["target_z"].iloc[0])
    target_pt = np.array([target_x, target_y, target_z])
    optimizer = str(df["optimizer"].iloc[0])
    repeat    = int(df["repeat"].iloc[0])

    evals     = df["evals_so_far"].values
    sel_best  = df["best_so_far"].values

    # parse best-so-far current vectors for every logged step
    currents_history = np.stack([parse_currents(c) for c in df["currents_best_so_far"]])
    # shape: (n_steps, N)

    # final (best-ever) currents
    best_row  = df.loc[df["best_so_far"].idxmax()]
    currents_final = parse_currents(best_row["currents_best_so_far"])

    sources_pos = layout_electrodes(n_rows, n_per_row, RADIUS, HEIGHT,
                                    -HEIGHT/2, HEIGHT/2)

    print(f"Loaded {len(df)} steps  |  N={N}  |  "
          f"best selectivity = {sel_best.max():.4f}  |  "
          f"target = ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")

    # ── Figure 1: currents over evaluations ──────────────────────────────────
    fig1, axes = plt.subplots(2, 1, figsize=(12, 8),
                              gridspec_kw={"height_ratios": [3, 1]})
    fig1.suptitle(
        f"{optimizer}  –  grid {n_rows}×{n_per_row} (N={N})  –  repeat {repeat}\n"
        f"target ({target_x*1e3:.1f}, {target_y*1e3:.1f}, {target_z*1e3:.1f}) mm",
        fontsize=11
    )

    ax_cur = axes[0]
    ax_sel = axes[1]

    # colour each electrode by its row (ring)
    cmap_elec = plt.get_cmap("tab10")
    row_colors = [cmap_elec(r / max(n_rows - 1, 1)) for r in range(n_rows)]

    # one thin line per electrode, coloured by ring
    for elec_idx in range(N):
        row = elec_idx // n_per_row
        col_in_row = elec_idx % n_per_row
        ax_cur.plot(
            evals,
            currents_history[:, elec_idx] * 1e3,   # → mA
            color=row_colors[row],
            alpha=0.55,
            linewidth=0.9,
            label=f"Row {row}" if col_in_row == 0 else "_",
        )

    ax_cur.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax_cur.set_ylabel("Current (mA)")
    ax_cur.set_title("Best-so-far electrode currents vs evaluations")
    ax_cur.legend(title="Ring (row)", loc="upper right", fontsize=7,
                  framealpha=0.7, ncol=min(n_rows, 5))
    ax_cur.set_xlim(evals[0], evals[-1])

    # selectivity convergence
    ax_sel.plot(evals, sel_best, color="steelblue", linewidth=1.5)
    ax_sel.set_xlabel("Evaluations")
    ax_sel.set_ylabel("Selectivity")
    ax_sel.set_title("Best selectivity vs evaluations")
    ax_sel.set_xlim(evals[0], evals[-1])

    fig1.tight_layout()
    out1 = os.path.join(
        OUT_DIR,
        f"{optimizer}_N{N}_r{repeat}_currents_over_evals.png"
    )
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved → {out1}")
    plt.close(fig1)

    # ── Figure 2: electric field in xy-plane ─────────────────────────────────
    print("Computing field grid (may take ~10-20 s) …")
    xx, yy, E_mag = compute_field_grid(currents_final, sources_pos,
                                       z_slice=target_z, res=140)

    fig2, ax = plt.subplots(figsize=(7, 6))

    # heatmap
    vmax = np.percentile(E_mag, 98)   # clip outliers near electrodes
    im = ax.pcolormesh(
        xx * 1e3, yy * 1e3, E_mag,
        cmap="inferno",
        norm=mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax),
        shading="auto",
        rasterized=True,
    )
    cb = fig2.colorbar(im, ax=ax, label="|E| (V/m)", pad=0.02)

    # cylinder boundary
    theta_c = np.linspace(0, 2*np.pi, 300)
    ax.plot(
        np.cos(theta_c) * RADIUS * 1e3,
        np.sin(theta_c) * RADIUS * 1e3,
        "w--", linewidth=1.2, label="Cuff boundary",
    )

    # electrode positions (cross-section ring; colour by current polarity)
    norm_cur = mcolors.TwoSlopeNorm(
        vmin=-1e-3, vcenter=0.0, vmax=1e-3
    )
    cmap_cur = plt.get_cmap("RdBu_r")

    # for 3-D cuff, plot each electrode at its (x,y) position at the z-row
    # that is closest to target_z – or just project all rings onto the xy plane
    for k, (pos, cur) in enumerate(zip(sources_pos, currents_final)):
        sc = ax.scatter(
            pos[0] * 1e3, pos[1] * 1e3,
            s=90,
            c=[cur],
            cmap=cmap_cur,
            norm=norm_cur,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        ax.text(
            pos[0] * 1e3 * 1.35, pos[1] * 1e3 * 1.35,
            str(k), fontsize=5, ha="center", va="center",
            color="white", zorder=6,
        )

    # add a discrete colorbar for electrode currents
    sm = plt.cm.ScalarMappable(cmap=cmap_cur, norm=norm_cur)
    sm.set_array([])
    cb2 = fig2.colorbar(sm, ax=ax, fraction=0.03, pad=0.12,
                        label="Electrode current (A)")

    # target point
    ax.scatter(
        target_x * 1e3, target_y * 1e3,
        marker="*", s=220, color="lime",
        edgecolors="black", linewidths=0.8,
        zorder=10, label=f"Target ({target_x*1e3:.1f}, {target_y*1e3:.1f}, {target_z*1e3:.1f}) mm",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(
        f"Final |E| in xy-plane at z = {target_z*1e3:.1f} mm\n"
        f"{optimizer}  N={N}  repeat {repeat}  "
        f"selectivity = {sel_best.max():.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="lower left", framealpha=0.7)

    fig2.tight_layout()
    out2 = os.path.join(
        OUT_DIR,
        f"{optimizer}_N{N}_r{repeat}_field_xy.png"
    )
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
    plt.close(fig2)

    print("Done.")


if __name__ == "__main__":
    main()
