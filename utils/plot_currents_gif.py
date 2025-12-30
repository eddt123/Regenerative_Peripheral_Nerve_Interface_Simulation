#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hard-coded GIF generator for MS_SHAPED_CMA N=12 run.

- Reads the CSV (with inconsistent column counts due to eval_index in stage 1)
  using the csv module (not pandas.read_csv).
- Constructs a clean DataFrame.
- Animates:
    * Left: |E| slice at the target z-plane inside the cylinder
    * Right: 12 electrodes (4x3 grid) coloured by current
    * Target point marked
    * Text box showing evals_so_far and best_so_far

Run from repo root:
    (phd) python utils/plot_currents_gif.py
"""

import csv
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# =========================================================
# HARD-CODED CSV PATH  🔴 EDIT THIS IF NEEDED
# =========================================================
CSV_PATH = Path(r"C:\Users\eddyt\Desktop\GitHub\Regenerative_Peripheral_Nerve_Interface_Simulation\data\benchmark_adaptive_increase\MS_SWEEP_PROG_CMA_N12_grid4x3_r1_x5.0_y0.0_z0.0.csv")

# e.g. if it lives in data/:
# CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "MS_SHAPED_CMA_N12_grid4x3_r0_x-3.0_y-2.0_z-15.0.csv"

OUT_GIF = CSV_PATH.with_suffix(".gif")


# =========================================================
# CSV LOADER (handles 14- and 15-column rows)
# =========================================================

def load_ms_shaped_csv(csv_path: Path) -> pd.DataFrame:
    """
    Custom CSV parser for MS_SHAPED_CMA logs.

    There are two row formats:
      - Stage 0 rows: 14 columns (no eval_index)
      - Stage 1 rows: 15 columns (extra eval_index before evals_so_far)

    This function normalises both into a DataFrame with columns:
      optimizer, stage, n_rows, n_per_row, N, repeat,
      eval_index, evals_so_far, step_best_so_far, best_so_far,
      best_found_at_eval, target_x, target_y, target_z, currents
    """
    rows = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        _header = next(reader, None)  # we ignore the original header

        for row in reader:
            if not row:
                continue

            if len(row) == 14:
                # No eval_index in these rows (stage 0)
                (
                    optimizer, stage, n_rows, n_per_row, N, repeat,
                    evals_so_far, step_best, best_so_far, best_found_at_eval,
                    target_x, target_y, target_z, currents_str
                ) = row
                eval_index = ""

            elif len(row) == 15:
                # With eval_index in these rows (stage 1)
                (
                    optimizer, stage, n_rows, n_per_row, N, repeat,
                    eval_index, evals_so_far, step_best, best_so_far,
                    best_found_at_eval, target_x, target_y, target_z,
                    currents_str
                ) = row

            else:
                # Unexpected row shape – just skip
                print(f"[WARN] Skipping row with unexpected length {len(row)}")
                continue

            rows.append(
                {
                    "optimizer": optimizer,
                    "stage": int(stage),
                    "n_rows": int(n_rows),
                    "n_per_row": int(n_per_row),
                    "N": int(N),
                    "repeat": int(repeat),
                    "eval_index": (
                        int(eval_index) if eval_index not in ("", None) else np.nan
                    ),
                    "evals_so_far": int(float(evals_so_far)),
                    "step_best_so_far": float(step_best),
                    "best_so_far": float(best_so_far),
                    "best_found_at_eval": int(best_found_at_eval),
                    "target_x": float(target_x),
                    "target_y": float(target_y),
                    "target_z": float(target_z),
                    "currents": currents_str,
                }
            )

    df = pd.DataFrame(rows)
    return df


# =========================================================
# Geometry & field helpers
# =========================================================

def layout_electrodes(
    n_rows: int = 4,
    n_per_row: int = 3,
    radius: float = 0.01,
    height: float = 0.04,
    rotation_offset: float = 0.0,
) -> np.ndarray:
    """
    4×3 grid: 4 rows in z, 3 electrodes per ring around the cylinder.
    Returns an array of shape (N, 3) with (x, y, z) positions.
    """
    z_vals = np.linspace(-height / 2.0, height / 2.0, n_rows)
    thetas = np.linspace(0, 2 * np.pi, n_per_row, endpoint=False) + rotation_offset

    positions = []
    for z in z_vals:
        for th in thetas:
            x = radius * np.cos(th)
            y = radius * np.sin(th)
            positions.append([x, y, z])
    return np.array(positions, dtype=float)


def electric_field_grid(
    points: np.ndarray,
    sources_pos: np.ndarray,
    currents: np.ndarray,
    sigma: float = 0.25,
    electrode_radius: float = 0.001,
) -> np.ndarray:
    """
    Vectorised computation of E-field at many points.

    points:      (P, 3)
    sources_pos: (N, 3)
    currents:    (N,)
    Returns E:   (P, 3)
    """
    points = np.asarray(points, dtype=float)
    sources_pos = np.asarray(sources_pos, dtype=float)
    currents = np.asarray(currents, dtype=float).reshape(-1)

    R = points[:, None, :] - sources_pos[None, :, :]  # (P,N,3)
    d2 = np.sum(R * R, axis=-1)                       # (P,N)
    a2 = electrode_radius ** 2
    d32 = (d2 + a2) ** 1.5                             # (P,N)

    coeff = currents / (4 * np.pi * sigma)            # (N,)
    E = (coeff[None, :, None] * R / d32[:, :, None]).sum(axis=1)  # (P,3)
    return E


# =========================================================
# GIF MAKER
# =========================================================

def make_cylinder_gif():
    # ------------------------------
    # Load & normalise CSV
    # ------------------------------
    df = load_ms_shaped_csv(CSV_PATH)

    # Sort by evals_so_far so the animation is in chronological order
    df = df.sort_values("evals_so_far").reset_index(drop=True)

    # Target point (same for all rows)
    tx = df["target_x"].iloc[0]
    ty = df["target_y"].iloc[0]
    tz = df["target_z"].iloc[0]

    # Parse currents: string -> np.array
    def parse_currents(s):
        return np.asarray(ast.literal_eval(str(s)), dtype=float)

    df["curr_array"] = df["currents"].apply(parse_currents)

    N = len(df["curr_array"].iloc[0])
    print(f"N electrodes detected: {N}")

    # ------------------------------
    # Electrode geometry & slice grid
    # ------------------------------
    n_rows = 4
    n_per_row = 3
    radius = 0.01
    height = 0.04
    grid_res = 64

    sources_pos = layout_electrodes(
        n_rows=n_rows,
        n_per_row=n_per_row,
        radius=radius,
        height=height,
        rotation_offset=0.0,
    )

    xs = np.linspace(-radius, radius, grid_res)
    ys = np.linspace(-radius, radius, grid_res)
    X, Y = np.meshgrid(xs, ys)
    slice_points = np.stack(
        [X.ravel(), Y.ravel(), np.full_like(X.ravel(), tz)], axis=1
    )

    # ------------------------------
    # Precompute E-field frames
    # ------------------------------
    field_frames = []
    all_currents = []
    max_Emag = 0.0

    for _, row in df.iterrows():
        currents = row["curr_array"]
        all_currents.append(currents)

        E = electric_field_grid(slice_points, sources_pos, currents)
        Emag = np.linalg.norm(E, axis=1).reshape(grid_res, grid_res)
        field_frames.append(Emag)
        max_Emag = max(max_Emag, float(Emag.max()))

    all_currents = np.stack(all_currents, axis=0)  # (T, N)
    max_I = float(np.max(np.abs(all_currents))) if np.any(all_currents) else 1.0

    # ------------------------------
    # Set up figure
    # ------------------------------
    fig = plt.figure(figsize=(10, 5))

    ax_field = fig.add_subplot(1, 2, 1)
    ax_geom = fig.add_subplot(1, 2, 2, projection="3d")

    # Field panel
    im = ax_field.imshow(
        field_frames[0],
        extent=[-radius, radius, -radius, radius],
        origin="lower",
        vmin=0.0,
        vmax=max_Emag,
        aspect="equal",
    )

    circ = plt.Circle((0, 0), radius, edgecolor="k", fill=False, lw=1.0)
    ax_field.add_patch(circ)
    ax_field.scatter([tx], [ty], marker="x", s=60, color="k", label="Target")
    ax_field.set_xlabel("x [m]")
    ax_field.set_ylabel("y [m]")
    ax_field.set_title("|E| at target z-slice")

    info_text = ax_field.text(
        0.02,
        0.98,
        "",
        transform=ax_field.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    cbar_E = fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04)
    cbar_E.set_label("|E| [V/m]")

    # Geometry panel
    sc = ax_geom.scatter(
        sources_pos[:, 0],
        sources_pos[:, 1],
        sources_pos[:, 2],
        c=all_currents[0],
        cmap="coolwarm",
        vmin=-max_I,
        vmax=max_I,
        s=60,
    )
    ax_geom.scatter([tx], [ty], [tz], marker="*", color="k", s=100, label="Target")

    ax_geom.set_xlabel("x [m]")
    ax_geom.set_ylabel("y [m]")
    ax_geom.set_zlabel("z [m]")
    ax_geom.set_title("Electrode currents on cylinder (4×3)")

    lim_xy = radius * 1.2
    ax_geom.set_xlim(-lim_xy, lim_xy)
    ax_geom.set_ylim(-lim_xy, lim_xy)
    ax_geom.set_zlim(-height / 2 * 1.2, height / 2 * 1.2)

    cbar_I = fig.colorbar(sc, ax=ax_geom, fraction=0.046, pad=0.04)
    cbar_I.set_label("Current [A]")

    # ------------------------------
    # Build frames
    # ------------------------------
    frames = []

    for i, (_, row) in enumerate(df.iterrows()):
        Emag = field_frames[i]
        currents = all_currents[i]

        # Update field
        im.set_data(Emag)

        # Update current colours
        sc.set_array(currents)

        # Text info
        evals_val = int(row["evals_so_far"])
        best_val = float(row["best_so_far"])
        info_text.set_text(f"evals: {evals_val}\nbest_so_far: {best_val:.3f}")

        # Render to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)

    # ------------------------------
    # Save GIF
    # ------------------------------
    imageio.mimsave(OUT_GIF, frames, duration=1.0 / 10.0)  # fps = 10
    plt.close(fig)

    print(f"Saved GIF → {OUT_GIF}")


# =========================================================
# Run immediately
# =========================================================

if __name__ == "__main__":
    make_cylinder_gif()
