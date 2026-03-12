#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hardcoded benchmark processing script (no CLI args).

- Ensures the benchmark data are available (extracts benchmark_paper.zip if needed)
- Reads each per-run CSV log (even if the file contains mixed row formats across stages)
- Extracts the final best selectivity per run (max "best_so_far")
- Aggregates across runs for each optimiser and dimension N
- Plots final best selectivity vs dimensions (mean ± 95% CI)

Run:
  python process_benchmark_plot_hardcoded.py

Outputs (in OUT_DIR):
  final_selectivity_vs_dimensions.png
  final_selectivity_vs_dimensions.pdf
  final_runs_extracted.csv
  final_summary_by_optimizer_and_N.csv
"""

import os
import glob
import math
import zipfile
import csv
import warnings
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Hardcoded paths (EDIT HERE)
# =========================
ZIP_PATH = "/mnt/data/benchmark_paper.zip"
EXTRACT_DIR = "/mnt/data/benchmark_paper"          # zip contains benchmark_paper/...
OUT_DIR = "/mnt/data/benchmark_paper_outputs"


FILENAME_RE = re.compile(
    r"^(?P<opt>.+?)_N(?P<N>\d+)_grid(?P<grid>\d+x\d+)_r(?P<repeat>\d+)"
    r"_x(?P<x>-?\d+(?:\.\d+)?)_y(?P<y>-?\d+(?:\.\d+)?)_z(?P<z>-?\d+(?:\.\d+)?)\.csv$"
)


def ensure_extracted(zip_path: str, extract_dir: str) -> None:
    """Extracts ZIP_PATH if EXTRACT_DIR doesn't already contain CSVs."""
    if os.path.isdir(extract_dir) and glob.glob(os.path.join(extract_dir, "*.csv")):
        return

    os.makedirs(extract_dir, exist_ok=True)

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP_PATH not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract into /mnt/data; zip itself contains 'benchmark_paper/' prefix.
        zf.extractall(os.path.dirname(extract_dir))


def parse_meta_from_filename(path: str) -> dict | None:
    base = os.path.basename(path)
    m = FILENAME_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    return {
        "run_file": base,
        "optimizer": d["opt"],
        "N": int(d["N"]),
        "grid": d["grid"],
        "repeat": int(d["repeat"]),
        "target_x": float(d["x"]),
        "target_y": float(d["y"]),
        "target_z": float(d["z"]),
    }


def _approx_equal(a: float, b: float, atol: float = 1e-9) -> bool:
    return abs(a - b) <= atol


def best_so_far_from_mixed_csv(path: str, tx: float, ty: float, tz: float) -> float:
    """
    Robustly extract max(best_so_far) from a run CSV, even if the file contains
    different row formats across stages (variable column counts).

    We locate the (target_x, target_y, target_z) triple and read best_so_far at j-2:
        ... step_best_selectivity, best_so_far, best_found_at_eval, target_x, target_y, target_z, ...
    """
    best = float("nan")

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # discard header
        if header is None:
            return best

        for row in reader:
            if not row or len(row) < 6:
                continue

            found = False
            for j in range(0, len(row) - 2):
                try:
                    x = float(row[j]); y = float(row[j + 1]); z = float(row[j + 2])
                except Exception:
                    continue

                if _approx_equal(x, tx) and _approx_equal(y, ty) and _approx_equal(z, tz):
                    if j >= 2:
                        try:
                            cand = float(row[j - 2])
                            if not math.isnan(cand):
                                best = cand if math.isnan(best) else max(best, cand)
                        except Exception:
                            pass
                    found = True
                    break

            if not found and header and "best_so_far" in header:
                idx = header.index("best_so_far")
                if idx < len(row):
                    try:
                        cand = float(row[idx])
                        best = cand if math.isnan(best) else max(best, cand)
                    except Exception:
                        pass

    return best


def summarize(df_runs: pd.DataFrame) -> pd.DataFrame:
    df = df_runs.copy()
    df = df[pd.to_numeric(df["best"], errors="coerce").notna()]
    df["best"] = pd.to_numeric(df["best"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce").astype(int)

    g = (
        df.groupby(["optimizer", "N"], as_index=False)
          .agg(
              n_runs=("best", "count"),
              mean_best=("best", "mean"),
              std_best=("best", "std"),
              median_best=("best", "median"),
              max_best=("best", "max"),
              min_best=("best", "min"),
          )
          .sort_values(["optimizer", "N"])
          .reset_index(drop=True)
    )

    ci = []
    for _, r in g.iterrows():
        n = int(r["n_runs"])
        sd = float(r["std_best"]) if not pd.isna(r["std_best"]) else 0.0
        ci_half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
        ci.append(ci_half)
    g["ci95_halfwidth"] = ci
    g["ci95_low"] = g["mean_best"] - g["ci95_halfwidth"]
    g["ci95_high"] = g["mean_best"] + g["ci95_halfwidth"]
    return g


def plot_final_vs_dimension(df_summary: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    plt.figure(figsize=(8.5, 5.0))
    for opt, grp in df_summary.groupby("optimizer"):
        grp = grp.sort_values("N")
        x = grp["N"].to_numpy()
        y = grp["mean_best"].to_numpy()
        ylo = grp["ci95_low"].to_numpy()
        yhi = grp["ci95_high"].to_numpy()

        plt.plot(x, y, marker="o", linewidth=2, label=opt)
        plt.fill_between(x, ylo, yhi, alpha=0.2)

    plt.xlabel("Dimensions (N)")
    plt.ylabel("Final best selectivity (mean ± 95% CI)")
    plt.title("Final selectivity vs dimensions (aggregated over targets/repeats)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.savefig(out_pdf)
    plt.close()


def main():
    ensure_extracted(ZIP_PATH, EXTRACT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.csv")))
    if not csvs:
        raise RuntimeError(f"No CSV files found in {EXTRACT_DIR}")

    rows = []
    for p in csvs:
        meta = parse_meta_from_filename(p)
        if meta is None:
            continue  # ignore non-run csvs like optimizer_summary.csv etc.

        best = best_so_far_from_mixed_csv(
            p,
            tx=meta["target_x"],
            ty=meta["target_y"],
            tz=meta["target_z"],
        )
        meta["best"] = best
        rows.append(meta)

    df_runs = pd.DataFrame(rows)
    df_runs = df_runs[pd.to_numeric(df_runs["best"], errors="coerce").notna()]

    runs_out = os.path.join(OUT_DIR, "final_runs_extracted.csv")
    df_runs.to_csv(runs_out, index=False)

    df_summary = summarize(df_runs)
    summary_out = os.path.join(OUT_DIR, "final_summary_by_optimizer_and_N.csv")
    df_summary.to_csv(summary_out, index=False)

    out_png = os.path.join(OUT_DIR, "final_selectivity_vs_dimensions.png")
    out_pdf = os.path.join(OUT_DIR, "final_selectivity_vs_dimensions.pdf")
    plot_final_vs_dimension(df_summary, out_png, out_pdf)

    print("\n=== Final best selectivity by optimiser and N ===")
    print(df_summary[["optimizer", "N", "n_runs", "mean_best", "std_best", "median_best"]].to_string(index=False))

    print("\nWrote:")
    print(f"  {runs_out}")
    print(f"  {summary_out}")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    main()
