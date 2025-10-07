from __future__ import annotations
import csv
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np


class GreedyRandomOptimizer:
    """
    Greedy-random search over electrode currents to maximize selectivity
    for a given spatial target index.
    """
    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        n_iters: int = 50,
        candidates_per_iter: int = 10
    ):
        self.current_ranges = current_ranges
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.n_iters = n_iters
        self.candidates_per_iter = candidates_per_iter

        # Prepare output CSV
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, mode="w", newline="") as fh:
                writer = csv.writer(fh)
                header = list(current_ranges.keys()) + ["target_idx", "selectivity"]
                writer.writerow(header)

    def _sample_random(self) -> Dict[str, float]:
        """Sample a random electrode current vector within the specified ranges."""
        return {
            name: float(np.random.uniform(low, high))
            for name, (low, high) in self.current_ranges.items()
        }

    def _evaluate_and_record(self, currents: Dict[str, float]) -> float:
        """
        Evaluate selectivity for the given currents, append the result to the CSV,
        and print the currents with their selectivity.
        Returns the selectivity score.
        """
        score = self.simulate_fn(currents, self.target_idx)

        # Record row: currents in order, then target_idx and score
        row = [currents[name] for name in self.current_ranges] + [self.target_idx, score]
        with open(self.out_csv, mode="a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(row)

        # Print immediate feedback
        curr_str = ", ".join(f"{k}={v:.2f}" for k, v in currents.items())
        print(f"Evaluated currents: {curr_str} | target={self.target_idx} | selectivity={score:.4f}")

        return score

    def optimize(self) -> tuple[Dict[str, float], float]:
        """
        Perform greedy-random optimization.

        Returns:
            best_currents: Dict[str, float] -- electrode currents achieving highest selectivity.
            best_score: float -- highest selectivity found.
        """
        # Initialize with one random sample
        best_currents = self._sample_random()
        best_score = self._evaluate_and_record(best_currents)
        print(f"Initial best selectivity={best_score:.4f}")

        # Iterative greedy-random search
        for it in range(1, self.n_iters + 1):
            print(f"\n--- Iteration {it}/{self.n_iters} ---")
            for c_idx in range(1, self.candidates_per_iter + 1):
                candidate = self._sample_random()
                score = self._evaluate_and_record(candidate)
                if score > best_score:
                    best_score = score
                    best_currents = candidate
                    print(f"  New overall best! selectivity={best_score:.4f}")
            print(f"End of iteration {it}, best selectivity so far={best_score:.4f}")

        print(f"\nOptimization complete: best selectivity={best_score:.4f}, currents={best_currents}")
        return best_currents, best_score
