from __future__ import annotations
from pathlib import Path
import csv
from typing import Callable, Dict, Tuple

class GradientDescentSimulator:
    """
    Simple gradient-descent optimizer for electrode currents to maximize selectivity.
    Uses finite-difference approximations of the gradient.
    """
    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        learning_rate: float = 0.1,
        n_iters: int = 50,
        eps: float = 1e-3
    ):
        self.current_ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.lr = learning_rate
        self.n_iters = n_iters
        self.eps = eps

        # Initialize parameters at midpoint of ranges
        self.params: Dict[str, float] = {
            name: (bounds[0] + bounds[1]) / 2.0
            for name, bounds in current_ranges.items()
        }

        # Prepare CSV
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, mode="w", newline="") as fh:
                writer = csv.writer(fh)
                header = self.param_names + ["target_idx", "selectivity"]
                writer.writerow(header)

    def _record(self, params: Dict[str, float], score: float) -> None:
        """Append a row to CSV."""
        row = [params[name] for name in self.param_names] + [self.target_idx, score]
        with open(self.out_csv, mode="a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(row)

    def optimize(self) -> tuple[Dict[str, float], float]:
        """
        Run gradient descent for n_iters iterations.
        Returns the best parameter set and its selectivity.
        """
        # Evaluate and record initial
        best_score = self.simulate_fn(self.params, self.target_idx)
        self._record(self.params, best_score)
        best_params = self.params.copy()

        for it in range(1, self.n_iters + 1):
            # Compute numerical gradient
            grads: Dict[str, float] = {}
            base_score = self.simulate_fn(self.params, self.target_idx)
            for name in self.param_names:
                # perturb
                orig = self.params[name]
                self.params[name] = orig + self.eps
                pos_score = self.simulate_fn(self.params, self.target_idx)
                # gradient approximation
                grads[name] = (pos_score - base_score) / self.eps
                # restore
                self.params[name] = orig

            # Update parameters
            for name in self.param_names:
                lb, ub = self.current_ranges[name]
                self.params[name] = max(lb, min(ub, self.params[name] + self.lr * grads[name]))

            # Evaluate and record
            score = self.simulate_fn(self.params, self.target_idx)
            self._record(self.params, score)

            # Track best
            if score > best_score:
                best_score = score
                best_params = self.params.copy()

            print(f"Iter {it}/{self.n_iters}: selectivity={score:.4f}")

        return best_params, best_score
