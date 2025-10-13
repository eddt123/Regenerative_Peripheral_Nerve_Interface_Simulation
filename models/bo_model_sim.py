from __future__ import annotations
from pathlib import Path
import numpy as np
import csv
from skopt import Optimizer
from skopt.space import Real
from typing import Callable, Dict, Tuple

class BOSimulation:
    """
    Bayesian Optimization for RPNI electrode current selection to maximise selectivity.
    Uses Lower Confidence Bound (LCB) acquisition with parameter κ.
    """
    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        n_iters: int = 50,
        candidates_per_iter: int = 20,
        n_initial_points: int = 10,
        acq_func: str = "LCB",          # changed default
        random_state: int = 42,
        kappa: float = 1.96,            # controls exploration (higher = more exploratory)
    ):
        # Store inputs
        self.simulate_fn = simulate_fn
        self.current_ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.out_csv = Path(out_csv)
        self.n_initial = n_initial_points
        self.n_iters = n_iters
        self.candidates_per_iter = candidates_per_iter  # unused, kept for compatibility

        # Prepare search space
        self.space = [
            Real(lb, ub, name=name)
            for name, (lb, ub) in current_ranges.items()
        ]

        # Initialize optimizer using LCB acquisition
        self.optimizer = Optimizer(
            dimensions=self.space,
            base_estimator="GP",
            acq_func=acq_func,                 # now "LCB"
            acq_func_kwargs={"kappa": kappa},  # use κ instead of ξ
            random_state=random_state,
            n_initial_points=self.n_initial
        )

        # Prepare CSV
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, mode="w", newline="") as fh:
                writer = csv.writer(fh)
                header = self.param_names + ["target_idx", "selectivity"]
                writer.writerow(header)

        # History tracking
        self.history = []  # list of (params, score)
        self.best_score = -np.inf
        self.best_params: Dict[str, float] = {}

    def suggest(self) -> Dict[str, float]:
        """Ask for the next point to evaluate."""
        x = self.optimizer.ask()
        return {name: val for name, val in zip(self.param_names, x)}

    def observe(self, params: Dict[str, float], score: float) -> None:
        """Tell optimizer the observed score for given parameters."""
        x = [params[name] for name in self.param_names]
        # skopt minimizes, so pass negative reward
        self.optimizer.tell(x, -score)
        self.history.append((params, score))

        # Record to CSV
        row = [params[name] for name in self.param_names] + [self.target_idx, score]
        with open(self.out_csv, mode="a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(row)

        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

    def optimize(self) -> tuple[Dict[str, float], float]:
        """
        Perform the full sequential BO loop.
        Returns best parameter set and best score.
        """
        # initial evaluations
        for _ in range(self.n_initial):
            params = self.suggest()
            score = self.simulate_fn(params, self.target_idx)
            self.observe(params, score)

        # sequential iterations
        for it in range(self.n_iters):
            params = self.suggest()
            score = self.simulate_fn(params, self.target_idx)
            self.observe(params, score)
            print(f"Iter {it+1}/{self.n_iters}: score={score:.4f}, best={self.best_score:.4f}")

        return self.best_params, self.best_score
