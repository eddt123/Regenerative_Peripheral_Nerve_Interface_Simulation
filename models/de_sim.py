from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, Dict, Tuple

class DESimulation:
    """
    Differential Evolution optimizer for electrode currents to maximize selectivity.

    Hyperparameters:
      - popsize: population size (number of candidate solutions per generation)
      - mutation: tuple of mutation factors (usually (0.5, 1.0))
      - recombination: crossover probability
      - maxiter: number of generations to run
    """

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        popsize: int = 15,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        maxiter: int = 100
    ):
        self.current_ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.maxiter = maxiter

        # Prepare CSV
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, mode="w", newline="") as fh:
                writer = csv.writer(fh)
                header = self.param_names + ["target_idx", "selectivity"]
                writer.writerow(header)

    def _record(self, params: Dict[str, float], score: float) -> None:
        row = [params[name] for name in self.param_names] + [self.target_idx, score]
        with open(self.out_csv, mode="a", newline="") as fh:
            csv.writer(fh).writerow(row)

    def optimize(self) -> tuple[Dict[str, float], float]:
        """
        Run differential evolution and return best parameters and selectivity.
        """
        # Objective for DE (minimize negative selectivity)
        def objective(x: np.ndarray) -> float:
            param_dict = {name: float(val) for name, val in zip(self.param_names, x)}
            score = self.simulate_fn(param_dict, self.target_idx)
            self._record(param_dict, score)
            return -score  # DE minimizes

        bounds = [self.current_ranges[name] for name in self.param_names]
        result = differential_evolution(
            objective,
            bounds,
            popsize=self.popsize,
            mutation=self.mutation,
            recombination=self.recombination,
            maxiter=self.maxiter,
            disp=True,
            polish=False
        )

        best_x = result.x
        best_score = -result.fun
        best_params = {name: float(val) for name, val in zip(self.param_names, best_x)}
        return best_params, best_score
    """
    Differential Evolution optimizer for electrode currents to maximize selectivity.

    Hyperparameters:
      - popsize: population size multiplier (popsize * len(params))
      - mutation: tuple of mutation factors (min, max)
      - recombination: crossover probability
      - maxiter: number of generations
    """
    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        popsize: int = 15,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        maxiter: int = 100,
    ):
        self.current_ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.maxiter = maxiter

        # Prepare CSV
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(self.param_names + ['target_idx', 'selectivity'])

    def _record(self, param: Dict[str, float], score: float) -> None:
        row = [param[name] for name in self.param_names] + [self.target_idx, score]
        with open(self.out_csv, 'a', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(row)

    def optimize(self) -> tuple[Dict[str, float], float]:
        """
        Run DE for maxiter generations. Returns best params and selectivity.
        """
        bounds = [(lb, ub) for lb, ub in self.current_ranges.values()]

        def _objective(x: np.ndarray) -> float:
            param = {name: float(val) for name, val in zip(self.param_names, x)}
            score = self.simulate_fn(param, self.target_idx)
            self._record(param, score)
            # DE minimizes, so return negative of selectivity
            return -score

        result = differential_evolution(
            _objective,
            bounds,
            strategy='best1bin',
            popsize=self.popsize,
            mutation=self.mutation,
            recombination=self.recombination,
            maxiter=self.maxiter,
            polish=False,
        )

        best_x = result.x
        best_score = -result.fun
        best_params = {name: float(val) for name, val in zip(self.param_names, best_x)}
        return best_params, best_score
