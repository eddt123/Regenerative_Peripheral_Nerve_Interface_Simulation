from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Callable
import csv
import cma

class CMAESOptimizer:
    """
    CMA-ES optimizer for electrode currents to maximize selectivity.
    """

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        sigma: float = 0.3,
        popsize: int | None = None,
        n_iters: int = 50
    ):
        self.current_ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.n_iters = n_iters

        # initial mean = midpoint of ranges
        self.mean = [(lb + ub) / 2.0 for (lb, ub) in current_ranges.values()]
        # initial sigma (step size)
        self.sigma = sigma
        # bounds
        self.bounds = [[lb for (lb, ub) in current_ranges.values()],
                       [ub for (lb, ub) in current_ranges.values()]]

        # prepare csv
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, mode="w", newline="") as fh:
                writer = csv.writer(fh)
                header = self.param_names + ["target_idx", "selectivity"]
                writer.writerow(header)

        # init CMA-ES
        opts = {"bounds": self.bounds}
        if popsize is not None:
            opts["popsize"] = popsize
        self.es = cma.CMAEvolutionStrategy(self.mean, self.sigma, opts)

        # track best
        self.best_score = -float("inf")
        self.best_params: Dict[str, float] = {}

    def _record(self, params: Dict[str, float], score: float):
        row = [params[n] for n in self.param_names] + [self.target_idx, score]
        with open(self.out_csv, mode="a", newline="") as fh:
            csv.writer(fh).writerow(row)

    def optimize(self) -> tuple[Dict[str, float], float]:
        for it in range(1, self.n_iters + 1):
            solutions = self.es.ask()
            scores = []
            for sol in solutions:
                param_dict = {name: val for name, val in zip(self.param_names, sol)}
                score = self.simulate_fn(param_dict, self.target_idx)
                self._record(param_dict, score)
                scores.append(-score)  # CMA-ES minimizes
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = param_dict.copy()
            self.es.tell(solutions, scores)
            print(f"[CMA-ES] Iter {it}/{self.n_iters} best_so_far={self.best_score:.4f}")
        return self.best_params, self.best_score
