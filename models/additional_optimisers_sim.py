# additional_optimizers.py
# -----------------------------------------------------------------------------
# Stand‑alone drop‑in optimizers that follow the same pattern you used for
# GreedyRandomOptimizer, BOSimulation, CMAESOptimizer, GradientDescentSimulator
# and DESimulation.  Each class exposes
#     optimize() -> tuple[Dict[str, float], float]
# and takes:
#     current_ranges : Dict[str, Tuple[float, float]]
#     target_idx     : int
#     simulate_fn    : Callable[[Dict[str, float], int], float]
#     out_csv        : Union[str, Path]
# -----------------------------------------------------------------------------
# NOTE: These are deliberately lightweight (no GPU / deep‑probabilistic libs) so
# they run out‑of‑the‑box.  They give you: 1) a tree‑based BO (TPE), 2) a random
#‑forest surrogate BO, 3) a simple Cross‑Entropy Method (CEM) evolutionary
# search, 4) a local trust‑region Powell search, and 5) a discrete UCB bandit.
# -----------------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import csv
import random
from typing import Callable, Dict, Tuple

# TuRBO dependencies ----------------------------------------------------------

# import torch                                         
# from turbo import Turbo1                              

import numpy as np
from scipy.optimize import minimize

# Optional: Install optuna & scikit‑optimize if you want the tree‑based models
try:
    import optuna  # type: ignore
except ModuleNotFoundError:
    optuna = None  # fallback stub

try:
    from skopt import Optimizer as SkOptimizer  # type: ignore
except ModuleNotFoundError:
    SkOptimizer = None  # fallback stub


# -----------------------------------------------------------------------------
# 1. Tree‑structured Parzen Estimator (TPE) via Optuna -------------------------
# -----------------------------------------------------------------------------
class TPESimulation:
    """Tree‑based Bayesian optimisation using Optuna's TPE sampler."""

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        n_trials: int = 200,
        random_seed: int | None = None,
    ) -> None:
        if optuna is None:
            raise ImportError("optuna is required for TPESimulation")

        self.ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.n_trials = n_trials
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, "w", newline="") as fh:
                csv.writer(fh).writerow(self.param_names + ["target_idx", "selectivity"])

        # Optuna study
        self.study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=random_seed),
            direction="maximize",
        )

    # ──────────────────────────────────────────────────────────────────
    def _record(self, params: Dict[str, float], score: float) -> None:
        with open(self.out_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([params[n] for n in self.param_names] + [self.target_idx, score])

    # ──────────────────────────────────────────────────────────────────
    def optimize(self) -> tuple[Dict[str, float], float]:
        def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore
            params = {
                name: trial.suggest_float(name, lb, ub)
                for name, (lb, ub) in self.ranges.items()
            }
            score = self.simulate_fn(params, self.target_idx)
            self._record(params, score)
            return score

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_params = self.study.best_trial.params  # type: ignore
        best_score = self.study.best_value  # type: ignore
        return best_params, best_score


# -----------------------------------------------------------------------------
# 2. Random‑Forest surrogate Bayesian optimisation via scikit‑optimize ----------
# -----------------------------------------------------------------------------
class RandomForestBOSimulation:
    """scikit‑optimize Optimizer with Random‑Forest surrogate and EI acquisition."""

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        n_calls: int = 200,
        random_seed: int | None = None,
    ) -> None:
        if SkOptimizer is None:
            raise ImportError("scikit‑optimize is required for RandomForestBOSimulation")

        self.ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.n_calls = n_calls
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, "w", newline="") as fh:
                csv.writer(fh).writerow(self.param_names + ["target_idx", "selectivity"])

        dimensions = [(lb, ub) for lb, ub in current_ranges.values()]
        self.opt = SkOptimizer(
            dimensions=dimensions,
            base_estimator="RF",
            acq_func="EI",
            random_state=random_seed,
        )

    def _record(self, params: Dict[str, float], score: float) -> None:
        with open(self.out_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([params[n] for n in self.param_names] + [self.target_idx, score])

    def optimize(self) -> tuple[Dict[str, float], float]:
        best_score = -np.inf
        best_params: Dict[str, float] = {}

        for _ in range(self.n_calls):
            x = self.opt.ask()
            params = {name: float(val) for name, val in zip(self.param_names, x)}
            score = self.simulate_fn(params, self.target_idx)
            self._record(params, score)
            self.opt.tell(x, -score)  # skopt minimizes
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score


# -----------------------------------------------------------------------------
# 3. Cross‑Entropy Method (CEM) -----------------------------------------------
# -----------------------------------------------------------------------------
class CEMOptimizer:
    """Simple Cross‑Entropy Method for continuous parameters."""

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        popsize: int = 50,
        elite_frac: float = 0.2,
        n_iters: int = 20,
        smoothing: float = 0.9,
        random_seed: int | None = None,
    ) -> None:
        self.ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.popsize = popsize
        self.elite_frac = elite_frac
        self.n_iters = n_iters
        self.smoothing = smoothing
        np.random.seed(random_seed)

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, "w", newline="") as fh:
                csv.writer(fh).writerow(self.param_names + ["target_idx", "selectivity"])

        # Initialise mean and std to mid‑range
        lbs, ubs = zip(*current_ranges.values())
        self.mean = np.array([(l + u) / 2 for l, u in zip(lbs, ubs)], dtype=float)
        self.std = np.array([(u - l) / 2 for l, u in zip(lbs, ubs)], dtype=float)

    def _record(self, params: Dict[str, float], score: float):
        with open(self.out_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([params[n] for n in self.param_names] + [self.target_idx, score])

    def optimize(self) -> Tuple[Dict[str, float], float]:
        best_score = -np.inf
        best_params: Dict[str, float] = {}
        D = len(self.param_names)
        lb = np.array([self.ranges[n][0] for n in self.param_names])
        ub = np.array([self.ranges[n][1] for n in self.param_names])

        for _ in range(self.n_iters):
            # Sample population
            pop = self.mean + self.std * np.random.randn(self.popsize, D)
            pop = np.clip(pop, lb, ub)
            scores = []
            for x in pop:
                params = {n: float(v) for n, v in zip(self.param_names, x)}
                score = self.simulate_fn(params, self.target_idx)
                self._record(params, score)
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_params = params
            # Select elites
            elite_idx = np.argsort(scores)[-int(self.elite_frac * self.popsize):]
            elite = pop[elite_idx]
            # Update mean/std with smoothing
            self.mean = self.smoothing * self.mean + (1 - self.smoothing) * elite.mean(axis=0)
            self.std  = self.smoothing * self.std  + (1 - self.smoothing) * elite.std(axis=0)
            self.std = np.maximum(self.std, 1e-6)
        return best_params, best_score


# -----------------------------------------------------------------------------
# 4. Local Trust‑Region (Powell) ----------------------------------------------
# -----------------------------------------------------------------------------
class PowellLocalOptimizer:
    """scipy.optimize.minimize with Powell method (box‑constrained)."""

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        n_restarts: int = 5,
        random_seed: int | None = None,
    ) -> None:
        self.ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.n_restarts = n_restarts
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, "w", newline="") as fh:
                csv.writer(fh).writerow(self.param_names + ["target_idx", "selectivity"])

    def _record(self, params: Dict[str, float], score: float):
        with open(self.out_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([params[n] for n in self.param_names] + [self.target_idx, score])

    def optimize(self) -> Tuple[Dict[str, float], float]:
        best_score = -np.inf
        best_params = {}
        bounds = [self.ranges[n] for n in self.param_names]
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        def objective(x: np.ndarray) -> float:
            params = {n: float(v) for n, v in zip(self.param_names, x)}
            score = self.simulate_fn(params, self.target_idx)
            self._record(params, score)
            return -score  # minimize

        for _ in range(self.n_restarts):
            x0 = lb + (ub - lb) * np.random.rand(len(bounds))
            res = minimize(objective, x0, method="Powell", bounds=bounds, options={"maxiter": 200})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = {n: float(v) for n, v in zip(self.param_names, res.x)}
        return best_params, best_score


# -----------------------------------------------------------------------------
# 5. Discrete UCB Multi‑Armed Bandit ------------------------------------------
# -----------------------------------------------------------------------------
class BanditOptimizer:
    """Upper‑Confidence‑Bound bandit over a discretised input space."""

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        resolution: int = 5,
        total_pulls: int = 200,
        c: float = 2.0,
    ) -> None:
        self.ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.resolution = resolution
        self.total_pulls = total_pulls
        self.c = c

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, "w", newline="") as fh:
                csv.writer(fh).writerow(self.param_names + ["target_idx", "selectivity"])

        # Build discrete grid
        grids = [np.linspace(lb, ub, resolution) for lb, ub in current_ranges.values()]
        self.actions = np.array(np.meshgrid(*grids)).T.reshape(-1, len(self.param_names))
        self.N = np.zeros(len(self.actions))
        self.S = np.zeros(len(self.actions))

    def _record(self, params: Dict[str, float], score: float):
        with open(self.out_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([params[n] for n in self.param_names] + [self.target_idx, score])

    def optimize(self) -> Tuple[Dict[str, float], float]:
        best_score = -np.inf
        best_params = {}
        for t in range(1, self.total_pulls + 1):
            # UCB selection
            ucb = np.where(
                self.N == 0,
                np.inf,
                self.S / self.N + self.c * np.sqrt(np.log(t) / self.N),
            )
            idx = int(np.argmax(ucb))
            x = self.actions[idx]
            params = {n: float(v) for n, v in zip(self.param_names, x)}
            score = self.simulate_fn(params, self.target_idx)
            self._record(params, score)
            # update stats
            self.N[idx] += 1
            self.S[idx] += score
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score
    
# -----------------------------------------------------------------------------
# 6. Particle Swarm Optimisation ------------------------------------------
# -----------------------------------------------------------------------------

class PSOSimulation:
    """
    Particle Swarm Optimization (PSO) for electrode current parameters.

    Hyperparameters:
      - popsize: number of particles in swarm
      - w: inertia weight
      - c1: cognitive coefficient
      - c2: social coefficient
      - n_iters: number of iterations (generations)
    """
    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        popsize: int = 20,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        n_iters: int = 20,
        random_seed: int | None = None,
    ):
        self.current_ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.out_csv = Path(out_csv)
        self.popsize = popsize
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_iters = n_iters
        if random_seed is not None:
            np.random.seed(random_seed)

        # Prepare CSV
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(self.param_names + ['target_idx', 'selectivity'])

    def _record(self, params: Dict[str, float], score: float) -> None:
        with open(self.out_csv, 'a', newline='') as fh:
            csv.writer(fh).writerow([params[n] for n in self.param_names] + [self.target_idx, score])

    def optimize(self) -> tuple[Dict[str, float], float]:
        D = len(self.param_names)
        # Initialize particle positions and velocities
        lb = np.array([b[0] for b in self.current_ranges.values()])
        ub = np.array([b[1] for b in self.current_ranges.values()])
        pos = lb + (ub - lb) * np.random.rand(self.popsize, D)
        vel = np.zeros((self.popsize, D))

        # Personal bests
        pbest_pos = pos.copy()
        pbest_score = np.full(self.popsize, -np.inf)

        # Global best
        gbest_pos = None
        gbest_score = -np.inf

        for it in range(1, self.n_iters + 1):
            scores = []
            for i in range(self.popsize):
                params = {name: float(val) for name, val in zip(self.param_names, pos[i])}
                score = self.simulate_fn(params, self.target_idx)
                self._record(params, score)
                scores.append(score)
                # update personal best
                if score > pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i] = pos[i].copy()
                # update global best
                if score > gbest_score:
                    gbest_score = score
                    gbest_pos = pos[i].copy()
            scores = np.array(scores)

            # Velocity & position update
            r1 = np.random.rand(self.popsize, D)
            r2 = np.random.rand(self.popsize, D)
            vel = (
                self.w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            pos = pos + vel
            pos = np.clip(pos, lb, ub)
            print(f"[PSO] Iter {it}/{self.n_iters} best_so_far={gbest_score:.4f}")

        best_params = {name: float(val) for name, val in zip(self.param_names, gbest_pos)}
        return best_params, gbest_score

# -----------------------------------------------------------------------------
# 7. Trust-Region Bayesian Optimisation (TuRBO-1) -----------------------------
# -----------------------------------------------------------------------------
#
#  Uses the reference implementation from: https://github.com/uber-research/TuRBO
#  Paper: Eriksson et al., “Scalable Global Optimisation via Local Bayesian
#  Optimisation”, NeurIPS 2020.
#
#  API matches your existing optimisers:
#      best_params, best_score = TuRBOBOSimulation(...).optimize()
# -----------------------------------------------------------------------------



class TuRBOBOSimulation:
    """
    Trust-Region Bayesian Optimisation (TuRBO-1) wrapper.

    Parameters
    ----------
    current_ranges : Dict[str, Tuple[float, float]]
        Parameter bounds, e.g. {"amp": (0, 100), "pw": (100, 500)}
    target_idx : int
        Index of the target region / channel for selectivity calc.
    simulate_fn : Callable[[Dict[str, float], int], float]
        Black-box function returning selectivity score.
    out_csv : str | Path
        CSV path to log each evaluation.
    n_init : int
        Number of Sobol initial points.
    max_evals : int
        Total evaluation budget (init + BO rounds).
    batch_size : int
        Parallel suggestions per BO step.
    random_seed : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        current_ranges: Dict[str, Tuple[float, float]],
        target_idx: int,
        simulate_fn: Callable[[Dict[str, float], int], float],
        out_csv: str | Path,
        n_init: int = 20,
        max_evals: int = 200,
        batch_size: int = 1,
        random_seed: int | None = None,
    ) -> None:
        if Turbo1 is None:
            raise ImportError(
                "TuRBOBOSimulation requires BoTorch + turbo-bo. "
                "Install with:  pip install botorch gpytorch turbo-bo"
            )

        self.ranges = current_ranges
        self.param_names = list(current_ranges.keys())
        self.target_idx = target_idx
        self.simulate_fn = simulate_fn
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, "w", newline="") as fh:
                csv.writer(fh).writerow(self.param_names + ["target_idx", "selectivity"])

        # TuRBO bounds (0–1 scaled inside algorithm)
        self.lb = np.array([b[0] for b in current_ranges.values()], dtype=np.float64)
        self.ub = np.array([b[1] for b in current_ranges.values()], dtype=np.float64)
        self.dim = len(self.param_names)

        # Initialise TuRBO state
        self._turbo = Turbo1(
            f=None,                          # we handle evaluation manually
            lb=self.lb,
            ub=self.ub,
            n_init=self.n_init,
            max_evals=self.max_evals,
            batch_size=self.batch_size,
            use_ard=True,
            max_cholesky_size=2000,
            verbose=False,
            device="cpu",
            dtype="float64",
            seed=random_seed,
        )

    # ──────────────────────────────────────────────────────────────────
    def _record(self, params: Dict[str, float], score: float) -> None:
        """Append a single evaluation to CSV log."""
        with open(self.out_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([params[n] for n in self.param_names] +
                                    [self.target_idx, score])

    # ──────────────────────────────────────────────────────────────────
    def optimize(self) -> tuple[Dict[str, float], float]:
        best_score = -np.inf
        best_params: Dict[str, float] = {}

        n_evals = 0
        while n_evals < self.max_evals:
            # --- TuRBO proposes next batch ----------------------------------
            X_next = self._turbo.ask()                # shape: (batch_size, dim)

            # Evaluate batch
            Y_next = np.zeros(self.batch_size)
            for i, x in enumerate(X_next):
                params = {n: float(v) for n, v in zip(self.param_names, x)}
                score = self.simulate_fn(params, self.target_idx)
                self._record(params, score)
                Y_next[i] = score
                if score > best_score:
                    best_score = score
                    best_params = params

            # NOTE:  Turbo1 assumes *minimisation*.  To maximise selectivity
            # we send -score.  Remove the negative sign if your objective is
            # naturally minimised.
            self._turbo.tell(-Y_next)
            n_evals += self.batch_size

            if self._turbo.restart_triggered:   # safety break
                break

        return best_params, best_score
