#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytic cylinder optimisation benchmark without external dependencies."""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Geometry and activation model
# ---------------------------------------------------------------------------

RADIUS = 1.5e-3  # [m]
SIGMA = 0.3  # conductivity [S/m]
Z_SPACING = 1.0e-3  # [m]
H_AF = 0.3e-3  # step size for activation function
ALPHA = 12.0  # sigmoid slope
BETA = 0.45  # weight applied to off-target activations
CURRENT_LIMIT = 1.0  # absolute bound on currents [A]
SELECTIVITY_THRESHOLD = 0.9995  # score defining a selective solution

# Derived constants
FOUR_PI_SIG = 4.0 * math.pi * SIGMA


Vector = List[float]


def electrode_positions(n_rows: int = 3, n_per_row: int = 4) -> List[Vector]:
    """Return evenly spaced electrodes on a cylindrical surface."""
    thetas = [2.0 * math.pi * i / n_per_row for i in range(n_per_row)]
    zs = [Z_SPACING * (i - (n_rows - 1) / 2.0) for i in range(n_rows)]
    positions: List[Vector] = []
    for z in zs:
        for theta in thetas:
            x = RADIUS * math.cos(theta)
            y = RADIUS * math.sin(theta)
            positions.append([x, y, z])
    return positions


ELECTRODE_POSITIONS = electrode_positions()


def _vector_add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def _vector_sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def _vector_scale(v: Sequence[float], scalar: float) -> Vector:
    return [scalar * x for x in v]


def _vector_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _outer(a: Sequence[float], b: Sequence[float]) -> List[List[float]]:
    return [[x * y for y in b] for x in a]


def _matrix_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def _matrix_scale(matrix: List[List[float]], scalar: float) -> List[List[float]]:
    return [[scalar * value for value in row] for row in matrix]


def _identity_matrix(size: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def _cholesky(matrix: List[List[float]]) -> List[List[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                lower[i][j] = math.sqrt(max(val, 1e-12))
            else:
                lower[i][j] = (matrix[i][j] - s) / lower[j][j]
    return lower


def _forward_substitution(lower: List[List[float]], b: Sequence[float]) -> List[float]:
    n = len(lower)
    y = [0.0] * n
    for i in range(n):
        s = sum(lower[i][k] * y[k] for k in range(i))
        y[i] = (b[i] - s) / lower[i][i]
    return y


def _backward_substitution(upper: List[List[float]], y: Sequence[float]) -> List[float]:
    n = len(upper)
    x = [0.0] * n
    for i in reversed(range(n)):
        s = sum(upper[i][k] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - s) / upper[i][i]
    return x


def _solve_cholesky(lower: List[List[float]], b: Sequence[float]) -> List[float]:
    y = _forward_substitution(lower, b)
    upper = [[lower[j][i] for j in range(len(lower))] for i in range(len(lower))]
    return _backward_substitution(upper, y)


def potential_at(point: Sequence[float], currents: Sequence[float]) -> float:
    """Return the quasi-static potential at *point* from electrode currents."""
    total = 0.0
    for pos, current in zip(ELECTRODE_POSITIONS, currents):
        dist = _vector_norm(_vector_sub(point, pos)) + 1e-12
        total += current / (FOUR_PI_SIG * dist)
    return total


def activation_function(point: Sequence[float], direction: Sequence[float], currents: Sequence[float]) -> float:
    """Second directional derivative of the potential (finite differences)."""
    norm_dir = _vector_norm(direction)
    if norm_dir == 0:
        raise ValueError("direction vector must be non-zero")
    unit_dir = [d / norm_dir for d in direction]
    offset = _vector_scale(unit_dir, H_AF)
    v_plus = potential_at(_vector_add(point, offset), currents)
    v_minus = potential_at(_vector_sub(point, offset), currents)
    v0 = potential_at(point, currents)
    return (v_plus - 2.0 * v0 + v_minus) / (H_AF ** 2)


def sigmoid(x: float) -> float:
    if x >= 60.0:
        return 1.0
    if x <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Fibre configuration
# ---------------------------------------------------------------------------


def make_fiber_points() -> Tuple[Dict[int, Vector], int, List[int], Vector]:
    rs = [0.45 * RADIUS, 0.6 * RADIUS, 0.7 * RADIUS, 0.85 * RADIUS, 0.3 * RADIUS]
    thetas = [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0, math.pi, 5.0 * math.pi / 6.0]
    zs = [0.0, 0.0, 0.0, 0.0, 0.0]
    fibres: Dict[int, Vector] = {}
    for idx, (r, theta, z) in enumerate(zip(rs, thetas, zs), start=20000):
        fibres[idx] = [r * math.cos(theta), r * math.sin(theta), z]
    target_idx = 20000
    off_indices = [idx for idx in fibres if idx != target_idx]
    direction = [0.0, 0.0, 1.0]
    return fibres, target_idx, off_indices, direction


FIBRES, DEFAULT_TARGET_IDX, DEFAULT_OFF_IDXS, FIBRE_DIR = make_fiber_points()


# ---------------------------------------------------------------------------
# Reward definition
# ---------------------------------------------------------------------------


def selectivity_reward(
    currents: Sequence[float],
    target_idx: int,
    *,
    off_indices: Sequence[int] = DEFAULT_OFF_IDXS,
    alpha: float = ALPHA,
    beta: float = BETA,
    l2_penalty: float = 0.02,
) -> float:
    """Selectivity reward based on target vs off-target activation."""
    target_point = FIBRES[target_idx]
    target_af = activation_function(target_point, FIBRE_DIR, currents)
    target_term = sigmoid(alpha * target_af)

    off_terms = []
    for idx in off_indices:
        off_point = FIBRES[idx]
        off_af = activation_function(off_point, FIBRE_DIR, currents)
        off_terms.append(sigmoid(alpha * off_af))
    off_term = sum(off_terms) / len(off_terms)

    l2 = sum(c * c for c in currents) / len(currents)
    reward = target_term - beta * off_term - l2_penalty * l2
    return reward


# Provide a dict-based interface for backwards compatibility

def simulate_selectivity(param_dict: Dict[str, float], target_idx: int) -> float:
    ordered_currents = [param_dict[f"I_d{i}"] for i in range(1, 13)]
    return selectivity_reward(ordered_currents, target_idx)


# ---------------------------------------------------------------------------
# Optimisation utilities (no external dependencies)
# ---------------------------------------------------------------------------


Bounds = List[Tuple[float, float]]


def clamp(value: float, bounds: Tuple[float, float]) -> float:
    lower, upper = bounds
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@dataclass
class OptimisationResult:
    best_currents: Vector
    best_score: float
    evaluations: int
    history: List[float]
    first_hit_evaluation: Optional[int]


class CylindricalTask:
    def __init__(self, target_idx: int = DEFAULT_TARGET_IDX):
        self.target_idx = target_idx

    def evaluate(self, currents: Sequence[float]) -> float:
        return selectivity_reward(currents, self.target_idx)


class BaseOptimizer:
    def __init__(self, task: CylindricalTask, bounds: Bounds, seed: int):
        self.task = task
        self.bounds = bounds
        self.rng = random.Random(seed)
        self.history: List[float] = []
        self.eval_count = 0
        self.first_hit_evaluation: Optional[int] = None

    def _random_vector(self) -> Vector:
        return [self.rng.uniform(lb, ub) for lb, ub in self.bounds]

    def _evaluate(self, currents: Sequence[float]) -> float:
        score = self.task.evaluate(currents)
        self.eval_count += 1
        self.history.append(score)
        if (
            self.first_hit_evaluation is None
            and score >= SELECTIVITY_THRESHOLD
        ):
            self.first_hit_evaluation = self.eval_count
        return score

    def optimise(self) -> OptimisationResult:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, task: CylindricalTask, bounds: Bounds, seed: int, budget: int):
        super().__init__(task, bounds, seed)
        self.budget = budget

    def optimise(self) -> OptimisationResult:
        best_score = -math.inf
        best_vector: Vector = []
        for _ in range(self.budget):
            candidate = self._random_vector()
            score = self._evaluate(candidate)
            if score > best_score:
                best_score = score
                best_vector = candidate
        return OptimisationResult(
            best_vector,
            best_score,
            self.eval_count,
            self.history,
            self.first_hit_evaluation,
        )


class HillClimbOptimizer(BaseOptimizer):
    def __init__(
        self,
        task: CylindricalTask,
        bounds: Bounds,
        seed: int,
        *,
        step_start: float,
        step_end: float,
        iters: int,
        proposals_per_iter: int,
    ):
        super().__init__(task, bounds, seed)
        self.step_start = step_start
        self.step_end = step_end
        self.iters = iters
        self.proposals_per_iter = proposals_per_iter

    def optimise(self) -> OptimisationResult:
        current = self._random_vector()
        current_score = self._evaluate(current)
        best = list(current)
        best_score = current_score
        for it in range(self.iters):
            t = it / max(1, self.iters - 1)
            step = self.step_start * (1.0 - t) + self.step_end * t
            temperature = max(1e-3, 1.0 - t)
            for _ in range(self.proposals_per_iter):
                candidate = [
                    clamp(curr + self.rng.gauss(0.0, step), bounds)
                    for curr, bounds in zip(current, self.bounds)
                ]
                score = self._evaluate(candidate)
                if score > current_score or self.rng.random() < math.exp((score - current_score) / max(1e-6, temperature)):
                    current = candidate
                    current_score = score
                if score > best_score:
                    best = candidate
                    best_score = score
        return OptimisationResult(
            best,
            best_score,
            self.eval_count,
            self.history,
            self.first_hit_evaluation,
        )


class ParticleSwarmOptimizer(BaseOptimizer):
    def __init__(
        self,
        task: CylindricalTask,
        bounds: Bounds,
        seed: int,
        *,
        swarm_size: int,
        iters: int,
        inertia: float,
        cognitive: float,
        social: float,
    ):
        super().__init__(task, bounds, seed)
        self.swarm_size = swarm_size
        self.iters = iters
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def optimise(self) -> OptimisationResult:
        particles = [self._random_vector() for _ in range(self.swarm_size)]
        velocities = [[0.0] * len(self.bounds) for _ in range(self.swarm_size)]
        personal_best = list(particles)
        personal_scores = [self._evaluate(p) for p in personal_best]
        best_index = max(range(self.swarm_size), key=lambda i: personal_scores[i])
        global_best = list(personal_best[best_index])
        global_score = personal_scores[best_index]

        for _ in range(self.iters):
            for i, particle in enumerate(particles):
                velocity = velocities[i]
                r1 = self.rng.random()
                r2 = self.rng.random()
                for d, bounds in enumerate(self.bounds):
                    cognitive_term = self.cognitive * r1 * (personal_best[i][d] - particle[d])
                    social_term = self.social * r2 * (global_best[d] - particle[d])
                    velocity[d] = self.inertia * velocity[d] + cognitive_term + social_term
                    particle[d] = clamp(particle[d] + velocity[d], bounds)
                score = self._evaluate(particle)
                if score > personal_scores[i]:
                    personal_scores[i] = score
                    personal_best[i] = list(particle)
                    if score > global_score:
                        global_score = score
                        global_best = list(particle)
        return OptimisationResult(
            global_best,
            global_score,
            self.eval_count,
            self.history,
            self.first_hit_evaluation,
        )


class DifferentialEvolutionOptimizer(BaseOptimizer):
    def __init__(
        self,
        task: CylindricalTask,
        bounds: Bounds,
        seed: int,
        *,
        pop_size: int,
        iters: int,
        differential_weight: float,
        crossover_prob: float,
    ):
        super().__init__(task, bounds, seed)
        self.pop_size = pop_size
        self.iters = iters
        self.differential_weight = differential_weight
        self.crossover_prob = crossover_prob

    def optimise(self) -> OptimisationResult:
        population = [self._random_vector() for _ in range(self.pop_size)]
        scores = [self._evaluate(ind) for ind in population]
        for _ in range(self.iters):
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = self.rng.sample(indices, 3)
                mutant = []
                for d, bounds in enumerate(self.bounds):
                    donor = population[a][d] + self.differential_weight * (
                        population[b][d] - population[c][d]
                    )
                    mutant.append(clamp(donor, bounds))
                cross_point = self.rng.randrange(len(self.bounds))
                trial = []
                for d, bounds in enumerate(self.bounds):
                    if self.rng.random() < self.crossover_prob or d == cross_point:
                        trial.append(mutant[d])
                    else:
                        trial.append(population[i][d])
                    trial[d] = clamp(trial[d], bounds)
                trial_score = self._evaluate(trial)
                if trial_score > scores[i]:
                    population[i] = trial
                    scores[i] = trial_score
        best_index = max(range(self.pop_size), key=lambda idx: scores[idx])
        return OptimisationResult(
            population[best_index],
            scores[best_index],
            self.eval_count,
            self.history,
            self.first_hit_evaluation,
        )


class CoordinateDescentOptimizer(BaseOptimizer):
    def __init__(
        self,
        task: CylindricalTask,
        bounds: Bounds,
        seed: int,
        *,
        iters: int,
        step_start: float,
        step_end: float,
        restarts: int,
    ):
        super().__init__(task, bounds, seed)
        self.iters = iters
        self.step_start = step_start
        self.step_end = step_end
        self.restarts = restarts

    def optimise(self) -> OptimisationResult:
        best_params: Vector = []
        best_score = -math.inf

        for _ in range(self.restarts):
            params = self._random_vector()
            current_score = self._evaluate(params)
            if current_score > best_score:
                best_score = current_score
                best_params = list(params)

            for it in range(self.iters):
                t = it / max(1, self.iters - 1)
                step = self.step_start * (1.0 - t) + self.step_end * t
                improved = False
                for d in range(len(self.bounds)):
                    candidates = []
                    for delta in (-step, 0.0, step):
                        candidate = list(params)
                        candidate[d] = clamp(candidate[d] + delta, self.bounds[d])
                        score = self._evaluate(candidate)
                        candidates.append((score, candidate))
                    best_local_score, best_local_params = max(candidates, key=lambda x: x[0])
                    if best_local_score > current_score:
                        params = list(best_local_params)
                        current_score = best_local_score
                        improved = True
                    if best_local_score > best_score:
                        best_score = best_local_score
                        best_params = list(best_local_params)
                if not improved and step < 0.08:
                    params = [
                        clamp(p + self.rng.gauss(0.0, step), bounds)
                        for p, bounds in zip(params, self.bounds)
                    ]
                    current_score = self._evaluate(params)
                    if current_score > best_score:
                        best_score = current_score
                        best_params = list(params)
                if current_score >= SELECTIVITY_THRESHOLD:
                    break

        return OptimisationResult(
            best_params,
            best_score,
            self.eval_count,
            self.history,
            self.first_hit_evaluation,
        )


class CMAESOptimizer(BaseOptimizer):
    def __init__(
        self,
        task: CylindricalTask,
        bounds: Bounds,
        seed: int,
        *,
        population: int,
        sigma_init: float,
        sigma_min: float,
        iters: int,
    ):
        super().__init__(task, bounds, seed)
        self.population = population
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.iters = iters

    def optimise(self) -> OptimisationResult:
        dim = len(self.bounds)
        mean = self._random_vector()
        sigma = self.sigma_init
        variances = [1.0] * dim
        lam = self.population
        mu = max(2, lam // 2)
        weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        mu_eff = 1.0 / sum(w * w for w in weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + c_sigma + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = [0.0] * dim
        p_c = [0.0] * dim
        best_vector = list(mean)
        best_score = -math.inf

        for _ in range(self.iters):
            samples: List[Tuple[Vector, Vector]] = []
            scores: List[float] = []
            for i in range(lam):
                z = [self.rng.gauss(0.0, 1.0) for _ in range(dim)]
                candidate = [
                    clamp(
                        mean[d] + sigma * math.sqrt(variances[d]) * z[d],
                        self.bounds[d],
                    )
                    for d in range(dim)
                ]
                score = self._evaluate(candidate)
                samples.append((candidate, z))
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_vector = list(candidate)
            order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            selected = [samples[i] for i in order[:mu]]
            old_mean = list(mean)
            mean = [0.0] * dim
            for weight, (candidate, _) in zip(weights, selected):
                for d in range(dim):
                    mean[d] += weight * candidate[d]

            y_k: List[Vector] = []
            for candidate, _ in selected:
                diff = []
                for d in range(dim):
                    denom = sigma * math.sqrt(variances[d]) + 1e-12
                    diff.append((candidate[d] - old_mean[d]) / denom)
                y_k.append(diff)

            y_w = [0.0] * dim
            for weight, diff in zip(weights, y_k):
                for d in range(dim):
                    y_w[d] += weight * diff[d]
            p_sigma = [
                (1 - c_sigma) * p_sigma[d]
                + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w[d]
                for d in range(dim)
            ]
            norm_p_sigma = _vector_norm(p_sigma)
            sigma *= math.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))
            sigma = max(sigma, self.sigma_min)

            eval_factor = self.eval_count / max(1, lam)
            decay = 1 - c_sigma
            h_sigma_cond = norm_p_sigma / math.sqrt(max(1e-12, 1 - decay ** (2 * eval_factor))) < (1.4 + 2 / (dim + 1)) * chi_n
            h_sigma = 1.0 if h_sigma_cond else 0.0
            p_c = [
                (1 - c_c) * p_c[d]
                + h_sigma
                * math.sqrt(c_c * (2 - c_c) * mu_eff)
                * (mean[d] - old_mean[d])
                / max(sigma, 1e-12)
                for d in range(dim)
            ]

            for d in range(dim):
                variance_update = 0.0
                for weight, diff in zip(weights, y_k):
                    variance_update += weight * diff[d] * diff[d]
                variances[d] = (
                    (1 - c1 - c_mu) * variances[d]
                    + c1 * (p_c[d] * p_c[d] + (1 - h_sigma) * c_c * (2 - c_c) * variances[d])
                    + c_mu * variance_update
                )
                variances[d] = max(variances[d], 1e-6)

        return OptimisationResult(
            best_vector,
            best_score,
            self.eval_count,
            self.history,
            self.first_hit_evaluation,
        )


def _rbf_kernel(x: Sequence[float], y: Sequence[float], length_scale: float, variance: float) -> float:
    diff = _vector_sub(x, y)
    return variance * math.exp(-0.5 * _dot(diff, diff) / (length_scale ** 2))


class BayesianOptimizer(BaseOptimizer):
    def __init__(
        self,
        task: CylindricalTask,
        bounds: Bounds,
        seed: int,
        *,
        init_points: int,
        iters: int,
        length_scale: float,
        variance: float,
        noise: float,
        candidates: int,
        max_points: int,
    ):
        super().__init__(task, bounds, seed)
        self.init_points = init_points
        self.iters = iters
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.candidates = candidates
        self.max_points = max_points

    def optimise(self) -> OptimisationResult:
        dim = len(self.bounds)
        X_points: List[Vector] = []
        y_values: List[float] = []
        best_score = -math.inf
        best_vector: Vector = []

        for _ in range(self.init_points):
            x = self._random_vector()
            score = self._evaluate(x)
            X_points.append(x)
            y_values.append(score)
            if score > best_score:
                best_score = score
                best_vector = list(x)

        for _ in range(self.iters):
            n = len(X_points)
            kernel_matrix = [
                [
                    _rbf_kernel(X_points[i], X_points[j], self.length_scale, self.variance)
                    for j in range(n)
                ]
                for i in range(n)
            ]
            for i in range(n):
                kernel_matrix[i][i] += self.noise ** 2
            L = _cholesky(kernel_matrix)
            alpha = _solve_cholesky(L, y_values)

            def acquisition(x: Vector) -> float:
                k_star = [
                    _rbf_kernel(x, x_j, self.length_scale, self.variance)
                    for x_j in X_points
                ]
                mean = sum(k * a for k, a in zip(k_star, alpha))
                v = _forward_substitution(L, k_star)
                var = max(self.variance - sum(val * val for val in v), 1e-12)
                std = math.sqrt(var)
                improvement = mean - best_score
                if std < 1e-9:
                    return 0.0
                z = improvement / std
                normal_cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
                normal_pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
                return improvement * normal_cdf + std * normal_pdf

            best_acq = -math.inf
            next_x = None
            for _ in range(self.candidates):
                candidate = [self.rng.uniform(lb, ub) for lb, ub in self.bounds]
                value = acquisition(candidate)
                if value > best_acq:
                    best_acq = value
                    next_x = candidate
            if next_x is None:
                break
            score = self._evaluate(next_x)
            X_points.append(next_x)
            y_values.append(score)
            if len(X_points) > self.max_points:
                X_points.pop(0)
                y_values.pop(0)
            if score > best_score:
                best_score = score
                best_vector = list(next_x)
            if best_score >= 0.99995 and len(X_points) >= self.init_points + 10:
                break

        return OptimisationResult(
            best_vector,
            best_score,
            self.eval_count,
            self.history,
            self.first_hit_evaluation,
        )


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    task = CylindricalTask(DEFAULT_TARGET_IDX)
    bounds = [(-CURRENT_LIMIT, CURRENT_LIMIT) for _ in range(12)]

    optimizers: Dict[str, Callable[[int], BaseOptimizer]] = {
        "RandomSearch": lambda seed: RandomSearchOptimizer(
            task, bounds, seed, budget=1500
        ),
        "HillClimb": lambda seed: HillClimbOptimizer(
            task,
            bounds,
            seed,
            step_start=0.4,
            step_end=0.05,
            iters=60,
            proposals_per_iter=25,
        ),
        "ParticleSwarm": lambda seed: ParticleSwarmOptimizer(
            task,
            bounds,
            seed,
            swarm_size=26,
            iters=90,
            inertia=0.63,
            cognitive=1.85,
            social=1.65,
        ),
        "DifferentialEvolution": lambda seed: DifferentialEvolutionOptimizer(
            task,
            bounds,
            seed,
            pop_size=32,
            iters=90,
            differential_weight=0.65,
            crossover_prob=0.85,
        ),
        "CoordinateDescent": lambda seed: CoordinateDescentOptimizer(
            task,
            bounds,
            seed,
            iters=140,
            step_start=0.35,
            step_end=0.01,
            restarts=3,
        ),
        "CMAES": lambda seed: CMAESOptimizer(
            task,
            bounds,
            seed,
            population=24,
            sigma_init=0.25,
            sigma_min=0.02,
            iters=80,
        ),
        "BayesianOpt": lambda seed: BayesianOptimizer(
            task,
            bounds,
            seed,
            init_points=18,
            iters=120,
            length_scale=0.4,
            variance=1.3,
            noise=0.02,
            candidates=280,
            max_points=70,
        ),
    }

    replicates = 12
    summary: Dict[str, Dict[str, float]] = {}
    full_results: Dict[str, List[Dict[str, float]]] = {}

    for name, factory in optimizers.items():
        replicate_scores: List[float] = []
        replicate_evals: List[int] = []
        first_hits_all: List[Optional[int]] = []
        replicate_histories: List[List[float]] = []
        for rep in range(replicates):
            seed = 10_000 + rep * 97 + hash(name) % 1000
            optimiser = factory(seed)
            result = optimiser.optimise()
            replicate_scores.append(result.best_score)
            replicate_evals.append(result.evaluations)
            replicate_histories.append(result.history)
            first_hits_all.append(result.first_hit_evaluation)
            print(f"[{name}] replicate {rep+1}/{replicates} -> best={result.best_score:.4f} evals={result.evaluations}")
        observed_hits = [fh for fh in first_hits_all if fh is not None]
        summary[name] = {
            "mean_score": statistics.mean(replicate_scores),
            "std_score": statistics.pstdev(replicate_scores),
            "best_score": max(replicate_scores),
            "median_score": statistics.median(replicate_scores),
            "mean_evals": statistics.mean(replicate_evals),
            "mean_first_hit": statistics.mean(observed_hits) if observed_hits else float("nan"),
            "hit_rate": len(observed_hits) / replicates,
        }
        full_results[name] = [
            {
                "best_score": s,
                "evaluations": e,
                "first_hit": fh,
            }
            for s, e, fh in zip(replicate_scores, replicate_evals, first_hits_all)
        ]

    output_dir = Path("data") / "RPNI_sim_analytic"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "replicates": full_results}, fh, indent=2)

    print("\n=== Optimiser Comparison (12 replicates each) ===")
    header = (
        f"{'Optimiser':<24}"
        f"{'Mean':>10}"
        f"{'Std':>10}"
        f"{'Best':>10}"
        f"{'Median':>10}"
        f"{'Mean evals':>14}"
        f"{'Hit rate':>10}"
        f"{'Mean hit':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, stats in summary.items():
        mean_hit = stats['mean_first_hit']
        hit_display = f"{stats['hit_rate']*100:>9.1f}%"
        mean_hit_display = f"{mean_hit:>11.1f}" if not math.isnan(mean_hit) else f"{'N/A':>11}"
        print(
            f"{name:<24}"
            f"{stats['mean_score']:>10.4f}"
            f"{stats['std_score']:>10.4f}"
            f"{stats['best_score']:>10.4f}"
            f"{stats['median_score']:>10.4f}"
            f"{stats['mean_evals']:>14.1f}"
            f"{hit_display}"
            f"{mean_hit_display}"
        )


if __name__ == "__main__":
    run_experiment()
