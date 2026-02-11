# me144_toolbox/optimization/ga.py
"""
ME144/244 Spring 2026 — Genetic Algorithm (Project 1)

Implements a simple, reusable GA with:
- elitism (keep top P)
- nearest-neighbor breeding among elites to produce K offspring
- optional mutation (disabled for Project 1 debugging)
- fills the rest of the generation with fresh random samples to keep population size S

Returns diagnostics requested by the assignment:
- Pi      : (G, S) array of sorted costs per generation
- Pi_min  : (G,) best cost per generation
- Pi_avg  : (G,) mean cost per generation
- Lambda  : (S, dv) final generation design strings (unsorted)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np


def _validate_limits(lim: np.ndarray, dv: int) -> np.ndarray:
    lim = np.asarray(lim, dtype=float)
    if lim.shape == (2,) and dv == 1:
        lim = lim.reshape(1, 2)
    if lim.shape != (dv, 2):
        raise ValueError(f"lim must have shape (dv,2). Got {lim.shape}, dv={dv}.")
    if np.any(lim[:, 1] <= lim[:, 0]):
        raise ValueError("Each lim row must be [lower, upper] with upper > lower.")
    return lim


def _sample_uniform(rng: np.random.Generator, S: int, lim: np.ndarray) -> np.ndarray:
    """
    Sample S design strings uniformly inside bounds.
    Returns shape (S, dv).
    """
    dv = lim.shape[0]
    lo = lim[:, 0]
    hi = lim[:, 1]
    u = rng.random((S, dv))
    return lo + (hi - lo) * u


def _evaluate_cost(cost_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
    """
    Evaluate cost for a population X of shape (S, dv).
    cost_fn should accept (S, dv) and return (S,) or (S,1).
    """
    Pi = np.asarray(cost_fn(X), dtype=float)
    if Pi.ndim == 2 and Pi.shape[1] == 1:
        Pi = Pi.ravel()
    if Pi.ndim != 1 or Pi.shape[0] != X.shape[0]:
        raise ValueError(f"cost_fn must return shape (S,) (or (S,1)). Got {Pi.shape}.")
    return Pi


def _nearest_neighbor_pairs(P: int, K: int) -> np.ndarray:
    """
    Produce K pairs (i, i+1) cycling through elites 0..P-1.
    If P < 2, no valid pairs.
    Returns shape (K, 2).
    """
    if P < 2:
        raise ValueError("Need at least P>=2 elites for nearest-neighbor breeding.")
    pairs = []
    for t in range(K):
        i = t % (P - 1)
        pairs.append((i, i + 1))
    return np.asarray(pairs, dtype=int)


def _uniform_crossover(
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
) -> np.ndarray:
    """
    Uniform crossover: each gene independently chosen from parent A or B.
    parent_a, parent_b: (dv,)
    returns child: (dv,)
    """
    mask = rng.random(parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child


def _mutate_gaussian(
    rng: np.random.Generator,
    X: np.ndarray,
    lim: np.ndarray,
    mutation_rate: float,
    sigma_frac: float,
) -> np.ndarray:
    """
    Gaussian mutation with clipping to bounds.
    mutation_rate: probability of mutating each gene
    sigma_frac: std dev as fraction of (upper-lower)
    """
    if mutation_rate <= 0.0:
        return X

    dv = lim.shape[0]
    lo = lim[:, 0]
    hi = lim[:, 1]
    span = hi - lo

    X_new = X.copy()
    mask = rng.random(X_new.shape) < mutation_rate
    noise = rng.normal(loc=0.0, scale=sigma_frac, size=X_new.shape) * span
    X_new[mask] = X_new[mask] + noise[mask]
    # clip to bounds
    X_new = np.clip(X_new, lo, hi)
    return X_new


def genetic_algorithm(
    cost_fn: Callable[[np.ndarray], np.ndarray],
    *,
    S: int,
    P: int,
    K: int,
    TOL: float,
    G: int,
    dv: int,
    lim: np.ndarray,
    seed: Optional[int] = None,
    mutation_rate: float = 0.0,
    sigma_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run GA.

    Parameters (match assignment):
      S, P, K, TOL, G, dv, lim
    Extras:
      seed: reproducibility
      mutation_rate: set 0.0 for "no mutations" (Project 1 debugging)
      sigma_frac: mutation magnitude

    Returns:
      Pi     : (G, S) sorted costs per generation
      Pi_min : (G,) best cost per generation
      Pi_avg : (G,) mean cost per generation
      Lambda : (S, dv) final generation design strings (unsorted)
    """
    if S <= 0:
        raise ValueError("S must be positive.")
    if dv <= 0:
        raise ValueError("dv must be positive.")
    if P <= 0 or P > S:
        raise ValueError("P must satisfy 1 <= P <= S.")
    if K < 0 or (P + K) > S:
        raise ValueError("K must satisfy K>=0 and P+K <= S.")
    if G <= 0:
        raise ValueError("G must be positive.")
    if TOL <= 0:
        raise ValueError("TOL must be positive.")

    lim = _validate_limits(lim, dv)
    rng = np.random.default_rng(seed)

    # Initial population
    Lambda = _sample_uniform(rng, S, lim)

    Pi_hist = np.zeros((G, S), dtype=float)
    Pi_min = np.zeros(G, dtype=float)
    Pi_avg = np.zeros(G, dtype=float)

    pairs = _nearest_neighbor_pairs(P, K) if K > 0 else np.zeros((0, 2), dtype=int)

    for g in range(G):
        # Evaluate and rank
        Pi_vals = _evaluate_cost(cost_fn, Lambda)
        order = np.argsort(Pi_vals)  # ascending (best first)

        Pi_sorted = Pi_vals[order]
        Pi_hist[g, :] = Pi_sorted
        Pi_min[g] = Pi_sorted[0]
        Pi_avg[g] = Pi_sorted.mean()

        # Stopping criterion (assignment: acceptable cost threshold)
        if Pi_min[g] <= TOL:
            # Trim history to actual generations run
            Pi_hist = Pi_hist[: g + 1, :]
            Pi_min = Pi_min[: g + 1]
            Pi_avg = Pi_avg[: g + 1]
            return Pi_hist, Pi_min, Pi_avg, Lambda

        # Elitism: keep top P designs (as "parents")
        elites = Lambda[order[:P], :]  # (P, dv)

        # Breed K offspring using nearest-neighbor pairing among elites
        offspring = np.zeros((K, dv), dtype=float)
        for t in range(K):
            i, j = pairs[t]
            offspring[t, :] = _uniform_crossover(rng, elites[i, :], elites[j, :])

        # Fill the rest with fresh random designs to maintain population size
        R = S - (P + K)
        newcomers = _sample_uniform(rng, R, lim) if R > 0 else np.zeros((0, dv), dtype=float)

        # Next generation (unsorted)
        Lambda = np.vstack([elites, offspring, newcomers])

        # Optional mutation (disabled for Project 1 debugging)
        Lambda = _mutate_gaussian(rng, Lambda, lim, mutation_rate, sigma_frac)

    return Pi_hist, Pi_min, Pi_avg, Lambda
