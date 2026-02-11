# me144_toolbox/optimization/ga_phi_psi.py
"""
ME144/244 — Zohdi-style GA using component-wise phi/psi crossover.

Implements the textbook GA crossover:

For nearest-ranked parents (lambda^i, lambda^{i+1}):

child #1: lambda^{i,i+1,1} = Phi ∘ lambda^i + (1-Phi) ∘ lambda^{i+1}
child #2: lambda^{i,i+1,2} = Psi ∘ lambda^i + (1-Psi) ∘ lambda^{i+1}

where Phi and Psi are vectors with components in [0,1], sampled independently,
and ∘ denotes component-wise multiplication.

This module keeps your original GA intact by providing a separate implementation.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
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
    dv = lim.shape[0]
    lo = lim[:, 0]
    hi = lim[:, 1]
    u = rng.random((S, dv))
    return lo + (hi - lo) * u


def _evaluate_cost(cost_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
    Pi = np.asarray(cost_fn(X), dtype=float)
    if Pi.ndim == 2 and Pi.shape[1] == 1:
        Pi = Pi.ravel()
    if Pi.ndim != 1 or Pi.shape[0] != X.shape[0]:
        raise ValueError(f"cost_fn must return shape (S,) or (S,1). Got {Pi.shape}.")
    return Pi


def _nearest_neighbor_pairs(P: int, num_pairs: int) -> np.ndarray:
    """
    Produce num_pairs nearest-neighbor pairs among elites:
    (0,1), (1,2), ... cycling.
    """
    if P < 2:
        raise ValueError("Need at least P>=2 elites for nearest-neighbor mating.")
    pairs = []
    for t in range(num_pairs):
        i = t % (P - 1)
        pairs.append((i, i + 1))
    return np.asarray(pairs, dtype=int)


def _phi_psi_children(
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zohdi STEP 4: produce two offspring from (parent_a, parent_b)
    using component-wise Phi and Psi in [0,1]^dv.

    child1 = Phi ∘ parent_a + (1-Phi) ∘ parent_b
    child2 = Psi ∘ parent_a + (1-Psi) ∘ parent_b
    """
    dv = parent_a.shape[0]

    Phi = rng.random(dv)          # components in [0,1]
    Psi = rng.random(dv)          # components in [0,1]

    child1 = Phi * parent_a + (1.0 - Phi) * parent_b
    child2 = Psi * parent_a + (1.0 - Psi) * parent_b
    return child1, child2


def genetic_algorithm_phi_psi(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zohdi-style GA with phi/psi crossover.

    Parameters (match Project 1 GA interface):
      S: population size
      P: number of elites (parents preserved)
      K: number of offspring produced each generation (total offspring count)
      TOL: stop if best cost <= TOL
      G: maximum generations
      dv: number of design variables per string
      lim: (dv,2) bounds for each variable
      seed: reproducibility

    Returns (as recommended in the prompt):
      Pi     : (G_run, S) ranked costs each generation
      Pi_min : (G_run,) best cost each generation
      Pi_avg : (G_run,) mean cost each generation
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
    if P < 2 and K > 0:
        raise ValueError("Need P>=2 for nearest-neighbor phi/psi mating.")

    lim = _validate_limits(lim, dv)
    rng = np.random.default_rng(seed)

    # STEP 1: initial population
    Lambda = _sample_uniform(rng, S, lim)  # (S, dv)

    # Storage
    Pi_hist = np.zeros((G, S), dtype=float)
    Pi_min = np.zeros(G, dtype=float)
    Pi_avg = np.zeros(G, dtype=float)

    # Each nearest-neighbor pair can produce 2 offspring.
    # We generate ceil(K/2) pairs and then truncate to K offspring total.
    num_pairs = (K + 1) // 2 if K > 0 else 0
    pairs = _nearest_neighbor_pairs(P, num_pairs) if num_pairs > 0 else np.zeros((0, 2), dtype=int)

    for g in range(G):
        # STEP 2: compute fitness
        costs = _evaluate_cost(cost_fn, Lambda)

        # STEP 3: rank
        order = np.argsort(costs)
        costs_sorted = costs[order]

        Pi_hist[g, :] = costs_sorted
        Pi_min[g] = costs_sorted[0]
        Pi_avg[g] = costs_sorted.mean()

        # Stop if acceptable threshold reached
        if Pi_min[g] <= TOL:
            Pi_hist = Pi_hist[: g + 1, :]
            Pi_min = Pi_min[: g + 1]
            Pi_avg = Pi_avg[: g + 1]
            return Pi_hist, Pi_min, Pi_avg, Lambda

        # STEP 5: keep top P parents
        elites = Lambda[order[:P], :]  # (P, dv)

        # STEP 4: mate nearest ranked pairs and produce offspring using phi/psi
        '''
        offspring_list = []
        for (i, j) in pairs:
            c1, c2 = _phi_psi_children(rng, elites[i, :], elites[j, :])
            offspring_list.append(c1)
            if len(offspring_list) < K:
                offspring_list.append(c2)
            if len(offspring_list) >= K:
                break
        '''
        offspring_list = []
        for i in range(0, P, 2):
            pa = elites[i, :]
            pb = elites[i + 1, :]

            Phi = rng.random(dv)
            Psi = rng.random(dv)

            child1 = Phi * pa + (1.0 - Phi) * pb
            child2 = Psi * pa + (1.0 - Psi) * pb

            offspring_list.append(child1)
            if len(offspring_list) < K:
                offspring_list.append(child2)
            if len(offspring_list) >= K:
                break
            

        offspring = np.array(offspring_list, dtype=float).reshape(-1, dv) if K > 0 else np.zeros((0, dv), dtype=float)

        # STEP 6: add M new random strings to maintain population size S
        R = S - (P + offspring.shape[0])
        newcomers = _sample_uniform(rng, R, lim) if R > 0 else np.zeros((0, dv), dtype=float)

        # Next generation (unsorted)
        Lambda = np.vstack([elites, offspring, newcomers])

    return Pi_hist, Pi_min, Pi_avg, Lambda
