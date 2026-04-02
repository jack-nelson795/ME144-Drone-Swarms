from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .config import CONFIG, Project5Config
from .hashin_shtrikman import clip_design, evaluate_design, random_design, total_cost


CaseName = Literal["A", "B", "C"]


@dataclass
class GARunResult:
    case: CaseName
    seed: int
    population: np.ndarray
    costs: np.ndarray
    history_best: np.ndarray
    history_top10_mean: np.ndarray
    sorted_costs_by_generation: np.ndarray


def evaluate_population(population: np.ndarray, cfg: Project5Config = CONFIG) -> np.ndarray:
    return np.array([total_cost(individual, cfg) for individual in population], dtype=float)


def initialize_population(rng: np.random.Generator, cfg: Project5Config = CONFIG) -> np.ndarray:
    population = np.zeros((cfg.S, cfg.dv), dtype=float)
    for i in range(cfg.S):
        population[i, :] = random_design(rng, cfg)
    return population


def breed_child(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    case: CaseName,
    cfg: Project5Config = CONFIG,
) -> np.ndarray:
    if case == "C":
        phi = rng.uniform(cfg.mutation_a, cfg.mutation_b)
    else:
        phi = rng.random()
    child = phi * parent_a + (1.0 - phi) * parent_b
    return clip_design(child, cfg)


def run_case(
    case: CaseName,
    seed: int | None = None,
    cfg: Project5Config = CONFIG,
    verbose: bool = False,
    progress_every: int = 250,
) -> GARunResult:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    if verbose:
        print(f"[Case {case}] Initializing population...")

    population = initialize_population(rng, cfg)
    costs = evaluate_population(population, cfg)

    history_best = np.zeros(cfg.G, dtype=float)
    history_top10_mean = np.zeros(cfg.G, dtype=float)
    sorted_costs_by_generation = np.zeros((cfg.G, cfg.S), dtype=float)

    for g in range(cfg.G):
        order = np.argsort(costs)
        population = population[order]
        costs = costs[order]
        history_best[g] = costs[0]
        history_top10_mean[g] = costs[: cfg.P].mean()
        sorted_costs_by_generation[g, :] = costs

        if verbose and (g == 0 or (g + 1) % progress_every == 0 or g == cfg.G - 1):
            print(
                f"[Case {case}] Generation {g + 1}/{cfg.G} | "
                f"best={costs[0]:.6f} | top10_mean={history_top10_mean[g]:.6f}"
            )

        if g == cfg.G - 1:
            break

        next_population = np.zeros_like(population)
        next_costs = np.zeros_like(costs)

        if case in ("A", "C"):
            next_population[: cfg.P, :] = population[: cfg.P, :]
            next_costs[: cfg.P] = costs[: cfg.P]
            child_start = cfg.P
            random_start = cfg.P + cfg.K
        elif case == "B":
            child_start = 0
            random_start = cfg.K
        else:
            raise ValueError(f"Unsupported case {case}")

        for p in range(0, cfg.P, 2):
            parent_a = population[p, :]
            parent_b = population[p + 1, :]
            child_one = breed_child(parent_a, parent_b, rng, case, cfg)
            child_two = breed_child(parent_a, parent_b, rng, case, cfg)
            next_population[child_start + p, :] = child_one
            next_population[child_start + p + 1, :] = child_two
            next_costs[child_start + p] = total_cost(child_one, cfg)
            next_costs[child_start + p + 1] = total_cost(child_two, cfg)

        for i in range(random_start, cfg.S):
            next_population[i, :] = random_design(rng, cfg)
            next_costs[i] = total_cost(next_population[i, :], cfg)

        population = next_population
        costs = next_costs

    if verbose:
        print(f"[Case {case}] Completed. Final best cost = {costs.min():.6f}")

    return GARunResult(
        case=case,
        seed=int(cfg.seed if seed is None else seed),
        population=population,
        costs=costs,
        history_best=history_best,
        history_top10_mean=history_top10_mean,
        sorted_costs_by_generation=sorted_costs_by_generation,
    )


def summarize_top_designs(result: GARunResult, top_n: int = 4, cfg: Project5Config = CONFIG) -> list[dict]:
    rows: list[dict] = []
    for rank in range(top_n):
        design = result.population[rank, :]
        evaluation = evaluate_design(design, cfg)
        rows.append(
            {
                "rank": rank + 1,
                "k2_GPa": design[0] / 1.0e9,
                "mu2_GPa": design[1] / 1.0e9,
                "sig2E_1e7_Spm": design[2] / 1.0e7,
                "K2_WpmK": design[3],
                "v2": design[4],
                "Pi_total": result.costs[rank],
                "k_eff_GPa": evaluation["effective_properties"]["k_eff"] / 1.0e9,
                "mu_eff_GPa": evaluation["effective_properties"]["mu_eff"] / 1.0e9,
                "sigE_eff_1e7_Spm": evaluation["effective_properties"]["sigE_eff"] / 1.0e7,
                "K_eff_WpmK": evaluation["effective_properties"]["K_eff"],
            }
        )
    return rows
