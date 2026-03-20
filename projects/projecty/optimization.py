from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Callable
import numpy as np

try:
    from .config import DEFAULT_MATERIAL, SimConfig
    from .design import DroneDesign
    from .flight import FlightResult, simulate_hostile_course
    from .geometry import build_voxel_drone
except ImportError:
    from config import DEFAULT_MATERIAL, SimConfig
    from design import DroneDesign
    from flight import FlightResult, simulate_hostile_course
    from geometry import build_voxel_drone


def _evaluate_candidate(args: tuple[int, int, DroneDesign, SimConfig]) -> tuple[int, float, DroneDesign, FlightResult]:
    generation, candidate_index, design, config = args
    drone = build_voxel_drone(design, DEFAULT_MATERIAL, config.voxel_resolution)
    result = simulate_hostile_course(drone, config, design_name=f"g{generation}_c{candidate_index}")
    return candidate_index, result.score, design, result


def optimize_design(
    config: SimConfig,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[DroneDesign, FlightResult, list[dict[str, float]], list[dict[str, object]]]:
    rng = np.random.default_rng(config.random_seed)
    population = [DroneDesign().clipped()]
    while len(population) < config.population_size:
        population.append(population[0].mutate(rng, scale=0.18))

    history: list[dict[str, float]] = []
    generation_bests: list[dict[str, object]] = []
    best_design = population[0]
    best_result: FlightResult | None = None

    for generation in range(config.optimizer_generations):
        if progress_callback is not None:
            progress_callback(f"Generation {generation + 1}/{config.optimizer_generations}: evaluating population", generation, -1)
        scored: list[tuple[float, DroneDesign, FlightResult]] = []
        candidate_jobs = [(generation, candidate_index, design, config) for candidate_index, design in enumerate(population)]
        if config.parallel_workers > 1:
            with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
                future_map = {executor.submit(_evaluate_candidate, job): job[1] for job in candidate_jobs}
                completed_count = 0
                for future in as_completed(future_map):
                    candidate_index, score, design, result = future.result()
                    scored.append((score, design, result))
                    completed_count += 1
                    if progress_callback is not None:
                        progress_callback(
                            f"Generation {generation + 1}/{config.optimizer_generations}, candidate {completed_count}/{len(population)} complete",
                            generation,
                            completed_count - 1,
                        )
        else:
            for candidate_index, design in enumerate(population):
                _, score, design_eval, result = _evaluate_candidate((generation, candidate_index, design, config))
                scored.append((score, design_eval, result))
                if progress_callback is not None:
                    progress_callback(
                        f"Generation {generation + 1}/{config.optimizer_generations}, candidate {candidate_index + 1}/{len(population)} complete",
                        generation,
                        candidate_index,
                    )

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_design_gen, best_result_gen = scored[0]
        avg_score = float(np.mean([item[0] for item in scored]))
        history.append(
            {
                "generation": float(generation),
                "best_score": float(best_score),
                "avg_score": avg_score,
                "progress": float(best_result_gen.progress),
                "max_stress": float(best_result_gen.max_stress),
            }
        )
        generation_bests.append(
            {
                "generation": int(generation),
                "design": best_design_gen,
                "score": float(best_score),
                "progress": float(best_result_gen.progress),
                "max_stress": float(best_result_gen.max_stress),
                "survived": bool(best_result_gen.survived),
            }
        )
        if progress_callback is not None:
            progress_callback(
                f"Generation {generation + 1} best score {best_score:.2f}, progress {best_result_gen.progress:.2f}, survived {best_result_gen.survived}",
                generation,
                len(population),
            )

        if best_result is None or best_score > best_result.score:
            best_design = best_design_gen
            best_result = best_result_gen

        elite_designs = [item[1] for item in scored[: config.elite_count]]
        population = elite_designs.copy()
        while len(population) < config.population_size:
            parent = elite_designs[len(population) % len(elite_designs)]
            mutate_scale = max(0.05, 0.16 - 0.02 * generation)
            population.append(parent.mutate(rng, scale=mutate_scale))

    assert best_result is not None
    return best_design, best_result, history, generation_bests


def format_design(design: DroneDesign) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(design).items()}
