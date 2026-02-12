from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import Any

import numpy as np
from multiprocessing import get_context
from multiprocessing.pool import Pool

from . import state

# ============================================================================
# COST FUNCTION (CONTROL-LAW OPTIMIZATION - HYBRID)
# ============================================================================

# Track generation for hybrid evaluation strategy
# Keep costs positive so GA doesn't stop immediately on negative values
COST_OFFSET = 1000.0
# Ensure heuristic-only candidates do not dominate selection
HEURISTIC_PENALTY = 200.0
# Simulate a fixed fraction of the population each generation
SIM_FRACTION = 0.30


_MP_CTX = get_context('spawn')
_POOL: Pool | None = None
_POOL_NPROCS: int | None = None


def _close_pool() -> None:
    global _POOL, _POOL_NPROCS
    if _POOL is None:
        return
    try:
        _POOL.close()
        _POOL.join()
    finally:
        _POOL = None
        _POOL_NPROCS = None


@atexit.register
def _cleanup_pool() -> None:
    _close_pool()


def _get_pool(num_workers: int) -> Pool:
    global _POOL, _POOL_NPROCS
    if _POOL is None or _POOL_NPROCS != num_workers:
        _close_pool()
        _POOL = _MP_CTX.Pool(processes=num_workers)
        _POOL_NPROCS = num_workers
    return _POOL


def _evaluate_single_candidate(args: Any) -> float:
    """
    Worker function for parallel fitness evaluation.
    Takes serializable tuple of (candidate_id, weights_flat, run_full_sim, targets, obstacles).
    Returns single cost value.

    This runs in worker processes/threads; keep dependencies local and light.
    """
    from io import StringIO
    import sys

    # Import here to keep worker startup light (and avoid matplotlib imports)
    from .simulator import DroneControlSimulator

    # Suppress all output from worker threads
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        candidate_id, weights_flat, run_full_sim, targets_list, obstacles_list, n_drones, n_control_weights, n_targets = args

        # Reconstruct targets from serialized form
        targets = np.array(targets_list, dtype=float)
        obstacles = [dict(obs) for obs in obstacles_list]

        weights_reshaped = weights_flat.reshape(n_drones, n_control_weights)
        weights_reshaped = np.clip(weights_reshaped, 0.0, 1.0)

        if run_full_sim:
            # Full 1000-step simulation
            simulator = DroneControlSimulator(weights_reshaped, obstacles, targets)
            history = simulator.run(max_steps=1000, sleep_on_completion=False)

            n_targets_visited = np.sum(history['targets_visited'][-1])
            n_alive = np.sum(history['alive'][-1])
            n_dead = n_drones - n_alive

            cost = (n_targets - n_targets_visited) * 50.0
            cost += n_dead * 50.0
            cost -= n_alive * 10.0

            # Time-to-completion penalty (earlier is better)
            targets_visited_hist = history['targets_visited']
            all_visited_steps = np.where(np.all(targets_visited_hist, axis=1))[0]
            if all_visited_steps.size > 0:
                t_complete = all_visited_steps[0] * simulator.dt
                cost += 2.0 * t_complete
            else:
                cost += 2.0 * (len(targets_visited_hist) * simulator.dt)

            # Continuous progress penalty: integral of remaining targets over time
            remaining = n_targets - np.sum(targets_visited_hist, axis=1)
            cost += 1.0 * np.sum(remaining) * simulator.dt

            # Continuous terms: distance to remaining targets and obstacle proximity
            final_pos = history['positions'][-1]
            alive_mask = history['alive'][-1]
            alive_pos = final_pos[alive_mask]
            if alive_pos.size > 0:
                unvisited = ~history['targets_visited'][-1]
                if np.any(unvisited):
                    unvisited_targets = targets[unvisited]
                    # Sum of min distances from drones to each unvisited target
                    dists = np.linalg.norm(alive_pos[:, None, :] - unvisited_targets[None, :, :], axis=2)
                    cost += 2.0 * np.sum(np.min(dists, axis=0))

                # Obstacle proximity penalty (continuous)
                if obstacles:
                    centers = np.array([obs['center'] for obs in obstacles], dtype=float)
                    radii = np.array([obs.get('radius', 0.0) for obs in obstacles], dtype=float)
                    d_obs = np.linalg.norm(alive_pos[:, None, :] - centers[None, :, :], axis=2)
                    # Penalize if within (radius + 2)
                    margin = radii[None, :] + 2.0
                    violations = np.maximum(0.0, margin - d_obs)
                    cost += 5.0 * np.sum(violations)

            cost += COST_OFFSET
            return float(cost)
        else:
            # Quick 200-step simulation
            simulator = DroneControlSimulator(weights_reshaped, obstacles, targets)
            history = simulator.run(max_steps=200, sleep_on_completion=False)

            n_targets_visited = np.sum(history['targets_visited'][-1])
            n_alive = np.sum(history['alive'][-1])
            n_dead = n_drones - n_alive

            cost = (n_targets - n_targets_visited) * 50.0
            cost += n_dead * 25.0
            cost -= n_alive * 5.0

            # Time-to-completion penalty (scaled down for short sim)
            targets_visited_hist = history['targets_visited']
            all_visited_steps = np.where(np.all(targets_visited_hist, axis=1))[0]
            if all_visited_steps.size > 0:
                t_complete = all_visited_steps[0] * simulator.dt
                cost += 1.0 * t_complete
            else:
                cost += 1.0 * (len(targets_visited_hist) * simulator.dt)

            # Continuous progress penalty (scaled down for short sim)
            remaining = n_targets - np.sum(targets_visited_hist, axis=1)
            cost += 0.5 * np.sum(remaining) * simulator.dt

            # Continuous terms (scaled down for short sim)
            final_pos = history['positions'][-1]
            alive_mask = history['alive'][-1]
            alive_pos = final_pos[alive_mask]
            if alive_pos.size > 0:
                unvisited = ~history['targets_visited'][-1]
                if np.any(unvisited):
                    unvisited_targets = targets[unvisited]
                    dists = np.linalg.norm(alive_pos[:, None, :] - unvisited_targets[None, :, :], axis=2)
                    cost += 1.0 * np.sum(np.min(dists, axis=0))

                if obstacles:
                    centers = np.array([obs['center'] for obs in obstacles], dtype=float)
                    radii = np.array([obs.get('radius', 0.0) for obs in obstacles], dtype=float)
                    d_obs = np.linalg.norm(alive_pos[:, None, :] - centers[None, :, :], axis=2)
                    margin = radii[None, :] + 2.0
                    violations = np.maximum(0.0, margin - d_obs)
                    cost += 2.0 * np.sum(violations)

            cost += COST_OFFSET
            return float(cost)
    except Exception as e:
        return 1000.0  # Penalty for failed evaluation
    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def swarm_control_cost(control_weights: np.ndarray, run_simulation: bool = False) -> np.ndarray:
    """
    Parallel evaluation of control law weights using multiprocessing.

    Parameters
    ----------
    control_weights : (pop_size, DV) or (DV,)
        Control weights for each drone
    run_simulation : bool
        If True, run full 1000-step simulation on all candidates

    Returns
    -------
    costs : (pop_size, 1)
        Cost for each candidate (lower is better)
    """
    control_weights = np.asarray(control_weights, dtype=float)

    if control_weights.ndim == 1:
        control_weights = control_weights.reshape(1, -1)

    pop_size = control_weights.shape[0]

    # Quick heuristic for ranking (fast, always computed)
    heuristics = np.zeros(pop_size, dtype=float)
    for s in range(pop_size):
        weights = np.clip(control_weights[s], 0.0, 1.0)
        weights_reshaped = weights.reshape(state.N_DRONES, state.N_CONTROL_WEIGHTS)
        # Simple heuristic: encourage obstacle avoidance + separation, lightly penalize large weights
        w1 = weights_reshaped[:, 0]
        w2 = weights_reshaped[:, 1]
        w3 = weights_reshaped[:, 2]
        w4 = weights_reshaped[:, 3]
        heuristics[s] = -(1.0 * np.mean(w1) + 3.0 * np.mean(w2) + 4.0 * np.mean(w3) + 0.1 * np.mean(w4))
        # Penalize large weights and imbalance to increase spread
        heuristics[s] += 0.2 * np.mean(weights_reshaped)
        heuristics[s] += 0.4 * np.std(weights_reshaped)
        heuristics[s] += 0.2 * np.mean(weights_reshaped ** 2)
        heuristics[s] += COST_OFFSET

    # Decide which candidates to simulate
    if run_simulation:
        eval_mask = np.ones(pop_size, dtype=bool)
    else:
        # Simulate top fraction by heuristic every generation (consistent evaluation policy)
        k = max(1, int(pop_size * SIM_FRACTION))
        best_idx = np.argsort(heuristics)[:k]
        eval_mask = np.zeros(pop_size, dtype=bool)
        eval_mask[best_idx] = True

    # Prepare work packages for parallel evaluation
    work_packages = []
    for s in range(pop_size):
        if not eval_mask[s]:
            continue
        weights_flat = control_weights[s]  # (DV,)

        # Serialize targets and obstacles
        targets_serialized = [t.tolist() for t in state.TARGET_ZONES]

        work = (
            s,  # candidate_id
            weights_flat,  # weights
            run_simulation,  # run_full_sim
            targets_serialized,  # targets (serialized)
            state.OBSTACLES,  # obstacles (already dicts)
            state.N_DRONES,
            state.N_CONTROL_WEIGHTS,
            state.N_TARGETS,
        )
        work_packages.append(work)

    # Parallel evaluation using multiprocessing with 'spawn' context for Windows
    num_workers = max(2, min(6, mp.cpu_count() - 1))  # Leave one core free
    num_workers = min(num_workers, max(1, len(work_packages)))

    print(f"  [GA] Evaluating {len(work_packages)}/{pop_size} candidates using {num_workers} processes...", flush=True)

    if len(work_packages) == 0:
        costs_list = []
    elif num_workers <= 1:
        costs_list = [_evaluate_single_candidate(w) for w in work_packages]
    else:
        try:
            # Reuse a single pool across calls; spawning every generation is very slow on Windows.
            pool = _get_pool(num_workers)
            costs_list = pool.map(_evaluate_single_candidate, work_packages)
        except Exception as e:
            print(f"  [GA] Multiprocessing failed: {e}, falling back to sequential", flush=True)
            costs_list = [_evaluate_single_candidate(w) for w in work_packages]

    print(f"  [GA] Evaluation complete", flush=True)

    # Fill final cost array: simulated candidates use sim results, others use penalized heuristic
    costs = heuristics.copy() + HEURISTIC_PENALTY
    for idx, work in enumerate(work_packages):
        s = work[0]
        costs[s] = costs_list[idx]

    return costs.reshape(-1, 1)
