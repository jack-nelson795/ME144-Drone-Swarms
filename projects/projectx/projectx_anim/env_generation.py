from __future__ import annotations

from typing import Optional

import numpy as np

from . import config


def generate_random_targets_and_obstacles(
    n_targets: Optional[int] = None,
    n_obstacles: Optional[int] = None,
    min_distance: float = 4.0,
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate random non-intersecting target and obstacle positions.
    Uses NUM_TARGETS, NUM_OBSTACLES, and RANDOM_SEED if not specified.
    All at z=0 (ground level).

    Parameters
    ----------
    n_targets : int
        Number of target zones
    n_obstacles : int
        Number of obstacles
    min_distance : float
        Minimum separation between any two objects
    seed : Optional[int]
        Random seed (None for random)

    Returns
    -------
    targets : list of np.ndarray
    obstacles : list of dict
    """
    # Use config defaults if not specified
    if n_targets is None:
        n_targets = config.NUM_TARGETS
    if n_obstacles is None:
        n_obstacles = config.NUM_OBSTACLES
    if seed is None:
        seed = config.RANDOM_SEED

    np.random.seed(seed)

    targets: list[np.ndarray] = []
    obstacles: list[dict] = []

    # Track all placed objects for collision checking (position + kind)
    placed: list[tuple[np.ndarray, str]] = []

    # Distance rules:
    # - Targets: stay well-separated from each other.
    # - Obstacles: can be denser.
    # - Target/obstacle separation is relaxed to avoid the “targets fill the field
    #   then obstacles get pushed to edges” artifact.
    obstacle_min_distance = min_distance * 0.5  # obstacle-obstacle
    target_obstacle_min_distance = min_distance * 0.5  # target-obstacle

    def _build_spawn_order(n_t: int, n_o: int) -> list[str]:
        """Build a balanced spawn order to avoid clustering artifacts.

        Uses a weighted round-robin schedule with a small anti-streak rule so
        we don't place long runs of only targets or only obstacles.
        """
        total = n_t + n_o
        if total <= 0:
            return []

        weights = {
            'target': int(max(0, n_t)),
            'obstacle': int(max(0, n_o)),
        }
        remaining = weights.copy()
        current = {'target': 0, 'obstacle': 0}

        order: list[str] = []
        last_kind: str | None = None
        run_len = 0
        max_run = 2  # With unequal counts, strict alternation isn't always possible.

        for _ in range(total):
            for k in current:
                current[k] += weights[k]

            candidates = [k for k in ('target', 'obstacle') if remaining[k] > 0]
            if not candidates:
                break

            # Sort by (score, random_tiebreak) descending
            candidates_sorted = sorted(
                candidates,
                key=lambda k: (current[k], np.random.random()),
                reverse=True,
            )

            pick: str | None = None
            for k in candidates_sorted:
                if last_kind == k and run_len >= max_run:
                    other = 'obstacle' if k == 'target' else 'target'
                    if remaining.get(other, 0) > 0:
                        continue
                pick = k
                break

            if pick is None:
                pick = candidates_sorted[0]

            order.append(pick)
            remaining[pick] -= 1
            current[pick] -= total

            if pick == last_kind:
                run_len += 1
            else:
                last_kind = pick
                run_len = 1

        return order

    spawn_order = _build_spawn_order(n_targets, n_obstacles)

    for kind in spawn_order:
        # Rejection sampling gets harder as we place more objects.
        # Increase attempts with crowding so we are less likely to silently
        # return far fewer objects than requested.
        max_attempts = 200 + 10 * len(placed)
        for _ in range(max_attempts):
            if kind == 'target':
                pos = np.array([
                    np.random.uniform(15, 85),
                    np.random.uniform(10, 90),
                    0.0,
                ])
            else:
                pos = np.array([
                    np.random.uniform(10, 90),
                    np.random.uniform(10, 90),
                    0.0,
                ])

            valid = True
            for existing_pos, existing_kind in placed:
                dist = np.linalg.norm(pos[:2] - existing_pos[:2])

                if kind == existing_kind == 'target':
                    required_dist = min_distance
                elif kind == existing_kind == 'obstacle':
                    required_dist = obstacle_min_distance
                else:
                    required_dist = target_obstacle_min_distance

                if dist < required_dist:
                    valid = False
                    break

            if not valid:
                continue

            if kind == 'target':
                targets.append(pos)
                placed.append((pos, 'target'))
            else:
                obstacles.append({
                    'center': pos,
                    'size': 3.0,
                    'radius': 1.5,
                })
                placed.append((pos, 'obstacle'))
            break

    return targets, obstacles
