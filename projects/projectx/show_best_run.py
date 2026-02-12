#!/usr/bin/env python
"""
Display the best drone swarm run with proper 3D visualization.
Shows targets (green cubes), obstacles (blue cubes), and final drone positions.
Random field layout with drones spawning at elevated base location.
"""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from me144_toolbox.utils.animation_3d import plot_swarm_3d_static

# Configuration
BOUNDS_3D = (0, 100, 0, 100, 0, 50)
BASE_POS = np.array([50.0, 50.0, 20.0])  # Center, elevated
N_DRONES = 12

# Random seed for reproducibility
np.random.seed(42)

def generate_random_targets_and_obstacles(
    n_targets: int = 15,
    n_obstacles: int = 30,
    min_distance: float = 8.0,
    seed: int = 42,
) -> tuple:
    """Generate random non-intersecting target and obstacle positions."""
    np.random.seed(seed)

    targets = []
    obstacles = []
    placed = []  # list of (pos, kind)
    obstacle_min_distance = min_distance * 0.5
    target_obstacle_min_distance = min_distance * 0.5

    def _build_spawn_order(n_t: int, n_o: int) -> list[str]:
        total = n_t + n_o
        weights = {'target': int(n_t), 'obstacle': int(n_o)}
        remaining = weights.copy()
        current = {'target': 0, 'obstacle': 0}
        order = []
        last_kind = None
        run_len = 0
        max_run = 2

        for _ in range(total):
            for k in current:
                current[k] += weights[k]

            candidates = [k for k in ('target', 'obstacle') if remaining[k] > 0]
            if not candidates:
                break

            candidates_sorted = sorted(
                candidates,
                key=lambda k: (current[k], np.random.random()),
                reverse=True,
            )

            pick = None
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

    for kind in _build_spawn_order(n_targets, n_obstacles):
        max_attempts = 200 + 10 * len(placed)
        for _ in range(max_attempts):
            if kind == 'target':
                pos = np.array([
                    np.random.uniform(15, 85),
                    np.random.uniform(10, 90),
                    0.0
                ])
            else:
                pos = np.array([
                    np.random.uniform(10, 90),
                    np.random.uniform(10, 90),
                    0.0
                ])

            valid = True
            for existing_pos, existing_kind in placed:
                dist = np.linalg.norm(pos[:2] - existing_pos[:2])
                if kind == existing_kind == 'target':
                    required = min_distance
                elif kind == existing_kind == 'obstacle':
                    required = obstacle_min_distance
                else:
                    required = target_obstacle_min_distance
                if dist < required:
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
                    'destroyed': False
                })
                placed.append((pos, 'obstacle'))
            break
    
    return targets, obstacles

# Generate field
TARGET_ZONES, OBSTACLES = generate_random_targets_and_obstacles(
    n_targets=15,
    n_obstacles=30,
    min_distance=8.0,
    seed=42
)

# Create example: all drones at base (starting configuration)
positions_start = np.tile(BASE_POS, (N_DRONES, 1))
alive_start = np.ones(N_DRONES, dtype=bool)

print(f"Rendering 3D scene with:")
print(f"  - Base position (red cube): {BASE_POS}")
print(f"  - Target zones (green cubes): {len(TARGET_ZONES)}")
print(f"  - Obstacles (blue cubes): {len(OBSTACLES)}")
print(f"  - Drones at base: {N_DRONES}")
print()

# Display static plot
plot_swarm_3d_static(
    positions_start,
    np.array(TARGET_ZONES),
    alive_start,
    obstacles=OBSTACLES,
    bounds=BOUNDS_3D,
    base_pos=BASE_POS,
    title="Hostile Drone Incursion Scenario\n(Initial Configuration at Base)",
)

print("✓ 3D visualization complete!")
print("  The scene shows:")
print("  - RED CUBE: Drone base (elevated at x=50, y=100, z=20)")
print("  - BLUE CUBES: Obstacles to be destroyed")
print("  - GREEN CUBES: Target zones to reach")
print("  - BLUE DOTS: Drones starting at base")

