# projects/projectx/run_projectx_3d_animation.py
"""
ME144/244 — ProjectX 3D Swarm Animation (Zohdi-Inspired)

Recreates the hostile drone incursion scenario from research:
- Drones start at base
- Swarm sweeps through obstacle field
- Visits all target zones
- GA optimizes the trajectory
- Outputs smooth 3D animation
"""

from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, get_context
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from me144_toolbox.optimization.ga import genetic_algorithm

# IMPORTANT: Import matplotlib ONLY in main process, not in workers
# matplotlib is very slow to import and causes deadlocks in multiprocessing


# ============================================================================
# CONFIGURATION - Set these once and they'll be used everywhere
# ============================================================================
NUM_DRONES = 100           # Number of drones in swarm
NUM_TARGETS = 50          # Number of target zones
NUM_OBSTACLES = 50        # Number of obstacles
RANDOM_SEED = 887215      # Use None for random, or set to integer for reproducibility


# ============================================================================
# ENVIRONMENT CONFIGURATION (Matching Research Image)
# ============================================================================

# 3D Domain
BOUNDS_3D = (0, 100, 0, 100, 0, 50)  # x, y, z
BASE_POS = np.array([50.0, 100.0, 20.0])  # Center at far edge, elevated

# Random generator
# Use the RANDOM_SEED config variable above
np.random.seed(RANDOM_SEED)

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
        n_targets = NUM_TARGETS
    if n_obstacles is None:
        n_obstacles = NUM_OBSTACLES
    if seed is None:
        seed = RANDOM_SEED
    
    np.random.seed(seed)
    
    targets = []
    obstacles = []
    all_positions = []  # Track all placed positions for collision checking
    
    # Generate targets first
    for _ in range(n_targets):
        max_attempts = 100
        for attempt in range(max_attempts):
            pos = np.array([
                np.random.uniform(15, 85),
                np.random.uniform(10, 90),
                0.0
            ])
            
            # Check distance to all existing objects
            valid = True
            for existing_pos in all_positions:
                dist = np.linalg.norm(pos[:2] - existing_pos[:2])
                if dist < min_distance:
                    valid = False
                    break
            
            if valid:
                targets.append(pos)
                all_positions.append(pos)
                break
    
    # Generate obstacles
    # Use relaxed min_distance for obstacles to allow denser placement
    obstacle_min_distance = min_distance * 0.5  # Allow obstacles closer together
    for _ in range(n_obstacles):
        max_attempts = 100
        for attempt in range(max_attempts):
            pos = np.array([
                np.random.uniform(10, 90),
                np.random.uniform(10, 90),
                0.0
            ])
            
            # Check distance to all existing objects
            valid = True
            for i, existing_pos in enumerate(all_positions):
                dist = np.linalg.norm(pos[:2] - existing_pos[:2])
                # Full min_distance from targets, relaxed distance from other obstacles
                is_target = i < len(targets)
                required_dist = min_distance if is_target else obstacle_min_distance
                if dist < required_dist:
                    valid = False
                    break
            
            if valid:
                obstacles.append({
                    'center': pos,
                    'size': 3.0,
                    'radius': 1.5,
                })
                all_positions.append(pos)
                break
    
    return targets, obstacles


# ============================================================================
# GLOBAL INITIALIZATION (only in main process)
# ============================================================================

from typing import List

# Initialize globals from config right away
N_DRONES = NUM_DRONES
SEED = RANDOM_SEED

TARGET_ZONES: np.ndarray = np.array([])
OBSTACLES: List = []
N_TARGETS: int = 0
N_CONTROL_WEIGHTS = 4
DV: int = N_DRONES * 4
LIM: np.ndarray = np.tile([0.0, 1.0], (DV, 1))
S = 75
P = 15
K = 20
G = 50  # Reduced for testing (will increase once working)
OUTPUT_DIR: Path = Path()
FIG_DIR: Path = Path()


def _init_globals():
    """Initialize global variables - called only in main process."""
    global TARGET_ZONES, OBSTACLES, N_TARGETS, N_DRONES, DV, LIM, OUTPUT_DIR, FIG_DIR, SEED
    
    # Set N_DRONES and SEED from config
    N_DRONES = NUM_DRONES
    SEED = RANDOM_SEED
    
    # Generate random seed if not set
    if SEED is None:
        SEED = np.random.randint(0, 1000000)
    
    # Generate field layout using config variables
    TARGET_ZONES, OBSTACLES = generate_random_targets_and_obstacles(
        n_targets=NUM_TARGETS,
        n_obstacles=NUM_OBSTACLES,
        min_distance=8.0,
        seed=SEED
    )
    
    N_TARGETS = len(TARGET_ZONES)
    DV = N_DRONES * N_CONTROL_WEIGHTS
    LIM = np.tile([0.0, 1.0], (DV, 1))
    
    OUTPUT_DIR = Path(__file__).parent / 'output'
    FIG_DIR = OUTPUT_DIR / 'figures'
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# Print environment (only if main process)
import os
_is_main_process = os.getenv('_IS_WORKER_PROCESS') is None

# Only print if this is the main process (not a worker)
# Note: Worker processes will skip this when _IS_WORKER_PROCESS is set
if _is_main_process and __name__ == '__main__':
    print(f"\nEnvironment:")
    print(f"  Base position: {BASE_POS}")
    print(f"  Drones: {NUM_DRONES}")
    print(f"  Target zones: {NUM_TARGETS}")
    print(f"  Obstacles: {NUM_OBSTACLES}")
    print(f"  Design variables (control weights): will be set on init")



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

def _evaluate_single_candidate(args):
    """
    Worker function for parallel fitness evaluation.
    Takes serializable tuple of (candidate_id, weights_flat, run_full_sim, targets, obstacles).
    Returns single cost value.
    
    This runs in worker processes/threads; keep dependencies local and light.
    """
    from io import StringIO
    import sys
    
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
            history = simulator.run(max_steps=1000)
            
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
            history = simulator.run(max_steps=200)
            
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


def swarm_control_cost(control_weights: np.ndarray, run_simulation=False) -> np.ndarray:
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
        weights_reshaped = weights.reshape(N_DRONES, N_CONTROL_WEIGHTS)
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
        targets_serialized = [t.tolist() for t in TARGET_ZONES]

        work = (
            s,  # candidate_id
            weights_flat,  # weights
            run_simulation,  # run_full_sim
            targets_serialized,  # targets (serialized)
            OBSTACLES,  # obstacles (already dicts)
            N_DRONES,
            N_CONTROL_WEIGHTS,
            N_TARGETS
        )
        work_packages.append(work)
    
    # Parallel evaluation using multiprocessing with 'spawn' context for Windows
    num_workers = max(2, min(6, mp.cpu_count() - 1))  # Leave one core free
    
    print(f"  [GA] Evaluating {len(work_packages)}/{pop_size} candidates using {num_workers} processes...", flush=True)
    
    try:
        # Use spawn context for Windows compatibility
        with get_context('spawn').Pool(processes=num_workers) as pool:
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


class DroneControlSimulator:
    """
    Simulates drones using decentralized control laws with local sensing.
    
    Each drone computes forces based on:
      w1: attraction to closest unvisited target
      w2: repulsion from nearby obstacles
      w3: separation from nearby drones
      w4: cohesion with swarm centroid
    """
    
    def __init__(
        self,
        control_weights: np.ndarray,  # (N_drones, 4) - [w1, w2, w3, w4] per drone
        obstacles: list,
        target_zones: np.ndarray,
        dt: float = 0.1,  # Timestep (larger for faster simulation)
    ):
        """
        Parameters
        ----------
        control_weights : (N_drones, 4)
            Control law weights for each drone [w1, w2, w3, w4]
        obstacles : list of dicts with 'center' and 'radius'
        target_zones : array of target positions
        """
        self.control_weights = control_weights  # (N_drones, 4)
        self.obstacles = obstacles
        self.target_zones = np.array(target_zones, dtype=float)
        self.N = len(control_weights)
        
        self.time = 0.0
        self.dt = dt  # Use provided timestep
        self.max_speed = 10.0  # Increased from 5.0 for faster movement
        self.sensor_range = 30.0  # How far drones can sense
        self.target_capture_distance = 5.0
        self.obstacle_danger_distance = 5.0  # Distance at which to start avoiding
        
        # Initialize drone states
        self.positions = np.array([BASE_POS.copy() for _ in range(self.N)])
        self.velocities = np.zeros((self.N, 3))
        self.alive = np.ones(self.N, dtype=bool)
        self.stall_steps = np.zeros(self.N, dtype=int)
        
        # Track target visitation
        self.targets_visited = np.zeros(len(target_zones), dtype=bool)
        self.all_targets_visited = False  # Phase transition flag
        
        # Collision detection tracking
        self.drone_collision_threshold = 2.0
        self.collision_check_interval = 5  # Check every 5 frames for performance
        self.step_count = 0
        
        # History
        self.history_positions = [self.positions.copy()]
        self.history_alive = [self.alive.copy()]
        self.history_targets_visited = [self.targets_visited.copy()]
    
    def _detect_drone_collisions_simple(self) -> None:
        """
        Simple brute-force drone collision detection with early-exit.
        """
        drone_collision_threshold_sq = self.drone_collision_threshold ** 2
        for d1 in range(self.N):
            if not self.alive[d1]:
                continue
            x1, y1, z1 = self.positions[d1]
            for d2 in range(d1 + 1, self.N):
                if not self.alive[d2]:
                    continue
                x2, y2, z2 = self.positions[d2]
                # Early-exit: check manhattan distance first (faster)
                if abs(x1 - x2) > self.drone_collision_threshold or \
                   abs(y1 - y2) > self.drone_collision_threshold or \
                   abs(z1 - z2) > self.drone_collision_threshold:
                    continue
                # Only compute full distance if close in all dimensions
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
                if dist_sq < drone_collision_threshold_sq:
                    self.alive[d1] = False
                    self.alive[d2] = False
    
    def _find_closest_target(self, drone_idx: int) -> tuple:
        """
        Find closest unvisited target.
        
        Returns
        -------
        target_pos : ndarray or None
            Position of closest target, or None if all visited
        distance : float
            Distance to target
        """
        unvisited_mask = ~self.targets_visited
        if not np.any(unvisited_mask):
            return None, float('inf')
        
        unvisited_targets = self.target_zones[unvisited_mask]
        distances = np.linalg.norm(
            unvisited_targets - self.positions[drone_idx], axis=1
        )
        closest_idx_local = np.argmin(distances)
        closest_idx_global = np.where(unvisited_mask)[0][closest_idx_local]
        
        return self.target_zones[closest_idx_global], distances[closest_idx_local]
    
    def _find_nearby_obstacles(self, drone_idx: int) -> list:
        """Find obstacles within sensor range (vectorized)."""
        drone_pos = self.positions[drone_idx]  # (3,)
        nearby = []
        
        # Vectorized distance calculation for all obstacles
        if self.obstacles:
            centers = np.array([obs['center'] for obs in self.obstacles])  # (n_obs, 3)
            dists = np.linalg.norm(centers - drone_pos, axis=1)  # (n_obs,)
            
            # Find indices within sensor range
            within_range = dists < self.sensor_range
            for i, obs in enumerate(self.obstacles):
                if within_range[i]:
                    nearby.append((obs['center'], dists[i]))
        
        return nearby
    
    def _find_nearby_drones(self, drone_idx: int) -> list:
        """Find other alive drones within sensor range."""
        nearby = []
        dx = self.positions[drone_idx, 0]
        dy = self.positions[drone_idx, 1]
        dz = self.positions[drone_idx, 2]
        for d in range(self.N):
            if d == drone_idx or not self.alive[d]:
                continue
            ox, oy, oz = self.positions[d]
            dist = ((dx - ox)**2 + (dy - oy)**2 + (dz - oz)**2) ** 0.5
            if dist < self.sensor_range:
                nearby.append((self.positions[d], dist))
        return nearby
    
    def _compute_control_force(self, drone_idx: int) -> np.ndarray:
        """
        Compute control force for drone using weighted sensory inputs.
        When all targets visited, redirect to base.
        
        Returns
        -------
        force : ndarray, shape (3,)
            Control force direction (normalized)
        """
        w1, w2, w3, w4 = self.control_weights[drone_idx]
        force = np.zeros(3)
        
        # If all targets visited, return to base (override normal behavior)
        if self.all_targets_visited:
            base_dir_dist = BASE_POS - self.positions[drone_idx]
            base_dist = np.linalg.norm(base_dir_dist)
            if base_dist > 1.0:  # Not at base yet
                base_direction = base_dir_dist / (base_dist + 0.01)
                force += 2.0 * base_direction  # Strong attraction to base
                # Still avoid obstacles and other drones
                nearby_obstacles = self._find_nearby_obstacles(drone_idx)
                for obs_pos, obs_dist in nearby_obstacles:
                    if obs_dist < self.obstacle_danger_distance:
                        repulsion_strength = (self.obstacle_danger_distance - obs_dist) / (self.obstacle_danger_distance + 0.01)
                        repulsion_dir = (self.positions[drone_idx] - obs_pos) / (obs_dist + 0.01)
                        force += w2 * repulsion_strength * repulsion_dir
                nearby_drones = self._find_nearby_drones(drone_idx)
                for drone_pos, drone_dist in nearby_drones:
                    if drone_dist < 10.0:
                        separation_strength = (10.0 - drone_dist) / 10.0
                        separation_dir = (self.positions[drone_idx] - drone_pos) / (drone_dist + 0.01)
                        force += w3 * separation_strength * separation_dir
            # Normalize and cap speed
            force_mag = np.linalg.norm(force)
            if force_mag > 0:
                force = force / force_mag
            return force
        
        # w1: Attraction to closest unvisited target
        target_pos, target_dist = self._find_closest_target(drone_idx)
        if target_pos is not None:
            target_direction = (target_pos - self.positions[drone_idx]) / (target_dist + 0.01)
            if target_dist < self.sensor_range:
                force += w1 * target_direction
            else:
                # Weak long-range pull to avoid stalling when no targets are in range
                force += 0.2 * w1 * target_direction
        
        # w2: Repulsion from nearby obstacles
        nearby_obstacles = self._find_nearby_obstacles(drone_idx)
        for obs_pos, obs_dist in nearby_obstacles:
            if obs_dist < self.obstacle_danger_distance:
                # Strong repulsion from close obstacles
                repulsion_strength = (self.obstacle_danger_distance - obs_dist) / (self.obstacle_danger_distance + 0.01)
                repulsion_dir = (self.positions[drone_idx] - obs_pos) / (obs_dist + 0.01)
                force += w2 * repulsion_strength * repulsion_dir
        
        # w3: Separation from nearby drones (avoid collisions)
        nearby_drones = self._find_nearby_drones(drone_idx)
        for drone_pos, drone_dist in nearby_drones:
            if drone_dist < 15.0:  # Larger separation distance - keep drones spread out
                separation_strength = (15.0 - drone_dist) / 15.0
                separation_dir = (self.positions[drone_idx] - drone_pos) / (drone_dist + 0.01)
                force += w3 * separation_strength * separation_dir
        
        # w4: Cohesion with swarm (move toward centroid of alive drones)
        if np.sum(self.alive) > 1:
            alive_positions = self.positions[self.alive == True]  # Get positions of alive drones
            centroid = np.mean(alive_positions, axis=0)
            centroid_direction = (centroid - self.positions[drone_idx]) / (np.linalg.norm(centroid - self.positions[drone_idx]) + 0.01)
            force += w4 * centroid_direction
        
        # Normalize and cap speed
        force_mag = np.linalg.norm(force)
        if force_mag > 0:
            force = force / force_mag
        
        return force
    
    def step(self) -> None:
        """Advance simulation by one timestep."""
        # Update phase: if all targets visited, enable return to base
        if np.all(self.targets_visited):
            self.all_targets_visited = True
        
        # Check if any alive drone is within immunity zone (10 units of base)
        # If so, ALL drones are immune to collisions this step
        alive_indices = np.where(self.alive)[0]
        immunity_active = False
        if len(alive_indices) > 0:
            distances_to_base = np.linalg.norm(self.positions[alive_indices] - BASE_POS, axis=1)
            immunity_active = np.any(distances_to_base < 10.0)
        
        for d in alive_indices:
            # Compute control force
            force = self._compute_control_force(d)

            # Stall recovery: if drone barely moves for a while, add a small nudge
            if not self.all_targets_visited:
                speed = np.linalg.norm(self.velocities[d])
                if speed < 0.05:
                    self.stall_steps[d] += 1
                else:
                    self.stall_steps[d] = 0
                if self.stall_steps[d] > 10:
                    force += 0.5 * (np.random.rand(3) - 0.5)
                    self.stall_steps[d] = 0
            
            # Update velocity with damping (for stability)
            self.velocities[d] = 0.8 * self.velocities[d] + 0.2 * force * self.max_speed
            
            # Check for collision at new position BEFORE updating position
            # This ensures instantaneous death when collision is detected
            new_pos = self.positions[d] + self.velocities[d] * self.dt
            
            collision = False
            if not immunity_active:  # Only check collisions if no drone is near base
                new_x, new_y, new_z = new_pos
                for obs_idx, obs in enumerate(self.obstacles):
                    obs_x, obs_y, obs_z = obs['center']
                    dist_sq = (new_x - obs_x)**2 + (new_y - obs_y)**2 + (new_z - obs_z)**2
                    collision_threshold_sq = obs['radius'] ** 2  # Exact collision with obstacle radius only
                    if dist_sq < collision_threshold_sq:
                        self.alive[d] = False
                        collision = True
                        break
            
            if collision:
                continue  # Drone dies, don't update position
            
            # Safe to update position (no collision)
            self.positions[d] = new_pos
            
            # Boundary conditions (vectorized)
            self.positions[d, 0] = np.clip(self.positions[d, 0], BOUNDS_3D[0], BOUNDS_3D[1])
            self.positions[d, 1] = np.clip(self.positions[d, 1], BOUNDS_3D[2], BOUNDS_3D[3])
            self.positions[d, 2] = np.clip(self.positions[d, 2], BOUNDS_3D[4], BOUNDS_3D[5])
            
            # Check target captures (vectorized distance) - only while not returning
            if not self.all_targets_visited:
                unvisited_targets = self.target_zones[~self.targets_visited]
                if len(unvisited_targets) > 0:
                    distances_to_targets = np.linalg.norm(unvisited_targets - self.positions[d], axis=1)
                    captured_mask = distances_to_targets < self.target_capture_distance
                    if np.any(captured_mask):
                        unvisited_indices = np.where(~self.targets_visited)[0]
                        self.targets_visited[unvisited_indices[captured_mask]] = True
        
        # Check for drone-drone collisions every step after warmup
        # Each drone can only kill once - dead drones can't kill
        # Skip if immunity zone is active (any drone near base)
        if self.time > 5.0 and not immunity_active:
            currently_alive = np.where(self.alive)[0]  # Recompute alive drones
            drone_collision_threshold_sq = 0.5 ** 2  # Very tight collision - nearly exact
            for i, d1 in enumerate(currently_alive):
                # Skip if d1 already died in this collision pass
                if not self.alive[d1]:
                    continue
                x1, y1, z1 = self.positions[d1]
                for d2 in currently_alive[i + 1:]:
                    # Skip if d2 already died in this collision pass
                    if not self.alive[d2]:
                        continue
                    x2, y2, z2 = self.positions[d2]
                    # Quick early-exit: check manhattan distance first (faster)
                    if abs(x1 - x2) > 0.5 or abs(y1 - y2) > 0.5 or abs(z1 - z2) > 0.5:
                        continue
                    # Only compute full distance if close in all dimensions
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
                    if dist_sq < drone_collision_threshold_sq:
                        dist = np.sqrt(dist_sq)
                        print(f"[DRONE COLLISION] Drone {d1} hit Drone {d2} at pos {self.positions[d1]} vs {self.positions[d2]}, dist: {dist:.2f}")
                        self.alive[d1] = False
                        self.alive[d2] = False
                        break  # d1 is now dead, stop checking it against other drones
        
        self.time += self.dt
        self.history_positions.append(self.positions.copy())
        self.history_alive.append(self.alive.copy())
        self.history_targets_visited.append(self.targets_visited.copy())
    
    def run(self, max_steps: int = 3000) -> dict:
        """
        Run simulation until all drones return to base or all drones dead.
        First visits all targets, then returns to base.
        
        Parameters
        ----------
        max_steps : int
            Maximum timesteps to run (default 3000 steps = 300 seconds at dt=0.1)
        
        Returns
        -------
        dict with 'positions', 'alive', 'targets_visited'
        """
        for step in range(max_steps):
            self.step()
            
            # Stop if all drones are dead
            if not np.any(self.alive):
                break
            
            # Stop if all targets visited AND all alive drones returned near base
            if self.all_targets_visited and np.any(self.alive):
                alive_positions = self.positions[self.alive == True]
                deltas = np.abs(alive_positions - BASE_POS)
                if np.all(deltas[:, 0] <= 20.0) and np.all(deltas[:, 1] <= 20.0) and np.all(deltas[:, 2] <= 20.0):
                    # Wait 1 real-time second before ending
                    time.sleep(1.0)
                    break
        
        return {
            'positions': np.array(self.history_positions),
            'alive': np.array(self.history_alive),
            'targets_visited': np.array(self.history_targets_visited),
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete pipeline."""
    # Initialize globals in main process
    _init_globals()
    
    print("\n" + "="*70)
    print("ME144/244 ProjectX: 3D Drone Swarm Incursion Simulation")
    print("(Control-Law Optimization - Emergent Swarm Behavior)")
    print("="*70)
    print(f"\nRandom Seed: {SEED}")
    
    # ========================================================================
    # STEP 1: Optimize drone control law weights using GA
    # ========================================================================
    
    print("\nStep 1: Optimizing control law weights...")
    
    Pi, Pi_min, Pi_avg, Lambda = genetic_algorithm(
        swarm_control_cost,  # Multiprocessing cost evaluation
        S=S,
        P=P,
        K=K,
        TOL=1e-2,
        G=G,
        dv=DV,
        lim=LIM,
        seed=SEED,
        mutation_rate=0.15,
        sigma_frac=0.10,
    )
    
    # Get best solution with full simulation for final evaluation
    costs_final = swarm_control_cost(Lambda, run_simulation=True).ravel()
    best_idx = np.argmin(costs_final)
    best_weights = Lambda[best_idx]
    
    print(f"  [OK] Optimization complete")
    print(f"    Best cost: {costs_final[best_idx]:.1f}")
    print(f"    Initial cost: {Pi_min[0]:.1f}")
    print(f"    Improvement: {Pi_min[0] / Pi_min[-1]:.2f}x")
    
    # Extract and clip control weights
    best_weights = np.clip(best_weights, 0.0, 1.0)
    best_weights = best_weights.reshape(N_DRONES, N_CONTROL_WEIGHTS)
    
    # Print learned weights (insight into what GA discovered)
    print(f"\n  Learned control weights (per drone):")
    print(f"    [w1=target, w2=obstacle, w3=separation, w4=cohesion]")
    for d in range(min(5, N_DRONES)):
        print(f"    Drone {d}: w1={best_weights[d,0]:.3f}, w2={best_weights[d,1]:.3f}, w3={best_weights[d,2]:.3f}, w4={best_weights[d,3]:.3f}")
    if N_DRONES > 5:
        print(f"    ... ({N_DRONES - 5} more drones)")
    
    # ========================================================================
    # STEP 2: Simulate best control weights with collision detection
    # ========================================================================
    
    print("\nStep 2: Simulating optimized control weights...")
    print("  (Running until all targets visited or all drones dead - no step limit)")
    
    # Use smaller timestep and faster simulation for optimization
    simulator = DroneControlSimulator(best_weights, OBSTACLES, TARGET_ZONES, dt=0.3)
    history = simulator.run(max_steps=3000)  # Full simulation for validation
    
    positions = history['positions']
    alive = history['alive']
    targets_visited = history['targets_visited']
    
    n_alive_final = np.sum(alive[-1])
    n_targets_final = np.sum(targets_visited[-1])
    print(f"  [OK] Simulation complete")
    print(f"    Drones alive at end: {n_alive_final}/{N_DRONES}")
    print(f"    Drones lost: {N_DRONES - n_alive_final}")
    print(f"    Targets visited: {n_targets_final}/{N_TARGETS}")
    print(f"    Simulation time: {simulator.time:.1f}s")
    
    # ========================================================================
    # STEP 3: Visualization
    # ========================================================================
    
    print("\nStep 3: Creating visualizations...")

    # Import animation helpers only now (matplotlib is slow)
    from me144_toolbox.utils.animation_3d import animate_swarm_3d, plot_swarm_3d_static
    
    # Static 3D plot
    print("  Creating static 3D plot...")
    final_pos = positions[-1]
    final_alive = alive[-1]
    
    plot_swarm_3d_static(
        final_pos,
        np.array(TARGET_ZONES),
        final_alive,
        obstacles=OBSTACLES,
        bounds=BOUNDS_3D,
        base_pos=BASE_POS,
        title=f"Final Positions (Emergent Control)\n(Alive: {n_alive_final}/{N_DRONES}, Targets: {n_targets_final}/{N_TARGETS})",
    )
    
    # Animation
    print("  Creating animation...")
    
    # Use half the frames for smooth playback
    positions_sampled = positions[::1]
    alive_sampled = alive[::1]
    targets_visited_sampled = targets_visited[::1]
    
    print(f"  Animation will have {len(positions_sampled)} frames")
    
    try:
        animate_swarm_3d(
            positions_sampled,
            alive_sampled,
            np.array(TARGET_ZONES),
            obstacles=OBSTACLES,
            bounds=BOUNDS_3D,
            base_pos=BASE_POS,
            save_path=str(FIG_DIR / 'swarm_control_emergent.gif'),
            interval=10000000000,  # Frame interval (ms)
            targets_visited=targets_visited_sampled,
        )
        print("  [OK] Animation created: swarm_control_emergent.gif")
    except KeyboardInterrupt:
        print("  [NOTE] Animation creation interrupted")
    except Exception as e:
        print(f"  [NOTE] Animation note: {str(e)[:60]}...")
    
    # Convergence and performance plots (import matplotlib only now)
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # GA convergence
    Pi_min = np.asarray(Pi_min, dtype=float).ravel()
    Pi_avg = np.asarray(Pi_avg, dtype=float).ravel()
    # Enforce monotonic best for visualization
    if Pi_min.size > 0:
        Pi_min = np.minimum.accumulate(Pi_min)
    gen = np.arange(Pi_min.size)

    finite_mask = np.isfinite(Pi_min) & np.isfinite(Pi_avg)
    if Pi_min.size == 0 or not np.any(finite_mask):
        ax1.text(0.5, 0.5, 'No valid GA history to plot', ha='center', va='center')
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Cost', fontsize=11)
        ax1.set_title('GA Control Weight Optimization', fontsize=12)
        ax1.grid(True, alpha=0.3)
        print(f"  [WARN] GA history invalid: len(Pi_min)={Pi_min.size}")
    else:
        Pi_min_plot = Pi_min.copy()
        Pi_avg_plot = Pi_avg.copy()
        Pi_min_plot[~finite_mask] = np.nan
        Pi_avg_plot[~finite_mask] = np.nan

        min_val = min(np.nanmin(Pi_min_plot), np.nanmin(Pi_avg_plot))
        max_val = max(np.nanmax(Pi_min_plot), np.nanmax(Pi_avg_plot))
        print(f"  [GA] Pi_min range: {min_val:.3g} to {max_val:.3g}")

        use_log = min_val > 0.0
        if use_log:
            ax1.semilogy(gen, Pi_min_plot, 'g-', linewidth=2.0, marker='o', markersize=3, label='Best Cost')
            ax1.semilogy(gen, Pi_avg_plot, 'r--', linewidth=1.5, marker='o', markersize=3, label='Mean Cost')
            ax1.set_ylabel('Cost (log scale)', fontsize=11)
        else:
            # Shift up if nonpositive to keep a visible plot
            shift = 1.0 - min_val if min_val <= 0.0 else 0.0
            y_min = min_val + shift
            y_max = max_val + shift
            pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            ax1.plot(gen, Pi_min_plot + shift, 'g-', linewidth=2.0, marker='o', markersize=3, label='Best Cost')
            ax1.plot(gen, Pi_avg_plot + shift, 'r--', linewidth=1.5, marker='o', markersize=3, label='Mean Cost')
            ax1.set_ylim(y_min - pad, y_max + pad)
            ax1.set_ylabel('Cost', fontsize=11)

        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_title('GA Control Weight Optimization', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Drone survival
    n_alive_per_step = np.sum(alive, axis=1)
    time_steps = np.arange(len(n_alive_per_step)) * simulator.dt
    ax2.plot(time_steps, n_alive_per_step, 'b-', linewidth=2.5)
    ax2.fill_between(time_steps, n_alive_per_step, alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Alive Drones', fontsize=11)
    ax2.set_title(f'Swarm Survival ({n_alive_final} survived)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, N_DRONES + 1)
    
    # Targets scored
    n_targets_per_step = np.sum(targets_visited, axis=1)
    ax3.plot(time_steps, n_targets_per_step, 'g-', linewidth=2.5)
    ax3.fill_between(time_steps, n_targets_per_step, alpha=0.3, color='green')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Targets Visited', fontsize=11)
    ax3.set_title(f'Target Acquisition ({n_targets_final}/{N_TARGETS} scored)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, N_TARGETS + 1)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'control_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved analysis plot")
    plt.close()
    
    # Log results
    log_file = OUTPUT_DIR / 'logs' / 'control_swarm_incursion.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Drone Swarm Incursion - Control Law Optimization\n")
        f.write("="*70 + "\n\n")
        f.write(f"Methodology: GA optimizes 4 control weights per drone\n")
        f.write(f"  w1 = attraction to closest target\n")
        f.write(f"  w2 = repulsion from obstacles\n")
        f.write(f"  w3 = separation from drones\n")
        f.write(f"  w4 = cohesion with swarm\n\n")
        f.write(f"Drones: {N_DRONES}\n")
        f.write(f"Target zones: {N_TARGETS}\n")
        f.write(f"Obstacles: {len(OBSTACLES)}\n")
        f.write(f"GA generations: {G}\n")
        f.write(f"Design variables: {DV} (control weights)\n\n")
        f.write(f"Best control cost: {costs_final[best_idx]:.1f}\n")
        f.write(f"GA improvement: {Pi_min[0] / Pi_min[-1]:.2f}x\n\n")
        f.write(f"Drones alive: {n_alive_final}/{N_DRONES}\n")
        f.write(f"Drones lost: {N_DRONES - n_alive_final}\n")
        f.write(f"Targets visited: {n_targets_final}/{N_TARGETS}\n")
        f.write(f"Simulation time: {simulator.time:.1f}s\n\n")
        f.write("Learned Control Weights:\n")
        for d in range(N_DRONES):
            f.write(f"  Drone {d:2d}: w1={best_weights[d,0]:.3f}, w2={best_weights[d,1]:.3f}, w3={best_weights[d,2]:.3f}, w4={best_weights[d,3]:.3f}\n")
    
    print(f"  [OK] Logged results")
    
    print("\n" + "="*70)
    print("[OK] Simulation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
