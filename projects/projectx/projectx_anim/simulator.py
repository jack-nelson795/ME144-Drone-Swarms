from __future__ import annotations

import time

import numpy as np

from . import config


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
        self.positions = np.array([config.BASE_POS.copy() for _ in range(self.N)])
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
            base_dir_dist = config.BASE_POS - self.positions[drone_idx]
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
            distances_to_base = np.linalg.norm(self.positions[alive_indices] - config.BASE_POS, axis=1)
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
            self.positions[d, 0] = np.clip(self.positions[d, 0], config.BOUNDS_3D[0], config.BOUNDS_3D[1])
            self.positions[d, 1] = np.clip(self.positions[d, 1], config.BOUNDS_3D[2], config.BOUNDS_3D[3])
            self.positions[d, 2] = np.clip(self.positions[d, 2], config.BOUNDS_3D[4], config.BOUNDS_3D[5])

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

    def run(self, max_steps: int = 3000, *, sleep_on_completion: bool = True) -> dict:
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
                deltas = np.abs(alive_positions - config.BASE_POS)
                if np.all(deltas[:, 0] <= 20.0) and np.all(deltas[:, 1] <= 20.0) and np.all(deltas[:, 2] <= 20.0):
                    if sleep_on_completion:
                        time.sleep(1.0)
                    break

        return {
            'positions': np.array(self.history_positions),
            'alive': np.array(self.history_alive),
            'targets_visited': np.array(self.history_targets_visited),
        }
