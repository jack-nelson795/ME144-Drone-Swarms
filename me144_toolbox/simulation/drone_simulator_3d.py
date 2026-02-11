# me144_toolbox/simulation/drone_simulator_3d.py
"""
3D Drone Swarm Simulator with Physics and Collision Detection

Simulates drones moving towards target formations, with:
- Collision detection (drone-drone, drone-obstacle)
- Drone death on impact
- 3D visualization and animation
- Physics-based movement
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Drone:
    """Represents a single drone in 3D space."""
    
    pos: np.ndarray  # [x, y, z] current position
    vel: np.ndarray  # [vx, vy, vz] velocity
    target: np.ndarray  # [x, y, z] target position
    alive: bool = True
    radius: float = 2.0  # collision radius
    max_speed: float = 5.0
    
    def distance_to(self, other: np.ndarray) -> float:
        """Distance to a point or another drone."""
        return float(np.linalg.norm(self.pos - other))
    
    def update(self, dt: float = 0.1) -> None:
        """Update drone position with physics."""
        if not self.alive:
            return
        
        # Direction to target
        direction = self.target - self.pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            # Move toward target
            direction = direction / distance
            self.vel = direction * self.max_speed
        else:
            # At target, slow down
            self.vel *= 0.9
        
        # Update position
        self.pos = self.pos + self.vel * dt
    
    def check_collision(self, obstacle_pos: np.ndarray, obstacle_radius: float) -> bool:
        """Check if drone collides with obstacle. Returns True if collided (drone dies)."""
        dist = self.distance_to(obstacle_pos)
        if dist < self.radius + obstacle_radius:
            self.alive = False
            return True
        return False
    
    def check_drone_collision(self, other: Drone) -> bool:
        """Check if this drone collides with another drone."""
        if not self.alive or not other.alive:
            return False
        
        dist = self.distance_to(other.pos)
        if dist < 2 * self.radius:
            self.alive = False
            other.alive = False
            return True
        return False


class DroneSwarm3D:
    """
    3D drone swarm simulator with physics and collision detection.
    """
    
    def __init__(
        self,
        initial_positions: np.ndarray,  # (N, 3) starting positions
        target_positions: np.ndarray,   # (N, 3) target positions
        bounds: Tuple[float, float, float, float, float, float] = (0, 100, 0, 100, 0, 50),
        obstacles: Optional[List[dict]] = None,
    ):
        """
        Parameters
        ----------
        initial_positions : (N, 3)
            Starting [x, y, z] for each drone
        target_positions : (N, 3)
            Target [x, y, z] for each drone
        bounds : (x_min, x_max, y_min, y_max, z_min, z_max)
            3D simulation domain
        obstacles : list of dicts
            Each dict: {'center': (x, y, z), 'radius': r}
        """
        self.bounds = bounds
        self.N = len(initial_positions)
        self.time = 0.0
        self.dt = 0.1
        
        # Create drones
        self.drones: List[Drone] = []
        for i in range(self.N):
            drone = Drone(
                pos=initial_positions[i].copy(),
                vel=np.array([0.0, 0.0, 0.0]),
                target=target_positions[i].copy(),
            )
            self.drones.append(drone)
        
        # Obstacles
        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = [
                {'center': np.array(o['center']), 'radius': o.get('radius', 10)}
                for o in obstacles
            ]
    
    def step(self) -> None:
        """Advance simulation by one timestep."""
        # Update all drone positions
        for drone in self.drones:
            drone.update(self.dt)
        
        # Check bounds
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        for drone in self.drones:
            if drone.alive:
                drone.pos[0] = np.clip(drone.pos[0], x_min, x_max)
                drone.pos[1] = np.clip(drone.pos[1], y_min, y_max)
                drone.pos[2] = np.clip(drone.pos[2], z_min, z_max)
        
        # Check drone-obstacle collisions
        for drone in self.drones:
            if drone.alive:
                for obs in self.obstacles:
                    drone.check_collision(obs['center'], obs['radius'])
        
        # Check drone-drone collisions
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.drones[i].check_drone_collision(self.drones[j])
        
        self.time += self.dt
    
    def run_until_converged(self, max_steps: int = 1000, tolerance: float = 1.0) -> int:
        """
        Run simulation until drones converge or max steps reached.
        
        Returns
        -------
        steps_run : int
            Number of steps executed
        """
        for step in range(max_steps):
            self.step()
            
            # Check convergence: all alive drones at targets
            all_converged = True
            for drone in self.drones:
                if drone.alive:
                    dist_to_target = drone.distance_to(drone.target)
                    if dist_to_target > tolerance:
                        all_converged = False
                        break
            
            if all_converged:
                return step + 1
        
        return max_steps
    
    def get_history(self, max_steps: int = 1000) -> dict:
        """
        Run simulation and record positions at each step.
        
        Returns
        -------
        dict with keys:
            'positions': (max_steps, N, 3) array of positions
            'alive': (max_steps, N) boolean array
            'times': array of time values
        """
        positions = np.zeros((max_steps, self.N, 3))
        alive_history = np.ones((max_steps, self.N), dtype=bool)
        times = np.zeros(max_steps)
        
        for step in range(max_steps):
            # Record current state
            for i, drone in enumerate(self.drones):
                positions[step, i] = drone.pos.copy()
                alive_history[step, i] = drone.alive
            
            times[step] = self.time
            
            # Step simulation
            self.step()
            
            # Check if all dead or converged
            n_alive = sum(1 for d in self.drones if d.alive)
            if n_alive == 0:
                positions = positions[:step+1]
                alive_history = alive_history[:step+1]
                times = times[:step+1]
                break
            
            # Check convergence
            all_converged = True
            for drone in self.drones:
                if drone.alive:
                    dist = drone.distance_to(drone.target)
                    if dist > 1.0:
                        all_converged = False
                        break
            
            if all_converged:
                positions = positions[:step+1]
                alive_history = alive_history[:step+1]
                times = times[:step+1]
                break
        
        return {
            'positions': positions,
            'alive': alive_history,
            'times': times,
        }
    
    def get_stats(self) -> dict:
        """Get simulation statistics."""
        n_alive = sum(1 for d in self.drones if d.alive)
        n_dead = self.N - n_alive
        
        avg_dist = 0.0
        for drone in self.drones:
            if drone.alive:
                avg_dist += drone.distance_to(drone.target)
        
        if n_alive > 0:
            avg_dist /= n_alive
        
        return {
            'alive': n_alive,
            'dead': n_dead,
            'avg_distance_to_target': avg_dist,
            'time': self.time,
        }
