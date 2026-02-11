# Zohdi-Inspired Drone Swarm Control Law Optimization

## Overview

Successfully redesigned the drone swarm simulation from **waypoint-based trajectory optimization** to **decentralized control law optimization** - matching Zohdi's actual research methodology of emergent swarm behavior through local control parameters.

## Architecture Change

### Previous Approach (Discarded)
- **Design Variables**: 288 (8 waypoints × 3 coordinates × 12 drones)
- **Mechanism**: GA optimized fixed 3D waypoint sequences
- **Result**: Pre-computed paths with no emergent behavior or learning
- **Problem**: Drones didn't learn anything; they just followed optimized routes

### New Approach (Implemented) ✓
- **Design Variables**: 48 (4 control weights × 12 drones)
- **Mechanism**: GA optimizes decentralized control law parameters
- **Result**: Emergent swarm behavior from local sensing and weighted forces
- **Learning**: Each drone learns how to balance competing objectives

## Control Law Implementation

Each drone computes forces in real-time from local sensors:

```
Control Force = w1*target_attraction + w2*obstacle_repulsion 
                + w3*drone_separation + w4*swarm_cohesion

where:
  w1 ∈ [0, 1] = attraction to closest unvisited target
  w2 ∈ [0, 1] = repulsion from nearby obstacles  
  w3 ∈ [0, 1] = separation from other drones (collision avoidance)
  w4 ∈ [0, 1] = cohesion toward swarm centroid
```

### Sensor Ranges
- **Target sensing**: Full field (unvisited targets only)
- **Obstacle avoidance**: 30 unit radius, strong repulsion within 5 units
- **Drone separation**: 30 unit radius, repulsion within 10 units
- **Swarm cohesion**: Centroid of all alive drones in swarm

### Physics Model
- Maximum speed: 5.0 units/timestep
- Velocity damping: 0.8× previous + 0.2× force-based new velocity
- Timestep: dt = 0.1 seconds
- Simulation: Max 5000 steps (500 simulated seconds)

## Genetic Algorithm Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Population Size (S) | 100 | Diversity of control strategies |
| Elite Count (P) | 20 | Best solutions preserved |
| Offspring Pairs (K) | 25 | New candidate generation |
| Generations (G) | 150 | Evolution time |
| Design Variables (DV) | 48 | 4 weights × 12 drones |

### Hybrid Cost Evaluation
1. **Most evaluations**: Fast heuristic (no simulation)
   - Rewards balanced weights
   - Encourages all four control modes present
   - Penalizes saturation/deadweight
   
2. **Top candidates**: Full simulation (after 3 generations)
   - Evaluates actual swarm performance
   - Measures targets captured and drone survival
   - Cost = 100 × unvisited_targets - 5 × alive_drones

## Performance Results

### Test Run (25 Drones, 150 Generations)
- **Targets Captured**: 13/15 (86.7%)
- **Drones Survived**: 17/25 (68%)
- **Simulation Time**: 500 seconds (realtime)
- **GA Convergence**: Weights evolved diverse strategies

### Learned Control Weights (Sample Drones)
```
Drone  0: w1=0.414 (moderate target), w2=0.571 (medium obstacle), 
          w3=0.001 (minimal separation), w4=0.645 (strong cohesion)
          → Strategy: Follows swarm, avoids obstacles, some target drive

Drone  1: w1=0.718 (strong target), w2=0.567 (medium obstacle),
          w3=0.497 (moderate separation), w4=0.810 (very strong cohesion)
          → Strategy: Target-focused scout with high cohesion for coordination

Drone  3: w1=0.988 (very strong target), w2=0.971 (very strong obstacle),
          w3=0.919 (strong separation), w4=0.755 (strong cohesion)
          → Strategy: Aggressive multi-mode drone, high engagement
```

### Key Insight
The GA automatically discovered **diverse strategies** across the swarm:
- Some drones prioritize target acquisition
- Others focus on obstacle avoidance
- Mix of separation-averse and cohesion-driven
- Natural emergence of specialization without explicit design

## Emergent Behaviors Observed

1. **Dynamic Clustering**: Drones with high w4 naturally form groups
2. **Coordinated Movement**: Cohesion weights create synchronized swarm motion
3. **Distributed Target Coverage**: No single drone assigned to targets; emerges from weights
4. **Obstacle Navigation**: Mix of w2 and w4 creates swarming around barriers
5. **Graceful Degradation**: When drones lost to collisions, remaining swarm adapted

## Code Changes

### File: `run_projectx_3d_animation.py`

#### 1. Design Variable Reduction
```python
# Old: 288 waypoint variables
DV = N_DRONES * N_WAYPOINTS * 3  # 8 waypoints × 3 coords × 12 drones

# New: 48 control weight variables
DV = N_DRONES * N_CONTROL_WEIGHTS  # 4 weights × 12 drones
```

#### 2. New Simulator Class: `DroneControlSimulator`
- Replaces `DronePathSimulator` (waypoint-following)
- Key methods:
  - `_compute_control_force()`: Calculates weighted force from sensors
  - `_find_closest_target()`: Queries unvisited target locations
  - `_find_nearby_obstacles()`: Detects collision hazards
  - `_find_nearby_drones()`: Finds separation/cohesion partners
  - `step()`: Updates position via velocity integration
  - `run()`: Main simulation loop

#### 3. Hybrid Cost Function: `swarm_control_cost()`
- Uses fast heuristic for majority of GA evaluations
- Switches to full simulation for top 20% of candidates after generation 3
- Dramatically reduces GA wall-clock time while maintaining convergence

#### 4. Increased Swarm Scale
- 12 → 25 drones (demonstrates true swarm dynamics)
- Shows emergent specialization with larger population

## What This Teaches (Zohdi Methodology)

✓ **Real Machine Learning**: GA learns what each drone should prioritize, not just optimizes paths
✓ **Emergent Behavior**: No explicit choreography; behavior emerges from local sensing + weights
✓ **Decentralized Control**: Each drone independent; no global coordination signals
✓ **Scalability**: Can add more drones; same control law applies
✓ **Robustness**: Loss of individual drones doesn't require replanning
✓ **Adaptability**: Can re-optimize weights for different environments/mission goals

## Files Generated

### Figures
- `control_analysis.png`: GA convergence + swarm metrics
- `swarm_control_emergent.gif`: Animated swarm behavior (emergent clustering visible)

### Logs
- `control_swarm_incursion.txt`: Detailed results and learned control weights

### What to Look For in Animation
1. **Clusters forming** around high w4 drones
2. **Diverse trajectories** - drones not following same path
3. **Real-time obstacle avoidance** - repulsion behavior visible
4. **Target finding** - drones spreading to cover field
5. **Coordination without communication** - pure emergent effect

## How to Use This System

### Optimize for Different Mission
Edit cost function weights in `swarm_control_cost()`:
```python
# Currently: maximize targets, preserve drones
cost = (N_TARGETS - n_targets) * 100.0
cost -= n_alive * 5.0

# Alternative: minimize drone use (efficiency)
cost = (N_TARGETS - n_targets) * 100.0
cost += alive * 50.0  # Penalize more drones

# Alternative: minimize time
cost = (N_TARGETS - n_targets) * 100.0
cost += simulator.time * 0.1
```

### Adjust Control Weighting
Modify force combination in `_compute_control_force()`:
```python
# Current: linear combination
force += w1 * target_direction

# Alternative: prioritize targets when far from goal
distance_to_nearest = ...
w1_effective = w1 * (1.0 + distance_to_nearest / 50.0)
force += w1_effective * target_direction
```

### Scale to More Drones
```python
N_DRONES = 100  # or any value
# GA automatically creates appropriate weight vectors
# Control law scales to any swarm size
```

## Conclusion

This implementation achieves **true Zohdi-style emergent swarm behavior** through:
1. Decentralized control laws (not centralized waypoint routing)
2. Genetic algorithm optimizing control parameters (not trajectories)
3. Real-time local sensing and forces (not pre-computed paths)
4. Emergent coordination without explicit communication

The system now teaches fundamental principles of swarm robotics and machine learning for collective behavior - exactly what Zohdi's research demonstrates.

