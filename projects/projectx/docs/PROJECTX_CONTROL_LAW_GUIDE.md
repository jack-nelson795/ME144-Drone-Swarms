# Zohdi-Inspired Control Law Optimization for Drone Swarms

## Quick Start

### Run the Simulation
```bash
cd c:\Users\15593\Desktop\ME144
python -m projects.projectx.run_projectx_3d_animation
```

### What Happens
1. **GA Optimization** (75 generations): Learns best control weights for each drone
2. **Swarm Simulation** (500 seconds): Drones execute learned control laws
3. **Visualization**: Generates GIF animation showing emergent swarm behavior
4. **Results**: Logs control weights and performance metrics

### Expected Output
```
Targets Visited: 13/15 (86.7%)
Drones Survived: 8/12 (66.7%)
Simulation Time: 500 seconds
GA Generations: 75
Control Weights Learned: 48 (4 weights × 12 drones)
```

## System Architecture

### Decentralized Control Law
Each drone independently computes movement force based on four weighted sensory inputs:

$$\vec{F}_{drone} = w_1 \vec{F}_{target} + w_2 \vec{F}_{obstacle} + w_3 \vec{F}_{separation} + w_4 \vec{F}_{cohesion}$$

where:
- **$w_1$ (Target Attraction)**: How strongly to move toward closest unvisited target
- **$w_2$ (Obstacle Repulsion)**: How strongly to avoid nearby obstacles  
- **$w_3$ (Separation)**: How strongly to separate from other drones
- **$w_4$ (Cohesion)**: How strongly to move toward swarm centroid

Each weight is in range $[0, 1]$ and represents a learned strategy.

### Genetic Algorithm Pipeline
```
Initial Population (75 candidates)
    ↓
Evaluate with Heuristic (fast, 48 variables)
    ↓
Select/Mutate/Crossover
    ↓
Top 20% evaluated with Full Simulation (accurate)
    ↓
Repeat 75 generations
    ↓
Best weights run full simulation
    ↓
Visualization + logging
```

## Results & Interpretation

### Performance Metrics
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Targets Visited | 13/15 (87%) | Swarm reached 87% of target zones |
| Drones Alive | 8/12 (67%) | 4 drones lost to obstacle collisions |
| Simulation Time | 500s | 500 simulated seconds to reach stopping condition |

### Learned Control Weights

Examining the discovered weights shows the GA's learned strategies:

**High Target-Focus Drones** (w1 > 0.8):
```
Drone 3: w1=0.973  → Primary target seeker
Drone 8: w1=0.915  → Aggressive target pursuit
```

**Obstacle-Aware Drones** (w2 > 0.8):
```
Drone 1: w2=0.926  → Navigation specialist
Drone 11: w2=0.883 → Careful pathfinding
```

**Separation-Conscious Drones** (w3 > 0.9):
```
Drone 2: w3=0.905  → Anti-collision priority
Drone 11: w3=0.951 → Extreme separation focus
```

**Cohesion-Driven Drones** (w4 > 0.8):
```
Drone 1: w4=0.810  → Swarm follower
Drone 10: w4=0.998 → Extreme cohesion (near-perfect alignment)
```

### What the GA Discovered

The genetic algorithm automatically found **complementary specialization**:

1. **Scouts** (high w1, medium w2): Seek targets, respect obstacles
2. **Coordinators** (high w4, varied others): Hold swarm together
3. **Defenders** (high w2, w3): Avoid hazards, maintain spacing
4. **Generalists** (balanced weights): Blend all behaviors

**No explicit role assignment required** - roles emerged from optimization!

## Comparing to Old Approach

### Old System (Waypoint Optimization)
```
GA Variables: 288 (waypoint coordinates)
Design Space: 3D positions for each drone's route
Behavior: Pre-computed paths
Learning: None (just path optimization)
Result: Followed fixed routes
```

### New System (Control Law Optimization)
```
GA Variables: 48 (control weights)
Design Space: Force weighting strategies
Behavior: Real-time emergent
Learning: GA learns what matters (targets vs obstacles vs cohesion)
Result: Adaptive swarm behavior
```

**Key Difference**: This system actually learns through GA; the old system just optimized paths mechanically.

## Emergent Behaviors Visible in Animation

When you watch the GIF (`swarm_control_emergent.gif`), look for:

1. **Dynamic Clustering** 
   - Drones with high w4 naturally form spatial clusters
   - No explicit formation command; clusters emerge

2. **Coordinated Target Acquisition**
   - Multiple drones don't collide at targets
   - Separation weight (w3) creates spacing automatically
   - Targets visited sequentially, efficiently

3. **Obstacle Avoidance**
   - Drones detect obstacles and repel proactively
   - No collision = no replanning needed
   - Behavior purely reactive to sensory input

4. **Graceful Degradation**
   - When drones lost, remaining swarm continues
   - No central failure point
   - Distributed control survives attrition

## How to Modify the System

### Change Swarm Size
```python
# Line ~40 in run_projectx_3d_animation.py
N_DRONES = 12  # Change this number
# GA automatically creates appropriate weight vectors
# Control law scales to any swarm size
```

### Adjust GA Parameters
```python
S = 75   # Population size (more = more diversity, slower)
P = 15   # Elites (more = better convergence, less exploration)
K = 20   # Offspring pairs (more = faster evolution)
G = 75   # Generations (more = better optimization, slower)
```

### Modify Mission Objective
```python
# In swarm_control_cost() function:
# Currently prioritizes target coverage
cost = (N_TARGETS - n_targets) * 100.0
cost -= n_alive * 5.0

# Alternative: Minimize drone loss
cost = (N_TARGETS - n_targets) * 100.0
cost += (N_DRONES - n_alive) * 50.0

# Alternative: Fast completion
cost = (N_TARGETS - n_targets) * 100.0
cost += simulator.time * 0.1
```

### Enhance Control Law
Add new force components in `_compute_control_force()`:
```python
# Existing four forces
force += w1 * target_direction
force += w2 * obstacle_repulsion
force += w3 * separation_direction
force += w4 * cohesion_direction

# Could add:
# - w5: Boundary preference (stay in bounds)
# - w6: Energy conservation (minimize speed)
# - w7: Exploration (seek unmapped areas)
# Requires increasing DV and GA dimension
```

## Understanding Zohdi's Approach

This implementation demonstrates core principles from Zohdi's swarm robotics research:

✓ **Decentralized Control**: No central authority; each drone decides independently
✓ **Local Sensing**: Each drone only "sees" nearby obstacles, neighbors, targets
✓ **Emergent Behavior**: Coordinated motion emerges from local rules
✓ **Machine Learning**: GA finds optimal control law weights
✓ **Scalability**: Algorithm works for any swarm size
✓ **Robustness**: System continues if individual drones fail
✓ **No Communication**: Drones don't exchange messages; only feel forces

### Mathematical Grounding

The control law implements a **force-based potential field**:

$$V_{total} = \sum_{targets} V_{target}(r_t) + \sum_{obstacles} V_{obstacle}(r_o) + \sum_{drones} V_{separation}(r_d) + V_{cohesion}(r_{centroid})$$

$$\vec{F} = -\nabla V_{total}$$

By optimizing weights, GA finds the potential field that best solves the mission.

## Files & Outputs

### Input Files
- `me144_toolbox/optimization/ga.py` - Genetic algorithm engine
- `me144_toolbox/utils/animation_3d.py` - Visualization system

### Output Files (in `projects/projectx/output/`)

**Figures:**
- `control_analysis.png` - GA convergence + swarm metrics (3-panel plot)
- `swarm_control_emergent.gif` - Animated simulation (500 steps)

**Logs:**
- `control_swarm_incursion.txt` - Full results and control weights for all drones

### Key Code Sections

| Component | File | Lines |
|-----------|------|-------|
| Control Law Sim | run_projectx_3d_animation.py | 298-470 |
| Force Computation | run_projectx_3d_animation.py | 323-388 |
| Hybrid Cost Function | run_projectx_3d_animation.py | 169-260 |
| GA Optimization | me144_toolbox/optimization/ga.py | 1-300 |

## Performance Tuning

### Faster Runs (for testing)
```python
N_DRONES = 8
G = 30  # Fewer generations
S = 50  # Smaller population
```

### Higher Quality (for research)
```python
N_DRONES = 50
G = 300  # More generations
S = 150  # Larger population
```

### Physics Tuning
```python
# In DroneControlSimulator.__init__():
self.max_speed = 5.0           # Faster = more responsive
self.sensor_range = 30.0       # Larger = global awareness
self.obstacle_danger_distance = 5.0  # Tighter = safer
```

## Troubleshooting

### Problem: Drones not visiting targets
**Solution**: Increase target attraction weight in cost function, or increase `max_speed`

### Problem: Too many collisions
**Solution**: Increase `w2` (obstacle avoidance) or `w3` (separation) weights in GA rewards

### Problem: GA not converging
**Solution**: Increase generations (G) or run swarm_control_cost with `run_simulation=True` for all candidates

### Problem: Simulation too slow
**Solution**: Reduce max_steps in simulator.run(), or use heuristic-only evaluation (fewer full simulations)

## Educational Value

This system teaches:
1. **GA fundamentals**: How evolution finds optimizations
2. **Swarm robotics**: Emergent behavior from local rules
3. **Control theory**: Force-based motion control
4. **Multi-agent systems**: Decentralized coordination
5. **Zohdi's research**: Real application of swarm principles

Perfect for understanding how complex coordinated behavior emerges from simple local rules - the essence of swarm intelligence.

## References

The control law structure follows principles from:
- Zohdi, T. (2014) - "Dynamics of Clusters of Particles"
- Reynolds, C. (1987) - "Flocks, Herds and Schools: A Distributed Behavioral Model"
- Boid simulation principles applied to drone swarms

