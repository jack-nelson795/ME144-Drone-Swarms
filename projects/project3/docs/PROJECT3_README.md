# Project 3: Swarm GA + Physics Simulation

This folder contains an implementation of the ME144/244 Project 3 assignment: a 3D multi-agent swarm simulation whose control parameters are tuned by a genetic algorithm.

## What This Implements (Math Model)

### Dynamics
Each agent is modeled as a point mass with Newton’s second law:

$$m\,\mathbf{a}_i = \mathbf{F}_{p,i} + \mathbf{F}_{d,i}$$

with propulsion of fixed magnitude $F_p$ and direction $\mathbf{n}^*_i$:

$$\mathbf{F}_{p,i} = F_p\,\mathbf{n}^*_i$$

and aerodynamic drag (with $\mathbf{v}_a = \mathbf{0}$ in the provided parameters):

$$\mathbf{F}_{d,i} = \tfrac12\rho_a C_d A\,\|\mathbf{v}_a-\mathbf{v}_i\|\,(\mathbf{v}_a-\mathbf{v}_i)$$

Time integration uses Semi-Implicit Euler:

$$\mathbf{v}_i^{k+1}=\mathbf{v}_i^k+\mathbf{a}_i^k\,\Delta t, \qquad \mathbf{r}_i^{k+1}=\mathbf{r}_i^k+\mathbf{v}_i^{k+1}\,\Delta t$$

### Control Law
For each agent $i$, interactions with targets, obstacles, and other agents are summed. For targets (mapped similarly for obstacles and other agents), the interaction uses an exponential distance weighting:

$$\hat{\mathbf{n}}_{i\to j}^{mt}=(w_{t1}e^{-a_1 d_{ij}}-w_{t2}e^{-a_2 d_{ij}})\,\mathbf{n}_{i\to j}$$

Total interaction:

$$\mathbf{N}_i^{tot}=W_{mt}\,\mathbf{N}_i^{mt}+W_{mo}\,\mathbf{N}_i^{mo}+W_{mm}\,\mathbf{N}_i^{mm},\qquad \mathbf{n}_i^*=\frac{\mathbf{N}_i^{tot}}{\|\mathbf{N}_i^{tot}\|}$$

The GA optimizes the 15 parameters:

$$\Lambda = \{W_{mt},W_{mo},W_{mm},w_{t1},w_{t2},w_{o1},w_{o2},w_{m1},w_{m2},a_1,a_2,b_1,b_2,c_1,c_2\}$$

with the assignment constraint $0\le \Lambda_n\le 2$.

### Cost Function
The scalar objective is

$$\Pi = 70\,M^* + 10\,T^* + 20\,L^*$$

where

$$M^*=\frac{\#\,\text{unmapped targets}}{\#\,\text{total targets}},\quad T^*=\frac{\text{used time}}{\text{total time}},\quad L^*=\frac{\#\,\text{crashed agents}}{\#\,\text{total agents}}.$$

## How to Run

See PROJECT3_QUICKSTART.md.

## Outputs

Generated artifacts land in:
- `projects/project3/figures/` (plots, MP4, snapshot PNGs)
- `projects/project3/ga_results.pkl` (GA histories)

---

## Project 3 vs ProjectX (Mathematical Differences)

Project 3 and ProjectX are both swarm/GA projects, but they optimize and simulate different mathematical models.

### 1) State update model
- Project 3: **second-order dynamics** with mass, forces, and drag:
  - $m\mathbf{a}=\mathbf{F}_{p}+\mathbf{F}_{d}$, integrated with Semi-Implicit Euler.
- ProjectX: **control-force / kinematic-style stepping** with velocity smoothing and speed caps:
  - A weighted direction “force” is computed and used to update velocities/positions with damping (no explicit mass/drag model).

### 2) Control-law parameterization
- Project 3: a **single 15-parameter global design string** that shapes exponential interaction fields across all agents.
  - Uses distance-scale parameters $(a_1,a_2,b_1,b_2,c_1,c_2)$ and attraction/repulsion weights.
- ProjectX: **per-drone weight vectors** (4 weights per drone):

$$\mathbf{F}_{drone}=w_1\mathbf{F}_{target}+w_2\mathbf{F}_{obstacle}+w_3\mathbf{F}_{separation}+w_4\mathbf{F}_{cohesion}$$

  - Weights are clipped to $[0,1]$ and can vary by drone.

### 3) Information assumptions
- Project 3: agents effectively use **global information** each timestep (interactions sum over all targets/obstacles/agents).
- ProjectX: agents use **local sensing** (sensor range) to compute nearby interactions.

### 4) Objective / cost structure
- Project 3: **single weighted sum** of three normalized star metrics: $(M^*,T^*,L^*)$.
- ProjectX: a **hybrid evaluation cost** (heuristic ranking + partial/full simulations) with continuous penalties (time-to-completion, remaining-target integral, obstacle proximity) and an offset to keep costs positive.

### 5) Task definition
- Project 3: map targets; stop when all targets mapped or all agents crash.
- ProjectX: visit targets and (optionally) return-to-base behavior after visiting all targets (phase transition).
