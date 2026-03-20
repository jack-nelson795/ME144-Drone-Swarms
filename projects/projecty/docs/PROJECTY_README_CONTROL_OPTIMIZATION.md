# ProjectY README: Control And Optimization

## Why This Document Exists

ProjectY is not just a rendering project. The central problem is an optimization problem under coupled flight and structural constraints. This document explains the relationship between:

- the controller
- the damage model
- the design variables
- the scoring function

## Design Variables Under Optimization

The optimizer can change:

- body planar dimensions
- body thickness
- body exponents
- arm length
- arm thickness
- motor size
- hole size
- thrust scale
- motor mass
- chassis mass target

These parameters affect both:

- geometry and visual form
- flight/structural performance

## What The Controller Wants

At run time, the controller wants:

- enough thrust
- enough torque authority
- enough stability
- enough structure to survive pulses
- enough speed to complete the course

## What The Optimizer Wants

The optimizer is trying to find designs that:

- complete the course
- pass the gates
- survive
- move quickly
- operate near but not far beyond yield
- avoid bloated low-utilization geometry

## Why Near-Yield Optimization Matters

A trivially thick drone is not interesting. The project is more meaningful when the optimizer searches for designs that are:

- structurally efficient
- visibly stressed
- close to the usable limit
- still able to complete the mission

That is why Project Y includes:

- explicit understress penalties
- near-yield rewards
- overstress penalties
- compactness penalties

## Why Completion Must Dominate

Without strong completion incentives, the optimizer can settle into bad local behaviors such as:

- hovering short of the finish
- surviving without finishing
- accepting poor progress if the stress state is favorable

That is why the scoring also includes:

- progress reward
- finish bonus
- finish error penalty
- explicit gate completion requirement

## Damage-Control Coupling

ProjectY also couples damage back into control.

When parts detach:

- they leave the main rigid body
- they become falling fragments
- mass and available control authority change
- motor clusters can lose thrust individually

This is important because otherwise the optimization would exploit unrealistic assumptions about still-attached actuation.

## Parallel Optimization

Candidate evaluation is embarrassingly parallel, so `optimization.py` uses `ProcessPoolExecutor` when:

- `parallel_workers > 1`

This improves runtime substantially on multi-core machines.

## Recommended Optimization Workflow

### For Fast Iteration

Use smaller:

- `optimizer_generations`
- `population_size`

### For Final Search

Increase:

- `optimizer_generations`
- `population_size`
- `parallel_workers`

and keep an eye on:

- total run time
- memory pressure
- quality of generation-best designs

## What To Improve Next

The most meaningful future upgrades here would be:

- richer mutation/recombination operators
- multi-objective optimization
- predictive/adversarial control
- robustness testing across randomized hazard realizations
