# ProjectY Summary

## Short Version

ProjectY is an end-to-end hostile-flight drone design study that evolves a voxel/DEM drone, simulates its structural and flight response under pressure-wave attack, and exports both static and animated visualizations.

## What Makes It Distinct

- The drone is rendered as a DEM-style lattice body with one sphere per voxel.
- Structural damage is visualized directly on the voxel spheres.
- Fragmentation remains visible as moving sphere clusters.
- The optimizer is pushed toward high stress utilization near yield instead of trivially overbuilt designs.
- The project includes both a final mission GIF and a generation-by-generation design evolution GIF.
- It also includes an interactive 3D snapshot viewer.

## Current Pipeline

1. Sample or mutate a `DroneDesign`.
2. Convert that design into a voxelized drone in [geometry.py](../geometry.py).
3. Simulate hostile flight in [flight.py](../flight.py).
4. Compute a scalar score from flight outcome and structural utilization.
5. Evolve the design population in [optimization.py](../optimization.py).
6. Export summary figures, GIFs, JSON, and snapshot archives in [visualization.py](../visualization.py).

## Main Visual Outputs

- `final_simulation.gif`
  Shows the optimized drone flying the hostile course, including pulse effects, damage coloring, and breakup.

- `design_evolution.gif`
  Shows the best design from each generation.

- `projecty_summary.png`
  A compact static dashboard showing the lattice chassis, hostile path, optimization history, and stress/tracking curves.

- `projecty_stress_snapshots.png`
  Static stress/damage snapshots rendered with the same DEM styling as the animation close-up.

## Main Physics Outputs

- trajectory history
- stress history
- gate completion state
- fragment history
- candidate score
- completion / survival status

## Design Variables

The design search currently spans:

- body planar radii
- body thickness
- body shape exponents
- arm length
- arm radius
- motor radius
- hole size
- thrust scale
- motor mass
- chassis mass target

## Main Constraints The Optimizer Is Trying To Balance

- get through the course
- pass the 50% and finish gates
- survive turret attacks
- stay near yield without going too far beyond it
- remain compact and lightweight
- preserve enough motor and control authority

## Why The DEM Presentation Matters

This project would be much less legible if it used a smooth mesh. The DEM-style lattice directly communicates:

- occupied voxels
- local thickness
- motor mass regions
- fragment breakup
- stress/damage gradients on discrete particles

That visual language is central to the project, not just cosmetic.
