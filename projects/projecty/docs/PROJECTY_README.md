# ProjectY README

## Overview

ProjectY is a voxelized drone design and hostile-flight simulation project inspired by the Chapter 4 textbook workflow. The project is built around a DEM-like visual language: each occupied voxel is shown as exactly one rendered sphere, so the drone reads as a lattice-built body instead of a smooth CAD shell.

The current implementation couples:

- a parameterized drone design space
- voxel-lattice geometry generation
- a simplified elastic stress solver
- damage accumulation and fragmentation
- rigid-body flight dynamics
- localized pressure-wave attacks from turrets
- optimization over the design parameters
- summary figures, GIF exports, and interactive 3D inspection

## Core Ideas

### 1. Geometry As A Voxel DEM Body

The drone body is not a triangle mesh. It is assembled from occupied voxels on a regular 3D lattice, with:

- a generalized ellipsoid / superellipsoid-inspired chassis envelope
- tapered arms
- connected motor blobs
- lightweight holes
- explicit body thickness

This is why the drone can be shown naturally as a cluster of tightly packed spheres.

### 2. Stress And Damage

For each candidate design, the simulation applies pressure-wave loading and rigid-body motion, then evaluates internal response on the voxel grid. The body tracks:

- von Mises stress
- integrity / damage state
- fragmentation thresholds
- detached connected components

### 3. Flight Through A Hostile Course

The drone flies a waypoint-based course while facing timed pressure-wave attacks. It must:

- survive structural loading
- maintain enough control authority
- pass through explicit green/red course gates
- complete the route instead of merely surviving in place

### 4. Optimization

The optimizer searches over geometry and mass/propulsion settings. The score rewards:

- course progress
- successful completion
- speed
- near-yield structural utilization

and penalizes:

- overstress
- understressed chunky designs
- excessive compactness penalties from oversized geometry
- failure and missed completion

## Main Files

- [run_projecty.py](../run_projecty.py) - orchestrates the whole pipeline
- [config.py](../config.py) - global configuration and output paths
- [design.py](../design.py) - design parameterization and mutation bounds
- [geometry.py](../geometry.py) - voxel chassis generation
- [elasticity.py](../elasticity.py) - stress computation and dynamic relaxation
- [flight.py](../flight.py) - hostile course simulation, gates, damage, fragments, and scoring
- [optimization.py](../optimization.py) - population evolution and candidate evaluation
- [visualization.py](../visualization.py) - summary plots, GIFs, and archive generation
- [interactive_viewer.py](../interactive_viewer.py) - local 3D snapshot browser

## Outputs

Everything is written under:

- `projects/projecty/output/`

Key artifacts:

- `final_simulation.gif`
- `design_evolution.gif`
- `projecty_summary.png`
- `projecty_stress_snapshots.png`
- `projecty_report.json`
- `optimization_history.json`
- `projecty_snapshot_archive.npz`

## Rendering Style

The rendering pipeline is intentionally consistent across:

- the close-up animation view
- the flight context view
- the interactive viewer
- the summary lattice chassis plot
- the stress snapshot plot

The important choices are:

- one sphere per voxel
- filled face colors
- darker edge outlines
- larger markers for motor voxels
- warm-to-cool damage coloring
- fragment clusters retained after detachment

## Running The Project

```powershell
cd c:\Users\15593\Desktop\ME144
python projects\projecty\run_projecty.py
```

## Viewing Snapshots

```powershell
python projects\projecty\interactive_viewer.py
```

## Textbook Mapping

See [TEXTBOOK_MAPPING.md](../TEXTBOOK_MAPPING.md) for the high-level mapping between the code and the Chapter 4 source concepts.
