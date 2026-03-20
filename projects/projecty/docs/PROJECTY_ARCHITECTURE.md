# ProjectY Architecture

## High-Level Flow

ProjectY follows a clean pipeline:

`design -> geometry -> flight/stress -> optimization -> visualization -> interactive review`

## Module Map

### [config.py](../config.py)

Defines:

- project output paths
- material constants
- simulation constants
- optimization budget
- pulse/timing/fragmentation settings

This file is the main knob panel for run time, aggressiveness, and output behavior.

### [design.py](../design.py)

Defines the `DroneDesign` parameter space.

Responsibilities:

- store geometric and mass/propulsion parameters
- clamp values to legal bounds
- mutate designs for evolutionary search
- expose a simple nominal volume property

This is the search-space definition for the optimizer.

### [geometry.py](../geometry.py)

Builds the voxelized drone body from a `DroneDesign`.

Responsibilities:

- create the voxel grid
- generate the chassis envelope
- generate tapered arms
- generate connected motor blobs
- cut lightweight holes
- compute mass, inertia, motor masks, thrust constants, and lattice spacing

The output is `VoxelDrone`, the central geometry/data object used by the rest of the project.

### [elasticity.py](../elasticity.py)

Implements the simplified stress solver.

Responsibilities:

- strain computation
- stress computation
- stress divergence
- dynamic relaxation
- von Mises extraction

The implementation is vectorized to avoid the very slow nested-voxel loop pattern.

### [flight.py](../flight.py)

This is the main simulation engine.

Responsibilities:

- define the course
- define course gates
- define turrets and pulse timing
- simulate rigid-body flight
- apply structural loading
- update damage/integrity
- detach fragments
- update fragment motion
- update motor availability after damage or separation
- compute candidate score

This module is where most of the project-level behavior lives.

### [optimization.py](../optimization.py)

Runs the evolutionary design search.

Responsibilities:

- spawn/evaluate candidates
- use multiprocessing when enabled
- keep elites
- mutate children
- track generation history
- record generation-best designs for animation

### [visualization.py](../visualization.py)

Exports all main visuals.

Responsibilities:

- final mission animation
- design evolution animation
- summary dashboard figure
- stress snapshot figure
- snapshot archive for the interactive viewer

The DEM rendering logic is centralized here so all outputs stay visually consistent.

### [interactive_viewer.py](../interactive_viewer.py)

Loads `projecty_snapshot_archive.npz` and provides an orbitable interactive 3D viewer.

Responsibilities:

- load saved snapshot frames
- render the exact stored DEM sphere colors, edges, and sizes
- allow slider/button snapshot navigation

### [run_projecty.py](../run_projecty.py)

Top-level orchestrator.

Responsibilities:

- instantiate config
- run optimization
- rebuild the best drone
- write figures/GIFs/JSON
- print progress and ETA

## Main Data Objects

### `DroneDesign`

Search-space parameter object.

### `VoxelDrone`

Geometry-and-physics carrier object. Contains:

- voxel coordinates
- solid mask
- surface mask
- mass distribution
- motor masks
- total mass
- inertia
- propulsion constants

### `FlightResult`

One candidate’s mission result. Contains:

- score
- progress
- speed/stress/error histories
- snapshot timings
- fragment history
- frame-wise render data
- summary metrics

## Output Architecture

Project Y uses a single output root:

- `projects/projecty/output/`

