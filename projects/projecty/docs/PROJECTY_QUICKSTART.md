# ProjectY Quickstart

## Goal

Run the Project Y pipeline, generate the optimized hostile-flight outputs, and inspect the resulting drone snapshots interactively.

## Requirements

- Python environment with the repo dependencies installed
- A working Matplotlib backend for static rendering
- Multiprocessing support if you want parallel candidate evaluation

## From Repo Root

```powershell
cd c:\Users\15593\Desktop\ME144
python projects\projecty\run_projecty.py
```

## What Happens During A Run

The pipeline performs these steps:

1. Builds an initial optimization population from the `DroneDesign` parameterization.
2. Evaluates each candidate by generating a voxel drone, simulating hostile flight, and scoring the result.
3. Evolves the population over multiple generations.
4. Rebuilds the best drone at the configured voxel resolution.
5. Writes static figures, JSON summaries, and GIF outputs.
6. Writes an interactive snapshot archive for later inspection.

## Live Console Feedback

`run_projecty.py` shows:

- a persistent progress bar
- current stage text
- per-generation candidate completion updates
- estimated remaining time in `MM:SS`

## Output Location

All outputs are written to:

- `projects/projecty/output/`

Expected files include:

- `projecty_summary.png`
- `projecty_stress_snapshots.png`
- `projecty_report.json`
- `optimization_history.json`
- `projecty_snapshot_archive.npz`
- `final_simulation.gif`
- `design_evolution.gif`

## Interactive Snapshot Viewer

After a run:

```powershell
python projects\projecty\interactive_viewer.py
```

The viewer lets you:

- orbit the DEM-style drone snapshots
- step through impact and course phases
- inspect the same per-sphere fill and edge styling used by the main animation outputs

## First Parameters To Tune

If you want shorter or longer runs, the main controls are in [config.py](../config.py):

- `optimizer_generations`
- `population_size`
- `parallel_workers`
- `voxel_resolution`
- `pulse_pressure_scale`
- `pulse_rigid_force_scale`
- `pulse_rigid_torque_scale`

## Recommended Fast Test Run

For a quick smoke test, temporarily lower:

- `optimizer_generations`
- `population_size`

Then rerun the pipeline and verify:

- `final_simulation.gif` renders
- `design_evolution.gif` renders
- `projecty_snapshot_archive.npz` exists

## If Something Looks Wrong

- If the drone shape looks off, inspect [geometry.py](../geometry.py).
- If it is not taking enough damage, inspect [flight.py](../flight.py) and [config.py](../config.py).
- If the GIF styling differs from the interactive viewer, inspect [visualization.py](../visualization.py) and [interactive_viewer.py](../interactive_viewer.py).
