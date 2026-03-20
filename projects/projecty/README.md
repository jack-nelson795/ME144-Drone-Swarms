# Project Y

This project is a chapter-4-inspired drone design study built around the textbook:

- voxel chassis generation using the generalized ellipsoid envelope in Equation 4.1
- union-based modular assembly from Equation 4.2
- voxel stress simulation from Equations 4.3-4.24
- rigid-body cluster flight dynamics from Equations 4.81-4.101
- targeted pressure-wave attacks using Equation 4.102
- moving-frame stress post-processing in the spirit of Equations 4.105-4.109

## Run

From the repository root:

```powershell
python projects\projecty\run_projecty.py
```

## Outputs

The script writes static summaries to `projects/projecty/output/` and the required animation GIF to `results/animations/`:

- `projecty_summary.png`
- `projecty_stress_snapshots.png`
- `projecty_report.json`
- `optimization_history.json`
- `results/animations/final_simulation.gif`
- `results/animations/design_evolution.gif`
- `projects/projecty/output/projecty_snapshot_archive.npz`

## Visualization Notes

- The chassis is rendered as a DEM-style lattice with one sphere marker per occupied voxel center.
- Sphere marker sizing is tuned so neighboring lattice voxels visually touch or slightly overlap in the oblique 3D view.
- Rotor/end masses are rendered as larger spheres at the arm tips.
- Per-voxel coloring transitions from warm orange/red for intact material toward green/blue as stress damage accumulates.
- Detached connected components remain visible as moving sphere clusters in the final GIF.
- Impact and course-phase snapshots are exported into `projecty_snapshot_archive.npz` for the interactive viewer.

## Interactive Viewer

Launch the local interactive 3D snapshot viewer after a run:

```powershell
python projects\projecty\interactive_viewer.py
```

- Drag to orbit the 3D view.
- Use the slider or Next/Prev buttons to inspect impact-phase and course snapshots.

## Performance Notes

- Candidate evaluation can run in parallel via `parallel_workers` in `config.py`.
- The design-evolution GIF playback speed is controlled by `evolution_hold_frames` and `evolution_gif_fps`.
