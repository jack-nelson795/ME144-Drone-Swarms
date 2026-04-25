# Project 6

This project models a LiDAR detection and reconstruction workflow for three oscillating surfaces:

- sinusoidal surface generation with analytic gradients
- ray-tracing with surface reflection and return-time reconstruction
- point-cloud recovery for three surface amplitudes
- notebook-driven plotting, error reporting, and optional animation preview

## Run

From the `projects/project6` folder:

```powershell
jupyter notebook main.ipynb
```

Run the notebook top-to-bottom to generate the true surfaces, reconstructed point clouds, and reconstruction errors.

## Outputs

The notebook writes figures to `projects/project6/output/`:

- `surfacePlot_A0.04.png`
- `surfacePlot_A0.16.png`
- `surfacePlot_A0.64.png`
- `point_cloud_reconstruction_A0.04.png`
- `point_cloud_reconstruction_A0.16.png`
- `point_cloud_reconstruction_A0.64.png`

## Project Files

- `main.ipynb` is the main entry point and report-style workflow.
- `lidar/surfaces.py` defines the surface equation and its gradient.
- `lidar/simulation.py` handles ray propagation, reflections, reconstruction, plotting, and animation.
- `lidar/__init__.py` exposes the notebook-facing package interface.

## Notes

- The notebook uses three amplitudes: `0.04`, `0.16`, and `0.64`.
- The animation preview is optional and intended for visualization/debugging rather than as a required output artifact.
