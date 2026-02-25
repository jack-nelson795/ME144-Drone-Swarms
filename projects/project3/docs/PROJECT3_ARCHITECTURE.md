# Project 3 Architecture

## Key Modules

- simulation.py
  - Implements the 3D swarm dynamics + interaction control law.
  - Handles target mapping, collisions, out-of-bounds crashes, and the cost function.

- ga_class.py
  - GA scaffold: sorting, parent mixing (breeding), immigrant injection, and metrics storage.

- swarmga.py
  - Project-specific GA subclass.
  - Evaluates each candidate design string by running `Simulation.run_simulation(...)`.
  - Tracks the star metrics ($M^*$, $T^*$, $L^*$) over generations.

- animation.py
  - Plots GA convergence and star-metric curves from `ga_results.pkl`.
  - Produces a 3D MP4 animation of the best design.

- export_frames.py
  - Extracts evenly spaced PNG frames from the MP4 using `ffprobe` + `ffmpeg`.

- test_scripts.ipynb
  - Orchestrates the full workflow:
    - regenerate parameters
    - run GA
    - save `ga_results.pkl`
    - replay best simulation
    - save plots, MP4, and snapshot PNGs

## Data / Artifacts

- parameters.pkl
  - Single dictionary of constants (domain, physics, GA hyperparameters).

- ga_results.pkl
  - Dictionary of GA histories and best parent strings.

- figures/
  - `cost_plot.png`
  - `LMTmin_plot.png`, `LMTPAve_plot.png`, `LMTAve_plot.png`
  - `{Nm}_agents_{No}_obs_{Nt}_tar.mp4`
  - `snapshot_01.png` … `snapshot_05.png`

## Design Interfaces

- GA → Simulation
  - GA passes a 15-element design string `LAM` (all values constrained to $[0,2]$).
  - `Simulation._read_LAM_vector_from_GA(...)` maps those 15 values to the control-law constants.
