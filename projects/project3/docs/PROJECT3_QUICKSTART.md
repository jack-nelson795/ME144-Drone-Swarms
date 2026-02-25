# Project 3 Quickstart (Swarm GA + Physics Simulation)

This project implements the ME144/244 Project 3 swarm simulation and genetic algorithm workflow.

## Quick Start (Recommended)

Run from the Project 3 folder (important because the code reads/writes local `*.pkl` files):

```bash
cd c:\Users\15593\Desktop\ME144\projects\project3
```

1) Write parameters

```bash
python write_parameters.py
```

2) Run the notebook end-to-end
- Open `test_scripts.ipynb`
- Run all cells

Outputs:
- Figures: `projects/project3/figures/`
- GA results pickle: `projects/project3/ga_results.pkl`
- Parameters pickle: `projects/project3/parameters.pkl`

## Dependencies

Python packages:
- `numpy`, `matplotlib` (plus Jupyter for the notebook)

System tools (optional but used for deliverables):
- FFmpeg (`ffmpeg`, `ffprobe`) for MP4 writing and for extracting evenly spaced snapshot frames.

## Common Pitfalls

- If you run from a different working directory, relative paths like `parameters.pkl` / `ga_results.pkl` may not be found.
- The MP4 writer uses Matplotlib’s `FFMpegWriter`; if FFmpeg isn’t installed, MP4 export may fail.
