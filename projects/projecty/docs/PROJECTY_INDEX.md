# ME144 ProjectY Index

## What This Is

ProjectY is a Chapter 4-inspired voxel/DEM drone design, attack-survivability, and flight-optimization sandbox. It combines:

- generalized-ellipsoid voxel geometry
- DEM-style one-sphere-per-voxel visualization
- structural stress and damage accumulation
- rigid-body flight with hostile pressure-wave attacks
- design optimization over geometry, mass, and propulsion parameters
- animated and interactive 3D outputs

## Start Here

- [run_projecty.py](../run_projecty.py) - main end-to-end Project Y pipeline
- [interactive_viewer.py](../interactive_viewer.py) - local interactive 3D snapshot viewer
- [README.md](../README.md) - concise project-level overview

## Recommended Reading Order

1. [PROJECTY_QUICKSTART.md](./PROJECTY_QUICKSTART.md)
2. [PROJECTY_README.md](./PROJECTY_README.md)
3. [PROJECTY_ARCHITECTURE.md](./PROJECTY_ARCHITECTURE.md)
4. [PROJECTY_CONTROL_LAW_GUIDE.md](./PROJECTY_CONTROL_LAW_GUIDE.md)
5. [PROJECTY_README_CONTROL_OPTIMIZATION.md](./PROJECTY_README_CONTROL_OPTIMIZATION.md)

## Full Documentation Set

- [PROJECTY_README.md](./PROJECTY_README.md)
- [PROJECTY_SUMMARY.md](./PROJECTY_SUMMARY.md)
- [PROJECTY_QUICKSTART.md](./PROJECTY_QUICKSTART.md)
- [PROJECTY_ARCHITECTURE.md](./PROJECTY_ARCHITECTURE.md)
- [PROJECTY_IMPLEMENTATION_SUMMARY.md](./PROJECTY_IMPLEMENTATION_SUMMARY.md)
- [PROJECTY_CONTROL_LAW_GUIDE.md](./PROJECTY_CONTROL_LAW_GUIDE.md)
- [PROJECTY_README_CONTROL_OPTIMIZATION.md](./PROJECTY_README_CONTROL_OPTIMIZATION.md)
- [PROJECTY_COMPLETION_REPORT.md](./PROJECTY_COMPLETION_REPORT.md)

## Core Source Files

- [config.py](../config.py)
- [design.py](../design.py)
- [geometry.py](../geometry.py)
- [elasticity.py](../elasticity.py)
- [flight.py](../flight.py)
- [optimization.py](../optimization.py)
- [visualization.py](../visualization.py)
- [interactive_viewer.py](../interactive_viewer.py)
- [TEXTBOOK_MAPPING.md](../TEXTBOOK_MAPPING.md)

## Main Outputs

All Project Y outputs are written under:

- `projects/projecty/output/`

Typical artifacts:

- `projecty_summary.png`
- `projecty_stress_snapshots.png`
- `projecty_report.json`
- `optimization_history.json`
- `projecty_snapshot_archive.npz`
- `final_simulation.gif`
- `design_evolution.gif`

## Typical Run Command

```powershell
cd c:\Users\15593\Desktop\ME144
python projects\projecty\run_projecty.py
```
