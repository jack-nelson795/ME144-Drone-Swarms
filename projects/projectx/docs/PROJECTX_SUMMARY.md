# ProjectX: Summary

## What It Is Now

ProjectX is a **3D emergent swarm control** sandbox. A genetic algorithm optimizes per-drone control weights (target attraction, obstacle avoidance, separation, cohesion). The optimized weights drive a time-domain 3D simulation with collisions, target capture, return-to-base behavior, and 3D animation output.

## Key Features

- **Decentralized control laws** (local sensing, no central planner)
- **GA optimization** of control weights per drone
- **Full 3D simulation** with collision detection
- **3D animation export** for visual inspection

## Main Entry Point

```
projectx/
├── run_projectx_3d_animation.py   # Main 3D emergent control pipeline
├── show_best_run.py               # Static 3D visualization helper
```

## How to Run

```bash
python projects\projectx\run_projectx_3d_animation.py
```

## Status

This is a sandbox for Project 3 ideas and will continue evolving.
