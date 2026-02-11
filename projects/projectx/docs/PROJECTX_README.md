# ProjectX: Emergent Swarm Control (3D)

**Objective**: Evolve decentralized control laws for a 3D drone swarm that captures targets, avoids obstacles, and returns to base. The system uses a genetic algorithm to optimize per-drone control weights and then runs a full 3D time-domain simulation with collision checks and animation output.

## Overview

ProjectX is a sandbox for multi-agent, Zohdi-inspired emergent control. The core idea is that each drone uses local sensing and a weighted sum of behaviors (target attraction, obstacle avoidance, separation, cohesion). The GA discovers weights that produce coordinated swarm behavior without centralized planning.

## File Structure

```
projectx/
├── run_projectx_3d_animation.py   # Main 3D emergent control pipeline
├── show_best_run.py               # Static 3D visualization helper
├── output/                        # Generated logs/figures (ignored in git)
```

## How to Run

```bash
cd c:\Users\15593\Desktop\ME144
python projects\projectx\run_projectx_3d_animation.py
```

Outputs:
- Console: optimization + simulation progress
- Logs: `projects/projectx/output/logs/`
- Figures/animations: `projects/projectx/output/figures/`

## Notes

This project is intentionally experimental and evolving. The authoritative entry point is `run_projectx_3d_animation.py`.
