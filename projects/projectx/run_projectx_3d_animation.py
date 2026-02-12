# projects/projectx/run_projectx_3d_animation.py
"""
ME144/244 — ProjectX 3D Swarm Animation (Zohdi-Inspired)

Recreates the hostile drone incursion scenario from research:
- Drones start at base
- Swarm sweeps through obstacle field
- Visits all target zones
- GA optimizes the trajectory
- Outputs smooth 3D animation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# IMPORTANT: Import matplotlib ONLY in main process, not in workers.
# matplotlib is very slow to import and can cause deadlocks in multiprocessing.

from projectx_anim import config
from projectx_anim.pipeline import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ProjectX 3D swarm GA + simulation + visualization")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a much smaller GA + quicker sims (intended for smoke tests / debugging)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization + GIF generation (saves time)",
    )
    args = parser.parse_args()

    print(f"\nEnvironment:")
    print(f"  Base position: {config.BASE_POS}")
    print(f"  Drones: {config.NUM_DRONES}")
    print(f"  Target zones: {config.NUM_TARGETS}")
    print(f"  Obstacles: {config.NUM_OBSTACLES}")
    print(f"  Design variables (control weights): will be set on init")

    main(fast=args.fast, create_viz=not args.no_viz)
