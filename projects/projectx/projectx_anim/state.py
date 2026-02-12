from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from . import config
from .env_generation import generate_random_targets_and_obstacles

# ============================================================================
# GLOBAL INITIALIZATION (only in main process)
# ============================================================================

# Initialize globals from config right away
N_DRONES = config.NUM_DRONES
SEED = config.RANDOM_SEED

TARGET_ZONES: np.ndarray = np.array([])
OBSTACLES: List = []
N_TARGETS: int = 0
N_CONTROL_WEIGHTS = config.N_CONTROL_WEIGHTS
DV: int = N_DRONES * N_CONTROL_WEIGHTS
LIM: np.ndarray = np.tile([0.0, 1.0], (DV, 1))

S = config.S
P = config.P
K = config.K
G = config.G

OUTPUT_DIR: Path = Path()
FIG_DIR: Path = Path()


def init_globals() -> None:
    """Initialize global variables - called only in main process."""
    global TARGET_ZONES, OBSTACLES, N_TARGETS, N_DRONES, DV, LIM, OUTPUT_DIR, FIG_DIR, SEED

    # Set N_DRONES and SEED from config
    N_DRONES = config.NUM_DRONES
    SEED = config.RANDOM_SEED

    # Generate random seed if not set
    if SEED is None:
        SEED = np.random.randint(0, 1000000)

    # Generate field layout using config variables
    TARGET_ZONES, OBSTACLES = generate_random_targets_and_obstacles(
        n_targets=config.NUM_TARGETS,
        n_obstacles=config.NUM_OBSTACLES,
        min_distance=8.0,
        seed=SEED,
    )

    N_TARGETS = len(TARGET_ZONES)
    DV = N_DRONES * N_CONTROL_WEIGHTS
    LIM = np.tile([0.0, 1.0], (DV, 1))

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
    FIG_DIR = OUTPUT_DIR / 'figures'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
