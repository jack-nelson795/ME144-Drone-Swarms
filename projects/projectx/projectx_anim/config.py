from __future__ import annotations

import os
import numpy as np

# ============================================================================
# CONFIGURATION - Set these once and they'll be used everywhere
# ============================================================================
NUM_DRONES = 25           # Number of drones in swarm
NUM_TARGETS = 100          # Number of target zones
NUM_OBSTACLES = 75        # Number of obstacles
RANDOM_SEED = 887215      # Use None for random, or set to integer for reproducibility

# ============================================================================
# ENVIRONMENT CONFIGURATION (Matching Research Image)
# ============================================================================

# 3D Domain
BOUNDS_3D = (0, 100, 0, 100, 0, 50)  # x, y, z
BASE_POS = np.array([50.0, 100.0, 20.0])  # Center at far edge, elevated

# Random generator
# Use the RANDOM_SEED config variable above
np.random.seed(RANDOM_SEED)

# Control law weights per drone
N_CONTROL_WEIGHTS = 4

# GA hyperparameters
S = 75
P = 15
K = 20
G = 50


# ============================================================================
# Optional speed controls (OFF by default)
#
# Windows multiprocessing + GIF rendering can be slow. Enable fast mode via:
#   PROJECTX_FAST=1 python projects/projectx/run_projectx_3d_animation.py
# ============================================================================

FAST_MODE = os.getenv('PROJECTX_FAST', '').strip().lower() in {'1', 'true', 'yes', 'y'}

# If True, skip the full visualization step (plots + GIF). Default False.
SKIP_VISUALIZATION = os.getenv('PROJECTX_SKIP_VIZ', '').strip().lower() in {'1', 'true', 'yes', 'y'}

# Downsample animation frames: use every Nth simulation step when making GIF.
# Default 1 preserves the original behavior.
ANIMATION_STRIDE = int(os.getenv('PROJECTX_ANIM_STRIDE', '1'))
if ANIMATION_STRIDE < 1:
	ANIMATION_STRIDE = 1


if FAST_MODE:
	# GA (smaller = much faster)
	S = int(os.getenv('PROJECTX_S', '10'))
	P = int(os.getenv('PROJECTX_P', '5'))
	K = int(os.getenv('PROJECTX_K', '3'))
	G = int(os.getenv('PROJECTX_G', '2'))

	# Visualization defaults for fast mode
	SKIP_VISUALIZATION = os.getenv('PROJECTX_SKIP_VIZ', '1').strip().lower() in {'1', 'true', 'yes', 'y'}
	ANIMATION_STRIDE = int(os.getenv('PROJECTX_ANIM_STRIDE', str(ANIMATION_STRIDE)))
	if ANIMATION_STRIDE < 1:
		ANIMATION_STRIDE = 1
