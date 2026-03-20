from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ANIMATION_DIR = OUTPUT_DIR


@dataclass(frozen=True)
class Material:
    young: float
    poisson: float
    density: float
    stress_limit: float


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.05
    horizon: float = 12.0
    gravity: float = 9.81
    voxel_resolution: int = 28
    stress_relaxation_steps: int = 4
    stress_relaxation_dt: float = 6.0e-5
    stress_damping: float = 30.0
    optimizer_generations: int = 100
    population_size: int = 100
    elite_count: int = 15
    random_seed: int = 144
    parallel_workers: int = 10
    fragment_integrity_threshold: float = 0.2
    damage_softening_threshold: float = 0.3
    damage_rate: float = 3.2
    debris_drag: float = 0.18
    debris_spin_decay: float = 0.75
    pulse_pressure_scale: float = 9.25
    pulse_rigid_force_scale: float = 0.009
    pulse_rigid_torque_scale: float = 0.14
    evolution_hold_frames: int = 4
    evolution_gif_fps: int = 12
    snapshot_event_cooldown_steps: int = 6


DEFAULT_MATERIAL = Material(
    young=1.8e7,
    poisson=0.31,
    density=1180.0,
    stress_limit=3.5e5,
)
