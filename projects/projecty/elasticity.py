from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from .geometry import VoxelDrone
except ImportError:
    from geometry import VoxelDrone


@dataclass
class ElasticState:
    u: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    body_force: np.ndarray


def lame_parameters(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return lam, mu


def initialize_elastic_state(drone: VoxelDrone) -> ElasticState:
    shape = drone.solid.shape
    return ElasticState(
        u=np.zeros(shape + (3,), dtype=float),
        v=np.zeros(shape + (3,), dtype=float),
        sigma=np.zeros(shape + (3, 3), dtype=float),
        body_force=np.zeros(shape + (3,), dtype=float),
    )


def _valid_mask(solid: np.ndarray) -> np.ndarray:
    valid = solid.copy()
    valid[0, :, :] = False
    valid[-1, :, :] = False
    valid[:, 0, :] = False
    valid[:, -1, :] = False
    valid[:, :, 0] = False
    valid[:, :, -1] = False
    return valid


def compute_strain(u: np.ndarray, h: float, solid: np.ndarray) -> np.ndarray:
    eps = np.zeros(u.shape[:-1] + (3, 3), dtype=float)
    valid = _valid_mask(solid)
    scale = 1.0 / (2.0 * h)
    du_dx = (u[2:, 1:-1, 1:-1, :] - u[:-2, 1:-1, 1:-1, :]) * scale
    du_dy = (u[1:-1, 2:, 1:-1, :] - u[1:-1, :-2, 1:-1, :]) * scale
    du_dz = (u[1:-1, 1:-1, 2:, :] - u[1:-1, 1:-1, :-2, :]) * scale
    grad = np.stack([du_dx, du_dy, du_dz], axis=-1)
    strain_core = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    eps[1:-1, 1:-1, 1:-1][valid[1:-1, 1:-1, 1:-1]] = strain_core[valid[1:-1, 1:-1, 1:-1]]
    return eps


def compute_stress(eps: np.ndarray, lam: float, mu: float, solid: np.ndarray) -> np.ndarray:
    sigma = np.zeros_like(eps)
    tr = np.trace(eps, axis1=-2, axis2=-1)
    sigma = lam * tr[..., None, None] * np.eye(3) + 2.0 * mu * eps
    sigma[~solid] = 0.0
    return sigma


def divergence_of_stress(sigma: np.ndarray, h: float, solid: np.ndarray) -> np.ndarray:
    div = np.zeros(sigma.shape[:-2] + (3,), dtype=float)
    valid = _valid_mask(solid)
    scale = 1.0 / (2.0 * h)
    d0 = (sigma[2:, 1:-1, 1:-1, :, 0] - sigma[:-2, 1:-1, 1:-1, :, 0]) * scale
    d1 = (sigma[1:-1, 2:, 1:-1, :, 1] - sigma[1:-1, :-2, 1:-1, :, 1]) * scale
    d2 = (sigma[1:-1, 1:-1, 2:, :, 2] - sigma[1:-1, 1:-1, :-2, :, 2]) * scale
    div_core = d0 + d1 + d2
    div[1:-1, 1:-1, 1:-1][valid[1:-1, 1:-1, 1:-1]] = div_core[valid[1:-1, 1:-1, 1:-1]]
    return div


def calc_von_mises(sigma: np.ndarray, solid: np.ndarray) -> np.ndarray:
    vm = np.zeros(solid.shape, dtype=float)
    tr = np.trace(sigma, axis1=-2, axis2=-1)
    dev = sigma - (tr[..., None, None] / 3.0) * np.eye(3)
    vm_all = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
    vm[solid] = vm_all[solid]
    return vm


def _anchor_mask(drone: VoxelDrone) -> np.ndarray:
    coords = drone.coords
    center_radius = 0.55 * min(drone.design.body_r1, drone.design.body_r2)
    return (
        drone.solid
        & (np.linalg.norm(coords[..., :2], axis=-1) <= center_radius)
        & (np.abs(coords[..., 2]) <= 0.5 * drone.design.body_r3)
    )


def relax_dynamic_stress(
    drone: VoxelDrone,
    state: ElasticState,
    body_force: np.ndarray,
    dt: float,
    steps: int,
    damping: float,
) -> tuple[np.ndarray, float]:
    lam, mu = lame_parameters(drone.material.young, drone.material.poisson)
    rho = drone.material.density
    h = drone.spacing
    solid = drone.solid
    anchor = _anchor_mask(drone)

    state.body_force[...] = 0.0
    state.body_force[solid] = body_force[solid]

    for _ in range(steps):
        eps = compute_strain(state.u, h, solid)
        state.sigma = compute_stress(eps, lam, mu, solid)
        div = divergence_of_stress(state.sigma, h, solid)
        valid = _valid_mask(solid)
        accel = div / rho + state.body_force / rho - damping * state.v
        state.v[valid] += dt * accel[valid]
        state.u[valid] += dt * state.v[valid]

        state.u[anchor] = 0.0
        state.v[anchor] = 0.0

    vm = calc_von_mises(state.sigma, solid)
    max_vm = float(vm[solid].max()) if np.any(solid) else 0.0
    return vm, max_vm
