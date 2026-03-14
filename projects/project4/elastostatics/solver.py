"""
Explicit finite-difference elastodynamics on a fixed grid.

The governing equations are:

  eps_ijk = 1/2 (grad(u_ijk) + grad(u_ijk).T) (eqn. 11.11)
  sigma_ijk = lambda * tr(eps) * I + 2 mu * eps_ijk (eqn. 11.65)
  rho * u_ddot = div(sigma) (eqn. 11.22)

Discretization: central finite differences on a uniform grid.
"""

import numpy as np
from elastostatics.grid import ElasticGrid


def _solid_cd_mask(solid: np.ndarray) -> np.ndarray:
    """Mask of voxels where central differences are valid.

    A voxel is valid if it is solid and not on the outer grid boundary.

    Note: For solid/air interfaces we still use central differences with the
    neighbor values (air voxels remain at zero displacement/stress). This is a
    simple approximation that ensures thin solid features (e.g., drone arms)
    still participate in the simulation.
    """
    valid = solid.copy()
    valid[0, :, :] = False
    valid[-1, :, :] = False
    valid[:, 0, :] = False
    valid[:, -1, :] = False
    valid[:, :, 0] = False
    valid[:, :, -1] = False
    return valid

def compute_strain(u, hx, hy, hz, solid: np.ndarray | None = None):
    """
    Return strain tensor eps[i,j,k,3,3] from displacement field.
    
    NOTE: we don't compute derivatives at the boundary to avoid introducing 1-sided derivatives or
          "ghost node" schemes. However, this means that the strain and stress at the boundaries
          will be zero, and so we will have to do some cheating to get our solution to be accurate.
          See extrapolate_sigma_to_z_boundaries() as well.
    """
    nx, ny, nz, _ = u.shape
    eps = np.zeros((nx, ny, nz, 3, 3))

    if solid is None:
        # Full-solid case (original behavior)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) / (2.0 * hx)
                    du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) / (2.0 * hy)
                    du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) / (2.0 * hz)

                    grad = np.stack([du_dx, du_dy, du_dz], axis=1)  # del u_i/ del x_j
                    eps[i, j, k] = 0.5 * (grad + grad.T)
    else:
        # Masked-solid case: only compute at valid solid voxels.
        valid = _solid_cd_mask(solid)
        for i, j, k in np.argwhere(valid):
            du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) / (2.0 * hx)
            du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) / (2.0 * hy)
            du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) / (2.0 * hz)

            grad = np.stack([du_dx, du_dy, du_dz], axis=1)  # del u_i/ del x_j
            eps[i, j, k] = 0.5 * (grad + grad.T)

    return eps


def compute_stress(eps, lam, mu, solid: np.ndarray | None = None):
    """Hooke's law for isotropic linear elasticity."""
    sigma = np.zeros_like(eps)

    if solid is None:
        for i in range(eps.shape[0]):
            for j in range(eps.shape[1]):
                for k in range(eps.shape[2]):
                    tr = np.trace(eps[i, j, k])
                    sigma[i, j, k] = lam * tr * np.eye(3) + 2.0 * mu * eps[i, j, k]
    else:
        for i, j, k in np.argwhere(solid):
            tr = np.trace(eps[i, j, k])
            sigma[i, j, k] = lam * tr * np.eye(3) + 2.0 * mu * eps[i, j, k]

    return sigma


def divergence_of_stress(sigma, hx, hy, hz, solid: np.ndarray | None = None):
    """Return div(sigma) as a vector field."""
    nx, ny, nz, _, _ = sigma.shape
    div = np.zeros((nx, ny, nz, 3))

    if solid is None:
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    for a in range(3):
                        d_sig_a0_dx = (sigma[i + 1, j, k, a, 0] - sigma[i - 1, j, k, a, 0]) / (2.0 * hx)
                        d_sig_a1_dy = (sigma[i, j + 1, k, a, 1] - sigma[i, j - 1, k, a, 1]) / (2.0 * hy)
                        d_sig_a2_dz = (sigma[i, j, k + 1, a, 2] - sigma[i, j, k - 1, a, 2]) / (2.0 * hz)
                        div[i, j, k, a] = d_sig_a0_dx + d_sig_a1_dy + d_sig_a2_dz
    else:
        valid = _solid_cd_mask(solid)
        for i, j, k in np.argwhere(valid):
            for a in range(3):
                d_sig_a0_dx = (sigma[i + 1, j, k, a, 0] - sigma[i - 1, j, k, a, 0]) / (2.0 * hx)
                d_sig_a1_dy = (sigma[i, j + 1, k, a, 1] - sigma[i, j - 1, k, a, 1]) / (2.0 * hy)
                d_sig_a2_dz = (sigma[i, j, k + 1, a, 2] - sigma[i, j, k - 1, a, 2]) / (2.0 * hz)
                div[i, j, k, a] = d_sig_a0_dx + d_sig_a1_dy + d_sig_a2_dz

    return div

def step_elastic_grid(grid: ElasticGrid, dt: float, damping: float = 0.0, enforce_bc=None):
    """
        Advance the elastic grid by one time step.

        Important: For second-order elastodynamics written as
            u' = v,
            v' = a(u, v, ...),
        strict (forward) explicit Euler on (u,v) is numerically unstable for oscillatory
        systems (e.g., undamped modes behave like harmonic oscillators and spiral outward).

        We therefore use a semi-implicit / symplectic Euler update, which remains simple
        but produces bounded energy behavior and allows damping to drive the system toward
        a quasi-static equilibrium:

            a^n = a(u^n, v^n, ...)
            v^{n+1} = v^n + dt * a^n
            u^{n+1} = u^n + dt * v^{n+1}
      
    Args:
        grid: ElasticGrid object
        dt: time step
        damping: damping coefficient (increase for stability)
        enforce_bc: callback function to re-apply boundary conditions
    """
    hx, hy, hz = grid.hx, grid.hy, grid.hz

    # Ensure boundary conditions are satisfied for the state at time n
    # before computing strain/stress/divergence.
    if enforce_bc is not None:
        enforce_bc(grid)

    eps = compute_strain(grid.u, hx, hy, hz, solid=grid.solid)
    grid.sigma = compute_stress(eps, grid.lam, grid.mu, solid=grid.solid)

    extrapolate_sigma_to_z_boundaries(grid)

    div = divergence_of_stress(grid.sigma, hx, hy, hz, solid=grid.solid)

    valid = _solid_cd_mask(grid.solid)
    for i, j, k in np.argwhere(valid):
        v_old = grid.v[i, j, k].copy()
        a = div[i, j, k] / grid.rho - damping * v_old
        v_new = v_old + dt * a
        grid.v[i, j, k] = v_new
        grid.u[i, j, k] += dt * v_new
  
    # Re-apply boundary conditions after update
    if enforce_bc is not None:
        enforce_bc(grid)


def extrapolate_sigma_to_z_boundaries(grid: ElasticGrid):
    """
    Copies stress from slices adjacent to the bottom and top.

    Why do we need this?
    compute_strain() only fills interior nodes (i=1..nx-2, j=1..ny-2, k=1..nz-2),
    so sigma at the boundary faces is never computed and stays at zero.
    So derivatives of stress at the boundary are artificially high,  creating a false force that 
    prevents the solution from converging. 
    """
    # Copy stresses into z-boundary slices to avoid spurious forces.
    # Respect the solid mask (air voxels remain zero).
    grid.sigma[:, :, 0, :, :] = grid.sigma[:, :, 1, :, :]
    grid.sigma[:, :, -1, :, :] = grid.sigma[:, :, -2, :, :]
    grid.sigma[~grid.solid] = 0.0


def compute_max_wavespeed(lam, mu, rho):
    """
    Compute maximum wave speed for CFL condition.
    
    In the same way that acoustic waves have a finite speed in sound,
    mechanical disturbances, like our pulling on the end of the block
    travel through the material at a certain speed which should inform the 
    rate at which we update the solution (aka the timestep duration dt).
    """
    # P-wave speed in isotropic medium
    c_p = np.sqrt((lam + 2*mu) / rho)
    return c_p


def compute_stable_dt(grid: ElasticGrid, safety_factor: float = 0.1):
    """
    Compute stable time step based on CFL condition.
    dt < (min_grid_spacing) / max_wavespeed

    Essentially we want to update the solution faster than the stress field can propagate between
    nodes, otherwise we "miss it"
    
    safety_factor: multiply CFL limit by this (0.1 = very conservative)
    """
    h_min = min(grid.hx, grid.hy, grid.hz)
    c_max = compute_max_wavespeed(grid.lam, grid.mu, grid.rho)
    dt_cfl = h_min / c_max
    return safety_factor * dt_cfl
