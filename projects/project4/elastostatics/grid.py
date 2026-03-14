"""
Fixed-grid continuum solid for linear elastodynamics.

Each grid node stores:
  u[i,j,k] : displacement vector
  v[i,j,k] : velocity vector
  sigma    : Cauchy stress tensor
"""

import numpy as np

class ElasticGrid:
    def __init__(self, shape, spacing, density, young, poisson, solid=None):
        """
        Parameters
        ----------
        shape   : (nx, ny, nz)
        spacing : grid spacing (dx, dy, dz)
        density : material density rho
        young   : Young's modulus E
        poisson : Poisson ratio nu
        solid   : optional boolean mask of shape (nx, ny, nz) indicating material voxels.
                  If None, the entire grid is treated as solid.
        """
        self.nx, self.ny, self.nz = shape
        self.hx, self.hy, self.hz = spacing
        self.rho = float(density)

        # fill in the Lamé parameters
        E = float(young)
        nu = float(poisson)
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Solid/air occupancy mask
        if solid is None:
            self.solid = np.ones((self.nx, self.ny, self.nz), dtype=bool)
        else:
            solid_arr = np.asarray(solid, dtype=bool)
            if solid_arr.shape != (self.nx, self.ny, self.nz):
                raise ValueError(
                    f"solid mask must have shape {(self.nx, self.ny, self.nz)}, got {solid_arr.shape}"
                )
            self.solid = solid_arr

        # State fields
        self.u = np.zeros((self.nx, self.ny, self.nz, 3))
        self.v = np.zeros_like(self.u)
        self.sigma = np.zeros((self.nx, self.ny, self.nz, 3, 3))
   
    def calc_von_mises(self):
        """return the von-mises stresses of all grid cells."""
        von_mises = np.zeros([self.nx, self.ny, self.nz])
        sig = self.sigma
        for i, j, k in np.ndindex(sig.shape[:3]):
            if not self.solid[i, j, k]:
                continue
            sig_ijk = sig[i, j, k, :, :]
            # von Mises stress from deviatoric stress: sqrt(3/2 * s_ij s_ij)
            tr = np.trace(sig_ijk)
            s = sig_ijk - (tr / 3.0) * np.eye(3)
            von_mises[i, j, k] = np.sqrt(1.5 * np.sum(s * s))
        return von_mises  