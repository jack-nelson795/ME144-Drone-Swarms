"""\
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems
Project 3: Swarm
Spring 2026 Semester
GSI - Tommy Hosmer

Student implementation: complete physics + control law swarm simulation.

The simulation accepts:
  - parameters: fixed environment/constants (from parameters.pkl)
  - LAM: 15-element design string optimized by the GA

Implements (per project spec):
  - Newton's 2nd law with propulsion + aerodynamic drag
  - Semi-Implicit Euler integration
  - Exponentially weighted interaction control law
  - Target mapping + collision logic with required priority
  - Cost: PI = 70*M* + 10*T* + 20*L*
"""

from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import numpy as np


class Simulation:
    def __init__(self, parameters: Dict, path_to_genes: str):
        np.random.seed(42)

        # Environment parameters are fixed so we can set them when we instantiate the class
        for key, value in parameters.items():
            # Ensure arrays are numpy arrays when needed
            if key == "va":
                value = np.asarray(value, dtype=float)
            setattr(self, key, value)

        # Save initial number of targets and agents for cost calculation
        self.nT0 = int(self.Nt)
        self.nA0 = int(self.Nm)
        self.path_to_genes = path_to_genes

        # Current counts (tracked for convenience)
        self.nT = int(self.Nt)
        self.nA = int(self.Nm)

        # Initialize obstacle locations
        self.obs = np.column_stack(
            [
                (self.locx - (-self.locx)) * np.random.rand(self.No) + (-self.locx),
                (self.locy - (-self.locy)) * np.random.rand(self.No) + (-self.locy),
                (self.locz - (-self.locz)) * np.random.rand(self.No) + (-self.locz),
            ]
        )

        # Initialize target locations
        self.tar = np.column_stack(
            [
                (self.locx - (-self.locx)) * np.random.rand(self.Nt) + (-self.locx),
                (self.locy - (-self.locy)) * np.random.rand(self.Nt) + (-self.locy),
                (self.locz - (-self.locz)) * np.random.rand(self.Nt) + (-self.locz),
            ]
        )

        # Initialize drone positions and velocities
        pos = np.array(
            [
                (self.xmax - 0.05 * self.xmax) * np.ones(self.Nm),
                np.linspace(-self.ymax + 0.05 * self.ymax, self.ymax - 0.05 * self.ymax, self.Nm),
                np.zeros(self.Nm),
            ]
        )
        self.pos = pos.T
        self.vel = np.zeros((self.Nm, 3), dtype=float)

        # Store initial conditions (copies, so each run starts clean)
        self.pos0 = self.pos.copy()
        self.vel0 = self.vel.copy()
        self.tar0 = self.tar.copy()


    def _read_optimal_string(self) -> None:
        with open(self.path_to_genes, "rb") as file:
            drone_genes = pickle.load(file)
        for key, value in drone_genes.items():
            setattr(self, key, value)


    def _read_LAM_vector_from_GA(self, LAM: List[float]) -> None:
        """Map 15-element design string into class parameters."""
        if len(LAM) != 15:
            raise ValueError(f"Expected 15 design variables, got {len(LAM)}")

        (
            self.Wmt,
            self.Wmo,
            self.Wmm,
            self.wt1,
            self.wt2,
            self.wo1,
            self.wo2,
            self.wm1,
            self.wm2,
            self.a1,
            self.a2,
            self.b1,
            self.b2,
            self.c1,
            self.c2,
        ) = [float(x) for x in LAM]


    @staticmethod
    def _safe_unit_vectors(diff: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        """Return (unit_vectors, norms) for a (...,3) diff array, safely handling zeros/NaNs."""
        norms = np.linalg.norm(diff, ord=2, axis=-1)
        unit = np.divide(
            diff,
            norms[..., np.newaxis],
            out=np.zeros_like(diff, dtype=float),
            where=(norms[..., np.newaxis] > eps) & np.isfinite(norms[..., np.newaxis]),
        )
        # If diff contains NaNs, unit should be zeros for those entries
        unit = np.nan_to_num(unit, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.nan_to_num(norms, nan=np.nan, posinf=np.nan, neginf=np.nan)
        return unit, norms


    def _compute_distances(self) -> None:
        """Vectorized drone-target, drone-drone, and drone-obstacle differences and distances."""
        # Differences (vector from drone i to object j)
        # Shapes:
        #   mtDiff: (Nm, Nt, 3)
        #   mmDiff: (Nm, Nm, 3)
        #   moDiff: (Nm, No, 3)
        self.mtDiff = self.tar[np.newaxis, :, :] - self.pos[:, np.newaxis, :]
        self.mmDiff = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :]
        self.moDiff = self.obs[np.newaxis, :, :] - self.pos[:, np.newaxis, :]

        # Distances
        self.mtDist = np.linalg.norm(self.mtDiff, ord=2, axis=2)
        self.mmDist = np.linalg.norm(self.mmDiff, ord=2, axis=2)
        self.moDist = np.linalg.norm(self.moDiff, ord=2, axis=2)

        # Ignore self-self interactions
        np.fill_diagonal(self.mmDist, np.nan)
        self.mmDiff[np.arange(self.mmDiff.shape[0]), np.arange(self.mmDiff.shape[1])] = np.nan


    def _check_collisions(self) -> None:
        """Compute mapped targets and crashed drones for the current timestep.

        Priority (when multiple events happen in same step):
          1) agent-target mapping
          2) agent-agent collisions
          3) agent-obstacle collisions

        Drones leaving domain bounds are treated as crashes.
        """

        # Step 2: mapping + collision hits from current distance matrices
        mtHit = np.where(self.mtDist <= self.agent_sight)
        moHit = np.where(self.moDist <= self.crash_range)
        mmHit = np.where(self.mmDist <= self.crash_range)

        # Step 3: out-of-bounds crashes (|x|<=xmax, |y|<=ymax, |z|<=zmax)
        xLost = np.where(np.abs(self.pos[:, 0]) > self.xmax)
        yLost = np.where(np.abs(self.pos[:, 1]) > self.ymax)
        zLost = np.where(np.abs(self.pos[:, 2]) > self.zmax)

        mLost = np.unique(np.hstack([xLost[0], yLost[0], zLost[0]]))

        # Targets mapped by any drone within sight
        self.tarMapped = np.unique(mtHit[1])

        # Agent-agent collisions: crash both participants
        mmCrash = np.unique(np.hstack([mmHit[0], mmHit[1]]))
        moCrash = np.unique(moHit[0])

        self.mCrash = np.unique(np.hstack([mmCrash, moCrash, mLost]))


    def _label_crash_and_map(self) -> None:
        """Label crashed drones and mapped targets by setting corresponding rows/cols to NaN."""
        if self.mCrash.size:
            self.mtDist[self.mCrash, :] = np.nan
            self.mtDiff[self.mCrash, :, :] = np.nan

            self.mmDist[self.mCrash, :] = np.nan
            self.mmDist[:, self.mCrash] = np.nan
            self.mmDiff[self.mCrash, :, :] = np.nan
            self.mmDiff[:, self.mCrash, :] = np.nan

            self.moDist[self.mCrash, :] = np.nan
            self.moDiff[self.mCrash, :, :] = np.nan

        if self.tarMapped.size:
            self.tar[self.tarMapped, :] = np.nan
            self.mtDist[:, self.tarMapped] = np.nan
            self.mtDiff[:, self.tarMapped, :] = np.nan


    def _compute_dynamics(self) -> None:
        """Compute interactions, forces, and apply Semi-Implicit Euler update."""

        # Unit direction vectors
        nMT, _ = self._safe_unit_vectors(self.mtDiff)
        nMO, _ = self._safe_unit_vectors(self.moDiff)
        nMM, _ = self._safe_unit_vectors(self.mmDiff)

        # Step 4: exponentially weighted interaction vectors (Eqns 13/17/21)
        # For NaN distances (mapped targets / dead drones), exponentials become NaN;
        # nan_to_num turns them into 0 contribution.
        nMThat_mag = self.wt1 * np.exp(-self.a1 * self.mtDist) - self.wt2 * np.exp(-self.a2 * self.mtDist)
        nMThat = np.nan_to_num(nMThat_mag[:, :, np.newaxis] * nMT)

        nMOhat_mag = self.wo1 * np.exp(-self.b1 * self.moDist) - self.wo2 * np.exp(-self.b2 * self.moDist)
        nMOhat = np.nan_to_num(nMOhat_mag[:, :, np.newaxis] * nMO)

        nMMhat_mag = self.wm1 * np.exp(-self.c1 * self.mmDist) - self.wm2 * np.exp(-self.c2 * self.mmDist)
        nMMhat = np.nan_to_num(nMMhat_mag[:, :, np.newaxis] * nMM)

        # Sum over all objects for each drone
        Nmt = np.nansum(nMThat, axis=1)
        Nmo = np.nansum(nMOhat, axis=1)
        Nmm = np.nansum(nMMhat, axis=1)

        # Step 5: total interaction and propulsion direction (Eqns 23/24)
        Ntot = self.Wmt * Nmt + self.Wmo * Nmo + self.Wmm * Nmm
        Ntot_norm = np.linalg.norm(Ntot, ord=2, axis=1)

        nProp = np.zeros_like(Ntot)
        nonzero = (Ntot_norm > 1e-12) & np.isfinite(Ntot_norm)
        nProp[nonzero] = Ntot[nonzero] / Ntot_norm[nonzero, np.newaxis]

        # Safe fallback if Ntot is ~0: no thrust
        # (keeps behavior stable and avoids random non-determinism)
        fProp = self.Fp * nProp

        # Drag force (Eqn for F_d)
        velocity_diff = np.nan_to_num(self.va - self.vel, nan=0.0)
        vNormDiff = np.linalg.norm(velocity_diff, ord=2, axis=1)[:, np.newaxis]
        fDrag = 0.5 * self.ra * self.Cdi * self.Ai * vNormDiff * velocity_diff

        fTot = fProp + fDrag

        # Step 6: Semi-Implicit Euler update (Eqns 8/9)
        alive = ~np.isnan(self.pos[:, 0])
        if np.any(alive):
            self.vel[alive, :] = self.vel[alive, :] + (fTot[alive, :] / self.mi) * self.dt
            self.pos[alive, :] = self.pos[alive, :] + self.vel[alive, :] * self.dt


    def _compute_cost(self, counter: int) -> Tuple[float, float, float, float]:
        """Compute PI, M*, T*, L* for a trial."""
        remaining_targets = int(np.count_nonzero(~np.isnan(self.tar[:, 0])))
        remaining_agents = int(np.count_nonzero(~np.isnan(self.pos[:, 0])))

        # Step 7: nondimensionalized cost terms (Eqns 25/26)
        Mstar = remaining_targets / float(self.nT0)
        Tstar = (counter * self.dt) / float(self.tf)
        Lstar = (self.nA0 - remaining_agents) / float(self.nA0)

        PI = self.w1 * Mstar + self.w2 * Tstar + self.w3 * Lstar
        return PI, Mstar, Tstar, Lstar


    def _remove_crash_and_map(self) -> None:
        """Remove crashed drones and mapped targets by setting state to NaN."""
        if self.mCrash.size:
            self.pos[self.mCrash, :] = np.nan
            self.vel[self.mCrash, :] = np.nan

        if self.tarMapped.size:
            self.tar[self.tarMapped, :] = np.nan

        # Step 8: update counts
        self.nT = int(np.count_nonzero(~np.isnan(self.tar[:, 0])))
        self.nA = int(np.count_nonzero(~np.isnan(self.pos[:, 0])))


    def run_simulation(self, read_LAM: bool = False, LAM: List[float] | np.ndarray = []):
        if read_LAM:
            if len(LAM) == 0:
                raise ValueError("LAM is empty but read_LAM is true.")
            self._read_LAM_vector_from_GA(list(LAM))
        else:
            self._read_optimal_string()

        # Reset state for a clean run
        self.pos = self.pos0.copy()
        self.vel = self.vel0.copy()
        self.tar = self.tar0.copy()
        self.nT = int(self.nT0)
        self.nA = int(self.nA0)

        tStep = int(np.ceil(self.tf / self.dt))
        counter = 0

        posData = [self.pos.copy()]
        tarData = [self.tar.copy()]
        obsData = self.obs.copy()

        for i in range(tStep):
            self.timestep = i

            # Step 9: method order per scaffold + spec
            self._compute_distances()
            self._check_collisions()
            self._label_crash_and_map()
            self._remove_crash_and_map()

            # Stop if all targets mapped OR all agents crashed
            if np.all(np.isnan(self.tar)) or np.all(np.isnan(self.pos)):
                break

            self._compute_dynamics()

            posData.append(self.pos.copy())
            tarData.append(self.tar.copy())
            counter += 1

        PI, Mstar, Tstar, Lstar = self._compute_cost(counter)
        return PI, posData, tarData, obsData, counter, Mstar, Tstar, Lstar


if __name__ == "__main__":
    with open("parameters.pkl", "rb") as file:
        parameters = pickle.load(file)

    sim = Simulation(parameters, path_to_genes="")
    LAM = [
        0.3988442888626489,
        1.949567329046,
        0.5777614822547401,
        1.064381542531382,
        0.2389800770766377,
        1.8583543294267468,
        0.08971249931120329,
        0.18965425881131015,
        1.3688472010259771,
        0.13871317151794504,
        0.9680810718049682,
        1.8708028880195542,
        1.3845315310998025,
        0.9138950782301458,
        0.5957025792584691,
    ]

    PI, *_ = sim.run_simulation(read_LAM=True, LAM=LAM)
    print(PI)
