"""
Simulates an applied displacement to a block of material, printing the internal stress state
in comparison to the analytical solution.
"""

# Allow running this file directly via `python test_scripts/uniaxial_tension_test.py`
# (in that case, Python's import root is `test_scripts/`, not the project4 folder).
if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from typing import Any, cast
from elastostatics.grid import ElasticGrid
import elastostatics.solver as solver


def main():
    """
    The parameters chosen below have shown to converge to the analytical solution in a reasonable
    amount of time, but feel free to play around with them.
    """
    E       = 1e6          # Young's modulus [Pa]
    nu      = 0.25         # Poisson ratio
    rho     = 1000.0       # density [kg/m³]
    h       = 0.001        # node spacing [m]
    nx = ny = nz = 10
    eps_zz  = 0.05        # applied axial strain (0.5%)

    grid = ElasticGrid(
        shape=(nx, ny, nz),
        spacing=(h, h, h),
        density=rho,
        young=E,
        poisson=nu,
    )

    # Physical bar length: node 0 to node nz-1
    L_bar          = (nz - 1) * h
    target_disp    = eps_zz * L_bar                   # applied z-displacement
    sig_analytical = E * eps_zz                       # analytical solution for resulting stress

    # Time step (CFL) and physically-motivated damping
    dt                = solver.compute_stable_dt(grid, safety_factor=0.1)
    c_p               = solver.compute_max_wavespeed(grid.lam, grid.mu, grid.rho)
    critical_damping  = 2 * c_p / L_bar
    damping           = 0.5 * critical_damping        # 50% of critical
    total_steps       = 3000

    print(f"Bar length : {L_bar*1e3:.1f} mm")
    print(f"max wavespeed        : {c_p:.2f} m/s")
    print(f"dt         : {dt:.3e} s")
    print(f"damping    : {damping:.1f} (50% of critical {critical_damping:.1f})")
    print(f"sigma_zz (analytical) : {sig_analytical:.2f} Pa")
    print(f"Total steps: {total_steps}\n")

    def apply_bcs(g):
        # z-displacement BCs
        g.u[:, :,  0, 2] = 0.0          # fix bottom face (z only)
        g.v[:, :,  0, :]  = 0.0          # zero velocity at bottom
        g.u[:, :, -1, 2]  = target_disp  # prescribed top displacement
        g.v[:, :, -1, 2]  = 0.0          # zero velocity at top

    # center node indices, we monitor the stress here
    ic = nx // 2
    jc = ny // 2
    kc = nz // 2

    update_3d, render_3d = init_von_mises_3d_animation(grid)

    for n in range(total_steps + 1):
        solver.step_elastic_grid(grid, dt, damping=damping, enforce_bc=apply_bcs)
        if n <= 500 and n % 25 == 0:
            update_3d(n)
        if n % 100 == 0:
            KE = 0.5 * rho * np.sum(grid.v ** 2)
            sigzz = grid.sigma[ic, jc, kc, 2, 2]
            error = abs(sigzz - sig_analytical) / sig_analytical * 100
            print(f"step {n:5d} | sigma_zz (center) = {sigzz:.4e} Pa "
                  f"| error = {error:.2f}% | KE = {KE:.2e}")

    # Render and display animations after the loop
    display(render_3d())

def init_von_mises_3d_animation(grid: ElasticGrid, mask: np.ndarray | None = None):
    """Initialise animation of 3D von Mises stress.

    Parameters
    ----------
    grid:
        The simulation grid.
    mask:
        Optional boolean array with shape ``(nx, ny, nz)`` selecting which voxels
        to include in the visualization (e.g., solid material voxels). If None,
        all voxels are included.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    xi, yi, zi = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    x_all = xi.flatten()
    y_all = yi.flatten()
    z_all = zi.flatten()

    if mask is None:
        keep = np.ones(nx * ny * nz, dtype=bool)
    else:
        if mask.shape != (nx, ny, nz):
            raise ValueError(f"mask shape {mask.shape} must match grid shape {(nx, ny, nz)}")
        keep = mask.astype(bool, copy=False).flatten()

    x_pts = x_all[keep]
    y_pts = y_all[keep]
    z_pts = z_all[keep]

    frames: list[tuple[int, np.ndarray]] = []  # list of (step, 1-D vm array)

    def update(step: int):
        vm = grid.calc_von_mises().flatten()
        frames.append((step, vm[keep].copy()))

    def render():
        if not frames:
            raise RuntimeError("No frames captured. Call update(step) at least once before render().")

        vmin = min(f[1].min() for f in frames)
        vmax = max(f[1].max() for f in frames)

        fig = plt.figure()
        ax = cast(Any, fig.add_subplot(111, projection="3d"))
        scatter = ax.scatter(
            x_pts,
            y_pts,
            z_pts,
            c=frames[0][1],
            cmap="viridis",
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        fig.colorbar(scatter, ax=ax, label="Von Mises Stress")

        def anim_update(i: int):
            step, vm_flat = frames[i]
            # Matplotlib uses a private 3D attribute here; use setattr to keep
            # Pylance (pyright) happy while preserving runtime behavior.
            setattr(scatter, "_offsets3d", (x_pts, y_pts, z_pts))
            scatter.set_array(vm_flat)
            ax.set_title(f"Von Mises Stress Field (3D) - Step {step}")
            return (scatter,)

        ani = FuncAnimation(fig, anim_update, frames=len(frames), interval=100, blit=False)
        plt.close(fig)
        try:
            ani_any = cast(Any, ani)
            return HTML(ani_any.to_html5_video())
        except Exception:
            # Fallback that doesn't require ffmpeg
            return HTML(cast(Any, ani).to_jshtml())

    return update, render



if __name__ == "__main__":
    main()
