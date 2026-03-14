"""
Simulates an applied displacement to a block of material, printing the internal stress state
in comparison to the analytical solution.
"""

# Allow running this file directly via `python test_scripts/uniaxial_tension_test_cube.py`
# (in that case, Python's import root is `test_scripts/`, not the project4 folder).
if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from IPython.display import HTML, display  # type: ignore
    from IPython import get_ipython  # type: ignore
except Exception:  # pragma: no cover
    HTML = None  # type: ignore

    def display(_obj):  # type: ignore
        return None

    def get_ipython():  # type: ignore
        return None

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
    eps_zz  = 0.005        # applied axial strain (0.5%)

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

    # Render and display animations after the loop.
    # In a notebook this will display inline; as a script we write an HTML file.
    html_obj_or_str = render_3d()
    if get_ipython() is not None and HTML is not None:
        display(html_obj_or_str)
    else:
        html_str = getattr(html_obj_or_str, "data", None) if HTML is not None else None
        if html_str is None:
            html_str = str(html_obj_or_str)
        out_path = "uniaxial_tension_cube_von_mises.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"\nWrote animation to: {out_path}")

def init_von_mises_3d_animation(grid):
    """
    Initialise animation of 3D von-mises stress
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    xi, yi, zi = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing='ij')
    x_pts = xi.flatten()
    y_pts = yi.flatten()
    z_pts = zi.flatten()

    frames = []   # list of (step, 1-D vm array)

    def update(step):
        vm = grid.calc_von_mises()
        frames.append((step, vm.flatten().copy()))

    def render():
        vmin = min(f[1].min() for f in frames)
        vmax = max(f[1].max() for f in frames)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_pts, y_pts, z_pts,
                             c=frames[0][1], cmap='viridis', alpha=0.6,
                             vmin=vmin, vmax=vmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        fig.colorbar(scatter, ax=ax, label='Von Mises Stress')

        def anim_update(i):
            step, vm_flat = frames[i]
            # Matplotlib uses a private 3D attribute here; use setattr so this
            # doesn't break on minor internal changes.
            setattr(scatter, "_offsets3d", (x_pts, y_pts, z_pts))
            scatter.set_array(vm_flat)
            ax.set_title(f'Von Mises Stress Field (3D) - Step {step}')
            return scatter,

        ani = FuncAnimation(fig, anim_update, frames=len(frames),
                            interval=100, blit=False)
        plt.close(fig)
        # Prefer HTML5 video; fall back to JS-only HTML if video export isn't available.
        try:
            html = ani.to_html5_video()
        except Exception:
            html = ani.to_jshtml()

        if HTML is not None:
            return HTML(html)
        return html

    return update, render



if __name__ == "__main__":
    main()
