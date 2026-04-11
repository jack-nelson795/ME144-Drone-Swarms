"""
ME144/244
LIDAR Simulation Module

Simulation initializes rays downwards from scanner. Rays increment in time and reflect off
the surface. When rays reach the original scanner height, they are considered "returned"
and can be used for time-of-flight calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, cast
from IPython.display import HTML
from matplotlib import animation
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

from lidar.surfaces import surfG, gradG, A_

###################################################################################################
############################### Simulation Constants and Parameters ###############################
###################################################################################################

c = 3e8                                         # speed of light (m/s)
z0 = 3                                          # scanner height (m)
nRays = 10000                                   # number of rays
dt = 0.001 * z0/c                               # time step size (s)
domainMax = 0.5
domainMin = -domainMax

error = 1e4 * np.ones(3)                        # initialize errors
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


###################################################################################################
##################################### Simulation Functions ########################################
###################################################################################################

def lidarsim(surface):
    """
    Lidar simulation - Ray tracing with reflections from oscillating surface

    Parameters
    ----------
    surface : int
        Index into A_ array specifying which surface amplitude to use (0, 1, or 2)

    Returns
    -------
    list
        List of ray position arrays at each recorded time step for animation
    """

    # Solution lists
    posTot = []
    timeTot = []

    rng = np.random.default_rng(144 + surface)
    phi1 = rng.uniform(0.0, 1.0, nRays)
    phi2 = rng.uniform(0.0, 1.0, nRays)
    r0 = np.column_stack((phi1 - 0.5, phi2 - 0.5, z0 * np.ones(nRays)))
    v0 = np.column_stack((np.zeros(nRays), np.zeros(nRays), -c * np.ones(nRays)))

    # Set current positions and velocities to initial conditions
    r = r0.copy()
    v = v0.copy()

    # Tracking arrays
    active = np.ones(nRays, dtype=bool)
    rtrn = np.zeros(nRays, dtype=bool)
    rfln = np.zeros(nRays)

    time = 0.0
    tj = np.zeros(nRays)
    rcontact = np.zeros((nRays, 3))
    n = np.zeros((nRays, 3))
    thetai = np.zeros((nRays, 1))
    vjperp = np.zeros((nRays, 3))
    vjref = np.zeros((nRays, 3))

    counter = 0
    while any(active):
        surface_height = surfG(r[:, 0], r[:, 1], A_[surface])
        idx = np.flatnonzero(active & (r[:, 2] <= surface_height))

        if idx.size != 0:
            rfln[idx] += 1

            # Snap the contact point to the surface before applying the reflection law.
            r[idx, 2] = surface_height[idx]
            rcontact[idx, :] = r[idx, :]

            gradx, grady = gradG(r[idx, 0], r[idx, 1], A_[surface])
            grad = np.column_stack((gradx, grady, -np.ones(idx.size)))
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            n[idx, :] = grad / grad_norm

            v_norm = np.linalg.norm(v[idx, :], axis=1, keepdims=True)
            cos_thetai = np.sum(v[idx, :] * n[idx, :], axis=1, keepdims=True) / (
                v_norm * np.linalg.norm(n[idx, :], axis=1, keepdims=True)
            )
            cos_thetai = np.clip(cos_thetai, -1.0, 1.0)
            thetai[idx] = np.arccos(cos_thetai)

            v_dot_n = np.sum(v[idx, :] * n[idx, :], axis=1, keepdims=True)
            vjperp[idx, :] = v_dot_n * n[idx, :]
            vjref[idx, :] = v[idx, :] - 2.0 * vjperp[idx, :]
            v[idx, :] = vjref[idx, :]

        idx_multiple = np.flatnonzero(active & (rfln > 1))
        if idx_multiple.size != 0:
            active[idx_multiple] = False

        r_prev = r.copy()
        r[active, :] += v[active, :] * dt
        time += dt

        idx_rtrn = np.flatnonzero(active & (r[:, 2] >= z0))
        if idx_rtrn.size != 0:
            tau = (z0 - r_prev[idx_rtrn, 2]) / v[idx_rtrn, 2]
            tau = np.clip(tau, 0.0, dt)
            r[idx_rtrn, :] = r_prev[idx_rtrn, :] + v[idx_rtrn, :] * tau[:, None]
            tj[idx_rtrn] = time - dt + tau
            active[idx_rtrn] = False
            rtrn[idx_rtrn] = True

        idx_domain = np.flatnonzero(
            active
            & (
                (r[:, 0] < domainMin)
                | (r[:, 0] > domainMax)
                | (r[:, 1] < domainMin)
                | (r[:, 1] > domainMax)
            )
        )
        if idx_domain.size != 0:
            active[idx_domain] = False

        if counter % 10 == 0:
            posTot.append(r.copy())
            timeTot.append(time)
        counter += 1

    valid = rtrn & (rfln == 1)
    r0_valid = r0[valid, :]
    r_valid = r[valid, :]
    rcontact_valid = rcontact[valid, :]
    vref_valid = vjref[valid, :]

    L = np.linalg.norm(r0_valid - r_valid, axis=1)

    v0_valid = v0[valid, :]
    cos_theta = np.sum(vref_valid * v0_valid, axis=1) / (
        np.linalg.norm(vref_valid, axis=1) * np.linalg.norm(v0_valid, axis=1)
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    sin_theta = np.sin(theta)
    d2 = np.empty_like(L)
    stable = np.abs(sin_theta) > 1e-12
    d2[stable] = L[stable] / sin_theta[stable]
    d2[~stable] = np.linalg.norm(r_valid[~stable] - rcontact_valid[~stable], axis=1)

    d1 = c * tj[valid] - d2
    rp = np.column_stack((r0_valid[:, 0], r0_valid[:, 1], z0 - d1))

    error[surface] = 1.0 - np.count_nonzero(valid) / nRays

    fig = plt.figure(figsize=(15, 10))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    cast(Any, ax).scatter(rp[:, 0], rp[:, 1], rp[:, 2], marker='o', s=25, color='red', edgecolor='grey')
    ax.set_xlabel('X (m)', fontsize=15)
    ax.set_ylabel('Y (m)', fontsize=15)
    ax.set_zlabel('Z (m)', fontsize=15)
    ax.set_xlim((-0.75, 0.75))
    ax.set_ylim((-0.75, 0.75))
    ax.set_title('Point Cloud Reconstruction - A={}'.format(A_[surface]))
    plt.savefig(OUTPUT_DIR / 'point_cloud_reconstruction_A{}.png'.format(A_[surface]))
    plt.show()

    print("Finished -> Error = ", error[surface])

    return posTot


def plot_surfaces():
    """
    Plot the three surface geometries before simulation
    """
    for i, _ in enumerate(A_):
        fig = plt.figure()
        ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
        x = y = np.arange(-0.5, 0.5 + 0.01, 0.01)
        X, Y = np.meshgrid(x, y)
        zs = np.array(surfG(np.ravel(X), np.ravel(Y), A_[i]))
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z, cmap='plasma')

        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.set_zlim(1.5, 4.0)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('Surface Geometry - A={}'.format(A_[i]))
        plt.savefig(OUTPUT_DIR / 'surfacePlot_A{}.png'.format(A_[i]))
        plt.show()


def animate_rays(posTot, surface, save_to_file=False, filename='anim.mp4'):
    """
    Create animation of ray trajectories

    Parameters
    ----------
    posTot : list
        List of ray position arrays at each time step
    surface : int
        Surface index for labeling
    save_to_file : bool, optional
        If True, save animation to an MP4 file. Default is False.
    filename : str, optional
        Filename for saved animation. Default is 'anim.mp4'.
        Only used if save_to_file=True.

    Returns
    -------
    IPython.display.HTML
        HTML5 video that displays in the notebook

    Note: Requires ffmpeg to be installed on the system when save_to_file=True
    """
    rc('animation', html='html5')

    def update(n):
        dots1.set_data_3d(posTot[n][:, 0], posTot[n][:, 1], posTot[n][:, 2])
        Title.set_text(f"Solution Animation: Time = {0:4f}".format(n * dt))
        return dots1,

    x = np.arange(-0.5, 0.5+0.01, 0.01)
    y = np.arange(-0.5, 0.5+0.01, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = surfG(X, Y, A_[surface])

    fig = plt.figure(figsize=(15, 12))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.6, edgecolor='none')

    ax.set_xlabel('X (m)', fontsize=15)
    ax.set_ylabel('Y (m)', fontsize=15)
    ax.set_zlabel('Z (m)', fontsize=15)
    ax.set_xlim((-0.75, 0.75))
    ax.set_ylim((-0.75, 0.75))
    ax.set_zlim((1.5, 4))
    ax.view_init(elev=35., azim=45)
    Title = ax.set_title(f'Ray Trajectories - Surface A={A_[surface]}')
    dots1 = cast(Line3D, ax.plot([], [], [], 'r.', ms=2)[0])
    ax.legend(['Ray Position'])

    anim = animation.FuncAnimation(fig, update, frames=len(posTot), interval=50, blit=True, repeat=True)

    if save_to_file:
        try:
            writervideo = animation.FFMpegWriter(fps=30)
            anim.save(filename, writer=writervideo)
            print(f" Animation saved to: {filename}")
        except Exception as e:
            print(f" Failed to save animation: {e}")
            print("   Make sure FFmpeg is installed on your system.")
            print("   macOS: brew install ffmpeg")
            print("   Ubuntu/Debian: sudo apt install ffmpeg")
    plt.close(fig)
    return HTML(anim.to_html5_video())
