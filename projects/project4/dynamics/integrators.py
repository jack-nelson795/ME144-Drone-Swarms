"""
Time integrators for rigid-body dynamics.
"""
from dynamics.state import RigidBodyState
import numpy as np

# helper functions
def hat(w: np.ndarray) -> np.ndarray:
    """
    Convert a 3-vector into a skew-symmetric matrix.

        w = [wx, wy, wz]

        hat(w) = [[ 0, -wz,  wy],
                  [ wz,  0, -wx],
                  [-wy, wx,   0]]

    This gives us:  hat(w) v = w × v
    """
    wx, wy, wz = w
    return np.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0],
    ])


def reorthonormalize(R: np.ndarray) -> np.ndarray:
    """
    Ensures rotation matrix is always orthonormal, correcting for numerical drift.
    
    You can think of U and V^T as containing the information relating to the "rotation" of R, while
    the singular value matrix Sigma is related to the scaling so we 'discard it' by setting it to 
    the identity.
    """
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


# ---------- Explicit Euler for rigid bodies ----------

def step_rigid_body(
    state: RigidBodyState,
    force: np.ndarray,
    torque: np.ndarray,
    dt: float,
    reproject_rotation: bool = True,
) -> RigidBodyState:
    """
    Advance a rigid body one time step using explicit Euler.

    Args:
        state (RigidBodyState): state object describing current position, velocity, 
                                angular velocity, etc.
        force : current force in N applied to the CG of the rigid-body in world-frame (3-vector)
        torque : current torque applied to the rigid-body in body-frame (3-vector)
        dt : time-step duration in seconds
        reproject_rotation (bool) : when set to true, the rotation matrix R will be 
                            re-orthonormalized at every time step. 
    Returns:
        (RigidBodyState) : the updated state
    """
    # Rigid-body strict explicit Euler update.
    x = state.position
    v = state.velocity
    R = state.rotation
    w = state.omega
    m = state.mass
    I = state.inertia

    # --- Translational rigid-body dynamics ---
    # force is given in the world frame
    a = force / m
    v_new = v + dt * a
    # Strict explicit Euler requires using the old velocity for position:
    #   x_{n+1} = x_n + dt * v_n
    # (not x_{n+1} = x_n + dt * v_{n+1}).
    x_new = x + dt * v

    # --- Rotational dynamics (body frame) ---
    # Hint: np.linalg.solve(A,b) == the vector x such that Ax = b
    # Euler's rigid body equation: I w_dot + w x (I w) = tau
    w_dot = np.linalg.solve(I, torque - np.cross(w, I @ w))
    w_new = w + dt * w_dot

    # Kinematic equation for orientation
    R_dot = R @ hat(w)
    R_new = R + dt * R_dot

    if reproject_rotation:
        R_new = reorthonormalize(R_new)

    return RigidBodyState(
        position=x_new,
        velocity=v_new,
        rotation=R_new,
        omega=w_new,
        mass=m,
        inertia=I,
    )
