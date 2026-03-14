"""
build a quadrotor drone by specifying design parameters.

without specifying any parameters, a default quadrotor is built.
"""

import numpy as np
from dynamics.quadrotor import Quadrotor

def build_simple_quad(
    # representative initial values
    total_mass: float = 1.0,
    arm_length: float = 0.2,
    body_radius: float = 0.05,
    motor_strength: float = 5.0,
    yaw_strength: float = 0.1,
    gravity: float = 9.81,
    arm_radius: float | None = None,
    motor_radius: float | None = None,
):
    """
    Build a simple quadrotor model with analytically-derived inertia.

    Parameters
    ----------
    total_mass     : total mass of the drone
    arm_length     : distance from center to each motor
    body_radius    : radius of the central body (sphere)
    motor_strength : thrust coefficient (N per unit command)
    yaw_strength   : yaw torque coefficient
    gravity        : gravitational acceleration

    Returns
    -------
    quad  : Quadrotor
    state : RigidBodyState (at rest at the origin)
    """

    # Split mass between body and motors
    body_mass = 0.4 * total_mass
    motor_mass = 0.6 * total_mass / 4.0

    # we use the inertia of solid sphere ((2/5) m r^2 I) to represent
    # the inertia of the body
    I_body = (2.0 / 5.0) * body_mass * body_radius**2 * np.eye(3)

    # Motor positions in the body frame
    motor_positions = np.array([
        [ arm_length,  0.0, 0.0],
        [ 0.0,  arm_length, 0.0],
        [-arm_length,  0.0, 0.0],
        [ 0.0, -arm_length, 0.0],
    ])

    # Inertia contribution of motors, treat as point masses
    # I = sum( m (||r||^2 I - r r^T) )
    I_motors = np.zeros((3, 3))
    for r in motor_positions:
        r2 = np.dot(r, r)
        I_motors += motor_mass * (r2 * np.eye(3) - np.outer(r, r))

    I = I_body + I_motors

    quad = Quadrotor(
        mass=total_mass,
        inertia=I,
        arm_length=arm_length,
        k_thrust=motor_strength,
        k_torque=yaw_strength,
        gravity=gravity,
    )

    # Attach simple geometry hints (used by voxel-grid demos / visualizations).
    # These are not required for rigid-body dynamics.
    quad.body_radius = float(body_radius)
    quad.arm_radius = float(arm_radius) if arm_radius is not None else float(0.25 * body_radius)
    quad.motor_radius = float(motor_radius) if motor_radius is not None else float(0.4 * body_radius)

    state = quad.initial_state(position=[0, 0, 1.0])

    return quad, state
