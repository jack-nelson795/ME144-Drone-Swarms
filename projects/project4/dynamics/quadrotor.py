"""
Quadrotor dynamics model.

A quadrotor is represented as a rigid body plus four rotors that
produce thrust along the body +z axis and reaction torques about z.
"""

import numpy as np
from dynamics.state import RigidBodyState
from dynamics.integrators import step_rigid_body

class Quadrotor:
    """
        Parameters
        ----------
        mass       : total mass of the vehicle
        inertia    : 3x3 body-frame inertia matrix
        arm_length : distance from center to each motor
        k_thrust   : thrust coefficient (N per unit command)
        k_torque   : yaw torque coefficient
        gravity    : gravitational acceleration
    """
    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,
        arm_length: float,
        k_thrust: float,
        k_torque: float,
        gravity: float = 9.81,
    ):
        self.mass = mass
        self.inertia = np.array(inertia, dtype=float)
        self.arm_length = arm_length
        self.k_thrust = k_thrust
        self.k_torque = k_torque
        self.g = gravity

        # Motor layout (body frame):
        #      2           ^   
        #      |           |   
        # 3 -- + -- 1      y
        #      |                x (forward) -->
        #      4
        self.motor_positions = np.array([
            [ self.arm_length,  0.0, 0.0],  # 1
            [ 0.0,  self.arm_length, 0.0],  # 2
            [-self.arm_length,  0.0, 0.0],  # 3
            [ 0.0, -self.arm_length, 0.0],  # 4
        ])

        # Spin directions (+1 or -1) for yaw torque
        self.spin = np.array([+1, -1, +1, -1])

    def forces_and_torques(self, state: RigidBodyState, u: np.ndarray):
        """
        Compute total force (world frame) and torque (body frame) from motor commands.

        u : array of 4 motor commands (dimensionless)
        """
        # Map motor commands to net force/torque.
        # Thrust from each motor
        thrusts = self.k_thrust * u

        # Initialize body force
        # Thrust acts along +z body axis
        F_body = np.array([0.0, 0.0, float(np.sum(thrusts))])

        # Add the body torque produced by each motor to tau_body
        tau_body = np.zeros(3)
        # Torque from thrust forces applied at motor positions: tau = r x F
        for r_i, f_i in zip(self.motor_positions, thrusts):
            tau_body += np.cross(r_i, np.array([0.0, 0.0, float(f_i)]))

        # add yaw reaction torques
        tau_body[2] += self.k_torque * float(np.dot(self.spin, u))

        # Rotate force into world frame
        F_world = state.rotation @ F_body

        # Add gravity in world frame
        F_world += np.array([0.0, 0.0, -self.mass * self.g])

        return F_world, tau_body

    def step(self, state: RigidBodyState, u: np.ndarray, dt: float) -> RigidBodyState:
        """
        Advance the quadrotor one time step.
        """
        F, tau = self.forces_and_torques(state, u)
        return step_rigid_body(state, F, tau, dt)

    def initial_state(self, position=None) -> RigidBodyState:
        """
        Return the initial state for a quadrotor at rest.
        """
        return RigidBodyState.at_rest(self.mass, self.inertia, position)
    