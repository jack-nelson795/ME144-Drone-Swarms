""" class definition of RigidBodyState
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class RigidBodyState:
    """
    State of a single rigid body in 3D.

    position : (3-vector) position in m
    velocity : (3-vector) velocity in m/s
    rotation : (3x3 matrix), body → world rotation matrix
    omega : (3-vector), angular velocity in body frame
    mass : mass of the rigid body in kg
    inertia : I (3x3 matrix), body-frame inertia matrix
    """

    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    omega: np.ndarray
    mass: float
    inertia: np.ndarray

    @staticmethod
    def at_rest(mass: float, inertia: np.ndarray, position=None):
        """
        Construct a body at rest with identity orientation.
        """
        if position is None:
            position = np.zeros(3)

        return RigidBodyState(
            position=np.array(position, dtype=float),
            velocity=np.zeros(3),
            rotation=np.eye(3),
            omega=np.zeros(3),
            mass=mass,
            inertia=np.array(inertia, dtype=float),
        )
