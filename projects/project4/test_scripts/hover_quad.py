"""
Quadrotor flight simulation and animation
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
from typing import Any, cast
from dynamics.quadrotor import Quadrotor

class QuadFlight():
    """
    Parameters
        ----------
        Quadrotor   : Quadrotor object
        T           : length of flight simulation (s)
        dt          : timestep duration (s)
    """
    def __init__(
            self,
            quad: Quadrotor,
            T: float = 2.0,
            dt: float = 0.01
        ):
        self.quad = quad
        self.T = T
        self.dt = dt

    def hover(self):
        """Get drone to hover by counteracting its weight with thrust.

        Returns:
            state_history (dict): contains trajectory data for position, velocity, omega, and R
        """

        quad = self.quad
        state = quad.initial_state(position=[0, 0, 10.0])
        dt = self.dt
        T = self.T
        mass = self.quad.mass

        steps = int(T / dt)
        state_history = {'x': np.zeros((steps,3)),
                         'v': np.zeros((steps,3)),
                         'omega': np.zeros((steps,3)),
                         'R': np.zeros((steps,3,3))}

        # Hover by commanding constant thrust that balances weight.
        for k in range(steps):
            # Command chosen to balance gravity
            u_hover = (mass * quad.g) / (4.0 * quad.k_thrust)
            # advance state by a step, providing the thrust command
            state = quad.step(state, u_hover * np.ones(4), dt)
            state_history['x'][k] = state.position
            state_history['v'][k] = state.velocity
            state_history['omega'][k] = state.omega
            state_history['R'][k] = state.rotation

            # print
            if k % 40 == 0:
                print(f"t={k*dt: .2f}  z={state.position[2]: .3f}  w={[f'{w: .4f}' for w in state.omega]}")

        return state_history

    def animate(self, state_history):
        '''
        Takes position, velocity, angular velocity, and rotation matrix history and animates flight
        '''
        dt = self.dt

        # Extract position history
        position_history = state_history['x']

        # Quadcopter size (for visualization)
        arm_length = self.quad.arm_length 

        # Set up the figure and 3D axis
        fig = plt.figure()
        # Type stubs for matplotlib's 3D axes are incomplete; cast to Any to avoid
        # false-positive Pylance attribute errors on 3D-only methods.
        ax = cast(Any, fig.add_subplot(111, projection='3d'))
        ax.set_xlim(np.min(position_history[:, 0]) - 1, np.max(position_history[:, 0]) + 1)
        ax.set_ylim(np.min(position_history[:, 1]) - 1, np.max(position_history[:, 1]) + 1)
        ax.set_zlim(np.min(position_history[:, 2]) - 1, np.max(position_history[:, 2]) + 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Quadcopter Flight Animation')
        # 3D axes don't reliably support set_aspect('equal') across matplotlib versions.
        # Prefer set_box_aspect when available; otherwise, just skip equal aspect.
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        # Initialize the quadcopter body (4 arms)
        body_lines = [ax.plot([], [], [], 'b-', lw=2)[0] for _ in range(4)]

        # Update function for animation
        def update(frame):
            # Current position and rotation matrix
            position = position_history[frame]
            R = state_history['R'][frame]

            # Define the endpoints of the quadcopter arms in the body frame
            arm_endpoints = np.array([
                [arm_length, 0, 0],
                [-arm_length, 0, 0],
                [0, arm_length, 0],
                [0, -arm_length, 0]
            ])

            # Transform arm endpoints to the world frame
            world_endpoints = (R @ arm_endpoints.T).T + position

            # Update the lines for each arm
            for i, line in enumerate(body_lines):
                x = [position[0], world_endpoints[i, 0]]
                y = [position[1], world_endpoints[i, 1]]
                z = [position[2], world_endpoints[i, 2]]
                line_any = cast(Any, line)
                line_any.set_data(x, y)
                line_any.set_3d_properties(z)

            return body_lines

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(position_history), interval=dt*1000, blit=True)
        plt.close(fig)

        # Display the animation in Jupyter Notebook
        try:
            return HTML(cast(Any, ani).to_html5_video())
        except Exception:
            # Fallback that doesn't require ffmpeg
            return HTML(cast(Any, ani).to_jshtml())
