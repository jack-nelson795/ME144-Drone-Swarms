"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 7: Aerial Firefighting
Spring 2026 Semester
"""
import os
import matplotlib

# Use a non-interactive backend by default so animation export works reliably.
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import pickle
from pathlib import Path
from typing import Any, Sequence, cast
"""
This script features the function to run the genetic algorithm.
It calls the simulation function from the Simulation class to evaluate the cost of each design.
"""

class Animation:
    
    def __init__(self, hist_plane, hist_particles, parameters):
        self.hist_plane: Sequence[np.ndarray] = hist_plane
        self.hist_particles: Sequence[np.ndarray] = hist_particles

        # environment parameters are fixed so we can set them once when we instantiate the class
        for key, value in parameters.items():
            setattr(self, key, value)

        # Make the parameter attributes explicit for static analyzers.
        self.FIRE_X_MIN: float = float(parameters["FIRE_X_MIN"])
        self.FIRE_X_MAX: float = float(parameters["FIRE_X_MAX"])
        self.FIRE_Z_MIN: float = float(parameters["FIRE_Z_MIN"])
        self.FIRE_Z_MAX: float = float(parameters["FIRE_Z_MAX"])


    def animate_firefighting(self, title = 'firesim', savefig = True, output_dir = '../animations'):
        
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        # ax.view_init(90,270)
        # Set Scene Limits
        ax.set_xlim(0, 400)
        ax.set_ylim(-200, 200)  # Z-axis in 3D plot usually maps to depth
        ax.set_zlim(0, 400)   # Vertical height
        
        # Labels (matching standard engineering notation)
        ax.set_xlabel("Downrange (X)")
        ax.set_ylabel("Crossrange (Z)")
        ax.set_zlabel("Altitude (Y)")
        ax.set_title("Digital Twin: Aerial Firefighting Optimization")
        
        # Draw Ground
        # (Just a visual plane at z=0)
        
        # Draw Fire Zone (Red Rectangle on ground)
        # X from 300 to 400, Z from -50 to 50
        xx = [self.FIRE_X_MIN, self.FIRE_X_MAX, self.FIRE_X_MAX, self.FIRE_X_MIN, self.FIRE_X_MIN]
        zz = [self.FIRE_Z_MIN, self.FIRE_Z_MIN, self.FIRE_Z_MAX, self.FIRE_Z_MAX, self.FIRE_Z_MIN]
        yy = [0, 0, 0, 0, 0]
        ax.plot(xx, zz, yy, 'r-', linewidth=2, label='Fire Zone (Updraft Source)')
        
        # 4. Dynamic Elements (Initialize empty)
        # Plane (Large Black Dot)
        plane_dot = cast(Line3D, ax.plot([], [], [], 'ko', markersize=8, label='Aircraft')[0])
        # Trail for Plane
        plane_trail = cast(Line3D, ax.plot([], [], [], 'k--', linewidth=0.5)[0])
        
        # Particles (Blue Dots) - "Water/Retardant"
        # Note: We use a single plot object for all particles
        particles_dots = cast(Line3D, ax.plot([], [], [], 'b.', markersize=3, alpha=0.6)[0])
        
        # 5. Update Function
        def update(frame: int) -> tuple[Any, Any, Any]:
            # Get data for this frame
            p_pos = self.hist_plane[frame]      # (3,)
            d_pos = self.hist_particles[frame]  # (N, 3)
            
            # Update Plane
            # Data mapping because we treated Y as vertical: 
            # Plot X = Data X
            # Plot Y = Data Z (Width)
            # Plot Z = Data Y (Height)
            
            # Plane
            plane_dot.set_data_3d([p_pos[0]], [p_pos[2]], [p_pos[1]])
            
            # Plane Trail (history up to this frame)
            # We need to slice the history list
            trail_data = np.array(self.hist_plane[:frame+1])
            if len(trail_data) > 0:
                plane_trail.set_data_3d(trail_data[:, 0], trail_data[:, 2], trail_data[:, 1])
            
            # Particles
            xs = d_pos[:, 0]
            ys = d_pos[:, 2] # Swap Y/Z for visualization
            zs = d_pos[:, 1]
            
            particles_dots.set_data_3d(xs, ys, zs)
            
            return plane_dot, particles_dots, plane_trail

        
        anim = animation.FuncAnimation(fig, update, frames=len(self.hist_plane), 
                                    interval=30, blit=False)
        
        plt.legend()
        

        if savefig:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            if shutil.which('ffmpeg'):
                anim.save(str(output_dir / f'{title}.mp4'), writer='ffmpeg', fps=30)
            else:
                anim.save(str(output_dir / f'{title}.gif'), writer='pillow', fps=30)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


    def plot_costs_over_generations(self, path_to_pickle: str, storage_directory:str ='')-> None:
        with open(path_to_pickle, 'rb') as f:
            ga_results = pickle.load(f)

        # Assuming ga_results is a dict with keys: 'avg_cost', 'best_cost', 'avg_parent_cost'
        avg_cost = ga_results['average_cost']
        best_cost = ga_results['best_cost']
        avg_parent_cost = ga_results['parents_average_cost']

        plt.figure(figsize=(10, 6))
        plt.plot(avg_cost, label='Average Cost')
        plt.plot(best_cost, label='Best Cost')
        plt.plot(avg_parent_cost, label='Average Parent Cost')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('GA Cost Metrics Over Generations')
        plt.legend()
        plt.grid(True)
        if len(storage_directory):
            output_dir = Path(storage_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'cost_plot.png')
            plt.close()
        else:
            plt.show()
            plt.close()
