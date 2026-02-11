# me144_toolbox/utils/animation_3d.py
"""
3D Animation utilities for drone swarm visualization.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple
import shutil
import subprocess
import tempfile
import os


def draw_cube(ax, center: np.ndarray, size: float = 2.0, color: str = 'blue', alpha: float = 0.7):
    """Draw a cube at center position and return the collection."""
    c = center
    s = size / 2
    
    # Define cube vertices
    vertices = np.array([
        [c[0]-s, c[1]-s, c[2]-s],
        [c[0]+s, c[1]-s, c[2]-s],
        [c[0]+s, c[1]+s, c[2]-s],
        [c[0]-s, c[1]+s, c[2]-s],
        [c[0]-s, c[1]-s, c[2]+s],
        [c[0]+s, c[1]-s, c[2]+s],
        [c[0]+s, c[1]+s, c[2]+s],
        [c[0]-s, c[1]+s, c[2]+s],
    ])
    
    # Define cube faces
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # bottom
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # top
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # front
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # back
    ]
    
    collection = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_collection3d(collection)
    return collection


def animate_swarm_3d(
    positions: np.ndarray,  # (T, N, 3)
    alive: np.ndarray,      # (T, N) boolean
    targets: np.ndarray,    # (M, 3)
    obstacles: Optional[list] = None,
    bounds: Tuple[float, ...] = (0, 100, 0, 100, 0, 50),
    base_pos: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    interval: int = 50,
    targets_visited: Optional[np.ndarray] = None,  # (T, M) boolean
) -> None:
    """
    Create 3D animation of drone swarm matching research image.
    
    Parameters
    ----------
    positions : (T, N, 3)
        Position history
    alive : (T, N)
        Boolean array of drone status
    targets : (M, 3)
        Target cube positions (as centers)
    obstacles : list of dicts, optional
        Each dict has 'center', 'radius'/'size', and optionally 'destroyed'
    bounds : tuple
    base_pos : (3,), optional
        Base position for visualization marker
    save_path : str, optional
    interval : int
    targets_visited : (T, M) optional
        Boolean array tracking which targets have been visited at each frame
    """
    T, N, _ = positions.shape
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)  # type: ignore
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)  # type: ignore
    ax.set_title('HOSTILE DRONE INCURSION', fontsize=14, fontweight='bold', pad=20)
    
    # Draw base position as a cube
    if base_pos is not None:
        draw_cube(ax, base_pos, size=3, color='red', alpha=0.9)
    
    # Plot target cubes (green boxes, exactly like obstacles but green)
    target_cube_artists = []
    if len(targets) > 0:
        for tidx, target in enumerate(targets):
            cube_artist = draw_cube(ax, target, size=2.0, color='green', alpha=0.7)
            target_cube_artists.append(cube_artist)
    
    # Plot obstacle cubes (blue)
    if obstacles:
        for idx, obs in enumerate(obstacles):
            center = obs['center']
            size = obs.get('size', obs.get('radius', 2.0) * 2)
            draw_cube(ax, center, size=size, color='lightblue', alpha=0.6)
    
    # Scatter artists for drones
    scatter_alive = ax.scatter([], [], [], c='blue', marker='o', s=100,  # type: ignore
                              alpha=0.8, label='Alive')
    scatter_dead = ax.scatter([], [], [], c='red', marker='x', s=200,  # type: ignore
                             alpha=0.5, label='Dead')
    
    title = fig.suptitle('', fontsize=12)
        
    def update(frame):
        """Update 3D plot."""
        pos = positions[frame]
        is_alive = alive[frame]
        
        alive_pos = pos[is_alive]
        dead_pos = pos[~is_alive]
        
        if len(alive_pos) > 0:
            scatter_alive._offsets3d = (alive_pos[:, 0], alive_pos[:, 1], alive_pos[:, 2])
        else:
            scatter_alive._offsets3d = ([], [], [])
        
        if len(dead_pos) > 0:
            scatter_dead._offsets3d = (dead_pos[:, 0], dead_pos[:, 1], dead_pos[:, 2])
        else:
            scatter_dead._offsets3d = ([], [], [])
        
        # Update target cube visibility based on visited status
        if targets_visited is not None and frame < len(targets_visited):
            visited_this_frame = targets_visited[frame]
            for t_idx, cube_artist in enumerate(target_cube_artists):
                if visited_this_frame[t_idx]:
                    # Make visited targets transparent (invisible)
                    cube_artist.set_alpha(0.0)
                else:
                    # Keep unvisited targets visible and green
                    cube_artist.set_alpha(0.7)
                    cube_artist.set_facecolor('green')
        
        # Update title with target counter prominently displayed
        n_alive = np.sum(is_alive)
        n_targets_scored = 0
        if targets_visited is not None and frame < len(targets_visited):
            n_targets_scored = np.sum(targets_visited[frame])
        n_targets_total = len(targets) if targets is not None else 0
        
        time_sec = frame * (interval / 1000.0)  # Convert interval ms to seconds
        # Display target counter prominently as the main information
        title.set_text(f'TARGETS: {n_targets_scored}/{n_targets_total}  |  Drones: {n_alive}/{N}  |  Time: {time_sec:.1f}s')
        
        return scatter_alive, scatter_dead, title
    
    ax.legend(loc='upper right')
    
    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=True, repeat=True)
    
    if save_path:
        ffmpeg_available = shutil.which('ffmpeg') is not None

        # If user asked for GIF and ffmpeg is available, use ffmpeg with all threads.
        if save_path.lower().endswith('.gif') and ffmpeg_available:
            print("  Using FFmpeg (multi-threaded) for GIF...")
            tmp_dir = tempfile.mkdtemp(prefix='swarm_gif_')
            tmp_mp4 = os.path.join(tmp_dir, 'temp_animation.mp4')
            palette = os.path.join(tmp_dir, 'palette.png')
            try:
                # Render MP4 using ffmpeg with all threads
                writer = FFMpegWriter(
                    fps=10,
                    codec='libx264',
                    bitrate=2000,
                    extra_args=['-threads', '0'],
                )
                anim.save(tmp_mp4, writer=writer, dpi=80)

                # Create palette for high-quality GIF
                subprocess.run(
                    [
                        'ffmpeg', '-y', '-i', tmp_mp4,
                        '-vf', 'fps=10,scale=640:-1:flags=lanczos,palettegen',
                        palette,
                    ],
                    check=True,
                )

                # Convert to GIF using all threads
                subprocess.run(
                    [
                        'ffmpeg', '-y', '-i', tmp_mp4, '-i', palette,
                        '-lavfi', 'fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse',
                        '-threads', '0',
                        save_path,
                    ],
                    check=True,
                )
                print(f"✓ GIF animation saved to {save_path}")
            except Exception as e:
                print(f"⚠ FFmpeg GIF export failed: {e}")
                # Fall back to Pillow
                try:
                    writer = PillowWriter(fps=10)
                    anim.save(save_path, writer=writer, dpi=80)
                    print(f"✓ GIF animation saved to {save_path}")
                except Exception as e2:
                    print(f"⚠ Could not save GIF: {e2}")
            finally:
                try:
                    if os.path.exists(tmp_mp4):
                        os.remove(tmp_mp4)
                    if os.path.exists(palette):
                        os.remove(palette)
                    os.rmdir(tmp_dir)
                except Exception:
                    pass
        elif ffmpeg_available:
            print(f"  Using FFmpeg for MP4...")
            try:
                writer = FFMpegWriter(fps=10, codec='libx264', bitrate=2000, extra_args=['-threads', '0'])
                anim.save(save_path, writer=writer, dpi=80)
                print(f"✓ MP4 animation saved to {save_path}")
            except Exception as e:
                print(f"⚠ FFmpeg export failed: {e}")
        else:
            # No FFmpeg, use GIF instead (single-threaded Pillow)
            print("  FFmpeg not found, using GIF (single-threaded)...")
            try:
                writer = PillowWriter(fps=10)
                anim.save(save_path, writer=writer, dpi=80)
                print(f"✓ GIF animation saved to {save_path}")
                print("  Tip: Install ffmpeg for multi-threaded GIF export: https://ffmpeg.org/download.html")
            except Exception as e:
                print(f"⚠ Could not save animation: {e}")
    
    plt.close(fig)


def plot_swarm_3d_static(
    positions_final: np.ndarray,
    targets: np.ndarray,
    alive: np.ndarray,
    obstacles: Optional[list] = None,
    bounds: Tuple[float, ...] = (0, 100, 0, 100, 0, 50),
    base_pos: Optional[np.ndarray] = None,
    title: str = "Final Swarm Configuration",
) -> None:
    """
    Static 3D plot of final swarm configuration.
    
    Parameters
    ----------
    positions_final : (N, 3)
        Final positions of drones
    targets : (M, 3)
        Target cube positions
    alive : (N,)
        Boolean array of which drones survived
    obstacles : list, optional
    bounds : tuple
    base_pos : (3,), optional
    title : str
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')  # type: ignore
    
    # Draw base
    if base_pos is not None:
        draw_cube(ax, base_pos, size=3, color='red', alpha=0.9)
    
    # Draw targets
    if len(targets) > 0:
        for target in targets:
            draw_cube(ax, target, size=2.0, color='green', alpha=0.7)
    
    # Draw obstacles
    if obstacles:
        for obs in obstacles:
            center = obs['center']
            size = obs.get('size', obs.get('radius', 2.0) * 2)
            draw_cube(ax, center, size=size, color='blue', alpha=0.6)
    
    # Plot drones
    alive_pos = positions_final[alive]
    dead_pos = positions_final[~alive]
    
    if len(alive_pos) > 0:
        ax.scatter(alive_pos[:, 0], alive_pos[:, 1], alive_pos[:, 2],
                  c='blue', marker='o', label=f'Alive ({len(alive_pos)})', depthshade=False)
    
    if len(dead_pos) > 0:
        ax.scatter(dead_pos[:, 0], dead_pos[:, 1], dead_pos[:, 2],
                  c='red', marker='x', label=f'Dead ({len(dead_pos)})', depthshade=False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)  # type: ignore
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.close(fig)
