from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


DEM_CMAP = LinearSegmentedColormap.from_list(
    "dem_damage",
    ["#1b4fd4", "#29b85f", "#efe65f", "#f27a1a", "#a61d1d"],
)
SPHERE_EDGE = "#6b210f"


def _voxel_marker_size(fig: Figure, ax: Axes3D, coords: np.ndarray, spacing: float, radius_scale: float = 1.18) -> float:
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axis_span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2]) * 1.2, spacing)
    points_per_unit = (72.0 * min(bbox.width, bbox.height)) / max(axis_span, 1.0e-6)
    marker_radius_pt = 0.5 * spacing * radius_scale * points_per_unit
    return float(np.pi * marker_radius_pt**2)


def _style_axes(ax: Axes3D, x_extent: float, y_extent: float, z_extent: float) -> None:
    ax.set_xlim(-1.05 * x_extent, 1.05 * x_extent)
    ax.set_ylim(-1.05 * y_extent, 1.05 * y_extent)
    ax.set_zlim(-1.2 * z_extent, 1.2 * z_extent)
    ax.set_box_aspect((2.1 * x_extent, 2.1 * y_extent, 2.4 * z_extent))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=21, azim=-48)
    ax.grid(False)
    ax.set_proj_type("persp")
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis = getattr(ax, axis_name, None)
        pane = getattr(axis, "pane", None)
        if pane is not None:
            pane.set_alpha(0.05)


def _load_snapshot_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def launch_snapshot_viewer(path: Path) -> None:
    archive = _load_snapshot_archive(path)
    coords = archive["coords"]
    rgba = archive["rgba"]
    edge_rgba = archive["edge_rgba"]
    sizes = archive["sizes"]
    labels = archive["labels"]
    times = archive["times"]
    spacing = float(np.asarray(archive.get("spacing", np.array([0.03], dtype=float))).ravel()[0])
    arm_length = float(np.asarray(archive.get("arm_length", np.array([0.35], dtype=float))).ravel()[0])
    body_r3 = float(np.asarray(archive.get("body_r3", np.array([0.06], dtype=float))).ravel()[0])

    fig = plt.figure(figsize=(12.5, 6.8))
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    plt.subplots_adjust(bottom=0.18)

    x_extent = max(float(np.max(np.abs(coords[..., 0]))) if coords.size else 0.5, 1.02 * arm_length)
    y_extent = max(float(np.max(np.abs(coords[..., 1]))) if coords.size else 0.5, 1.02 * arm_length)
    z_extent = max(float(np.max(np.abs(coords[..., 2]))) if coords.size else 0.3, 4.0 * body_r3)
    _style_axes(ax, x_extent, y_extent, z_extent)
    ax.set_title("Project Y Interactive Snapshot Viewer")

    marker_size = _voxel_marker_size(fig, ax, coords[0], spacing)
    scatter: Any = cast(Any, ax).scatter(
        [],
        [],
        zs=[],
        s=0,
        c="#d96d1d",
        edgecolors=SPHERE_EDGE,
        linewidths=0.2,
        alpha=0.96,
        depthshade=True,
    )
    status = fig.text(0.5, 0.96, "", ha="center", va="top", fontsize=12)
    help_text = fig.text(
        0.5,
        0.02,
        "Drag to orbit. Use slider or Next/Prev to switch impact and course snapshots.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    slider_ax = fig.add_axes((0.18, 0.08, 0.52, 0.03))
    prev_ax = fig.add_axes((0.73, 0.073, 0.08, 0.05))
    next_ax = fig.add_axes((0.82, 0.073, 0.08, 0.05))
    slider = Slider(slider_ax, "Snapshot", 0, len(coords) - 1, valinit=0, valstep=1)
    prev_button = Button(prev_ax, "Prev")
    next_button = Button(next_ax, "Next")

    def render(index: int) -> None:
        idx = int(index)
        frame_coords = np.asarray(coords[idx], dtype=float)
        frame_rgba = np.asarray(rgba[idx], dtype=float)
        frame_edges = np.asarray(edge_rgba[idx], dtype=float)
        frame_sizes = marker_size * np.asarray(sizes[idx], dtype=float)
        scatter._offsets3d = (frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2])
        scatter.set_facecolor(frame_rgba.tolist())
        scatter.set_edgecolor(frame_edges.tolist())
        scatter.set_sizes(frame_sizes)
        status.set_text(f"{labels[idx]}   t = {times[idx]:.2f} s")
        fig.canvas.draw_idle()

    def on_slider_change(value: float) -> None:
        render(int(value))

    def on_prev(_event) -> None:
        slider.set_val(max(0, int(slider.val) - 1))

    def on_next(_event) -> None:
        slider.set_val(min(len(coords) - 1, int(slider.val) + 1))

    slider.on_changed(on_slider_change)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    render(0)
    plt.show()
    _ = help_text


def main() -> None:
    if len(sys.argv) > 1:
        archive_path = Path(sys.argv[1]).resolve()
    else:
        archive_path = Path(__file__).resolve().parent / "output" / "projecty_snapshot_archive.npz"
    launch_snapshot_viewer(archive_path)


if __name__ == "__main__":
    main()
