from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

try:
    from .config import ANIMATION_DIR, DEFAULT_MATERIAL, OUTPUT_DIR, SimConfig
    from .design import DroneDesign
    from .flight import FlightResult, Gate, Turret, make_turrets, turret_can_reach_drone, turret_pulse_active
    from .geometry import VoxelDrone, build_voxel_drone
    from .optimization import format_design
except ImportError:
    from config import ANIMATION_DIR, DEFAULT_MATERIAL, OUTPUT_DIR, SimConfig
    from design import DroneDesign
    from flight import FlightResult, Gate, Turret, make_turrets, turret_can_reach_drone, turret_pulse_active
    from geometry import VoxelDrone, build_voxel_drone
    from optimization import format_design


DEM_CMAP = LinearSegmentedColormap.from_list(
    "dem_damage",
    ["#1b4fd4", "#29b85f", "#efe65f", "#f27a1a", "#a61d1d"],
)

SPHERE_EDGE = "#6b210f"
FRAGMENT_EDGE = "#16366f"


def _dem_rgba(values: np.ndarray) -> np.ndarray:
    return DEM_CMAP(np.clip(values, 0.0, 1.0))


def _voxel_marker_size(fig: Figure, ax: Axes3D, drone: VoxelDrone, radius_scale: float = 1.18) -> float:
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axis_span = max(
        np.ptp(drone.coords[..., 0]),
        np.ptp(drone.coords[..., 1]),
        np.ptp(drone.coords[..., 2]) * 1.2,
    )
    points_per_unit = (72.0 * min(bbox.width, bbox.height)) / max(axis_span, 1.0e-6)
    marker_radius_pt = 0.5 * drone.spacing * radius_scale * points_per_unit
    return float(np.pi * marker_radius_pt**2)


def _style_axes(ax: Axes3D, limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]], title: str):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.view_init(elev=24, azim=-52)
    ax.grid(False)
    ax.set_proj_type("persp")
    try:
        ax.set_box_aspect((limits[0][1] - limits[0][0], limits[1][1] - limits[1][0], 1.1 * (limits[2][1] - limits[2][0])))
    except Exception:
        pass
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis = getattr(ax, axis_name, None)
        pane = getattr(axis, "pane", None)
        if pane is not None:
            pane.set_alpha(0.05)


def _dem_color_values(stress: np.ndarray, integrity: np.ndarray, stress_limit: float) -> np.ndarray:
    normalized_stress = np.clip(stress / max(stress_limit, 1.0), 0.0, 1.6)
    warmth = 0.78 * integrity + 0.22 * np.clip(1.0 - normalized_stress, 0.0, 1.0)
    heat = np.clip(0.6 * normalized_stress + 0.4 * (1.0 - integrity), 0.0, 1.0)
    return np.clip(0.85 * warmth - 0.55 * heat + 0.15, 0.0, 1.0)


def _body_world_points(drone: VoxelDrone, position: np.ndarray, rotation: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    local_coords = drone.coords[active_mask]
    if len(local_coords) == 0:
        return np.zeros((0, 3), dtype=float)
    return position + local_coords @ rotation.T


def _scatter_voxels(
    ax: Axes3D,
    coords: np.ndarray,
    values: np.ndarray | None,
    marker_size: float,
    title: str,
):
    if len(coords) == 0:
        ax.set_title(title)
        return None
    scalar_values = np.full(len(coords), 0.86) if values is None else values
    plot = cast(Any, ax).scatter(
        coords[:, 0],
        coords[:, 1],
        zs=coords[:, 2],
        c=_dem_rgba(scalar_values),
        s=marker_size,
        alpha=0.96,
        edgecolors=SPHERE_EDGE,
        linewidths=0.2,
        depthshade=True,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    try:
        ax.set_box_aspect((1, 1, 0.82))
    except Exception:
        pass
    return plot


def _closeup_bounds(drone: VoxelDrone) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    local_half_span = max(0.36, 1.02 * drone.design.arm_length)
    local_z_span = max(0.22, 4.0 * drone.design.body_r3)
    return (
        (-local_half_span, local_half_span),
        (-local_half_span, local_half_span),
        (-local_z_span, local_z_span),
    )


def _render_dem_closeup(
    fig: Figure,
    ax: Axes3D,
    drone: VoxelDrone,
    coords: np.ndarray,
    rgba: np.ndarray,
    size_scale: np.ndarray,
    title: str,
    edge_rgba: np.ndarray | None = None,
) -> None:
    _style_axes(ax, _closeup_bounds(drone), title)
    ax.view_init(elev=21, azim=-48)
    marker_size = _voxel_marker_size(fig, ax, drone, radius_scale=1.18)
    if edge_rgba is None:
        edge_rgba = np.tile(to_rgba(SPHERE_EDGE), (len(coords), 1))
    cast(Any, ax).scatter(
        coords[:, 0],
        coords[:, 1],
        zs=coords[:, 2],
        s=np.asarray(marker_size * size_scale, dtype=float),
        c=rgba.tolist(),
        edgecolors=edge_rgba.tolist(),
        linewidths=0.2,
        alpha=0.96,
        depthshade=True,
    )


def _set_offsets3d(scatter: Any, coords: np.ndarray) -> None:
    if len(coords) == 0:
        scatter._offsets3d = ([], [], [])
    else:
        scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])


def _draw_turret_assets(ax: Axes3D, turrets: list[Turret]) -> None:
    for turret in turrets:
        x, y, z = turret.position
        dx = 0.12
        dy = 0.09
        foot_z = 0.01
        platform_z = 0.085
        neck_z0 = 0.11
        neck_z1 = z - 0.15
        head_z = z

        base_ring = np.array(
            [
                [x - dx, y - dy, foot_z],
                [x + dx, y - dy, foot_z],
                [x + dx, y + dy, foot_z],
                [x - dx, y + dy, foot_z],
                [x - dx, y - dy, foot_z],
            ]
        )
        inset_ring = np.array(
            [
                [x - 0.7 * dx, y - 0.7 * dy, platform_z],
                [x + 0.7 * dx, y - 0.7 * dy, platform_z],
                [x + 0.7 * dx, y + 0.7 * dy, platform_z],
                [x - 0.7 * dx, y + 0.7 * dy, platform_z],
                [x - 0.7 * dx, y - 0.7 * dy, platform_z],
            ]
        )
        for edge_idx in range(4):
            ax.plot(
                [base_ring[edge_idx, 0], inset_ring[edge_idx, 0]],
                [base_ring[edge_idx, 1], inset_ring[edge_idx, 1]],
                [base_ring[edge_idx, 2], inset_ring[edge_idx, 2]],
                color="#4d5561",
                lw=2.2,
                alpha=0.95,
            )
        ax.plot(base_ring[:, 0], base_ring[:, 1], base_ring[:, 2], color="#1b1f25", lw=3.2, alpha=0.98)
        ax.plot(inset_ring[:, 0], inset_ring[:, 1], inset_ring[:, 2], color="#606a77", lw=3.0, alpha=0.98)

        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            ax.plot(
                [x + 0.48 * sx * dx, x + 0.9 * sx * dx],
                [y + 0.48 * sy * dy, y + 0.9 * sy * dy],
                [platform_z, foot_z],
                color="#7b8695",
                lw=1.4,
                alpha=0.8,
            )

        mast_offsets = [(-0.03, -0.024), (0.03, -0.024), (0.03, 0.024), (-0.03, 0.024)]
        for ox, oy in mast_offsets:
            ax.plot(
                [x + ox, x + 0.6 * ox],
                [y + oy, y + 0.6 * oy],
                [platform_z, neck_z1],
                color="#838d9a",
                lw=2.4,
                alpha=0.92,
            )
        ax.plot([x, x], [y, y], [neck_z0, neck_z1], color="#3d4653", lw=7.2, alpha=0.98)
        ax.plot([x, x], [y, y], [neck_z0, neck_z1], color="#97a3b2", lw=2.0, alpha=0.8)

        fin_span = 0.1
        fin_drop = 0.06
        for axis in range(2):
            if axis == 0:
                xline = [x - fin_span, x + fin_span]
                yline = [y, y]
            else:
                xline = [x, x]
                yline = [y - fin_span, y + fin_span]
            ax.plot(xline, yline, [head_z - fin_drop, head_z - 0.018], color="#c45b2d", lw=4.2, alpha=0.92)
            ax.plot(xline, yline, [head_z - fin_drop, head_z - 0.018], color="#ffd37a", lw=1.1, alpha=0.55)

        cast(Any, ax).scatter([x], [y], zs=[head_z - 0.025], s=250, c="#7a1116", edgecolors="#2d0406", linewidths=1.0, depthshade=False)
        cast(Any, ax).scatter([x], [y], zs=[head_z], s=185, c="#c92d24", edgecolors="#4f0807", linewidths=1.0, depthshade=False)
        cast(Any, ax).scatter([x], [y], zs=[head_z + 0.012], s=60, c="#ffe68a", edgecolors="none", alpha=0.95, depthshade=False)
        cast(Any, ax).scatter([x], [y], zs=[head_z], s=420, c="#ff6b2e", alpha=0.12, edgecolors="none", depthshade=False)
        cast(Any, ax).scatter([x], [y], zs=[head_z], s=780, c="#ffd94a", alpha=0.055, edgecolors="none", depthshade=False)


def _pulse_volume_points(turret: Turret, drone_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    incident = drone_position - turret.position
    dist = float(np.linalg.norm(incident))
    if dist < 1.0e-8:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    direction = incident / dist
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(direction, helper)
    b1 /= max(np.linalg.norm(b1), 1.0e-8)
    b2 = np.cross(direction, b1)

    axial_stop = max(0.25, min(4.0, dist))
    axial_samples = np.linspace(0.25, axial_stop, 10)
    radial_samples = np.linspace(0.0, 0.42, 5)
    angle_samples = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)

    points: list[np.ndarray] = []
    intensities: list[float] = []
    for axial in axial_samples:
        for radial in radial_samples:
            for theta in angle_samples:
                offset = radial * (np.cos(theta) * b1 + np.sin(theta) * b2)
                point = turret.position + axial * direction + offset
                if float(np.dot(point - turret.position, direction)) > dist:
                    continue
                intensity = np.exp(-0.9 * axial) * np.exp(-turret.decay * radial)
                if intensity < 0.015:
                    continue
                points.append(point)
                intensities.append(float(intensity))
    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    return np.vstack(points), np.asarray(intensities, dtype=float)


def _gate_ring_points(gate: Gate, samples: int = 120) -> np.ndarray:
    normal = gate.normal / max(np.linalg.norm(gate.normal), 1.0e-8)
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(normal, helper)
    b1 /= max(np.linalg.norm(b1), 1.0e-8)
    b2 = np.cross(normal, b1)
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    return gate.center[None, :] + gate.radius * (
        np.cos(theta)[:, None] * b1[None, :] + np.sin(theta)[:, None] * b2[None, :]
    )


def _draw_gate_rings(ax: Axes3D, gates: list[Gate], passed: np.ndarray | None = None) -> list[Any]:
    artists: list[Any] = []
    for idx, gate in enumerate(gates):
        ring = _gate_ring_points(gate)
        is_passed = bool(passed[idx]) if passed is not None and idx < len(passed) else False
        color = "#d72828" if not is_passed else "#35d04f"
        alpha = 0.95 if not is_passed else 0.85
        line = cast(Any, ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, lw=2.4, alpha=alpha)[0])
        artists.append(line)
    return artists


def _animation_limits(result: FlightResult, turret_positions: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    position_history = result.state_history["position"]
    x_all = np.concatenate([position_history[:, 0], turret_positions[:, 0]])
    y_all = np.concatenate([position_history[:, 1], turret_positions[:, 1]])
    z_all = np.concatenate([position_history[:, 2], turret_positions[:, 2], np.array([0.0])])
    return (
        (float(np.min(x_all) - 1.2), float(np.max(x_all) + 1.2)),
        (float(np.min(y_all) - 1.9), float(np.max(y_all) + 1.9)),
        (-0.15, float(np.max(z_all) + 1.0)),
    )


def save_animation(
    drone: VoxelDrone,
    result: FlightResult,
    path: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = 12
    hold_frames = fps
    base_frame_count = len(result.state_history["position"])
    frame_count = base_frame_count + hold_frames
    turret_positions = np.asarray(cast(np.ndarray, result.frame_history["turrets"]), dtype=float)
    turret_specs_raw = cast(list[dict[str, object]], result.frame_history["turret_specs"])
    turrets = [
        Turret(
            position=np.asarray(spec["position"], dtype=float),
            pressure_peak=float(cast(float, spec["pressure_peak"])),
            decay=float(cast(float, spec["decay"])),
            frequency=float(cast(float, spec["frequency"])),
            pulse_width=float(cast(float, spec["pulse_width"])),
            phase_offset=float(cast(float, spec["phase_offset"])),
            x_window=cast(tuple[float, float], spec["x_window"]),
        )
        for spec in turret_specs_raw
    ]
    gate_specs_raw = cast(list[dict[str, object]], result.frame_history["gates"])
    gates = [
        Gate(
            center=np.asarray(spec["center"], dtype=float),
            normal=np.asarray(spec["normal"], dtype=float),
            radius=float(cast(float, spec["radius"])),
            label=str(spec["label"]),
        )
        for spec in gate_specs_raw
    ]
    limits = _animation_limits(result, turret_positions)
    fig = plt.figure(figsize=(12.5, 6.8))
    ax_close = cast(Axes3D, fig.add_subplot(121, projection="3d"))
    ax_world = cast(Axes3D, fig.add_subplot(122, projection="3d"))
    _style_axes(ax_world, limits, "Flight Context")
    marker_size_close = _voxel_marker_size(fig, ax_close, drone, radius_scale=1.18)
    marker_size_world = max(2.8, 0.18 * marker_size_close)
    local_half_span = max(0.36, 1.02 * drone.design.arm_length)
    local_z_span = max(0.22, 4.0 * drone.design.body_r3)

    fig.suptitle("Hostile Flight DEM Animation", y=0.98)
    way = result.waypoints
    ax_world.plot(way[:, 0], way[:, 1], way[:, 2], color="#444444", linestyle="--", lw=1.2, alpha=0.7)
    _draw_turret_assets(ax_world, turrets)
    gate_artists = _draw_gate_rings(ax_world, gates)

    _style_axes(
        ax_close,
        ((-local_half_span, local_half_span), (-local_half_span, local_half_span), (-local_z_span, local_z_span)),
        "DEM Body Close-Up",
    )
    ax_close.view_init(elev=21, azim=-48)
    ax_world.view_init(elev=24, azim=-52)

    body_close_scatter: Any = cast(Any, ax_close).scatter([], [], zs=[], s=marker_size_close, c="#d96d1d", edgecolors=SPHERE_EDGE, linewidths=0.2, alpha=0.96)
    fragment_close_scatter: Any = cast(Any, ax_close).scatter([], [], zs=[], s=marker_size_close, c="#2d68d8", edgecolors=FRAGMENT_EDGE, linewidths=0.18, alpha=0.95)
    body_world_scatter: Any = cast(Any, ax_world).scatter([], [], zs=[], s=marker_size_world, c="#d96d1d", edgecolors=SPHERE_EDGE, linewidths=0.14, alpha=0.95)
    fragment_world_scatter: Any = cast(Any, ax_world).scatter([], [], zs=[], s=marker_size_world, c="#2d68d8", edgecolors=FRAGMENT_EDGE, linewidths=0.12, alpha=0.94)
    trail_line = cast(Any, ax_world.plot([], [], [], color="#276fbf", lw=1.6, alpha=0.85)[0])
    pulse_scatters = [
        cast(Any, cast(Any, ax_world).scatter([], [], zs=[], s=[], c="#f9f06b", edgecolors="none", alpha=0.0, depthshade=False))
        for _ in range(len(turrets))
    ]
    pulse_lines = [
        cast(Any, ax_world.plot([], [], [], color="#f9f06b", lw=1.1, alpha=0.0)[0])
        for _ in range(len(turrets))
    ]
    status_text = fig.text(0.03, 0.93, "")

    def update(frame_idx: int):
        state_idx = min(frame_idx, base_frame_count - 1)
        position = result.state_history["position"][state_idx]
        rotation = result.state_history["rotation"][state_idx]
        active_mask_history = cast(list[np.ndarray], result.frame_history["body_active_mask"])
        integrity_history = cast(list[np.ndarray], result.frame_history["integrity"])
        stress_history = cast(list[np.ndarray], result.frame_history["von_mises"])
        active_mask = np.asarray(active_mask_history[state_idx], dtype=bool)
        integrity = np.asarray(integrity_history[state_idx], dtype=float)
        stress = np.asarray(stress_history[state_idx], dtype=float)

        body_world_coords = _body_world_points(drone, position, rotation, active_mask)
        active_integrity = integrity[active_mask]
        active_stress = stress[active_mask]
        body_colors = _dem_color_values(active_stress, active_integrity, drone.material.stress_limit)
        body_rgba = _dem_rgba(body_colors)
        main_size_scale = np.where(drone.motor_mask[active_mask], 2.4, 1.0)

        close_world = body_world_coords - position
        _set_offsets3d(body_close_scatter, close_world)
        body_close_scatter.set_color(body_rgba.tolist())
        body_close_scatter.set_edgecolor(np.tile(to_rgba(SPHERE_EDGE), (len(close_world), 1)).tolist())
        body_close_scatter.set_sizes(np.asarray(marker_size_close * main_size_scale, dtype=float))
        _set_offsets3d(body_world_scatter, body_world_coords)
        body_world_scatter.set_color(body_rgba.tolist())
        body_world_scatter.set_edgecolor(np.tile(to_rgba(SPHERE_EDGE), (len(body_world_coords), 1)).tolist())
        body_world_scatter.set_sizes(np.asarray(marker_size_world * main_size_scale, dtype=float))

        fragment_coords_list: list[np.ndarray] = []
        fragment_color_list: list[np.ndarray] = []
        fragment_size_scale_list: list[np.ndarray] = []
        fragments_history = cast(list[list[dict[str, np.ndarray]]], result.frame_history["fragments"])
        for fragment in fragments_history[state_idx]:
            coords = np.asarray(fragment["coords"], dtype=float)
            fragment_coords_list.append(coords)
            fragment_color_list.append(
                _dem_color_values(np.zeros(len(coords)), np.asarray(fragment["integrity"], dtype=float), drone.material.stress_limit)
            )
            fragment_size_scale_list.append(np.asarray(fragment["size_scale"], dtype=float))
        if fragment_coords_list:
            fragment_coords = np.vstack(fragment_coords_list)
            fragment_rgba = _dem_rgba(np.concatenate(fragment_color_list))
            fragment_scales = np.concatenate(fragment_size_scale_list)
            fragment_close_coords = fragment_coords - position
            in_close_bounds = (
                (np.abs(fragment_close_coords[:, 0]) <= local_half_span)
                & (np.abs(fragment_close_coords[:, 1]) <= local_half_span)
                & (np.abs(fragment_close_coords[:, 2]) <= local_z_span)
            )
            fragment_close_visible = fragment_close_coords[in_close_bounds]
            fragment_close_rgba = fragment_rgba[in_close_bounds]
            fragment_close_scales = fragment_scales[in_close_bounds]
            _set_offsets3d(fragment_close_scatter, fragment_close_visible)
            fragment_close_scatter.set_color(fragment_close_rgba.tolist())
            fragment_close_scatter.set_edgecolor(np.tile(to_rgba(FRAGMENT_EDGE), (len(fragment_close_visible), 1)).tolist())
            fragment_close_scatter.set_sizes(np.asarray(marker_size_close * fragment_close_scales, dtype=float))
            _set_offsets3d(fragment_world_scatter, fragment_coords)
            fragment_world_scatter.set_color(fragment_rgba.tolist())
            fragment_world_scatter.set_edgecolor(np.tile(to_rgba(FRAGMENT_EDGE), (len(fragment_coords), 1)).tolist())
            fragment_world_scatter.set_sizes(np.asarray(marker_size_world * fragment_scales, dtype=float))
        else:
            _set_offsets3d(fragment_close_scatter, np.empty((0, 3)))
            fragment_close_scatter.set_color(np.empty((0, 4)))
            fragment_close_scatter.set_edgecolor(np.empty((0, 4)))
            fragment_close_scatter.set_sizes(np.array([]))
            _set_offsets3d(fragment_world_scatter, np.empty((0, 3)))
            fragment_world_scatter.set_color(np.empty((0, 4)))
            fragment_world_scatter.set_edgecolor(np.empty((0, 4)))
            fragment_world_scatter.set_sizes(np.array([]))

        trail = result.state_history["position"][: state_idx + 1]
        trail_line.set_data(trail[:, 0], trail[:, 1])
        trail_line.set_3d_properties(trail[:, 2])

        current_time = float(result.state_history["time"][state_idx])
        pulse_state = result.state_history["pulse"][state_idx]
        gate_state = np.asarray(result.state_history["gate"][state_idx], dtype=float) > 0.5
        for artist in gate_artists:
            try:
                artist.remove()
            except Exception:
                pass
        gate_artists[:] = _draw_gate_rings(ax_world, gates, gate_state)
        for pulse_idx, pulse_scatter in enumerate(pulse_scatters):
            pulse_line = pulse_lines[pulse_idx]
            if pulse_state[pulse_idx] > 0.5 and turret_pulse_active(turrets[pulse_idx], current_time) and turret_can_reach_drone(turrets[pulse_idx], position):
                pulse_points, pulse_intensity = _pulse_volume_points(turrets[pulse_idx], position)
                _set_offsets3d(pulse_scatter, pulse_points)
                if len(pulse_points):
                    pulse_colors = np.column_stack(
                        [
                            np.full(len(pulse_intensity), 0.98),
                            np.full(len(pulse_intensity), 0.93),
                            0.18 + 0.7 * pulse_intensity,
                            0.04 + 0.32 * pulse_intensity,
                        ]
                    )
                    pulse_scatter.set_color(pulse_colors.tolist())
                    pulse_scatter.set_sizes(np.asarray(14.0 + 26.0 * pulse_intensity, dtype=float))
                    pulse_scatter.set_alpha(1.0)
                    turret = turrets[pulse_idx].position
                    pulse_line.set_data([turret[0], position[0]], [turret[1], position[1]])
                    pulse_line.set_3d_properties([turret[2], position[2]])
                    pulse_line.set_alpha(0.35)
                else:
                    _set_offsets3d(pulse_scatter, np.empty((0, 3)))
                    pulse_scatter.set_alpha(0.0)
                    pulse_line.set_data([], [])
                    pulse_line.set_3d_properties([])
                    pulse_line.set_alpha(0.0)
            else:
                _set_offsets3d(pulse_scatter, np.empty((0, 3)))
                pulse_scatter.set_alpha(0.0)
                pulse_line.set_data([], [])
                pulse_line.set_3d_properties([])
                pulse_line.set_alpha(0.0)

        status_text.set_text(
            f"t = {result.state_history['time'][state_idx]:.2f} s\n"
            f"stress = {result.state_history['stress'][state_idx] / max(drone.material.stress_limit, 1.0):.2f}x limit\n"
            f"gates = {int(np.count_nonzero(gate_state))}/{len(gates)}"
        )
        return [
            body_close_scatter,
            fragment_close_scatter,
            body_world_scatter,
            fragment_world_scatter,
            trail_line,
            status_text,
            *pulse_scatters,
            *pulse_lines,
            *gate_artists,
        ]

    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=70, blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=fps), progress_callback=progress_callback)
    plt.close(fig)
    return path


def save_design_evolution_animation(
    generation_bests: list[dict[str, object]],
    config: SimConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path | None:
    if not generation_bests:
        return None

    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    drones = [
        build_voxel_drone(cast(DroneDesign, row["design"]), DEFAULT_MATERIAL, config.voxel_resolution)
        for row in generation_bests
    ]
    max_extent_xy = max(np.max(np.abs(drone.coords[..., :2])) for drone in drones)
    max_extent_z = max(np.max(np.abs(drone.coords[..., 2])) for drone in drones)
    path = ANIMATION_DIR / "design_evolution.gif"

    fig = plt.figure(figsize=(8.4, 6.8))
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    limits = (
        (-1.08 * max_extent_xy, 1.08 * max_extent_xy),
        (-1.08 * max_extent_xy, 1.08 * max_extent_xy),
        (-1.55 * max_extent_z, 1.55 * max_extent_z),
    )
    _style_axes(ax, limits, "Best Drone Per Generation")
    ax.view_init(elev=23, azim=-47)

    first_drone = drones[0]
    marker_size = _voxel_marker_size(fig, ax, first_drone, radius_scale=1.18)
    scatter: Any = cast(Any, ax).scatter([], [], zs=[], s=marker_size, c="#d96d1d", edgecolors=SPHERE_EDGE, linewidths=0.2, alpha=0.96)
    title_text = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=14)
    subtitle_text = fig.text(0.5, 0.93, "", ha="center", va="top", fontsize=10)

    hold_frames = max(config.evolution_hold_frames, 1)
    total_frames = hold_frames * len(generation_bests)

    def update(frame_idx: int):
        generation_idx = min(frame_idx // hold_frames, len(generation_bests) - 1)
        drone = drones[generation_idx]
        row = generation_bests[generation_idx]
        coords = drone.coords[drone.solid]
        colors = np.full(len(coords), 0.86)
        colors[drone.motor_mask[drone.solid]] = 0.93
        sizes = marker_size * np.where(drone.motor_mask[drone.solid], 2.4, 1.0)

        _set_offsets3d(scatter, coords)
        scatter.set_color(_dem_rgba(colors).tolist())
        scatter.set_edgecolor(np.tile(to_rgba(SPHERE_EDGE), (len(coords), 1)).tolist())
        scatter.set_sizes(np.asarray(sizes, dtype=float))
        title_text.set_text(f"Generation {row['generation']}")
        subtitle_text.set_text(
            f"score = {row['score']:.2f}   progress = {row['progress']:.2f}   survived = {row['survived']}"
        )
        return [scatter, title_text, subtitle_text]

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=140, blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=config.evolution_gif_fps), progress_callback=progress_callback)
    plt.close(fig)
    return path


def save_snapshot_archive(drone: VoxelDrone, result: FlightResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_count = len(result.snapshot_frame_indices)
    if snapshot_count == 0:
        solid_coords = drone.coords[drone.solid]
        motor_scale = np.where(drone.motor_mask[drone.solid], 2.4, 1.0).astype(float)
        np.savez(
            path,
            coords=np.array([solid_coords], dtype=object),
            rgba=np.array([_dem_rgba(np.full(len(solid_coords), 0.86))], dtype=object),
            edge_rgba=np.array([np.tile(to_rgba(SPHERE_EDGE), (len(solid_coords), 1))], dtype=object),
            sizes=np.array([motor_scale], dtype=object),
            labels=np.array(["Initial"], dtype=object),
            times=np.array([0.0], dtype=float),
            spacing=np.array([drone.spacing], dtype=float),
            arm_length=np.array([drone.design.arm_length], dtype=float),
            body_r3=np.array([drone.design.body_r3], dtype=float),
        )
        return path

    coords_stack: list[np.ndarray] = []
    rgba_stack: list[np.ndarray] = []
    edge_stack: list[np.ndarray] = []
    size_stack: list[np.ndarray] = []
    body_masks = cast(list[np.ndarray], result.frame_history["body_active_mask"])
    integrity_history = cast(list[np.ndarray], result.frame_history["integrity"])
    stress_history = cast(list[np.ndarray], result.frame_history["von_mises"])
    fragments_history = cast(list[list[dict[str, np.ndarray]]], result.frame_history["fragments"])
    positions = np.asarray(result.state_history["position"], dtype=float)
    rotations = np.asarray(result.state_history["rotation"], dtype=float)

    for frame_idx in result.snapshot_frame_indices:
        position = positions[frame_idx]
        rotation = rotations[frame_idx]
        active_mask = np.asarray(body_masks[frame_idx], dtype=bool)
        integrity = np.asarray(integrity_history[frame_idx], dtype=float)
        stress = np.asarray(stress_history[frame_idx], dtype=float)

        body_coords = _body_world_points(drone, position, rotation, active_mask) - position
        body_colors = _dem_rgba(
            _dem_color_values(stress[active_mask], integrity[active_mask], drone.material.stress_limit)
        )
        body_sizes = np.where(drone.motor_mask[active_mask], 2.4, 1.0).astype(float)
        body_edges = np.tile(to_rgba(SPHERE_EDGE), (len(body_coords), 1))

        fragment_coords_list: list[np.ndarray] = []
        fragment_rgba_list: list[np.ndarray] = []
        fragment_edge_list: list[np.ndarray] = []
        fragment_size_list: list[np.ndarray] = []
        for fragment in fragments_history[frame_idx]:
            coords = np.asarray(fragment["coords"], dtype=float) - position
            fragment_coords_list.append(coords)
            fragment_rgba_list.append(
                _dem_rgba(
                    _dem_color_values(
                        np.zeros(len(coords), dtype=float),
                        np.asarray(fragment["integrity"], dtype=float),
                        drone.material.stress_limit,
                    )
                )
            )
            fragment_edge_list.append(np.tile(to_rgba(FRAGMENT_EDGE), (len(coords), 1)))
            fragment_size_list.append(np.asarray(fragment["size_scale"], dtype=float))

        if fragment_coords_list:
            frame_coords = np.vstack([body_coords, *fragment_coords_list])
            frame_rgba = np.vstack([body_colors, *fragment_rgba_list])
            frame_edges = np.vstack([body_edges, *fragment_edge_list])
            frame_sizes = np.concatenate([body_sizes, *fragment_size_list])
        else:
            frame_coords = body_coords
            frame_rgba = body_colors
            frame_edges = body_edges
            frame_sizes = body_sizes

        coords_stack.append(frame_coords)
        rgba_stack.append(frame_rgba)
        edge_stack.append(frame_edges)
        size_stack.append(frame_sizes)

    labels = np.array(result.snapshot_labels or [f"Snapshot {idx}" for idx in range(snapshot_count)], dtype=object)
    times = np.array(result.snapshot_times, dtype=float)
    np.savez(
        path,
        coords=np.array(coords_stack, dtype=object),
        rgba=np.array(rgba_stack, dtype=object),
        edge_rgba=np.array(edge_stack, dtype=object),
        sizes=np.array(size_stack, dtype=object),
        labels=labels,
        times=times,
        spacing=np.array([drone.spacing], dtype=float),
        arm_length=np.array([drone.design.arm_length], dtype=float),
        body_r3=np.array([drone.design.body_r3], dtype=float),
    )
    return path


def save_summary_artifacts(
    drone: VoxelDrone,
    design: DroneDesign,
    result: FlightResult,
    history: list[dict[str, float]],
    animation_progress_callback: Callable[[int, int], None] | None = None,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    ax1 = cast(Axes3D, fig.add_subplot(221, projection="3d"))
    design_coords = drone.coords[drone.solid]
    design_values = np.full(np.count_nonzero(drone.solid), 0.86)
    design_values[drone.motor_mask[drone.solid]] = 0.93
    design_rgba = _dem_rgba(design_values)
    design_scales = np.where(drone.motor_mask[drone.solid], 2.4, 1.0).astype(float)
    _render_dem_closeup(fig, ax1, drone, design_coords, design_rgba, design_scales, "DEM Lattice Chassis")

    ax2 = cast(Axes3D, fig.add_subplot(222, projection="3d"))
    traj = result.trajectory
    way = result.waypoints
    gates = [
        Gate(
            center=np.asarray(spec["center"], dtype=float),
            normal=np.asarray(spec["normal"], dtype=float),
            radius=float(cast(float, spec["radius"])),
            label=str(spec["label"]),
        )
        for spec in cast(list[dict[str, object]], result.frame_history["gates"])
    ]
    ax2.plot(way[:, 0], way[:, 1], way[:, 2], "k--", lw=1.5, label="course")
    ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", lw=2.0, label="best flight")
    for turret in make_turrets():
        ax2.scatter(*turret.position, color="crimson", s=80)
    _draw_gate_rings(ax2, gates, np.asarray(result.state_history["gate"][-1], dtype=float) > 0.5 if len(result.state_history["gate"]) else None)
    ax2.set_title("Hostile Flight Path")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend(loc="upper left")
    try:
        ax2.set_box_aspect((2.0, 1.2, 1.0))
    except Exception:
        pass

    ax3 = fig.add_subplot(223)
    generations = [row["generation"] for row in history]
    best = [row["best_score"] for row in history]
    avg = [row["avg_score"] for row in history]
    ax3.plot(generations, best, marker="o", label="best")
    ax3.plot(generations, avg, marker="s", label="average")
    ax3.set_title("Optimization History")
    ax3.set_xlabel("generation")
    ax3.set_ylabel("score")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(224)
    steps = np.arange(len(result.state_history["stress"]))
    ax4.plot(steps, result.state_history["stress"], label="max von Mises")
    ax4.plot(steps, result.state_history["track_error"], label="track error")
    ax4.axhline(drone.material.stress_limit, color="crimson", linestyle="--", label="stress limit")
    ax4.set_title("Stress + Tracking")
    ax4.set_xlabel("time step")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "projecty_summary.png", dpi=180)
    plt.close(fig)

    if result.vm_snapshots:
        cols = len(result.vm_snapshots)
        fig = plt.figure(figsize=(4.2 * cols, 4.8))
        active_masks = cast(list[np.ndarray], result.frame_history["body_active_mask"])
        integrity_history = cast(list[np.ndarray], result.frame_history["integrity"])
        for index, (vm, t) in enumerate(zip(result.vm_snapshots, result.snapshot_times), start=1):
            ax = fig.add_subplot(1, cols, index, projection="3d")
            frame_idx = result.snapshot_frame_indices[index - 1]
            active_mask = np.asarray(active_masks[frame_idx], dtype=bool)
            integrity = np.asarray(integrity_history[frame_idx], dtype=float)
            coords = drone.coords[active_mask]
            values = _dem_color_values(vm[active_mask], integrity[active_mask], drone.material.stress_limit)
            rgba = _dem_rgba(values)
            scales = np.where(drone.motor_mask[active_mask], 2.4, 1.0).astype(float)
            _render_dem_closeup(fig, ax, drone, coords, rgba, scales, f"t = {t:.2f} s")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "projecty_stress_snapshots.png", dpi=180)
        plt.close(fig)

    snapshot_archive_path = save_snapshot_archive(drone, result, OUTPUT_DIR / "projecty_snapshot_archive.npz")
    gif_path = save_animation(
        drone,
        result,
        ANIMATION_DIR / "final_simulation.gif",
        progress_callback=animation_progress_callback,
    )

    report = {
        "design": format_design(design),
        "score": result.score,
        "summary": result.summary,
        "survived": result.survived,
        "artifacts": {
            "gif": str(gif_path),
            "summary": str(OUTPUT_DIR / "projecty_summary.png"),
            "stress_snapshots": str(OUTPUT_DIR / "projecty_stress_snapshots.png"),
            "snapshot_archive": str(snapshot_archive_path),
        },
        "visualization_notes": {
            "voxel_rendering": "One rendered sphere marker per occupied voxel center on a regular lattice.",
            "radius_choice": "Marker radius is scaled to make neighboring lattice spheres touch or slightly overlap in the oblique camera view.",
            "fragmentation": "Detached connected components remain visible as moving sphere clusters with ballistic debris motion.",
        },
        "textbook_mapping": {
            "geometry": "Eq. 4.1, Eq. 4.2",
            "stress": "Eq. 4.3-4.24, Eq. 4.105-4.109",
            "rigid_body": "Eq. 4.81-4.101",
            "pressure_wave": "Eq. 4.102",
        },
    }
    with open(OUTPUT_DIR / "projecty_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
