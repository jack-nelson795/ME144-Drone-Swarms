from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from .config import Material
    from .design import DroneDesign
except ImportError:
    from config import Material
    from design import DroneDesign


@dataclass
class VoxelDrone:
    design: DroneDesign
    material: Material
    spacing: float
    coords: np.ndarray
    solid: np.ndarray
    surface: np.ndarray
    mass_per_voxel: np.ndarray
    total_mass: float
    inertia: np.ndarray
    motor_positions: np.ndarray
    motor_mask: np.ndarray
    motor_masks: np.ndarray
    k_thrust: float
    k_yaw: float
    max_total_thrust: float
    thrust_to_weight: float
    voxel_area: float


def _grid(resolution: int, extent_xy: float, extent_z: float):
    x = np.linspace(-extent_xy, extent_xy, resolution)
    y = np.linspace(-extent_xy, extent_xy, resolution)
    z = np.linspace(-extent_z, extent_z, max(10, resolution // 2))
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])
    hz = float(z[1] - z[0])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([xx, yy, zz], axis=-1)
    return coords, (hx, hy, hz)


def _superellipsoid_mask(coords: np.ndarray, center: np.ndarray, radii: np.ndarray, powers: np.ndarray):
    shifted = coords - center
    field = (
        np.abs(shifted[..., 0] / radii[0]) ** powers[0]
        + np.abs(shifted[..., 1] / radii[1]) ** powers[1]
        + np.abs(shifted[..., 2] / radii[2]) ** powers[2]
    )
    return field <= 1.0


def _sphere_mask(coords: np.ndarray, center: np.ndarray, radius: float):
    shifted = coords - center
    return np.linalg.norm(shifted, axis=-1) <= radius


def _frame_body_mask(coords: np.ndarray, design: DroneDesign) -> np.ndarray:
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]
    a = max(0.78 * design.arm_length, 1.8 * design.body_r1)
    b = max(0.78 * design.arm_length, 1.8 * design.body_r2)
    # Use the inverse superellipse exponent in-planform so larger textbook
    # shape parameters create a more concave membrane between the arms.
    px = np.clip(1.0 / max(design.body_p1, 1.0e-6), 0.42, 0.92)
    py = np.clip(1.0 / max(design.body_p2, 1.0e-6), 0.42, 0.92)
    pz = max(design.body_p3, 0.75)

    planar_field = (np.abs(x / a) ** px + np.abs(y / b) ** py)
    planar_fraction = np.clip(planar_field, 0.0, None)
    thickness = design.body_r3 * (
        0.28
        + 1.05 * np.clip(1.0 - planar_fraction, 0.0, 1.0) ** (0.62 / pz)
    )
    membrane = (planar_field <= 1.0) & (np.abs(z) <= thickness)

    inner_soft_field = (
        np.abs(x / max(0.56 * a, 1.0e-6)) ** max(1.05, 1.0 / px)
        + np.abs(y / max(0.56 * b, 1.0e-6)) ** max(1.05, 1.0 / py)
        + np.abs(z / max(1.15 * design.body_r3, 1.0e-6)) ** max(0.95, pz)
    )

    rib_half_width = 0.42 * design.arm_radius
    ribs = (
        ((np.abs(y) <= rib_half_width) & (np.abs(x) <= 0.8 * design.arm_length))
        | ((np.abs(x) <= rib_half_width) & (np.abs(y) <= 0.8 * design.arm_length))
    ) & (np.abs(z) <= 0.9 * design.body_r3)

    center_core = _superellipsoid_mask(
        coords,
        center=np.zeros(3),
        radii=np.array([0.66 * design.body_r1, 0.66 * design.body_r2, 1.18 * design.body_r3]),
        powers=np.array([max(1.15, 1.0 / px), max(1.15, 1.0 / py), max(0.9, pz)]),
    )

    return membrane | (inner_soft_field <= 1.0) | ribs | center_core


def _tapered_arm_mask(coords: np.ndarray, design: DroneDesign, axis: int) -> np.ndarray:
    axial = coords[..., axis]
    radial_axis = 1 if axis == 0 else 0
    radial = np.sqrt(coords[..., radial_axis] ** 2 + coords[..., 2] ** 2)
    root_start = 0.08 * design.arm_length
    s = np.clip((np.abs(axial) - root_start) / (0.92 * design.arm_length), 0.0, 1.0)
    root_radius = 1.75 * design.arm_radius + 0.38 * design.body_r3
    tip_radius = max(0.72 * design.arm_radius, 0.3 * design.body_r3)
    radius = (1.0 - s) * root_radius + s * tip_radius
    axial_limit = np.abs(axial) <= (design.arm_length + 0.2 * design.motor_radius)
    return axial_limit & (radial <= radius)


def _motor_bridge_mask(coords: np.ndarray, motor_center: np.ndarray, design: DroneDesign) -> np.ndarray:
    axis = int(np.argmax(np.abs(motor_center[:2])))
    sign = 1.0 if motor_center[axis] >= 0.0 else -1.0
    axial = sign * coords[..., axis]
    other_axis = 1 - axis
    radial = np.sqrt(coords[..., other_axis] ** 2 + coords[..., 2] ** 2)

    start = 0.64 * design.arm_length
    stop = design.arm_length + 0.24 * design.motor_radius
    s = np.clip((axial - start) / max(stop - start, 1.0e-6), 0.0, 1.0)
    radius = (1.0 - s) * (1.1 * design.arm_radius + 0.18 * design.body_r3) + s * (0.68 * design.motor_radius)
    return (axial >= start) & (axial <= stop) & (radial <= radius)


def _lightweight_holes(coords: np.ndarray, design: DroneDesign) -> np.ndarray:
    if design.lightweight_hole <= 0.0:
        return np.zeros(coords.shape[:-1], dtype=bool)

    hole_radius = 0.62 * design.lightweight_hole
    offset = 1.35 * design.lightweight_hole
    mask = np.zeros(coords.shape[:-1], dtype=bool)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            center = np.array([sx * offset, sy * offset, 0.0])
            shifted = coords - center
            radial = np.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)
            mask |= (radial <= hole_radius) & (np.abs(shifted[..., 2]) <= 1.15 * design.body_r3)
    return mask


def _make_surface_mask(solid: np.ndarray) -> np.ndarray:
    surface = np.zeros_like(solid, dtype=bool)
    nx, ny, nz = solid.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not solid[i, j, k]:
                    continue
                if i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1):
                    surface[i, j, k] = True
                    continue
                neighborhood = solid[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                surface[i, j, k] = not neighborhood.all()
    return surface


def build_voxel_drone(design: DroneDesign, material: Material, resolution: int) -> VoxelDrone:
    design = design.clipped()
    extent_xy = design.arm_length + 1.8 * design.motor_radius
    extent_z = max(1.8 * design.body_r3, design.motor_radius)
    coords, spacing = _grid(resolution, extent_xy, extent_z)
    hx, hy, hz = spacing

    body = _frame_body_mask(coords, design)
    arm_x = _tapered_arm_mask(coords, design, axis=0)
    arm_y = _tapered_arm_mask(coords, design, axis=1)
    motor_positions = np.array(
        [
            [design.arm_length, 0.0, 0.0],
            [0.0, design.arm_length, 0.0],
            [-design.arm_length, 0.0, 0.0],
            [0.0, -design.arm_length, 0.0],
        ]
    )
    motors = np.zeros(body.shape, dtype=bool)
    motor_masks: list[np.ndarray] = []
    motor_bridges = np.zeros(body.shape, dtype=bool)
    for motor_center in motor_positions:
        motor_component = _sphere_mask(coords, motor_center, design.motor_radius)
        motors |= motor_component
        motor_masks.append(motor_component)
        motor_bridges |= _motor_bridge_mask(coords, motor_center, design)

    solid = body | arm_x | arm_y | motor_bridges | motors

    hole_mask = _lightweight_holes(coords, design)
    solid &= ~hole_mask
    solid |= motors

    surface = _make_surface_mask(solid)
    voxel_volume = hx * hy * hz
    active_count = max(int(np.count_nonzero(solid)), 1)
    chassis_mass = design.chassis_mass_target
    chassis_mass_per_voxel = chassis_mass / active_count

    mass_per_voxel = np.zeros(solid.shape, dtype=float)
    mass_per_voxel[solid] = chassis_mass_per_voxel

    total_mass = chassis_mass + 4.0 * design.motor_mass_each
    inertia = np.zeros((3, 3), dtype=float)
    active_coords = coords[solid]
    for point in active_coords:
        r2 = float(np.dot(point, point))
        inertia += chassis_mass_per_voxel * (r2 * np.eye(3) - np.outer(point, point))
    for motor_center in motor_positions:
        r2 = float(np.dot(motor_center, motor_center))
        inertia += design.motor_mass_each * (r2 * np.eye(3) - np.outer(motor_center, motor_center))

    hover_force = total_mass * 9.81 / 4.0
    radius_factor = np.clip((design.motor_radius / 0.047) ** 2.15, 0.6, 1.9)
    motor_mass_factor = np.clip((design.motor_mass_each / 0.145) ** 0.4, 0.82, 1.22)
    thrust_radius_gain = 0.72 + 0.28 * radius_factor
    thrust_mass_gain = 0.78 + 0.22 * motor_mass_factor
    k_thrust = design.thrust_scale * hover_force * thrust_radius_gain * thrust_mass_gain
    max_total_thrust = 4.0 * k_thrust
    thrust_to_weight = max_total_thrust / max(total_mass * 9.81, 1.0e-8)
    k_yaw = 0.092 * k_thrust * design.arm_length * np.clip(radius_factor ** 0.45, 0.8, 1.35)
    voxel_area = voxel_volume ** (2.0 / 3.0)

    return VoxelDrone(
        design=design,
        material=material,
        spacing=float((hx + hy + hz) / 3.0),
        coords=coords,
        solid=solid,
        surface=surface,
        mass_per_voxel=mass_per_voxel,
        total_mass=total_mass,
        inertia=inertia,
        motor_positions=motor_positions,
        motor_mask=motors,
        motor_masks=np.stack(motor_masks, axis=0),
        k_thrust=k_thrust,
        k_yaw=k_yaw,
        max_total_thrust=max_total_thrust,
        thrust_to_weight=thrust_to_weight,
        voxel_area=voxel_area,
    )
