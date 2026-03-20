from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

try:
    from .config import SimConfig
    from .elasticity import ElasticState, initialize_elastic_state, relax_dynamic_stress
    from .geometry import VoxelDrone
except ImportError:
    from config import SimConfig
    from elasticity import ElasticState, initialize_elastic_state, relax_dynamic_stress
    from geometry import VoxelDrone


def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def reorthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    return u @ vt


@dataclass
class RigidState:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class Turret:
    position: np.ndarray
    pressure_peak: float
    decay: float
    frequency: float
    pulse_width: float
    phase_offset: float
    x_window: tuple[float, float]


@dataclass(frozen=True)
class Gate:
    center: np.ndarray
    normal: np.ndarray
    radius: float
    label: str


@dataclass
class FlightResult:
    design_name: str
    score: float
    progress: float
    avg_speed: float
    max_stress: float
    max_track_error: float
    energy: float
    survived: bool
    trajectory: np.ndarray
    waypoints: np.ndarray
    vm_snapshots: list[np.ndarray]
    snapshot_times: list[float]
    snapshot_labels: list[str]
    snapshot_frame_indices: list[int]
    state_history: dict[str, np.ndarray]
    frame_history: dict[str, object]
    summary: dict[str, float]


@dataclass
class Fragment:
    local_coords: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    omega: np.ndarray
    integrity: np.ndarray
    size_scale: np.ndarray


def _rotation_step(rotation: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    rotation_pred = rotation + dt * rotation @ hat(omega)
    return reorthonormalize(rotation_pred)


def make_course() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.9],
            [2.0, 0.0, 1.1],
            [4.0, 1.3, 1.2],
            [6.2, 0.4, 1.0],
            [8.0, -1.1, 1.25],
            [10.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_course_gates(waypoints: np.ndarray) -> list[Gate]:
    segment_vectors = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = max(float(cumulative[-1]), 1.0e-8)
    gate_specs = [(0.5 * total_length, "50% Gate"), (total_length, "Finish Gate")]
    gates: list[Gate] = []

    for target_length, label in gate_specs:
        seg_idx = int(np.searchsorted(cumulative[1:], target_length, side="left"))
        seg_idx = min(seg_idx, len(segment_vectors) - 1)
        seg_start_length = cumulative[seg_idx]
        seg_length = max(float(segment_lengths[seg_idx]), 1.0e-8)
        alpha = np.clip((target_length - seg_start_length) / seg_length, 0.0, 1.0)
        center = (1.0 - alpha) * waypoints[seg_idx] + alpha * waypoints[seg_idx + 1]
        gates.append(Gate(center=center, normal=np.array([1.0, 0.0, 0.0]), radius=0.62, label=label))
    return gates


def make_turrets(config: SimConfig | None = None) -> list[Turret]:
    pressure_scale = 1.0 if config is None else config.pulse_pressure_scale
    return [
        Turret(
            np.array([2.3, 2.8, 1.15]),
            pressure_peak=12000.0 * pressure_scale,
            decay=8.8,
            frequency=0.88,
            pulse_width=0.22,
            phase_offset=0.00,
            x_window=(-0.5, 3.45),
        ),
        Turret(
            np.array([5.6, -2.9, 1.0]),
            pressure_peak=16500.0 * pressure_scale,
            decay=9.8,
            frequency=0.88,
            pulse_width=0.22,
            phase_offset=0.34,
            x_window=(3.45, 6.55),
        ),
        Turret(
            np.array([7.1, 3.6, 1.3]),
            pressure_peak=18500.0 * pressure_scale,
            decay=10.6,
            frequency=0.88,
            pulse_width=0.22,
            phase_offset=0.68,
            x_window=(6.55, 9.2),
        ),
    ]


def turret_pulse_active(turret: Turret, t: float) -> bool:
    phase = (t * turret.frequency + turret.phase_offset) % 1.0
    return phase <= turret.pulse_width


def turret_can_reach_drone(turret: Turret, drone_position: np.ndarray) -> bool:
    x = float(drone_position[0])
    return turret.x_window[0] <= x < turret.x_window[1]


def _initial_state() -> RigidState:
    return RigidState(
        position=np.array([0.0, 0.0, 0.9], dtype=float),
        velocity=np.zeros(3, dtype=float),
        rotation=np.eye(3, dtype=float),
        omega=np.zeros(3, dtype=float),
    )


def _course_target(waypoints: np.ndarray, progress: int) -> tuple[np.ndarray, np.ndarray]:
    idx = min(progress, len(waypoints) - 2)
    target = waypoints[idx + 1]
    tangent = waypoints[idx + 1] - waypoints[idx]
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent = tangent / norm
    return target, tangent


def _desired_rotation(force_world: np.ndarray, yaw_heading: float) -> np.ndarray:
    b3 = force_world / max(np.linalg.norm(force_world), 1e-9)
    yaw_axis = np.array([math.cos(yaw_heading), math.sin(yaw_heading), 0.0])
    b2 = np.cross(b3, yaw_axis)
    if np.linalg.norm(b2) < 1e-9:
        b2 = np.array([0.0, 1.0, 0.0])
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    return np.column_stack((b1, b2, b3))


def _lookahead_target(waypoints: np.ndarray, progress: int, state: RigidState) -> tuple[np.ndarray, np.ndarray]:
    idx = min(progress, len(waypoints) - 2)
    next_wp = waypoints[idx + 1]
    if idx + 2 < len(waypoints):
        future_wp = waypoints[idx + 2]
        blend = np.clip(np.linalg.norm(next_wp - state.position) / 2.4, 0.2, 1.0)
        target = blend * next_wp + (1.0 - blend) * future_wp
        tangent = future_wp - waypoints[idx]
    else:
        target = next_wp
        tangent = next_wp - waypoints[idx]
    tangent_norm = np.linalg.norm(tangent)
    tangent = tangent / tangent_norm if tangent_norm > 1.0e-9 else np.array([1.0, 0.0, 0.0])
    return target, tangent


def _path_progress_fraction(waypoints: np.ndarray, progress_idx: int, position: np.ndarray) -> float:
    segment_count = max(len(waypoints) - 1, 1)
    seg_idx = min(progress_idx, len(waypoints) - 2)
    start = waypoints[seg_idx]
    end = waypoints[seg_idx + 1]
    segment = end - start
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq <= 1.0e-12:
        local_progress = 0.0
    else:
        local_progress = float(np.clip(np.dot(position - start, segment) / seg_len_sq, 0.0, 1.0))
    return np.clip((seg_idx + local_progress) / segment_count, 0.0, 1.0)


def _distance_to_finish(waypoints: np.ndarray, position: np.ndarray) -> float:
    return float(np.linalg.norm(waypoints[-1] - position))


def _gate_contains_position(gate: Gate, position: np.ndarray) -> bool:
    rel = position - gate.center
    axial = abs(float(np.dot(rel, gate.normal)))
    radial = float(np.linalg.norm(rel - np.dot(rel, gate.normal) * gate.normal))
    return axial <= 0.34 and radial <= gate.radius


def _next_unpassed_gate(gates: list[Gate], gate_passed: np.ndarray) -> tuple[int | None, Gate | None]:
    for idx, gate in enumerate(gates):
        if not gate_passed[idx]:
            return idx, gate
    return None, None


def _rotation_error(rotation: np.ndarray, desired: np.ndarray) -> np.ndarray:
    err = 0.5 * (desired.T @ rotation - rotation.T @ desired)
    return np.array([err[2, 1], err[0, 2], err[1, 0]])


def _mix_controls(drone: VoxelDrone, collective: float, torque: np.ndarray) -> np.ndarray:
    l = drone.design.arm_length
    ky = max(drone.k_yaw, 1e-8)
    mixer = np.array(
        [
            [0.25, 0.0, -0.25 / l, 0.25 / ky],
            [0.25, 0.25 / l, 0.0, -0.25 / ky],
            [0.25, 0.0, 0.25 / l, 0.25 / ky],
            [0.25, -0.25 / l, 0.0, -0.25 / ky],
        ]
    )
    rhs = np.array([collective, torque[0], torque[1], torque[2]], dtype=float)
    thrusts = mixer @ rhs
    thrusts = np.clip(thrusts, 0.0, drone.k_thrust)
    return thrusts / drone.k_thrust


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int, int]) -> np.ndarray:
    keep = np.zeros_like(mask, dtype=bool)
    if not mask[seed]:
        return keep
    stack = [seed]
    keep[seed] = True
    while stack:
        idx = stack.pop()
        for nxt in _neighbors(*idx, mask.shape):
            if mask[nxt] and not keep[nxt]:
                keep[nxt] = True
                stack.append(nxt)
    return keep


def _effective_vehicle_properties(drone: VoxelDrone, active_mask: np.ndarray) -> dict[str, np.ndarray | float]:
    active_count = max(int(np.count_nonzero(active_mask & drone.solid)), 1)
    chassis_fraction = active_count / max(int(np.count_nonzero(drone.solid)), 1)
    active_mass = drone.design.chassis_mass_target * chassis_fraction

    motor_health = np.zeros(4, dtype=float)
    for idx in range(4):
        motor_voxels = drone.motor_masks[idx]
        motor_total = max(int(np.count_nonzero(motor_voxels)), 1)
        attached = int(np.count_nonzero(active_mask & motor_voxels))
        motor_health[idx] = attached / motor_total
    motor_available = np.clip((motor_health - 0.22) / 0.78, 0.0, 1.0)

    active_mass += float(np.sum(drone.design.motor_mass_each * motor_available))
    avg_motor = float(np.mean(motor_available))
    active_k_thrust = drone.k_thrust * motor_available
    active_max_total_thrust = float(np.sum(active_k_thrust))
    active_thrust_to_weight = active_max_total_thrust / max(active_mass * 9.81, 1.0e-8)
    active_k_yaw = drone.k_yaw * avg_motor
    return {
        "mass": active_mass,
        "motor_health": motor_health,
        "motor_available": motor_available,
        "avg_motor": avg_motor,
        "k_thrust_per_motor": active_k_thrust,
        "k_yaw": active_k_yaw,
        "max_total_thrust": active_max_total_thrust,
        "thrust_to_weight": active_thrust_to_weight,
    }


def _mix_controls_with_availability(
    drone: VoxelDrone,
    collective: float,
    torque: np.ndarray,
    motor_available: np.ndarray,
    k_yaw_active: float,
) -> np.ndarray:
    l = drone.design.arm_length
    ky = max(k_yaw_active, 1.0e-8)
    mixer = np.array(
        [
            [0.25, 0.0, -0.25 / l, 0.25 / ky],
            [0.25, 0.25 / l, 0.0, -0.25 / ky],
            [0.25, 0.0, 0.25 / l, 0.25 / ky],
            [0.25, -0.25 / l, 0.0, -0.25 / ky],
        ]
    )
    rhs = np.array([collective, torque[0], torque[1], torque[2]], dtype=float)
    thrusts = mixer @ rhs
    max_per_motor = drone.k_thrust * np.clip(motor_available, 0.0, 1.0)
    thrusts = np.clip(thrusts, 0.0, max_per_motor)
    return thrusts / max(drone.k_thrust, 1.0e-8)


def _pressure_field(
    drone: VoxelDrone,
    state: RigidState,
    turrets: list[Turret],
    t: float,
    config: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_force = np.zeros(drone.coords.shape, dtype=float)
    net_force_world = np.zeros(3, dtype=float)
    net_torque_body = np.zeros(3, dtype=float)
    surface_indices = np.argwhere(drone.surface)
    rigid_force_scale = config.pulse_rigid_force_scale
    rigid_torque_scale = config.pulse_rigid_torque_scale
    structural_force_scale = 10.0
    voxel_volume = max(drone.spacing**3, 1.0e-12)
    force_density_scale = drone.voxel_area / voxel_volume

    for turret in turrets:
        if not turret_pulse_active(turret, t) or not turret_can_reach_drone(turret, state.position):
            continue

        incident = state.position - turret.position
        dist_center = max(np.linalg.norm(incident), 1e-8)
        if dist_center > 4.5:
            continue
        direction_world = incident / dist_center
        direction_body = state.rotation.T @ direction_world

        for i, j, k in surface_indices:
            point = drone.coords[i, j, k]
            axial = float(np.dot(point, direction_body))
            perp = float(np.linalg.norm(point - axial * direction_body))
            front_factor = 1.0 / (1.0 + math.exp(20.0 * axial))
            base_pressure = (
                turret.pressure_peak
                * math.exp(-0.9 * dist_center)
                * math.exp(-turret.decay * perp)
                * front_factor
            )
            structural_force_body = (
                structural_force_scale
                * base_pressure
                * force_density_scale
                * direction_body
            )
            rigid_force_body = base_pressure * drone.voxel_area * direction_body
            body_force[i, j, k] += structural_force_body
            net_force_world += rigid_force_scale * (state.rotation @ rigid_force_body)
            net_torque_body += rigid_torque_scale * np.cross(point, rigid_force_body)

    return body_force, net_force_world, net_torque_body


def _damage_from_state(
    drone: VoxelDrone,
    config: SimConfig,
    vm: np.ndarray,
    body_force_body: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    stress_ratio = np.zeros_like(vm)
    stress_ratio[active_mask] = vm[active_mask] / max(drone.material.stress_limit, 1.0)
    load_norm = np.linalg.norm(body_force_body, axis=-1)
    load_scale = np.percentile(load_norm[active_mask], 90) if np.any(active_mask) else 1.0
    load_ratio = np.zeros_like(vm)
    if load_scale > 1.0e-9:
        load_ratio[active_mask] = load_norm[active_mask] / load_scale

    damage = np.zeros_like(vm)
    damage[active_mask] = config.dt * config.damage_rate * (
        1.15 * np.clip(stress_ratio[active_mask] - config.damage_softening_threshold, 0.0, None) ** 1.2
        + 0.18 * np.clip(load_ratio[active_mask] - 0.55, 0.0, None)
    )
    return damage


def _neighbors(i: int, j: int, k: int, shape: tuple[int, int, int]):
    nx, ny, nz = shape
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ii = i + di
        jj = j + dj
        kk = k + dk
        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
            yield ii, jj, kk


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int, int]]] = []
    for seed in map(tuple, np.argwhere(mask)):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        component: list[tuple[int, int, int]] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for nxt in _neighbors(*idx, mask.shape):
                if mask[nxt] and not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(component)
    return components


def _anchor_component_mask(drone: VoxelDrone, mask: np.ndarray) -> np.ndarray:
    center_weight = np.linalg.norm(drone.coords, axis=-1)
    masked_weight = np.where(mask, center_weight, np.inf)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    seed = tuple(np.unravel_index(np.argmin(masked_weight), mask.shape))
    keep = np.zeros_like(mask, dtype=bool)
    stack = [seed]
    keep[seed] = True
    while stack:
        idx = stack.pop()
        for nxt in _neighbors(*idx, mask.shape):
            if mask[nxt] and not keep[nxt]:
                keep[nxt] = True
                stack.append(nxt)
    return keep


def _spawn_fragment(
    drone: VoxelDrone,
    state: RigidState,
    indices: list[tuple[int, int, int]],
    integrity: np.ndarray,
    body_force_body: np.ndarray,
) -> Fragment | None:
    if not indices:
        return None
    idx = np.array(indices, dtype=int)
    local_coords = drone.coords[idx[:, 0], idx[:, 1], idx[:, 2]]
    center_local = np.mean(local_coords, axis=0)
    centered = local_coords - center_local
    center_world = state.position + state.rotation @ center_local
    inherited_velocity = state.velocity + state.rotation @ np.cross(state.omega, center_local)
    impulse = np.mean(body_force_body[idx[:, 0], idx[:, 1], idx[:, 2]], axis=0)
    fragment_velocity = inherited_velocity + 0.0045 * (state.rotation @ impulse)
    integrity_values = np.clip(0.35 * integrity[idx[:, 0], idx[:, 1], idx[:, 2]], 0.0, 1.0)
    size_scale = np.where(
        drone.motor_mask[idx[:, 0], idx[:, 1], idx[:, 2]],
        2.4,
        1.0,
    ).astype(float)
    return Fragment(
        local_coords=centered,
        center=center_world,
        velocity=fragment_velocity,
        rotation=state.rotation.copy(),
        omega=0.55 * state.omega.copy(),
        integrity=integrity_values.copy(),
        size_scale=size_scale,
    )


def _integrate_fragments(fragments: list[Fragment], config: SimConfig) -> None:
    gravity = np.array([0.0, 0.0, -config.gravity], dtype=float)
    alive: list[Fragment] = []
    for fragment in fragments:
        fragment.velocity += config.dt * (gravity - config.debris_drag * fragment.velocity)
        fragment.center += config.dt * fragment.velocity
        fragment.omega *= max(0.0, 1.0 - config.dt * config.debris_spin_decay)
        fragment.rotation = _rotation_step(fragment.rotation, fragment.omega, config.dt)
        if fragment.center[2] + np.max(np.abs(fragment.local_coords[:, 2])) > -0.02:
            alive.append(fragment)
    fragments[:] = alive


def _integrate_rigid_body(
    state: RigidState,
    commanded_force_world: np.ndarray,
    desired_rotation: np.ndarray,
    torque_body_cmd: np.ndarray,
    force_world_ext: np.ndarray,
    torque_body_ext: np.ndarray,
    active_mass: float,
    torque_scale: float,
    dt: float,
    gravity: float,
) -> tuple[RigidState, np.ndarray]:
    force_world = (
        commanded_force_world
        + force_world_ext
        + np.array([0.0, 0.0, -active_mass * gravity])
        - 0.35 * active_mass * state.velocity
    )
    accel = force_world / max(active_mass, 1.0e-8)
    velocity_new = state.velocity + dt * accel
    position_new = state.position + dt * velocity_new

    torque_limit = np.clip(4.0 * torque_scale, 0.35, 4.0)
    torque_body = np.clip(torque_body_cmd + torque_body_ext - 1.2 * torque_scale * state.omega, -torque_limit, torque_limit)
    omega_new = np.clip(state.omega + dt * torque_body, -6.0, 6.0)
    rotation_pred = state.rotation + dt * state.rotation @ hat(omega_new)
    rotation_new = reorthonormalize(0.82 * rotation_pred + 0.18 * desired_rotation)

    return RigidState(position_new, velocity_new, rotation_new, omega_new), accel


def simulate_hostile_course(drone: VoxelDrone, config: SimConfig, design_name: str) -> FlightResult:
    state = _initial_state()
    elastic_state: ElasticState = initialize_elastic_state(drone)
    waypoints = make_course()
    gates = make_course_gates(waypoints)
    turrets = make_turrets(config)
    steps = int(config.horizon / config.dt)

    progress_idx = 0
    position_history = np.zeros((steps, 3), dtype=float)
    rotation_history = np.zeros((steps, 3, 3), dtype=float)
    time_history = np.zeros(steps, dtype=float)
    stress_history = np.zeros(steps, dtype=float)
    error_history = np.zeros(steps, dtype=float)
    command_history = np.zeros((steps, 4), dtype=float)
    speed_history = np.zeros(steps, dtype=float)
    pulse_history = np.zeros((steps, len(turrets)), dtype=float)
    gate_history = np.zeros((steps, len(gates)), dtype=float)
    vm_snapshots: list[np.ndarray] = []
    snapshot_times: list[float] = []
    snapshot_labels: list[str] = []
    snapshot_frame_indices: list[int] = []
    snapshot_marks = {steps // 4, steps // 2, 3 * steps // 4, steps - 1}
    last_event_snapshot_step = -config.snapshot_event_cooldown_steps
    energy = 0.0
    survived = True
    reached_finish = False
    gate_passed = np.zeros(len(gates), dtype=bool)
    active_mask = _anchor_component_mask(drone, drone.solid)
    integrity = np.zeros(drone.solid.shape, dtype=float)
    integrity[drone.solid] = 1.0
    main_body_history: list[np.ndarray] = []
    main_integrity_history: list[np.ndarray] = []
    main_stress_history: list[np.ndarray] = []
    fragment_history: list[list[dict[str, np.ndarray]]] = []
    fragments: list[Fragment] = []

    initially_detached = drone.solid & ~active_mask
    for component in _connected_components(initially_detached):
        fragment = _spawn_fragment(drone, state, component, integrity, np.zeros(drone.coords.shape, dtype=float))
        if fragment is not None:
            fragments.append(fragment)
    integrity[~active_mask] = np.clip(integrity[~active_mask], 0.0, 0.35)

    for step in range(steps):
        t = step * config.dt
        vehicle = _effective_vehicle_properties(drone, active_mask)
        target, tangent = _lookahead_target(waypoints, progress_idx, state)
        next_gate_idx, next_gate = _next_unpassed_gate(gates, gate_passed)
        gate_seek_boost = 0.0
        if next_gate is not None:
            gate_error_vec = next_gate.center - state.position
            gate_signed_offset = float(np.dot(state.position - next_gate.center, next_gate.normal))
            gate_distance = float(np.linalg.norm(gate_error_vec))
            gate_window_open = (
                state.position[0] >= next_gate.center[0] - 1.25
                or gate_distance <= 1.9
                or gate_signed_offset > 0.0
                or (next_gate_idx is not None and next_gate_idx > 0 and gate_passed[next_gate_idx - 1])
            )
            if gate_window_open:
                target = next_gate.center.copy()
                if gate_signed_offset > 0.08:
                    tangent = -next_gate.normal.copy()
                else:
                    tangent = next_gate.normal.copy()
                gate_seek_boost = 0.85
        to_target = target - state.position
        track_error = float(np.linalg.norm(to_target))
        finish_error = _distance_to_finish(waypoints, state.position)
        if track_error < 0.55 and progress_idx < len(waypoints) - 2:
            progress_idx += 1
            target, tangent = _lookahead_target(waypoints, progress_idx, state)
            to_target = target - state.position
            track_error = float(np.linalg.norm(to_target))
            finish_error = _distance_to_finish(waypoints, state.position)

        for gate_idx, gate in enumerate(gates):
            if not gate_passed[gate_idx] and _gate_contains_position(gate, state.position):
                gate_passed[gate_idx] = True
        gate_history[step] = gate_passed.astype(float)

        agility_margin = max(float(vehicle["thrust_to_weight"]) - 1.0, 0.02)
        finish_pull = np.clip(1.8 - 0.22 * finish_error, 0.0, 1.2)
        desired_speed = (
            2.1
            + 1.55 * np.sqrt(agility_margin)
            + 0.35 * (1.0 / max(float(vehicle["mass"]), 1.0e-6))
            + finish_pull
            + gate_seek_boost
        )
        vel_target = desired_speed * tangent
        lateral_accel_cap = np.clip(2.2 + 4.2 * agility_margin, 2.4, 8.0)
        vertical_accel_cap = np.clip(1.6 + 3.3 * agility_margin, 1.8, 6.0)
        accel_cmd = np.array(
            [
                2.45 * to_target[0] + 1.45 * (vel_target[0] - state.velocity[0]),
                2.45 * to_target[1] + 1.45 * (vel_target[1] - state.velocity[1]),
                1.9 * to_target[2] + 1.9 * (vel_target[2] - state.velocity[2]),
            ]
        )
        accel_cmd = np.clip(
            accel_cmd,
            [-lateral_accel_cap, -lateral_accel_cap, -vertical_accel_cap],
            [lateral_accel_cap, lateral_accel_cap, vertical_accel_cap],
        )
        desired_force = float(vehicle["mass"]) * (accel_cmd + np.array([0.0, 0.0, config.gravity]))
        yaw_heading = math.atan2(tangent[1], tangent[0])
        desired_rotation = _desired_rotation(desired_force, yaw_heading)
        rot_err = _rotation_error(state.rotation, desired_rotation)
        motor_authority = float(vehicle["avg_motor"])
        torque_cmd = (-8.4 * rot_err - 1.35 * state.omega) * motor_authority
        collective = max(float(desired_force @ state.rotation[:, 2]), 0.0)
        command = _mix_controls_with_availability(
            drone,
            collective,
            torque_cmd,
            np.asarray(vehicle["motor_available"], dtype=float),
            float(vehicle["k_yaw"]),
        )
        thrust_ratio = min(
            float(np.sum(np.asarray(vehicle["k_thrust_per_motor"], dtype=float) * command)) / max(collective, 1e-8),
            min(1.18, float(vehicle["thrust_to_weight"])),
        )
        commanded_force_world = thrust_ratio * desired_force

        body_force_body, pressure_force_world, pressure_torque_body = _pressure_field(drone, state, turrets, t, config)
        for turret_idx, turret in enumerate(turrets):
            pulse_history[step, turret_idx] = 1.0 if (turret_pulse_active(turret, t) and turret_can_reach_drone(turret, state.position)) else 0.0
        state, rigid_accel = _integrate_rigid_body(
            state=state,
            commanded_force_world=commanded_force_world,
            desired_rotation=desired_rotation,
            torque_body_cmd=torque_cmd,
            force_world_ext=pressure_force_world,
            torque_body_ext=pressure_torque_body,
            active_mass=float(vehicle["mass"]),
            torque_scale=max(0.25, motor_authority),
            dt=config.dt,
            gravity=config.gravity,
        )

        inertial_body = state.rotation.T @ rigid_accel
        effective_force = body_force_body.copy()
        effective_force[drone.solid] -= drone.material.density * inertial_body
        vm, max_vm = relax_dynamic_stress(
            drone=drone,
            state=elastic_state,
            body_force=effective_force,
            dt=config.stress_relaxation_dt,
            steps=config.stress_relaxation_steps,
            damping=config.stress_damping,
        )
        vm_active = np.zeros_like(vm)
        vm_active[active_mask] = vm[active_mask]
        damage = _damage_from_state(drone, config, vm_active, body_force_body, active_mask)
        integrity[active_mask] = np.clip(integrity[active_mask] - damage[active_mask], 0.0, 1.0)

        overstressed = active_mask & (
            (integrity <= config.fragment_integrity_threshold) | (vm_active >= 1.04 * drone.material.stress_limit)
        )
        if np.any(overstressed):
            detaching = active_mask & overstressed
            keep_mask = active_mask & ~detaching
            keep_main = _anchor_component_mask(drone, keep_mask)

            released_components = _connected_components(detaching | (keep_mask & ~keep_main))
            for component in released_components:
                fragment = _spawn_fragment(drone, state, component, integrity, body_force_body)
                if fragment is not None:
                    fragments.append(fragment)
            active_mask = keep_main
            integrity[~active_mask] = np.clip(integrity[~active_mask], 0.0, 0.35)

        keep_main = _anchor_component_mask(drone, active_mask)
        passive_release = active_mask & ~keep_main
        if np.any(passive_release):
            for component in _connected_components(passive_release):
                fragment = _spawn_fragment(drone, state, component, integrity, body_force_body)
                if fragment is not None:
                    fragments.append(fragment)
            active_mask = keep_main
            integrity[~active_mask] = np.clip(integrity[~active_mask], 0.0, 0.35)

        _integrate_fragments(fragments, config)

        position_history[step] = state.position
        rotation_history[step] = state.rotation
        time_history[step] = t
        stress_history[step] = float(vm_active[active_mask].max()) if np.any(active_mask) else 0.0
        error_history[step] = track_error
        command_history[step] = command
        speed_history[step] = np.linalg.norm(state.velocity)
        energy += float(np.sum((drone.k_thrust * command) ** 2) * config.dt)
        main_body_history.append(active_mask.copy())
        main_integrity_history.append(integrity.copy())
        main_stress_history.append(vm_active.copy())
        fragment_history.append(
            [
                {
                    "coords": fragment.center + (fragment.local_coords @ fragment.rotation.T),
                    "integrity": fragment.integrity.copy(),
                    "size_scale": fragment.size_scale.copy(),
                }
                for fragment in fragments
            ]
        )

        if step in snapshot_marks:
            vm_snapshots.append(vm_active.copy())
            snapshot_times.append(t)
            snapshot_labels.append(f"Course Snapshot @ {t:.2f}s")
            snapshot_frame_indices.append(step)

        active_pulses = np.flatnonzero(pulse_history[step] > 0.5)
        if len(active_pulses) and step - last_event_snapshot_step >= config.snapshot_event_cooldown_steps:
            peak_idx = int(active_pulses[0])
            vm_snapshots.append(vm_active.copy())
            snapshot_times.append(t)
            snapshot_labels.append(f"Impact Phase {peak_idx + 1} @ {t:.2f}s")
            snapshot_frame_indices.append(step)
            last_event_snapshot_step = step

        if bool(np.all(gate_passed)):
            reached_finish = True
            position_history = position_history[: step + 1]
            rotation_history = rotation_history[: step + 1]
            time_history = time_history[: step + 1]
            stress_history = stress_history[: step + 1]
            error_history = error_history[: step + 1]
            command_history = command_history[: step + 1]
            speed_history = speed_history[: step + 1]
            pulse_history = pulse_history[: step + 1]
            gate_history = gate_history[: step + 1]
            break

        if state.position[2] < 0.15 or max_vm > 1.12 * drone.material.stress_limit or not np.any(active_mask):
            survived = False
            position_history = position_history[: step + 1]
            rotation_history = rotation_history[: step + 1]
            time_history = time_history[: step + 1]
            stress_history = stress_history[: step + 1]
            error_history = error_history[: step + 1]
            command_history = command_history[: step + 1]
            speed_history = speed_history[: step + 1]
            pulse_history = pulse_history[: step + 1]
            gate_history = gate_history[: step + 1]
            break

    progress = float(_path_progress_fraction(waypoints, progress_idx, state.position))
    avg_speed = float(np.mean(speed_history)) if len(speed_history) else 0.0
    max_stress = float(np.max(stress_history)) if len(stress_history) else 0.0
    max_track_error = float(np.max(error_history)) if len(error_history) else 0.0
    finish_error = _distance_to_finish(waypoints, state.position)
    stress_ratio = max_stress / max(drone.material.stress_limit, 1.0)
    target_stress_ratio = 0.965
    capped_stress_ratio = min(stress_ratio, 1.0)
    near_yield_reward = float(np.exp(-((capped_stress_ratio - target_stress_ratio) / 0.06) ** 2))
    stress_utilization_reward = 165.0 * capped_stress_ratio
    understress_penalty = 240.0 * max(0.0, 0.84 - stress_ratio) ** 1.85
    overstress_penalty = 1200.0 * max(0.0, stress_ratio - 1.0) ** 1.2
    compactness_penalty = (
        72.0 * drone.design.chassis_mass_target
        + 12.0 * drone.design.arm_length
        + 46.0 * (drone.design.body_r1 + drone.design.body_r2)
        + 34.0 * drone.design.body_r3
        + 46.0 * drone.design.arm_radius
        + 12.0 * drone.design.motor_radius
    )
    score = (
        465.0 * progress
        + 52.0 * avg_speed
        - 4.0 * max_track_error
        - overstress_penalty
        - 0.002 * energy
        + stress_utilization_reward
        + 260.0 * near_yield_reward
        + (180.0 if survived else -260.0)
        + (320.0 if reached_finish else 0.0)
        - 62.0 * finish_error
        - compactness_penalty
        - understress_penalty
    )

    summary = {
        "progress": progress,
        "avg_speed": avg_speed,
        "max_stress": max_stress,
        "stress_ratio": stress_ratio,
        "near_yield_reward": near_yield_reward,
        "stress_utilization_reward": stress_utilization_reward,
        "understress_penalty": understress_penalty,
        "overstress_penalty": overstress_penalty,
        "compactness_penalty": compactness_penalty,
        "max_track_error": max_track_error,
        "finish_error": finish_error,
        "reached_finish": reached_finish,
        "energy": energy,
    }
    state_history = {
        "position": position_history,
        "rotation": rotation_history,
        "time": time_history,
        "stress": stress_history,
        "track_error": error_history,
        "command": command_history,
        "speed": speed_history,
        "pulse": pulse_history,
        "gate": gate_history,
    }
    frame_history = {
        "body_active_mask": main_body_history,
        "integrity": main_integrity_history,
        "von_mises": main_stress_history,
        "fragments": fragment_history,
        "turrets": np.array([turret.position for turret in turrets], dtype=float),
        "turret_specs": [
            {
                "position": turret.position.copy(),
                "decay": turret.decay,
                "frequency": turret.frequency,
                "pulse_width": turret.pulse_width,
                "phase_offset": turret.phase_offset,
                "pressure_peak": turret.pressure_peak,
                "x_window": turret.x_window,
            }
            for turret in turrets
        ],
        "gates": [
            {
                "center": gate.center.copy(),
                "normal": gate.normal.copy(),
                "radius": gate.radius,
                "label": gate.label,
            }
            for gate in gates
        ],
    }

    return FlightResult(
        design_name=design_name,
        score=score,
        progress=progress,
        avg_speed=avg_speed,
        max_stress=max_stress,
        max_track_error=max_track_error,
        energy=energy,
        survived=survived,
        trajectory=position_history,
        waypoints=waypoints,
        vm_snapshots=vm_snapshots,
        snapshot_times=snapshot_times,
        snapshot_labels=snapshot_labels,
        snapshot_frame_indices=snapshot_frame_indices,
        state_history=state_history,
        frame_history=frame_history,
        summary=summary,
    )
