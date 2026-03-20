from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class DroneDesign:
    body_r1: float = 0.102
    body_r2: float = 0.102
    body_r3: float = 0.027
    body_p1: float = 1.72
    body_p2: float = 1.72
    body_p3: float = 1.08
    arm_length: float = 0.392
    arm_radius: float = 0.015
    motor_radius: float = 0.04
    lightweight_hole: float = 0.064
    thrust_scale: float = 1.48
    motor_mass_each: float = 0.145
    chassis_mass_target: float = 0.34

    def clipped(self) -> "DroneDesign":
        return replace(
            self,
            body_r1=min(max(self.body_r1, 0.05), 0.14),
            body_r2=min(max(self.body_r2, 0.05), 0.14),
            body_r3=min(max(self.body_r3, 0.012), 0.045),
            body_p1=min(max(self.body_p1, 1.15), 2.8),
            body_p2=min(max(self.body_p2, 1.15), 2.8),
            body_p3=min(max(self.body_p3, 0.75), 1.8),
            arm_length=min(max(self.arm_length, 0.34), 0.48),
            arm_radius=min(max(self.arm_radius, 0.009), 0.028),
            motor_radius=min(max(self.motor_radius, 0.026), 0.052),
            lightweight_hole=min(max(self.lightweight_hole, 0.015), 0.105),
            thrust_scale=min(max(self.thrust_scale, 1.1), 2.4),
            motor_mass_each=min(max(self.motor_mass_each, 0.07), 0.24),
            chassis_mass_target=min(max(self.chassis_mass_target, 0.18), 0.68),
        )

    def mutate(self, rng, scale: float = 0.08) -> "DroneDesign":
        def bump(value: float, rel: float = 1.0) -> float:
            sigma = scale * rel * max(abs(value), 0.05)
            return float(value + rng.normal(0.0, sigma))

        return DroneDesign(
            body_r1=bump(self.body_r1),
            body_r2=bump(self.body_r2),
            body_r3=bump(self.body_r3, 0.6),
            body_p1=bump(self.body_p1, 0.5),
            body_p2=bump(self.body_p2, 0.5),
            body_p3=bump(self.body_p3, 0.5),
            arm_length=bump(self.arm_length),
            arm_radius=bump(self.arm_radius, 0.6),
            motor_radius=bump(self.motor_radius, 0.6),
            lightweight_hole=bump(self.lightweight_hole, 0.7),
            thrust_scale=bump(self.thrust_scale, 0.35),
            motor_mass_each=bump(self.motor_mass_each, 0.5),
            chassis_mass_target=bump(self.chassis_mass_target, 0.4),
        ).clipped()

    @property
    def nominal_volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.body_r1 * self.body_r2 * self.body_r3
