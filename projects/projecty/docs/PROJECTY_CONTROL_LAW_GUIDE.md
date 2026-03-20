# ProjectY Control Law Guide

## Purpose

This document explains how Project Y currently steers the drone through the hostile course.

## Controller Style

The controller is not a full nonlinear optimal controller. It is a structured waypoint/gate-following controller with:

- lookahead path tracking
- desired force construction
- attitude alignment to the force vector
- mixer-based motor command generation
- damage-aware control authority degradation

## Control Pipeline

### 1. Path Target Selection

The sim first chooses a lookahead target on the course using the current progress index.

Near important course gates, the target can be overridden so the drone explicitly seeks the next unpassed gate rather than drifting near it.

This is especially important at the finish:

- if the drone overshoots the finish gate plane
- and the finish gate has not yet been passed
- the controller actively targets the gate center again instead of stalling nearby

### 2. Desired Translational Behavior

The controller constructs:

- a target velocity along the local path direction
- a desired acceleration from position error and velocity error
- a desired force equal to mass times desired acceleration plus gravity compensation

Desired speed depends on:

- thrust-to-weight margin
- active mass
- finish pull
- gate-seeking behavior

### 3. Desired Attitude

The controller chooses a desired rotation so the body thrust axis aligns with the desired world force.

Yaw is guided by the path heading.

This means tilt is an emergent behavior:

- larger lateral acceleration demands produce larger tilt
- the controller does not need a separate "tilt mode"

### 4. Attitude Error To Torque Command

Attitude error is computed from the current and desired rotations.

A proportional-plus-rate structure generates the torque command:

- rotation error term
- angular-rate damping term

When motors are damaged or detached, torque authority is reduced accordingly.

### 5. Mixer

The mixer converts:

- collective thrust demand
- body torque demand

into four motor commands.

The motor commands are then clipped by the per-motor availability:

- a healthy motor gets full authority
- a detached or mostly detached motor cluster contributes little or nothing

## Gate Completion Logic

Project Y now uses explicit visual/non-physical gates:

- one at 50% course progress
- one at 100% course progress

The run does not count as complete until both gates have been crossed.

## Failure Modes The Controller Must Handle

- pulse-induced translation disturbances
- pulse-induced rotation disturbances
- late-course overshoot
- reduced thrust after motor damage
- reduced yaw/moment authority after motor loss
- reduced structural mass after fragmentation

## Practical Interpretation

The current controller is best understood as:

- a reasonably capable mission-following controller
- with damage-aware authority degradation
- plus explicit gate-seeking logic

It is not yet a predictive adversarial controller, but it is well beyond a static path follower.
