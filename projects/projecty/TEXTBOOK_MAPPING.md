# Textbook Mapping

This project follows Chapter 4 of the Zohdi drone text as the primary model source.

## Geometry

- Equation 4.1: generalized ellipsoidal envelope for voxel activation
- Equation 4.2: union of overlapping subdomains for modular chassis construction

## Stress model

- Equation 4.3: linear momentum balance on the voxel grid
- Equation 4.4: linear elastic constitutive law
- Equation 4.5: infinitesimal-deformation structural dynamics
- Equations 4.6-4.12: trapezoidal `phi`-scheme logic for time stepping
- Equations 4.13-4.24: central-difference voxel derivatives
- Equations 4.105-4.109: moving-frame stress evaluation after rigid motion

## Flight model

- Equations 4.81-4.98: DEM-cluster translational and rotational dynamics
- Equations 4.99-4.101: hover, pitch, roll, and yaw force/torque balance

## Hostile environment

- Equation 4.102: exponential pressure-wave decay used for turret attacks
