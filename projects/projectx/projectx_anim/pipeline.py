from __future__ import annotations

import numpy as np

from me144_toolbox.optimization.ga import genetic_algorithm

from . import state
from .cost import swarm_control_cost
from .simulator import DroneControlSimulator
from .visualization import create_visualizations_and_reports


def main(*, fast: bool = False, create_viz: bool = True) -> None:
    """Run complete pipeline.

    Parameters
    ----------
    fast:
        If True, run a much smaller GA and avoid the expensive full-population
        full-simulation scoring pass. Intended for quick debugging.
    create_viz:
        If False, skip plotting/animation/report generation.
    """
    # Initialize globals in main process
    state.init_globals()

    print("\n" + "="*70)
    print("ME144/244 ProjectX: 3D Drone Swarm Incursion Simulation")
    print("(Control-Law Optimization - Emergent Swarm Behavior)")
    print("="*70)
    print(f"\nRandom Seed: {state.SEED}")

    # ========================================================================
    # STEP 1: Optimize drone control law weights using GA
    # ========================================================================

    print("\nStep 1: Optimizing control law weights...")

    # NOTE: Runtime is dominated by GA hyperparameters.
    # For quick debugging, fast mode reduces the GA workload significantly.
    S = 12 if fast else state.S
    P = 6 if fast else state.P
    K = 4 if fast else state.K
    G = 4 if fast else state.G
    if fast:
        print(f"  [FAST] Overriding GA params: S={S}, P={P}, K={K}, G={G}")

    Pi, Pi_min, Pi_avg, Lambda = genetic_algorithm(
        swarm_control_cost,  # Multiprocessing cost evaluation
        S=S,
        P=P,
        K=K,
        TOL=1e-2,
        G=G,
        dv=state.DV,
        lim=state.LIM,
        seed=state.SEED,
        mutation_rate=0.15,
        sigma_frac=0.10,
    )

    # Get best solution.
    # Full-population full-simulation scoring is very expensive; fast mode
    # skips it and uses the same hybrid scoring used during GA.
    if fast:
        costs_final = swarm_control_cost(Lambda, run_simulation=False).ravel()
        best_idx = int(np.argmin(costs_final))
        best_weights = Lambda[best_idx]
        print("  [FAST] Using hybrid scoring for final selection (no full-population full-sim pass)")
    else:
        costs_final = swarm_control_cost(Lambda, run_simulation=True).ravel()
        best_idx = int(np.argmin(costs_final))
        best_weights = Lambda[best_idx]

    print(f"  [OK] Optimization complete")
    print(f"    Best cost: {costs_final[best_idx]:.1f}")
    print(f"    Initial cost: {Pi_min[0]:.1f}")
    print(f"    Improvement: {Pi_min[0] / Pi_min[-1]:.2f}x")

    # Extract and clip control weights
    best_weights = np.clip(best_weights, 0.0, 1.0)
    best_weights = best_weights.reshape(state.N_DRONES, state.N_CONTROL_WEIGHTS)

    # Print learned weights (insight into what GA discovered)
    print(f"\n  Learned control weights (per drone):")
    print(f"    [w1=target, w2=obstacle, w3=separation, w4=cohesion]")
    for d in range(min(5, state.N_DRONES)):
        print(
            f"    Drone {d}: w1={best_weights[d,0]:.3f}, w2={best_weights[d,1]:.3f}, w3={best_weights[d,2]:.3f}, w4={best_weights[d,3]:.3f}"
        )
    if state.N_DRONES > 5:
        print(f"    ... ({state.N_DRONES - 5} more drones)")

    # ========================================================================
    # STEP 2: Simulate best control weights with collision detection
    # ========================================================================

    print("\nStep 2: Simulating optimized control weights...")
    print("  (Running until all targets visited or all drones dead - no step limit)")

    # Use smaller timestep and faster simulation for optimization
    simulator = DroneControlSimulator(best_weights, state.OBSTACLES, state.TARGET_ZONES, dt=0.3)
    sim_steps = 1200 if fast else 3000
    history = simulator.run(max_steps=sim_steps)  # Full simulation for validation

    positions = history['positions']
    alive = history['alive']
    targets_visited = history['targets_visited']

    n_alive_final = int(np.sum(alive[-1]))
    n_targets_final = int(np.sum(targets_visited[-1]))
    print(f"  [OK] Simulation complete")
    print(f"    Drones alive at end: {n_alive_final}/{state.N_DRONES}")
    print(f"    Drones lost: {state.N_DRONES - n_alive_final}")
    print(f"    Targets visited: {n_targets_final}/{state.N_TARGETS}")
    print(f"    Simulation time: {simulator.time:.1f}s")

    # ========================================================================
    # STEP 3: Visualization
    # ========================================================================
    if not create_viz:
        print("\nStep 3: Skipping visualizations (--no-viz)")
        return

    create_visualizations_and_reports(
        positions=positions,
        alive=alive,
        targets_visited=targets_visited,
        simulator_dt=simulator.dt,
        Pi_min=Pi_min,
        Pi_avg=Pi_avg,
        costs_final=costs_final,
        best_idx=best_idx,
        best_weights=best_weights,
    )
