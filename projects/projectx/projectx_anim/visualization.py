from __future__ import annotations

from typing import Any

import numpy as np

from . import config
from . import state


def create_visualizations_and_reports(
    *,
    positions: np.ndarray,
    alive: np.ndarray,
    targets_visited: np.ndarray,
    simulator_dt: float,
    Pi_min: Any,
    Pi_avg: Any,
    costs_final: np.ndarray,
    best_idx: int,
    best_weights: np.ndarray,
) -> None:
    print("\nStep 3: Creating visualizations...")

    # Import animation helpers only now (matplotlib is slow)
    from me144_toolbox.utils.animation_3d import animate_swarm_3d, plot_swarm_3d_static

    n_alive_final = int(np.sum(alive[-1]))
    n_targets_final = int(np.sum(targets_visited[-1]))

    # Static 3D plot
    print("  Creating static 3D plot...")
    final_pos = positions[-1]
    final_alive = alive[-1]

    plot_swarm_3d_static(
        final_pos,
        np.array(state.TARGET_ZONES),
        final_alive,
        obstacles=state.OBSTACLES,
        bounds=config.BOUNDS_3D,
        base_pos=config.BASE_POS,
        title=f"Final Positions (Emergent Control)\n(Alive: {n_alive_final}/{state.N_DRONES}, Targets: {n_targets_final}/{state.N_TARGETS})",
    )

    # Animation
    print("  Creating animation...")

    # Use half the frames for smooth playback
    stride = int(getattr(config, 'ANIMATION_STRIDE', 1))
    if stride < 1:
        stride = 1
    positions_sampled = positions[::stride]
    alive_sampled = alive[::stride]
    targets_visited_sampled = targets_visited[::stride]

    print(f"  Animation will have {len(positions_sampled)} frames")

    try:
        animate_swarm_3d(
            positions_sampled,
            alive_sampled,
            np.array(state.TARGET_ZONES),
            obstacles=state.OBSTACLES,
            bounds=config.BOUNDS_3D,
            base_pos=config.BASE_POS,
            save_path=str(state.FIG_DIR / 'swarm_control_emergent.gif'),
            interval=10000000000,  # Frame interval (ms)
            targets_visited=targets_visited_sampled,
        )
        print("  [OK] Animation created: swarm_control_emergent.gif")
    except KeyboardInterrupt:
        print("  [NOTE] Animation creation interrupted")
    except Exception as e:
        print(f"  [NOTE] Animation note: {str(e)[:60]}...")

    # Convergence and performance plots (import matplotlib only now)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # GA convergence
    Pi_min = np.asarray(Pi_min, dtype=float).ravel()
    Pi_avg = np.asarray(Pi_avg, dtype=float).ravel()
    # Enforce monotonic best for visualization
    if Pi_min.size > 0:
        Pi_min = np.minimum.accumulate(Pi_min)
    gen = np.arange(Pi_min.size)

    finite_mask = np.isfinite(Pi_min) & np.isfinite(Pi_avg)
    if Pi_min.size == 0 or not np.any(finite_mask):
        ax1.text(0.5, 0.5, 'No valid GA history to plot', ha='center', va='center')
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Cost', fontsize=11)
        ax1.set_title('GA Control Weight Optimization', fontsize=12)
        ax1.grid(True, alpha=0.3)
        print(f"  [WARN] GA history invalid: len(Pi_min)={Pi_min.size}")
    else:
        Pi_min_plot = Pi_min.copy()
        Pi_avg_plot = Pi_avg.copy()
        Pi_min_plot[~finite_mask] = np.nan
        Pi_avg_plot[~finite_mask] = np.nan

        min_val = min(np.nanmin(Pi_min_plot), np.nanmin(Pi_avg_plot))
        max_val = max(np.nanmax(Pi_min_plot), np.nanmax(Pi_avg_plot))
        print(f"  [GA] Pi_min range: {min_val:.3g} to {max_val:.3g}")

        use_log = min_val > 0.0
        if use_log:
            ax1.semilogy(gen, Pi_min_plot, 'g-', linewidth=2.0, marker='o', markersize=3, label='Best Cost')
            ax1.semilogy(gen, Pi_avg_plot, 'r--', linewidth=1.5, marker='o', markersize=3, label='Mean Cost')
            ax1.set_ylabel('Cost (log scale)', fontsize=11)
        else:
            # Shift up if nonpositive to keep a visible plot
            shift = 1.0 - min_val if min_val <= 0.0 else 0.0
            y_min = min_val + shift
            y_max = max_val + shift
            pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            ax1.plot(gen, Pi_min_plot + shift, 'g-', linewidth=2.0, marker='o', markersize=3, label='Best Cost')
            ax1.plot(gen, Pi_avg_plot + shift, 'r--', linewidth=1.5, marker='o', markersize=3, label='Mean Cost')
            ax1.set_ylim(y_min - pad, y_max + pad)
            ax1.set_ylabel('Cost', fontsize=11)

        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_title('GA Control Weight Optimization', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Drone survival
    n_alive_per_step = np.sum(alive, axis=1)
    time_steps = np.arange(len(n_alive_per_step)) * simulator_dt
    ax2.plot(time_steps, n_alive_per_step, 'b-', linewidth=2.5)
    ax2.fill_between(time_steps, n_alive_per_step, alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Alive Drones', fontsize=11)
    ax2.set_title(f'Swarm Survival ({n_alive_final} survived)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, state.N_DRONES + 1)

    # Targets scored
    n_targets_per_step = np.sum(targets_visited, axis=1)
    ax3.plot(time_steps, n_targets_per_step, 'g-', linewidth=2.5)
    ax3.fill_between(time_steps, n_targets_per_step, alpha=0.3, color='green')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Targets Visited', fontsize=11)
    ax3.set_title(f'Target Acquisition ({n_targets_final}/{state.N_TARGETS} scored)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, state.N_TARGETS + 1)

    plt.tight_layout()
    plt.savefig(state.FIG_DIR / 'control_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved analysis plot")
    plt.close()

    # Log results
    log_file = state.OUTPUT_DIR / 'logs' / 'control_swarm_incursion.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Drone Swarm Incursion - Control Law Optimization\n")
        f.write("="*70 + "\n\n")
        f.write(f"Methodology: GA optimizes 4 control weights per drone\n")
        f.write(f"  w1 = attraction to closest target\n")
        f.write(f"  w2 = repulsion from obstacles\n")
        f.write(f"  w3 = separation from drones\n")
        f.write(f"  w4 = cohesion with swarm\n\n")
        f.write(f"Drones: {state.N_DRONES}\n")
        f.write(f"Target zones: {state.N_TARGETS}\n")
        f.write(f"Obstacles: {len(state.OBSTACLES)}\n")
        f.write(f"GA generations: {state.G}\n")
        f.write(f"Design variables: {state.DV} (control weights)\n\n")
        f.write(f"Best control cost: {costs_final[best_idx]:.1f}\n")
        f.write(f"GA improvement: {Pi_min[0] / Pi_min[-1]:.2f}x\n\n")
        f.write(f"Drones alive: {n_alive_final}/{state.N_DRONES}\n")
        f.write(f"Drones lost: {state.N_DRONES - n_alive_final}\n")
        f.write(f"Targets visited: {n_targets_final}/{state.N_TARGETS}\n")
        f.write(f"Simulation time: {time_steps[-1]:.1f}s\n\n")
        f.write("Learned Control Weights:\n")
        for d in range(state.N_DRONES):
            f.write(f"  Drone {d:2d}: w1={best_weights[d,0]:.3f}, w2={best_weights[d,1]:.3f}, w3={best_weights[d,2]:.3f}, w4={best_weights[d,3]:.3f}\n")

    print(f"  [OK] Logged results")

    print("\n" + "="*70)
    print("[OK] Simulation Complete!")
    print("="*70)
