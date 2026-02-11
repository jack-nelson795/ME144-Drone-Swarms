# projects/project1/run_ga_project1.py
"""
Project 1 — Genetic Algorithm (Problem 2)

python -m projects.project1.run_ga_project1

Runs the GA on Πa and Πb with the debug parameters from the prompt:
S = 50, P = 12, K = 12, lim = [-20, 20], dv = 1, TOL = 1e-6, G = 100
Nearest-neighbor breeding, no mutations, no inbreeding prevention.

Outputs:
- logs in projects/project1/output/logs/
- figures in projects/project1/output/figures/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from me144_toolbox.optimization.ga import genetic_algorithm
from me144_toolbox.objectives.project1 import Pi_a, Pi_b
from me144_toolbox.utils.plotting import plot_ga_best_mean


def cost_Pi_a(X: np.ndarray) -> np.ndarray:
    # X is (S, dv). For dv=1, take column 0.
    x = X[:, 0]
    return Pi_a(x).ravel()


def cost_Pi_b(X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    return Pi_b(x).ravel()


def main() -> None:
    proj_dir = Path(__file__).resolve().parent
    out_fig = proj_dir / "output" / "figures"
    out_log = proj_dir / "output" / "logs"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_log.mkdir(parents=True, exist_ok=True)

    # Prompt parameters (debug settings)
    S = 50
    P = 12
    K = 12
    dv = 1
    lim = np.array([[-20.0, 20.0]])
    TOL = 1e-6
    G = 100

    # No mutations, no inbreeding prevention
    mutation_rate = 0.0

    # Use a fixed seed so your plots are reproducible
    seed = 123

    # Run GA on Πa (mostly for debugging / sanity)
    Pi_a_hist, Pi_a_min, Pi_a_avg, Lambda_a = genetic_algorithm(
        cost_Pi_a,
        S=S, P=P, K=K, TOL=TOL, G=G, dv=dv, lim=lim,
        seed=seed,
        mutation_rate=mutation_rate,
    )

    # Run GA on Πb (the main one they care about in (b))
    Pi_b_hist, Pi_b_min, Pi_b_avg, Lambda_b = genetic_algorithm(
        cost_Pi_b,
        S=S, P=P, K=K, TOL=TOL, G=G, dv=dv, lim=lim,
        seed=seed,
        mutation_rate=mutation_rate,
    )

    # Save a simple log
    lines = []
    lines.append("Project 1 — GA Runs\n")
    lines.append(f"S={S}, P={P}, K={K}, dv={dv}, lim=[-20,20], TOL={TOL}, G={G}\n")
    lines.append("Nearest-neighbor breeding, no mutations\n\n")

    lines.append(f"[Πa] generations_run={len(Pi_a_min)} best_cost={Pi_a_min[-1]:.6e}\n")
    lines.append(f"[Πb] generations_run={len(Pi_b_min)} best_cost={Pi_b_min[-1]:.6e}\n")

    (out_log / "ga_runs.txt").write_text("".join(lines), encoding="utf-8")

    # Plot best and mean for Πb (as requested) on semilog-y (plotting handled by utility)
    plot_ga_best_mean(
        best=Pi_b_min,
        mean=Pi_b_avg,
        title=r"GA on $\Pi_b$: Best and Mean Cost per Generation",
        xlabel="Generation",
        ylabel=r"$\Pi_b$",
        save_path=out_fig / "ga_Pib_best_mean.png",
        semilogy=True,
    )


if __name__ == "__main__":
    main()
