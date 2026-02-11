# projects/project1/run_ga_phi_psi_project1.py
"""
Project 1 — Genetic Algorithm using Zohdi phi/psi crossover
python -m projects.project1.run_ga_phi_psi_project1

Runs the phi/psi GA on Πa and Πb with the debugging parameters:
S = 50, P = 12, K = 12, lim = [-20, 20], dv = 1, TOL = 1e-6, G = 100
Nearest-neighbor mating, phi/psi component-wise convex crossover, no mutation.

Outputs:
- logs in projects/project1/output/logs/
- figures in projects/project1/output/figures/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from me144_toolbox.optimization.ga_phi_psi import genetic_algorithm_phi_psi
from me144_toolbox.objectives.project1 import Pi_a, Pi_b
from me144_toolbox.utils.plotting import plot_ga_best_mean


def cost_Pi_a(X: np.ndarray) -> np.ndarray:
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

    # Debug parameters from Project 1 prompt
    S = 50
    P = 12
    K = 12
    dv = 1
    lim = np.array([[-20.0, 20.0]])
    TOL = 1e-6
    G = 100

    seed = 123  # reproducibility

    # Run phi/psi GA on Πa (debug / sanity)
    Pi_a_hist, Pi_a_min, Pi_a_avg, Lambda_a = genetic_algorithm_phi_psi(
        cost_Pi_a,
        S=S, P=P, K=K, TOL=TOL, G=G, dv=dv, lim=lim,
        seed=seed,
    )

    # Run phi/psi GA on Πb (main)
    Pi_b_hist, Pi_b_min, Pi_b_avg, Lambda_b = genetic_algorithm_phi_psi(
        cost_Pi_b,
        S=S, P=P, K=K, TOL=TOL, G=G, dv=dv, lim=lim,
        seed=seed,
    )

    # Log
    lines = []
    lines.append("Project 1 — GA (phi/psi) Runs\n")
    lines.append(f"S={S}, P={P}, K={K}, dv={dv}, lim=[-20,20], TOL={TOL}, G={G}\n")
    lines.append("Nearest-neighbor mating, phi/psi component-wise convex crossover, no mutation\n\n")
    lines.append(f"[Πa] generations_run={len(Pi_a_min)} best_cost={Pi_a_min[-1]:.6e}\n")
    lines.append(f"[Πb] generations_run={len(Pi_b_min)} best_cost={Pi_b_min[-1]:.6e}\n")

    (out_log / "ga_phi_psi_runs.txt").write_text("".join(lines), encoding="utf-8")

    # Plot best and mean for Πb (as requested), semilog-y
    plot_ga_best_mean(
        best=Pi_b_min,
        mean=Pi_b_avg,
        title=r"GA (phi/psi) on $\Pi_b$: Best and Mean Cost per Generation",
        xlabel="Generation",
        ylabel=r"$\Pi_b$",
        save_path=out_fig / "ga_phi_psi_Pib_best_mean.png",
        semilogy=True,
    )


if __name__ == "__main__":
    main()
