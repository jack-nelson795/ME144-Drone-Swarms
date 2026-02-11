# projects/project1/run_project1.py
"""
Project 1 — Newton's Method deliverables (Parts a–d)

python -m projects.project1.run_project1

Generates:
(a) Objectives Πa and Πb on [-20, 20]
(c) Gradients and Hessians on [-20, 20]
(d) Newton runs for x0 = 2*10^k, k in {-1,0,1} with TOL=1e-8, maxit=20
    and plots Π(hist)

Outputs saved to:
projects/project1/output/figures/
projects/project1/output/logs/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from me144_toolbox.objectives.project1 import (
    Pi_a, Pi_b,
    grad_Pi_a, grad_Pi_b,
    hess_Pi_a, hess_Pi_b
)
from me144_toolbox.optimization.newton import myNewton
from me144_toolbox.utils.plotting import plot_two_curves, plot_convergence_curves


def main() -> None:
    # Paths
    proj_dir = Path(__file__).resolve().parent
    out_fig = proj_dir / "output" / "figures"
    out_log = proj_dir / "output" / "logs"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_log.mkdir(parents=True, exist_ok=True)

    # Domain for plots
    x = np.linspace(-20.0, 20.0, 2001)

    # (a) Plot objectives on same axes
    Pia = Pi_a(x).ravel()
    Pib = Pi_b(x).ravel()
    plot_two_curves(
        x=x, y1=Pia, y2=Pib,
        label1=r"$\Pi_a(x)=x^2$",
        label2=r"$\Pi_b(x)=(x+\frac{\pi}{2}\sin x)^2$",
        title="Project 1: Objective Functions",
        xlabel="x",
        ylabel=r"$\Pi(x)$",
        save_path=out_fig / "objectives.png",
    )

    # (c1) Plot gradients
    dPia = grad_Pi_a(x).ravel()
    dPib = grad_Pi_b(x).ravel()
    plot_two_curves(
        x=x, y1=dPia, y2=dPib,
        label1=r"$d\Pi_a/dx$",
        label2=r"$d\Pi_b/dx$",
        title="Project 1: Gradients",
        xlabel="x",
        ylabel=r"$d\Pi/dx$",
        save_path=out_fig / "gradients.png",
    )

    # (c2) Plot Hessians
    ddPia = np.diag(hess_Pi_a(x)).ravel()
    ddPib = np.diag(hess_Pi_b(x)).ravel()
    plot_two_curves(
        x=x, y1=ddPia, y2=ddPib,
        label1=r"$d^2\Pi_a/dx^2$",
        label2=r"$d^2\Pi_b/dx^2$",
        title="Project 1: Hessians",
        xlabel="x",
        ylabel=r"$d^2\Pi/dx^2$",
        save_path=out_fig / "hessians.png",
    )

    # (d) Newton runs
    TOL = 1e-8
    maxit = 20
    x0_list = [2 * 10 ** k for k in (-1, 0, 1)]  # 0.2, 2, 20

    log_lines = []
    log_lines.append("Project 1 Newton Runs\n")
    log_lines.append(f"TOL = {TOL}, maxit = {maxit}\n")
    log_lines.append(f"x0 list = {x0_list}\n\n")

    curves_a = {}
    curves_b = {}

    for x0 in x0_list:
        # Πa
        sol_a, its_a, hist_a = myNewton(grad_Pi_a, hess_Pi_a, x0, TOL, maxit)
        Pia_hist = Pi_a(hist_a.T).ravel()
        iters_a = np.arange(Pia_hist.size)
        curves_a[f"x0={x0:g}"] = (iters_a, Pia_hist)

        log_lines.append(
            f"[Πa] x0={x0:g} -> sol={sol_a.ravel()[0]:.12g}, "
            f"its={its_a}, |grad|={np.linalg.norm(grad_Pi_a(sol_a)):.3e}\n"
        )

        # Πb
        sol_b, its_b, hist_b = myNewton(grad_Pi_b, hess_Pi_b, x0, TOL, maxit)
        Pib_hist = Pi_b(hist_b.T).ravel()
        iters_b = np.arange(Pib_hist.size)
        curves_b[f"x0={x0:g}"] = (iters_b, Pib_hist)

        log_lines.append(
            f"[Πb] x0={x0:g} -> sol={sol_b.ravel()[0]:.12g}, "
            f"its={its_b}, |grad|={np.linalg.norm(grad_Pi_b(sol_b)):.3e}\n"
        )

    plot_convergence_curves(
        curves=curves_a,
        title=r"Newton Convergence: $\Pi_a(\mathrm{hist})$",
        xlabel="Iteration",
        ylabel=r"$\Pi_a$",
        save_path=out_fig / "newton_Pia_convergence.png",
        semilogy=True,
    )

    plot_convergence_curves(
        curves=curves_b,
        title=r"Newton Convergence: $\Pi_b(\mathrm{hist})$",
        xlabel="Iteration",
        ylabel=r"$\Pi_b$",
        save_path=out_fig / "newton_Pib_convergence.png",
        semilogy=True,
    )

    # FIX: explicitly use UTF-8 so Greek symbols (Π) work on Windows
    (out_log / "newton_runs.txt").write_text(
        "".join(log_lines),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
