# Project 1 Architecture

## Modules

- me144_toolbox/objectives/project1.py
  - Pi_a, Pi_b
  - grad and Hessian functions
- me144_toolbox/optimization/newton.py
  - myNewton implementation
- me144_toolbox/optimization/ga.py
  - Standard GA implementation
- me144_toolbox/optimization/ga_phi_psi.py
  - Phi/Psi crossover GA
- me144_toolbox/utils/plotting.py
  - Plot helpers for objectives and convergence

## Script Roles

- run_project1.py
  - Generates objectives, gradients, and Hessians
  - Runs Newton iterations from multiple initial guesses
  - Saves plots and logs

- run_ga_project1.py
  - Runs the standard GA on Pi_a and Pi_b
  - Saves logs and best/mean plots

- run_ga_phi_psi_project1.py
  - Runs the phi/psi GA on Pi_a and Pi_b
  - Saves logs and best/mean plots

## Output Conventions

All outputs are written to projects/project1/output/figures and projects/project1/output/logs.
