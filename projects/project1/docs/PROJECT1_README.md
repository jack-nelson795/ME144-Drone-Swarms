# Project 1: Optimization Methods (Complete)

Project 1 implements Newton's method and Genetic Algorithms (standard and Zohdi phi/psi crossover) for the two objective functions used in the assignment.

## Files

- run_project1.py: Newton's method tasks (objectives, gradients, Hessians, convergence plots)
- run_ga_project1.py: Standard GA for Pi_a and Pi_b
- run_ga_phi_psi_project1.py: Phi/Psi GA for Pi_a and Pi_b

## How to Run

From the repo root:

  python -m projects.project1.run_project1
  python -m projects.project1.run_ga_project1
  python -m projects.project1.run_ga_phi_psi_project1

## Outputs

Generated artifacts are saved to:
- projects/project1/output/figures/
- projects/project1/output/logs/

## Notes

- All runs are seeded for reproducibility.
- The objective functions and derivatives live in me144_toolbox/objectives/project1.py.
