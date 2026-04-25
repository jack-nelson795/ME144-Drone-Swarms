# Project 7

This project implements an aerial dispersal optimization and sensitivity-analysis workflow:

- fixed-vector verification against the assignment scaffold
- genetic-algorithm search over the 12-parameter lambda design vector
- convergence plotting and best-design summary export
- optimized-design visualization plus baseline-versus-optimized sensitivity studies

## Run

From the `projects/project7` folder:

```powershell
python run_project7.py
```

This generates the parameter files, runs the verification and GA workflow, and saves the summary figures and data products.

## Outputs

The pipeline writes results to `projects/project7/project7_outputs/`:

- `fixed_lam_verification.txt`
- `fixed_lam_verification.json`
- `ga_best_design_summary.json`
- `ga_best_design_vector.txt`
- `ga_best_cost_vs_generation.png`
- `ga_average_cost_vs_generation.png`
- `ga_parents_average_cost_vs_generation.png`
- `ga_convergence_combined.png`
- `optimized_design_screenshot.png`
- `default_sensitivity_surface.png`
- `default_sensitivity_heatmap.png`
- `optimized_sensitivity_surface.png`
- `optimized_sensitivity_heatmap.png`
- `terminal_summary.txt`
- `writeup_inputs.md`

## Project Files

- `run_project7.py` is the main end-to-end entry point for the project.
- `simulation.py` contains the active simulation used by the pipeline.
- `student_simulation.py` preserves the student-facing simulation version before syncing.
- `ga_class.py` and `geneticalgorithm.py` contain the GA implementation used for the search.
- `aerial_sensitivity_analysis.py` generates the sensitivity sweeps and summary JSON files.
- `animation.py` supports the saved optimized-design visualization outputs.
- `main.ipynb` is the notebook workspace for the project.

## Notes

- The random seed used for the saved results is `144`.
- The best recorded design in the saved outputs reaches a cost of about `0.1874` with hit fraction about `0.8043`.
- The output folder also includes `README_PROJECT7_OUTPUTS.md` with a more detailed breakdown of the generated artifacts.
