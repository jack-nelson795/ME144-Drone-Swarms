# README_PROJECT7_OUTPUTS

This folder contains the saved outputs for ME144/244 Project 7.

## Main files
- `fixed_lam_verification.txt/.json/.csv`: fixed-design verification numbers. Use these in the debugging/validation discussion.
- `ga_results.pkl`: saved GA results dictionary from the completed simulation.
- `ga_best_design_summary.json/.csv` and `ga_best_design_vector.txt/.csv`: final optimized design values and metrics.
- `ga_best_cost_vs_generation.png`: best cost per generation.
- `ga_average_cost_vs_generation.png`: average cost per generation.
- `ga_parents_average_cost_vs_generation.png`: parents' average cost per generation.
- `ga_convergence_combined.png`: combined convergence figure for the GA section.
- `optimized_design_screenshot.png`: representative frame from the optimized simulation for the figure requested in the assignment.
- `default_sensitivity_surface.png` and `default_sensitivity_heatmap.png`: sensitivity plots using the default baseline design.
- `optimized_sensitivity_surface.png` and `optimized_sensitivity_heatmap.png`: sensitivity plots using the GA-best design.
- `ga_terminal_output.txt`: GA progress printout copied from the terminal run.
- `terminal_summary.txt`: overall runner summary copied from the terminal run.
- `writeup_inputs.md`: short ready-to-use list of values, figure filenames, and observations for the report.

## Reproducibility
- Random seed used for reproducible runs: `144`.
- Re-run the workflow with `python run_project7.py` from `projects/project7`.

## Notes / limitations
- The PDF bounds and notebook debug cell use a flow-rate entry near 8.21 m^3/s. Using 8.21e-6 releases zero tracked droplets with the provided super-particle scaling, so the assignment-consistent benchmark uses 8.21.
- A static screenshot was saved instead of a movie file. This keeps the workflow dependable inside the current environment.
- Final optimized best cost: 0.187418.
- Default sensitivity sampled cost range: 0.780000.
- Optimized sensitivity sampled cost range: 0.841304.