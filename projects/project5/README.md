# Project 5

This project implements the Hashin-Shtrikman composite optimization workflow for the Project 5 brief:

- exact Hashin-Shtrikman property bounds and effective-property calculations
- three GA search cases for the design-variable studies
- convergence plotting, top-design tables, and summary exports
- report-ready data packaging and audit files for the final writeup

## Run

From the repository root:

```powershell
python -m projects.project5.run_project5
```

## Outputs

The main pipeline writes figures, raw data, tables, and report-support files under `projects/project5/output/`:

- `output/figures/case_a_convergence.png`
- `output/figures/case_b_convergence.png`
- `output/figures/case_c_convergence.png`
- `output/data/case_a_history.csv`
- `output/data/case_b_history.csv`
- `output/data/case_c_history.csv`
- `output/data/project5_summary.json`
- `output/tables/case_a_top4_report_ready.csv`
- `output/tables/case_b_top4_report_ready.csv`
- `output/tables/case_c_top4_report_ready.csv`
- `output/report_data/report_summary.json`
- `output/report_data/report_data_manifest.json`

## Project Files

- `run_project5.py` is the one-stop entry point for the full workflow.
- `hashin_shtrikman.py` contains the property, bound, and cost-function implementation.
- `ga.py` contains the genetic algorithm used for Cases A, B, and C.
- `export_report_data.py` and `build_report_data.py` generate the report-ready and audited support files.
- `Hashin_Shtrikman_provided.ipynb` is the notebook version of the project workspace.

## Optional Regeneration

If you only want to rebuild the report-support layers:

```powershell
python -m projects.project5.export_report_data
python -m projects.project5.build_report_data
```
