# Project 5

This folder contains a complete Project 5 solution for the Hashin-Shtrikman composite optimization brief.

Main files:

- `Hashin-Shtrikman_INCOMPLETE.ipynb`: completed notebook based on the provided GSI starter notebook name
- `run_project5.py`: runs the three GA cases, generates convergence histories, plots, and top-4 tables
- `hashin_shtrikman.py`: exact property-bound, effective-property, concentration-factor, and cost-function implementation from the PDF
- `ga.py`: genetic algorithm implementation for Cases A, B, and C
- `export_report_data.py`: generates a report-ready JSON package and plain-text summary for drafting the final writeup elsewhere
- `build_report_data.py`: generates the audited report-data package requested for the final submission support workflow
- `output/figures/`: saved convergence plots
- `output/data/`: saved histories, top-4 tables, and summary JSON
- `output/tables/`: full-precision and report-ready tables
- `output/report_data/`: audits, summaries, cost breakdowns, and technical notes

How to regenerate everything:

```powershell
python -m projects.project5.run_project5
```

`run_project5.py` is now the main one-stop entry point: it generates the base GA outputs, figures, report-ready tables, and the audited files under `output/report_data/`.

Optional:

```powershell
python -m projects.project5.export_report_data
python -m projects.project5.build_report_data
```

Use those only if you want to regenerate the report-data layers separately.

The most useful handoff files for writing the final report are:

- `output/data/report_ready_package.json`
- `output/data/report_ready_summary.txt`

These bundle the numerical results, top designs, comparison notes, and file references to aid in writing the report.
