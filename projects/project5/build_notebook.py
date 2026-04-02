from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "Hashin-Shtrikman_INCOMPLETE.ipynb"


def build_notebook() -> Path:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# ME144/244 Project 5: Optimization of Advanced Materials for Drones\n"
            "\n"
            "This notebook completes the provided Hashin-Shtrikman starter by implementing the exact equations, "
            "GA cases, plots, and top-design tables required in the brief."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "\n"
            "from projects.project5.config import CONFIG\n"
            "from projects.project5.hashin_shtrikman import evaluate_design\n"
            "from projects.project5.run_project5 import main as run_project5_main\n"
            "\n"
            "project_dir = Path('projects/project5') if Path('projects/project5').exists() else Path('.')\n"
            "summary_path = project_dir / 'output' / 'data' / 'project5_summary.json'\n"
            "if not summary_path.exists():\n"
            "    run_project5_main()\n"
            "summary = json.loads(summary_path.read_text(encoding='utf-8'))\n"
            "project_dir.resolve()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Step 0: Project Parameters"))
    cells.append(
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {'parameter': 'P', 'value': CONFIG.P, 'units': '-', 'description': 'number of parents'},\n"
            "    {'parameter': 'K', 'value': CONFIG.K, 'units': '-', 'description': 'number of offspring'},\n"
            "    {'parameter': 'G', 'value': CONFIG.G, 'units': '-', 'description': 'max generations'},\n"
            "    {'parameter': 'S', 'value': CONFIG.S, 'units': '-', 'description': 'population size'},\n"
            "    {'parameter': 'k1', 'value': CONFIG.k1, 'units': 'Pa', 'description': 'phase 1 bulk modulus'},\n"
            "    {'parameter': 'mu1', 'value': CONFIG.mu1, 'units': 'Pa', 'description': 'phase 1 shear modulus'},\n"
            "    {'parameter': 'sig1E', 'value': CONFIG.sig1E, 'units': 'S/m', 'description': 'phase 1 electrical conductivity'},\n"
            "    {'parameter': 'K1', 'value': CONFIG.K1, 'units': 'W/m/K', 'description': 'phase 1 thermal conductivity'},\n"
            "    {'parameter': 'k_effD', 'value': CONFIG.k_effD, 'units': 'Pa', 'description': 'desired effective bulk modulus'},\n"
            "    {'parameter': 'mu_effD', 'value': CONFIG.mu_effD, 'units': 'Pa', 'description': 'desired effective shear modulus'},\n"
            "    {'parameter': 'sigE_effD', 'value': CONFIG.sigE_effD, 'units': 'S/m', 'description': 'desired effective electrical conductivity'},\n"
            "    {'parameter': 'K_effD', 'value': CONFIG.K_effD, 'units': 'W/m/K', 'description': 'desired effective thermal conductivity'},\n"
            "    {'parameter': 'gamma', 'value': CONFIG.gamma, 'units': '-', 'description': 'Hashin-Shtrikman averaging constant'}\n"
            "])"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Steps 1-6: Exact Property, Concentration, and Cost Evaluation\n"
            "\n"
            "The full implementation now lives in `projects/project5/hashin_shtrikman.py`. "
            "The example below evaluates the best Case C design and exposes the effective properties, "
            "concentration factors, and cost decomposition."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "best_case_c = summary['C']['best_design']\n"
            "case_c_eval = evaluate_design(best_case_c)\n"
            "pd.DataFrame({\n"
            "    'effective_property': case_c_eval['effective_properties'].keys(),\n"
            "    'value': case_c_eval['effective_properties'].values(),\n"
            "})"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "pd.DataFrame({\n"
            "    'concentration_factor': case_c_eval['concentration_factors'].keys(),\n"
            "    'value': case_c_eval['concentration_factors'].values(),\n"
            "})"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "pd.DataFrame({\n"
            "    'cost_component': case_c_eval['cost_terms'].keys(),\n"
            "    'value': case_c_eval['cost_terms'].values(),\n"
            "})"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Steps 7-9: GA Cases A, B, and C\n"
            "\n"
            "The full GA logic is implemented in `projects/project5/ga.py` and called through `run_project5.py`. "
            "The saved results generated from the 5000-generation runs are shown below."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "for case in ('A', 'B', 'C'):\n"
            "    print(case, summary[case]['final_best_cost'], summary[case]['final_top10_mean_cost'])"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(project_dir / 'output' / 'figures' / 'case_a_convergence.png')))\n"
            "display(Image(filename=str(project_dir / 'output' / 'figures' / 'case_b_convergence.png')))\n"
            "display(Image(filename=str(project_dir / 'output' / 'figures' / 'case_c_convergence.png')))"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Question 3: Top 4 Designs"))
    cells.append(
        nbf.v4.new_code_cell(
            "for case in ('a', 'b', 'c'):\n"
            "    print(f'Case {case.upper()}')\n"
            "    display(pd.read_csv(project_dir / 'output' / 'data' / f'case_{case}_top4.csv').round(4))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Reproduction\n"
            "\n"
            "To rerun the full project from scratch, execute:\n"
            "\n"
            "```python\n"
            "run_project5_main()\n"
            "```"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    path = build_notebook()
    print(f"Wrote {path}")
