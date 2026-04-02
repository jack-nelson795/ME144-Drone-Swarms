from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from projects.project5.config import CONFIG
    from projects.project5.build_report_data import main as build_report_data_main
    from projects.project5.ga import run_case, summarize_top_designs
    from projects.project5.hashin_shtrikman import evaluate_design
else:
    from .config import CONFIG
    from .build_report_data import main as build_report_data_main
    from .ga import run_case, summarize_top_designs
    from .hashin_shtrikman import evaluate_design


RunnerCaseName = Literal["A", "B", "C"]


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def plot_convergence(case_label: str, generations, best, top10_mean) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(generations, best, linewidth=2.0, label="Best design cost")
    ax.loglog(generations, top10_mean, linewidth=2.0, label="Mean cost of top 10 designs")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")
    ax.set_title(f"Project 5 GA convergence: Case {case_label}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"case_{case_label.lower()}_convergence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def export_case_outputs(case_label: RunnerCaseName, seed: int) -> dict[str, Any]:
    print(f"Running Case {case_label}...")
    result = run_case(case_label, seed=seed, cfg=CONFIG, verbose=True, progress_every=250)
    generations = pd.Series(range(1, CONFIG.G + 1), name="generation")
    history_df = pd.DataFrame(
        {
            "generation": generations,
            "best_cost": result.history_best,
            "top10_mean_cost": result.history_top10_mean,
        }
    )
    history_path = DATA_DIR / f"case_{case_label.lower()}_history.csv"
    history_df.to_csv(history_path, index=False)

    top_design_rows = summarize_top_designs(result, top_n=4, cfg=CONFIG)
    top_design_df = pd.DataFrame(top_design_rows)
    table_path = DATA_DIR / f"case_{case_label.lower()}_top4.csv"
    top_design_df.round(4).to_csv(table_path, index=False)

    fig_path = plot_convergence(case_label, generations, result.history_best, result.history_top10_mean)
    best_eval = evaluate_design(result.population[0, :], CONFIG)
    print(f"Case {case_label} outputs saved to {history_path} and {table_path}")

    summary = {
        "case": case_label,
        "seed": seed,
        "final_best_cost": float(result.history_best[-1]),
        "final_top10_mean_cost": float(result.history_top10_mean[-1]),
        "best_design": [float(x) for x in result.population[0, :]],
        "best_effective_properties": {
            key: float(value) for key, value in best_eval["effective_properties"].items()
        },
        "best_concentration_factors": {
            key: float(value) for key, value in best_eval["concentration_factors"].items()
        },
        "history_csv": str(history_path.relative_to(PROJECT_DIR)),
        "top4_csv": str(table_path.relative_to(PROJECT_DIR)),
        "convergence_figure": str(fig_path.relative_to(PROJECT_DIR)),
    }

    return summary


def main() -> None:
    ensure_dirs()

    # Same seed for all cases so the first generation is initialized comparably.
    seed = CONFIG.seed
    summaries = {}
    for case_label in ("A", "B", "C"):
        summaries[case_label] = export_case_outputs(case_label, seed)

    summary_path = DATA_DIR / "project5_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("Base Project 5 outputs written to:")
    print(FIGURES_DIR)
    print(DATA_DIR)

    # Also generate the full audited report-data package and report-ready tables.
    print("Building audited report-data package...")
    build_report_data_main()
    print("All Project 5 outputs are complete.")


if __name__ == "__main__":
    main()
