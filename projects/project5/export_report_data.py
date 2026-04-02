from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CONFIG
from .hashin_shtrikman import evaluate_design


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "output" / "data"
FIGURES_DIR = PROJECT_DIR / "output" / "figures"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _case_metrics(history: pd.DataFrame) -> dict:
    best = history["best_cost"]
    top10 = history["top10_mean_cost"]
    final_best = float(best.iloc[-1])
    final_top10 = float(top10.iloc[-1])
    first_within_5pct = next((int(g) for g, val in zip(history["generation"], best) if val <= 1.05 * final_best), None)
    first_within_1pct = next((int(g) for g, val in zip(history["generation"], best) if val <= 1.01 * final_best), None)
    return {
        "initial_best_cost": float(best.iloc[0]),
        "initial_top10_mean_cost": float(top10.iloc[0]),
        "final_best_cost": final_best,
        "final_top10_mean_cost": final_top10,
        "final_gap_top10_minus_best": final_top10 - final_best,
        "generation_first_within_5pct_of_final_best": first_within_5pct,
        "generation_first_within_1pct_of_final_best": first_within_1pct,
    }


def build_report_package() -> dict:
    summary = _load_json(DATA_DIR / "project5_summary.json")

    package: dict = {
        "project": "ME144/244 Spring 2026 Project 5",
        "assignment_focus": "Optimization of Advanced Materials for Drones",
        "report_purpose": "Structured results package for drafting a writeup/report in another tool.",
        "method_summary": {
            "design_string": ["kappa_2", "mu_2", "sigma_c_2", "IK_2", "v_2"],
            "effective_property_method": "Hashin-Shtrikman bounds with gamma = 0.5 average of lower and upper bounds",
            "electrical_bounds_equation": 5,
            "thermal_bounds_equation": 6,
            "bulk_bounds_equation": 3,
            "shear_bounds_equation": 4,
            "effective_property_equation": 2,
            "concentration_factor_equations": [23, 24, 25, 26, 27, 28, 9, 11],
            "unilateral_weight_equations": [34, 35, 37, 38, 29, 30, 31, 32],
            "cost_equations": {
                "electrical": 33,
                "thermal": 36,
                "mechanical": "mechanical objective on page 6 of the brief",
                "total": 39,
            },
            "ga_cases": {
                "A": "Keep 10 parents, produce 10 offspring, generate 180 new random strings",
                "B": "Do not keep 10 parents, produce 10 offspring, generate 190 new random strings",
                "C": "Keep 10 parents, produce 10 offspring with mutation phi in [-1/2, 3/2], generate 180 new random strings",
            },
        },
        "project_parameters": {
            "P": CONFIG.P,
            "K": CONFIG.K,
            "G": CONFIG.G,
            "S": CONFIG.S,
            "W1": CONFIG.W1,
            "W2": CONFIG.W2,
            "W3": CONFIG.W3,
            "w1": CONFIG.w1,
            "wj": CONFIG.wj,
            "gamma": CONFIG.gamma,
            "TOL_k": CONFIG.TOL_k,
            "TOL_mu": CONFIG.TOL_mu,
            "TOL_K": CONFIG.TOL_K,
            "TOL_sig": CONFIG.TOL_sig,
        },
        "cases": {},
        "comparison_points": {},
        "files_to_reference": {
            "case_a_plot": str((FIGURES_DIR / "case_a_convergence.png").relative_to(PROJECT_DIR)),
            "case_b_plot": str((FIGURES_DIR / "case_b_convergence.png").relative_to(PROJECT_DIR)),
            "case_c_plot": str((FIGURES_DIR / "case_c_convergence.png").relative_to(PROJECT_DIR)),
        },
    }

    targets = {
        "k_eff": CONFIG.k_effD,
        "mu_eff": CONFIG.mu_effD,
        "sigE_eff": CONFIG.sigE_effD,
        "K_eff": CONFIG.K_effD,
    }

    for case in ("A", "B", "C"):
        history = _load_csv(DATA_DIR / f"case_{case.lower()}_history.csv")
        top4 = _load_csv(DATA_DIR / f"case_{case.lower()}_top4.csv")
        best_design = summary[case]["best_design"]
        best_eval = evaluate_design(best_design)
        props = best_eval["effective_properties"]
        percent_errors = {
            key: 100.0 * (float(props[key]) - target) / target for key, target in targets.items()
        }
        package["cases"][case] = {
            "summary": summary[case],
            "history_metrics": _case_metrics(history),
            "top4_designs_table": top4.round(4).to_dict(orient="records"),
            "best_design_full_evaluation": {
                "effective_properties": {k: float(v) for k, v in best_eval["effective_properties"].items()},
                "concentration_factors": {k: float(v) for k, v in best_eval["concentration_factors"].items()},
                "cost_terms": {k: float(v) for k, v in best_eval["cost_terms"].items()},
                "active_unilateral_weights": {k: float(v) for k, v in best_eval["weights"].items()},
                "percent_error_vs_targets": percent_errors,
            },
            "discussion_notes": [],
        }

    package["cases"]["A"]["discussion_notes"] = [
        "Case A uses elitism by retaining the top 10 parents each generation.",
        "The best-cost curve and top-10-mean curve become identical at the end of the run, indicating collapse toward one dominant elite design.",
        "This case converged quickly and reached nearly the same minimum cost as Case C.",
    ]
    package["cases"]["B"]["discussion_notes"] = [
        "Case B discards the parents each generation, so good designs only persist through offspring and random reseeding.",
        "This preserves more diversity and leaves a nonzero final gap between the best design and the top-10 average.",
        "Case B converged more slowly and finished at a slightly worse final best cost than Cases A and C.",
    ]
    package["cases"]["C"]["discussion_notes"] = [
        "Case C keeps the parents like Case A but allows mutation-style offspring weights phi in [-1/2, 3/2].",
        "The mutation step slightly improved the best final cost without preventing convergence.",
        "In this run, Case C was the best-performing GA variant overall.",
    ]

    package["comparison_points"] = {
        "best_case_by_final_cost": min(("A", "B", "C"), key=lambda c: package["cases"][c]["history_metrics"]["final_best_cost"]),
        "case_a_vs_case_c": "Cases A and C are effectively tied, but Case C achieved the lowest final best cost by a very small margin.",
        "case_b_behavior": "Case B remained more diverse but paid for that diversity with slower convergence and a slightly worse optimum.",
        "dominant_cost_behavior": "Electrical matching is comparatively good in all best designs, while thermal and mechanical penalties remain active and dominate the total cost.",
    }

    return package


def build_text_summary(package: dict) -> str:
    lines: list[str] = []
    lines.append("ME144/244 Spring 2026 Project 5")
    lines.append("Report-ready summary package")
    lines.append("")
    lines.append("Method used")
    lines.append("- Hashin-Shtrikman bounds with gamma = 0.5 to compute effective properties.")
    lines.append("- Electrical, thermal, bulk, and shear bounds used exactly from Eqs. 5, 6, 3, and 4.")
    lines.append("- Concentration factors used exactly from Eqs. 23, 24, 25, 26, 27, 28, 9, and 11.")
    lines.append("- Unilateral penalty weights used exactly from Eqs. 34, 35, 37, 38, 29, 30, 31, and 32.")
    lines.append("- Total cost used exactly from Eq. 39.")
    lines.append("")
    for case in ("A", "B", "C"):
        item = package["cases"][case]
        metrics = item["history_metrics"]
        best_eval = item["best_design_full_evaluation"]
        lines.append(f"Case {case}")
        lines.append(f"- Final best cost: {metrics['final_best_cost']:.6f}")
        lines.append(f"- Final top-10 mean cost: {metrics['final_top10_mean_cost']:.6f}")
        lines.append(f"- Final top10-best gap: {metrics['final_gap_top10_minus_best']:.6f}")
        lines.append(f"- First generation within 5% of final best: {metrics['generation_first_within_5pct_of_final_best']}")
        lines.append(f"- First generation within 1% of final best: {metrics['generation_first_within_1pct_of_final_best']}")
        lines.append("- Best effective properties:")
        lines.append(f"  k*: {best_eval['effective_properties']['k_eff'] / 1e9:.4f} GPa")
        lines.append(f"  mu*: {best_eval['effective_properties']['mu_eff'] / 1e9:.4f} GPa")
        lines.append(f"  sigma_c*: {best_eval['effective_properties']['sigE_eff'] / 1e7:.4f} x10^7 S/m")
        lines.append(f"  IK*: {best_eval['effective_properties']['K_eff']:.4f} W/m/K")
        lines.append("- Discussion notes:")
        for note in item["discussion_notes"]:
            lines.append(f"  - {note}")
        lines.append("")
    lines.append("Overall comparison")
    for key, value in package["comparison_points"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Reference files")
    lines.append(f"- Case A plot: {package['files_to_reference']['case_a_plot']}")
    lines.append(f"- Case B plot: {package['files_to_reference']['case_b_plot']}")
    lines.append(f"- Case C plot: {package['files_to_reference']['case_c_plot']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    package = build_report_package()
    json_path = DATA_DIR / "report_ready_package.json"
    txt_path = DATA_DIR / "report_ready_summary.txt"
    json_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    txt_path.write_text(build_text_summary(package), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
