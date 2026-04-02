from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .config import CONFIG
from .ga import GARunResult, run_case
from .hashin_shtrikman import evaluate_design


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_DATA_DIR = OUTPUT_DIR / "report_data"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _generation_index_first_reach(values: np.ndarray, target: float, atol: float = 1e-15) -> int | None:
    idx = np.where(np.isclose(values, target, atol=atol, rtol=0.0))[0]
    return int(idx[0] + 1) if idx.size else None


def _generation_index_all_remaining_equal(values: np.ndarray, target: float, atol: float = 1e-15) -> int | None:
    matches = np.isclose(values, target, atol=atol, rtol=0.0)
    for i in range(len(values)):
        if np.all(matches[i:]):
            return i + 1
    return None


def _is_monotone_nonincreasing(values: np.ndarray, atol: float = 1e-15) -> bool:
    return bool(np.all(np.diff(values) <= atol))


def _unique_rows_count(array: np.ndarray) -> int:
    rounded = np.round(array.astype(float), decimals=14)
    return int(np.unique(rounded, axis=0).shape[0])


def _rows_exactly_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.round(a, 14), np.round(b, 14)))


def _report_unit_vector(design: np.ndarray) -> np.ndarray:
    return np.array(
        [design[0] / 1e9, design[1] / 1e9, design[2] / 1e7, design[3], design[4]],
        dtype=float,
    )


def make_plot(case: str, result: GARunResult) -> Path:
    generations = np.arange(1, CONFIG.G + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(generations, result.history_best, linewidth=2.0, label="Best design cost")
    ax.loglog(generations, result.history_top10_mean, linewidth=2.0, label="Mean cost of top 10 designs")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")
    ax.set_title(f"Project 5 GA convergence: Case {case}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"case_{case.lower()}_convergence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_top_tables(case: str, result: GARunResult) -> dict[str, Any]:
    top4_rows = []
    for rank in range(4):
        design = result.population[rank, :]
        evaluation = evaluate_design(design, CONFIG)
        props = evaluation["effective_properties"]
        top4_rows.append(
            {
                "rank": rank + 1,
                "kappa_2_GPa": float(design[0] / 1e9),
                "mu_2_GPa": float(design[1] / 1e9),
                "sigma_2_c_1e7_S_per_m": float(design[2] / 1e7),
                "IK_2_W_per_mK": float(design[3]),
                "v_2": float(design[4]),
                "effective_k_star_GPa": float(props["k_eff"] / 1e9),
                "effective_mu_star_GPa": float(props["mu_eff"] / 1e9),
                "effective_sigma_c_star_1e7_S_per_m": float(props["sigE_eff"] / 1e7),
                "effective_IK_star_W_per_mK": float(props["K_eff"]),
                "total_cost": float(result.costs[rank]),
            }
        )

    full_df = pd.DataFrame(top4_rows)
    rounded_df = full_df.copy()
    for col in rounded_df.columns:
        if col != "rank":
            rounded_df[col] = rounded_df[col].map(lambda x: float(f"{x:.4g}"))

    full_path = TABLES_DIR / f"case_{case.lower()}_top4_full_precision.csv"
    rounded_path = TABLES_DIR / f"case_{case.lower()}_top4_report_ready.csv"
    full_df.to_csv(full_path, index=False)
    rounded_df.to_csv(rounded_path, index=False)

    top10_rows = []
    for rank in range(10):
        row: dict[str, float | int] = {"rank": rank + 1}
        genome = result.population[rank, :]
        for i, key in enumerate(["kappa_2_Pa", "mu_2_Pa", "sigma_2_c_S_per_m", "IK_2_W_per_mK", "v_2"]):
            row[key] = float(genome[i])
        row["total_cost"] = float(result.costs[rank])
        top10_rows.append(row)
    top10_path = TABLES_DIR / f"case_{case.lower()}_top10_full_precision.csv"
    pd.DataFrame(top10_rows).to_csv(top10_path, index=False)

    return {
        "top4_full_precision_csv": str(full_path.relative_to(PROJECT_DIR)),
        "top4_report_ready_csv": str(rounded_path.relative_to(PROJECT_DIR)),
        "top10_full_precision_csv": str(top10_path.relative_to(PROJECT_DIR)),
        "top4_full_precision_records": full_df.to_dict(orient="records"),
        "top4_report_ready_records": rounded_df.to_dict(orient="records"),
    }


def build_cost_breakdown(case: str, result: GARunResult) -> dict[str, Any]:
    evaluation = evaluate_design(result.population[0, :], CONFIG)
    path = REPORT_DATA_DIR / f"case_{case.lower()}_cost_breakdown.json"
    payload = {
        "case": case,
        "best_design": {k: float(v) for k, v in evaluation["design"].items()},
        "cost_terms": {k: float(v) for k, v in evaluation["cost_terms"].items()},
        "cost_breakdown": {k: float(v) for k, v in evaluation["cost_breakdown"].items()},
        "active_unilateral_weights": {k: float(v) for k, v in evaluation["weights"].items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_convergence_audit(case: str, result: GARunResult) -> dict[str, Any]:
    best = result.history_best
    top10 = result.history_top10_mean
    final_best = float(best[-1])
    final_top10 = float(top10[-1])
    generation_final_best_first = _generation_index_first_reach(best, final_best)
    generation_top10_final_first = _generation_index_first_reach(top10, final_top10)
    generation_top10_stabilized = _generation_index_all_remaining_equal(top10, final_top10)
    best_diff = np.diff(best)
    improve_idx = np.where(best_diff < -1e-15)[0]
    last_best_improvement_gen = int(improve_idx[-1] + 2) if improve_idx.size else 1
    stagnant_best_generations = int(CONFIG.G - generation_final_best_first) if generation_final_best_first else None
    plateau_threshold = 500
    stagnation_notes = []
    if stagnant_best_generations is not None and stagnant_best_generations >= plateau_threshold:
        stagnation_notes.append(
            f"Best cost did not improve after generation {generation_final_best_first}, leaving {stagnant_best_generations} stagnant generations."
        )
    if generation_top10_stabilized is not None and CONFIG.G - generation_top10_stabilized >= plateau_threshold:
        stagnation_notes.append(
            f"Top-10 mean stayed at its final value from generation {generation_top10_stabilized} onward."
        )
    if not stagnation_notes:
        stagnation_notes.append("No long plateau above the chosen 500-generation threshold was detected.")

    payload = {
        "case": case,
        "total_generations": CONFIG.G,
        "best_cost_monotone_nonincreasing": _is_monotone_nonincreasing(best),
        "top10_mean_monotone_nonincreasing": _is_monotone_nonincreasing(top10),
        "generation_final_best_cost_first_reached": generation_final_best_first,
        "generation_final_top10_mean_first_reached": generation_top10_final_first,
        "generation_top10_mean_stabilized_exact_final_value": generation_top10_stabilized,
        "last_generation_with_best_cost_improvement": last_best_improvement_gen,
        "evidence_of_premature_convergence_or_stagnation": stagnation_notes,
    }
    path = REPORT_DATA_DIR / f"case_{case.lower()}_convergence_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_diversity_audit(results: dict[str, GARunResult]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for case, result in results.items():
        top4 = result.population[:4, :]
        top10 = result.population[:10, :]
        unique_top4 = _unique_rows_count(top4)
        unique_top10 = _unique_rows_count(top10)
        payload[case] = {
            "case": case,
            "top4_all_unique": unique_top4 == 4,
            "top4_contains_duplicates": unique_top4 < 4,
            "top10_all_unique": unique_top10 == 10,
            "top10_contains_duplicates": unique_top10 < 10,
            "number_of_unique_genomes_in_final_top4": unique_top4,
            "number_of_unique_genomes_in_final_top10": unique_top10,
        }

    a_top10 = results["A"].population[:10, :]
    c_top10 = results["C"].population[:10, :]
    a_top10_report_units = np.array([_report_unit_vector(row) for row in a_top10])
    c_top10_report_units = np.array([_report_unit_vector(row) for row in c_top10])
    a_best_report = _report_unit_vector(results["A"].population[0, :])
    c_best_report = _report_unit_vector(results["C"].population[0, :])
    best_cost_diff = float(results["A"].costs[0] - results["C"].costs[0])
    best_design_norm_diff_report_units = float(np.linalg.norm(a_best_report - c_best_report))
    cross_case = {
        "cases_A_and_C_top10_exactly_identical": _rows_exactly_equal(a_top10, c_top10),
        "cases_A_and_C_top10_equal_to_4_significant_digits_in_report_units": _rows_exactly_equal(
            np.array([[float(f"{x:.4g}") for x in row] for row in a_top10_report_units]),
            np.array([[float(f"{x:.4g}") for x in row] for row in c_top10_report_units]),
        ),
        "cases_A_and_C_best_design_exactly_identical": _rows_exactly_equal(results["A"].population[0, :], results["C"].population[0, :]),
        "cases_A_and_C_best_design_equal_to_4_significant_digits_in_report_units": _rows_exactly_equal(
            np.array([float(f"{x:.4g}") for x in a_best_report]),
            np.array([float(f"{x:.4g}") for x in c_best_report]),
        ),
        "case_C_minus_case_A_best_cost": float(results["C"].costs[0] - results["A"].costs[0]),
        "absolute_best_cost_difference_A_vs_C": abs(best_cost_diff),
        "best_design_l2_norm_difference_A_vs_C_in_report_units": best_design_norm_diff_report_units,
        "mutation_had_measurable_effect_in_case_C": abs(best_cost_diff) > 0.0 or best_design_norm_diff_report_units > 0.0,
        "mutation_effect_note": (
            "Case C is not exactly identical to Case A in this run. The best cost is lower by "
            f"{abs(best_cost_diff):.12g} and the best-design L2 difference in report units is {best_design_norm_diff_report_units:.12g}. "
            "However, the two outcomes are indistinguishable at the 4-significant-digit report-ready table precision."
        ),
    }

    full_payload = {"per_case": payload, "cross_case": cross_case}
    path = REPORT_DATA_DIR / "diversity_duplication_audit.json"
    path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    return full_payload


def build_figure_audit(case_paths: dict[str, Path]) -> dict[str, Any]:
    payload: dict[str, Any] = {"required_log_log_scale_from_brief": True, "cases": {}}
    for case, path in case_paths.items():
        with Image.open(path) as img:
            width, height = img.size
        payload["cases"][case] = {
            "figure_path": str(path.relative_to(PROJECT_DIR)),
            "title": f"Project 5 GA convergence: Case {case}",
            "legend_entries": ["Best design cost", "Mean cost of top 10 designs"],
            "x_axis_label": "Generation",
            "y_axis_label": "Cost",
            "x_axis_log_scale": True,
            "y_axis_log_scale": True,
            "resolution_pixels": {"width": width, "height": height},
            "sufficient_for_pdf_inclusion": width >= 1200 and height >= 700,
        }
    path = REPORT_DATA_DIR / "figure_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_case_summary(case: str, result: GARunResult, convergence_audit: dict[str, Any], cost_breakdown: dict[str, Any]) -> dict[str, Any]:
    best_eval = evaluate_design(result.population[0, :], CONFIG)
    return {
        "case": case,
        "final_best_cost": float(result.history_best[-1]),
        "final_top10_mean_cost": float(result.history_top10_mean[-1]),
        "final_top10_best_gap": float(result.history_top10_mean[-1] - result.history_best[-1]),
        "first_generation_within_5pct_of_final_best": next(
            (int(i + 1) for i, value in enumerate(result.history_best) if value <= 1.05 * result.history_best[-1]),
            None,
        ),
        "first_generation_within_1pct_of_final_best": next(
            (int(i + 1) for i, value in enumerate(result.history_best) if value <= 1.01 * result.history_best[-1]),
            None,
        ),
        "best_design_variables": {
            "kappa_2_Pa": float(result.population[0, 0]),
            "mu_2_Pa": float(result.population[0, 1]),
            "sigma_2_c_S_per_m": float(result.population[0, 2]),
            "IK_2_W_per_mK": float(result.population[0, 3]),
            "v_2": float(result.population[0, 4]),
            "kappa_2_GPa": float(result.population[0, 0] / 1e9),
            "mu_2_GPa": float(result.population[0, 1] / 1e9),
            "sigma_2_c_1e7_S_per_m": float(result.population[0, 2] / 1e7),
        },
        "best_effective_properties": {
            "k_star_Pa": float(best_eval["effective_properties"]["k_eff"]),
            "mu_star_Pa": float(best_eval["effective_properties"]["mu_eff"]),
            "sigma_c_star_S_per_m": float(best_eval["effective_properties"]["sigE_eff"]),
            "IK_star_W_per_mK": float(best_eval["effective_properties"]["K_eff"]),
            "k_star_GPa": float(best_eval["effective_properties"]["k_eff"] / 1e9),
            "mu_star_GPa": float(best_eval["effective_properties"]["mu_eff"] / 1e9),
            "sigma_c_star_1e7_S_per_m": float(best_eval["effective_properties"]["sigE_eff"] / 1e7),
        },
        "core_scalar_outputs": {
            "generation_final_best_first_reached": convergence_audit["generation_final_best_cost_first_reached"],
            "generation_top10_mean_stabilized_exact_final_value": convergence_audit["generation_top10_mean_stabilized_exact_final_value"],
            "Pi_elec": cost_breakdown["cost_terms"]["Pi_elec"],
            "Pi_thermo": cost_breakdown["cost_terms"]["Pi_thermo"],
            "Pi_mech": cost_breakdown["cost_terms"]["Pi_mech"],
        },
    }


def build_report_notes(master_summary: dict[str, Any], diversity: dict[str, Any]) -> str:
    a = master_summary["cases"]["A"]
    b = master_summary["cases"]["B"]
    c = master_summary["cases"]["C"]
    notes = [
        "# Report Notes",
        "",
        "## Supported By The Data",
        f"- Case C has the lowest final best cost in this run: {c['final_best_cost']:.12f}.",
        f"- Case A is extremely close to Case C: {a['final_best_cost']:.12f} versus {c['final_best_cost']:.12f}.",
        f"- Case B is measurably worse in final best cost: {b['final_best_cost']:.12f}.",
        "- Case B retains more diversity in the final elite set than Cases A and C.",
        "- Cases A and C both collapse to duplicate elite designs in the final top 10.",
        "- Electrical matching is comparatively strong in the best designs, while thermal and mechanical penalties remain active.",
        "",
        "## Not Supported By The Data",
        "- Do not claim that Case C is dramatically better than Case A; the difference is tiny in this run.",
        "- Do not claim that the top four designs are unique for every case; duplicates are present.",
        "- Do not claim that mutation produced a qualitatively different optimum in Case C at report-table precision.",
        "",
        "## Case A vs Case C",
        f"- Exact best-cost difference |A - C| = {abs(a['final_best_cost'] - c['final_best_cost']):.12g}.",
        f"- Diversity audit says exact A/C elite identity is {diversity['cross_case']['cases_A_and_C_top10_exactly_identical']}.",
        f"- Diversity audit says A/C elite sets are equal at 4-significant-digit report precision: {diversity['cross_case']['cases_A_and_C_top10_equal_to_4_significant_digits_in_report_units']}.",
        "- Safe interpretation: Case C is marginally better numerically, but essentially tied with Case A at displayed report precision.",
        "",
        "## Duplicate Designs",
        f"- Case A final top-4 unique genome count: {diversity['per_case']['A']['number_of_unique_genomes_in_final_top4']}.",
        f"- Case B final top-4 unique genome count: {diversity['per_case']['B']['number_of_unique_genomes_in_final_top4']}.",
        f"- Case C final top-4 unique genome count: {diversity['per_case']['C']['number_of_unique_genomes_in_final_top4']}.",
        "- Explicitly mention duplicate elite designs when discussing convergence and repeatability.",
        "",
        "## Caveats",
        "- All statements should be framed as applying to this specific deterministic run with the recorded seed and settings.",
        "- If discussing convergence speed, use the exported first-within-5% and first-within-1% metrics rather than vague wording.",
        "- If discussing stagnation, rely on the convergence audit JSON files and mention the exact plateau generations.",
    ]
    return "\n".join(notes) + "\n"


def main() -> None:
    ensure_dirs()

    print("Preparing audited report-data package...")
    results: dict[str, GARunResult] = {}
    for case in ("A", "B", "C"):
        print(f"Re-running Case {case} for audited exports...")
        results[case] = run_case(case, seed=CONFIG.seed, cfg=CONFIG, verbose=True, progress_every=500)
    figure_paths = {case: make_plot(case, result) for case, result in results.items()}
    print("Figures regenerated for report-data package.")

    master_summary: dict[str, Any] = {
        "project": "ME144/244 Spring 2026 Project 5",
        "method": {
            "used_exact_project_workflow": True,
            "gamma": CONFIG.gamma,
            "population_size_S": CONFIG.S,
            "parents_P": CONFIG.P,
            "offspring_K": CONFIG.K,
            "generations_G": CONFIG.G,
            "equations_used": {
                "electrical_bounds": 5,
                "thermal_bounds": 6,
                "bulk_bounds": 3,
                "shear_bounds": 4,
                "effective_property_average": 2,
                "concentration_factors": [23, 24, 25, 26, 27, 28, 9, 11],
                "unilateral_weights": [34, 35, 37, 38, 29, 30, 31, 32],
                "electrical_cost": 33,
                "thermal_cost": 36,
                "total_cost": 39,
            },
        },
        "cases": {},
    }

    for case, result in results.items():
        print(f"Building tables and audits for Case {case}...")
        build_top_tables(case, result)
        cost_breakdown = build_cost_breakdown(case, result)
        convergence_audit = build_convergence_audit(case, result)
        master_summary["cases"][case] = build_case_summary(case, result, convergence_audit, cost_breakdown)

    diversity = build_diversity_audit(results)
    build_figure_audit(figure_paths)
    print("Cross-case diversity and figure audits completed.")

    summary_path = REPORT_DATA_DIR / "report_summary.json"
    summary_path.write_text(json.dumps(master_summary, indent=2), encoding="utf-8")

    notes_path = REPORT_DATA_DIR / "report_notes_for_chatgpt.md"
    notes_path.write_text(build_report_notes(master_summary, diversity), encoding="utf-8")

    manifest = {
        "report_summary": str(summary_path.relative_to(PROJECT_DIR)),
        "figure_audit": str((REPORT_DATA_DIR / "figure_audit.json").relative_to(PROJECT_DIR)),
        "diversity_audit": str((REPORT_DATA_DIR / "diversity_duplication_audit.json").relative_to(PROJECT_DIR)),
        "notes_for_chatgpt": str(notes_path.relative_to(PROJECT_DIR)),
        "tables_dir": str(TABLES_DIR.relative_to(PROJECT_DIR)),
        "figures_dir": str(FIGURES_DIR.relative_to(PROJECT_DIR)),
    }
    manifest_path = REPORT_DATA_DIR / "report_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {notes_path}")


if __name__ == "__main__":
    main()
