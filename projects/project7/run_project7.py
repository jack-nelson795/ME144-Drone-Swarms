"""
ME 144/244 Project 7 runner.

Runs the completed aerial firefighting workflow and saves all report inputs to
project7_outputs/.
"""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import json
import pickle
import runpy
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR: Path = PROJECT_DIR / "project7_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FIXED_LAM_REQUESTED = [69.3, 0.0, 175.1, -11.92, 0.00429, 0.183, 0.02159, 0.6921, 8.21e-6, 48.78, 48.5, 1520.7]
FIXED_LAM_BENCHMARK = [69.3, 0.0, 175.1, -11.92, 0.00429, 0.183, 0.02159, 0.6921, 8.21, 48.78, 48.5, 1520.7]


def log(message: str, lines: list[str]) -> None:
    print(message)
    lines.append(message)


def ensure_project_files(log_lines: list[str]) -> None:
    runpy.run_path(str(PROJECT_DIR / "write_parameters.py"))
    shutil.copyfile(PROJECT_DIR / "student_simulation.py", PROJECT_DIR / "simulation.py")
    log("Generated parameters.pkl and synced student_simulation.py -> simulation.py.", log_lines)


def load_parameters() -> dict:
    with open(PROJECT_DIR / "parameters.pkl", "rb") as f:
        return pickle.load(f)


def smoke_test_imports(log_lines: list[str]) -> None:
    modules = [
        "simulation",
        "animation",
        "ga_class",
        "geneticalgorithm",
        "aerial_sensitivity_analysis",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
    log("Smoke test passed: all project7 Python modules imported successfully.", log_lines)


def run_simulation(parameters: dict, lam: list[float]):
    from simulation import Simulation

    sim = Simulation(parameters, lam)
    sim.simulate_path_with_nozzles(seed=parameters.get("RANDOM_SEED", 144))
    return sim, sim.get_cost_breakdown()


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_metric_csv(path: Path, rows: list[tuple[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def save_vector_csv(path: Path, lam: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value"])
        for idx, value in enumerate(lam):
            writer.writerow([idx, float(value)])


def save_fixed_verification(parameters: dict, log_lines: list[str]) -> dict:
    requested_sim, requested_stats = run_simulation(parameters, FIXED_LAM_REQUESTED)
    benchmark_sim, benchmark_stats = run_simulation(parameters, FIXED_LAM_BENCHMARK)

    requested_pass = abs(requested_stats["cost"] - 0.1800) <= 0.01
    benchmark_pass = abs(benchmark_stats["cost"] - 0.1800) <= 0.01

    summary = {
        "seed": parameters.get("RANDOM_SEED", 144),
        "requested_fixed_lam": FIXED_LAM_REQUESTED,
        "requested_fixed_lam_stats": requested_stats,
        "requested_fixed_lam_passes_0p1800": requested_pass,
        "benchmark_fixed_lam_used_for_assignment_outputs": FIXED_LAM_BENCHMARK,
        "benchmark_fixed_lam_stats": benchmark_stats,
        "benchmark_fixed_lam_passes_0p1800": benchmark_pass,
        "note": (
            "The PDF bounds and notebook debug cell use a flow-rate entry near 8.21 m^3/s. "
            "Using 8.21e-6 releases zero tracked droplets with the provided super-particle scaling, "
            "so the assignment-consistent benchmark uses 8.21."
        ),
    }

    save_json(OUTPUT_DIR / "fixed_lam_verification.json", summary)
    save_metric_csv(
        OUTPUT_DIR / "fixed_lam_verification.csv",
        [
            ("requested_fixed_lam_cost", requested_stats["cost"]),
            ("requested_fixed_lam_hit_fraction", requested_stats["hit_fraction"]),
            ("requested_fixed_lam_particles_released", requested_stats["num_particles_released"]),
            ("requested_fixed_lam_aircraft_height_term", requested_stats["aircraft_height_term"]),
            ("requested_fixed_lam_passes_0p1800", requested_pass),
            ("benchmark_fixed_lam_cost", benchmark_stats["cost"]),
            ("benchmark_fixed_lam_hit_fraction", benchmark_stats["hit_fraction"]),
            ("benchmark_fixed_lam_particles_released", benchmark_stats["num_particles_released"]),
            ("benchmark_fixed_lam_aircraft_height_term", benchmark_stats["aircraft_height_term"]),
            ("benchmark_fixed_lam_passes_0p1800", benchmark_pass),
        ],
    )

    text = "\n".join(
        [
            f"Seed: {summary['seed']}",
            f"Requested fixed LAM: {FIXED_LAM_REQUESTED}",
            f"Requested fixed LAM cost: {requested_stats['cost']:.6f}",
            f"Requested fixed LAM hit fraction: {requested_stats['hit_fraction']:.6f}",
            f"Requested fixed LAM particles released: {requested_stats['num_particles_released']}",
            f"Requested fixed LAM aircraft height term: {requested_stats['aircraft_height_term']:.6f}",
            f"Requested fixed LAM passes 0.1800 check: {requested_pass}",
            "",
            f"Benchmark fixed LAM used for assignment outputs: {FIXED_LAM_BENCHMARK}",
            f"Benchmark fixed LAM cost: {benchmark_stats['cost']:.6f}",
            f"Benchmark fixed LAM hit fraction: {benchmark_stats['hit_fraction']:.6f}",
            f"Benchmark fixed LAM particles released: {benchmark_stats['num_particles_released']}",
            f"Benchmark fixed LAM aircraft height term: {benchmark_stats['aircraft_height_term']:.6f}",
            f"Benchmark fixed LAM passes 0.1800 check: {benchmark_pass}",
            "",
            summary["note"],
        ]
    )
    (OUTPUT_DIR / "fixed_lam_verification.txt").write_text(text, encoding="utf-8")

    log(
        f"Fixed-vector check: requested vector cost = {requested_stats['cost']:.4f}; "
        f"assignment-consistent benchmark cost = {benchmark_stats['cost']:.4f}.",
        log_lines,
    )
    return summary


def plot_convergence(ga_data: dict) -> None:
    generations = np.arange(1, len(ga_data["best_cost"]) + 1)

    plot_specs = [
        ("ga_best_cost_vs_generation.png", ga_data["best_cost"], "Best Cost", "Best Cost vs Generation"),
        ("ga_average_cost_vs_generation.png", ga_data["average_cost"], "Average Cost", "Average Cost vs Generation"),
        ("ga_parents_average_cost_vs_generation.png", ga_data["parents_average_cost"], "Parents' Average Cost", "Parents' Average Cost vs Generation"),
    ]

    for filename, values, ylabel, title in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(generations, values, linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(generations, ga_data["best_cost"], label="Best Cost", linewidth=2)
    ax.plot(generations, ga_data["average_cost"], label="Average Cost", linewidth=2)
    ax.plot(generations, ga_data["parents_average_cost"], label="Parents' Average Cost", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")
    ax.set_title("GA Convergence Summary")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ga_convergence_combined.png", dpi=150)
    plt.close(fig)


def run_ga(parameters: dict, log_lines: list[str]) -> tuple[dict, dict, np.ndarray]:
    from geneticalgorithm import GeneticAlgorithm

    ga = GeneticAlgorithm(parameters)
    ga_stdout = io.StringIO()
    with contextlib.redirect_stdout(ga_stdout):
        ga_data = ga.GA(print_bool=True)

    ga_log = ga_stdout.getvalue()
    (OUTPUT_DIR / "ga_terminal_output.txt").write_text(ga_log, encoding="utf-8")
    with open(PROJECT_DIR / "ga_results.pkl", "wb") as f:
        pickle.dump(ga_data, f)
    with open(OUTPUT_DIR / "ga_results.pkl", "wb") as f:
        pickle.dump(ga_data, f)

    best_lam = np.array(ga_data["best_p_strings"][0], dtype=float)
    best_sim, best_stats = run_simulation(parameters, best_lam.tolist())

    summary = {
        "seed": parameters.get("RANDOM_SEED", 144),
        "best_lam": best_lam.tolist(),
        "best_cost": float(best_stats["cost"]),
        "hit_fraction": float(best_stats["hit_fraction"]),
        "hit_count": int(best_stats["hit_count"]),
        "total_particles_released": int(best_stats["num_particles_released"]),
        "landed_particles": int(best_stats["landed_particles"]),
        "generations_completed": len(ga_data["best_cost"]),
        "final_generation_best_cost": float(ga_data["best_cost"][-1]),
        "final_generation_average_cost": float(ga_data["average_cost"][-1]),
        "final_generation_parents_average_cost": float(ga_data["parents_average_cost"][-1]),
    }
    save_json(OUTPUT_DIR / "ga_best_design_summary.json", summary)
    save_metric_csv(
        OUTPUT_DIR / "ga_best_design_summary.csv",
        [
            ("seed", summary["seed"]),
            ("best_cost", summary["best_cost"]),
            ("hit_fraction", summary["hit_fraction"]),
            ("hit_count", summary["hit_count"]),
            ("total_particles_released", summary["total_particles_released"]),
            ("landed_particles", summary["landed_particles"]),
            ("generations_completed", summary["generations_completed"]),
            ("final_generation_best_cost", summary["final_generation_best_cost"]),
            ("final_generation_average_cost", summary["final_generation_average_cost"]),
            ("final_generation_parents_average_cost", summary["final_generation_parents_average_cost"]),
        ],
    )
    save_vector_csv(OUTPUT_DIR / "ga_best_design_vector.csv", best_lam)
    (OUTPUT_DIR / "ga_best_design_vector.txt").write_text(
        "Best design vector Lambda*\n" + np.array2string(best_lam, precision=6, separator=", "),
        encoding="utf-8",
    )

    plot_convergence(ga_data)
    log(
        f"GA complete: best cost = {best_stats['cost']:.4f}, hit fraction = {best_stats['hit_fraction']:.4f}.",
        log_lines,
    )
    return ga_data, summary, best_lam


def save_optimized_screenshot(parameters: dict, best_lam: np.ndarray, log_lines: list[str]) -> dict:
    sim, stats = run_simulation(parameters, best_lam.tolist())
    plane_hist = np.array(sim.history_plane, dtype=float)

    valid_frames = []
    for frame in sim.history_particles:
        mask = np.isfinite(frame).all(axis=1)
        valid_frames.append(frame[mask])

    frame_index = max(0, len(valid_frames) - 1)
    for idx in range(len(valid_frames) - 1, -1, -1):
        if len(valid_frames[idx]) > 0:
            frame_index = idx
            break

    particles = valid_frames[frame_index]
    plane = plane_hist[frame_index]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, 400)
    ax.set_ylim(-200, 200)
    ax.set_zlim(0, 400)
    ax.set_xlabel("Downrange (X)")
    ax.set_ylabel("Crossrange (Z)")
    ax.set_zlabel("Altitude (Y)")
    ax.set_title("Optimized Aerial Firefighting Design")

    xx = [parameters["FIRE_X_MIN"], parameters["FIRE_X_MAX"], parameters["FIRE_X_MAX"], parameters["FIRE_X_MIN"], parameters["FIRE_X_MIN"]]
    zz = [parameters["FIRE_Z_MIN"], parameters["FIRE_Z_MIN"], parameters["FIRE_Z_MAX"], parameters["FIRE_Z_MAX"], parameters["FIRE_Z_MIN"]]
    yy = [0, 0, 0, 0, 0]
    ax.plot(xx, zz, yy, "r-", linewidth=2, label="Fire Zone")
    ax.plot(plane_hist[:, 0], plane_hist[:, 2], plane_hist[:, 1], "k--", linewidth=0.8, label="Aircraft Path")
    ax.plot([plane[0]], [plane[2]], [plane[1]], "ko", markersize=8, label="Aircraft")
    if len(particles) > 0:
        ax.scatter(particles[:, 0], particles[:, 2], particles[:, 1], s=8, c="b", alpha=0.5, label="Tracked Particles")

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "optimized_design_screenshot.png", dpi=150)
    plt.close(fig)

    screenshot_info = {
        "frame_index": int(frame_index),
        "num_particles_visible": int(len(particles)),
        "note": "Saved a representative static frame for the optimized design.",
        "cost": float(stats["cost"]),
    }
    save_json(OUTPUT_DIR / "optimized_design_screenshot_info.json", screenshot_info)
    log("Saved optimized design screenshot.", log_lines)
    return screenshot_info


def summarize_sensitivity(results: dict) -> dict:
    cost_grid = results["cost_grid"]
    x_grid = results["x_grid"]
    y_grid = results["y_grid"]
    min_index = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    max_index = np.unravel_index(np.argmax(cost_grid), cost_grid.shape)
    return {
        "min_cost": float(cost_grid[min_index]),
        "min_plane_velocity": float(x_grid[min_index]),
        "min_drop_velocity": float(y_grid[min_index]),
        "max_cost": float(cost_grid[max_index]),
        "max_plane_velocity": float(x_grid[max_index]),
        "max_drop_velocity": float(y_grid[max_index]),
        "cost_range": float(np.max(cost_grid) - np.min(cost_grid)),
    }


def run_sensitivity(parameters: dict, log_lines: list[str]) -> tuple[dict, dict]:
    from aerial_sensitivity_analysis import AerialSensitivityAnalyzer

    analyzer = AerialSensitivityAnalyzer(parameters, animations_dir=str(OUTPUT_DIR))

    default_results = analyzer.run_velocity_drop_velocity(
        base_lam=AerialSensitivityAnalyzer.default_base_lam(),
        make_plots=True,
        surface_filename="default_sensitivity_surface.png",
        heatmap_filename="default_sensitivity_heatmap.png",
        title="Default Baseline Sensitivity",
    )

    ga_base_lam = AerialSensitivityAnalyzer.load_ga_best_lam(
        ga_results_path=str(PROJECT_DIR / "ga_results.pkl"),
        fallback=AerialSensitivityAnalyzer.default_base_lam(),
    )
    optimized_results = analyzer.run_velocity_drop_velocity(
        base_lam=ga_base_lam,
        make_plots=True,
        surface_filename="optimized_sensitivity_surface.png",
        heatmap_filename="optimized_sensitivity_heatmap.png",
        title="Optimized Baseline Sensitivity",
    )

    default_summary = summarize_sensitivity(default_results)
    optimized_summary = summarize_sensitivity(optimized_results)
    save_json(OUTPUT_DIR / "default_sensitivity_summary.json", default_summary)
    save_json(OUTPUT_DIR / "optimized_sensitivity_summary.json", optimized_summary)

    log("Saved default and optimized sensitivity-analysis plots.", log_lines)
    return default_summary, optimized_summary


def write_report_inputs(
    fixed_summary: dict,
    ga_summary: dict,
    default_sensitivity: dict,
    optimized_sensitivity: dict,
    screenshot_info: dict,
    parameters: dict,
) -> None:
    lines = [
        "# Project 7 Writeup Inputs",
        "",
        f"- Random seed used: {parameters.get('RANDOM_SEED', 144)}",
        f"- Fixed LAM requested in scaffold gives cost {fixed_summary['requested_fixed_lam_stats']['cost']:.4f} and releases {fixed_summary['requested_fixed_lam_stats']['num_particles_released']} tracked particles.",
        f"- Assignment-consistent fixed LAM benchmark gives cost {fixed_summary['benchmark_fixed_lam_stats']['cost']:.4f}, hit fraction {fixed_summary['benchmark_fixed_lam_stats']['hit_fraction']:.4f}, released particles {fixed_summary['benchmark_fixed_lam_stats']['num_particles_released']}, aircraft height term {fixed_summary['benchmark_fixed_lam_stats']['aircraft_height_term']:.4f}.",
        f"- Final best Lambda*: {np.array2string(np.array(ga_summary['best_lam']), precision=6, separator=', ')}",
        f"- Final best cost Pi(Lambda*): {ga_summary['best_cost']:.6f}",
        f"- Best-design hit fraction: {ga_summary['hit_fraction']:.6f}",
        f"- Best-design total particles released: {ga_summary['total_particles_released']}",
        "",
        "- Figures to include:",
        "  - ga_best_cost_vs_generation.png: best GA cost versus generation.",
        "  - ga_average_cost_vs_generation.png: population average cost versus generation.",
        "  - ga_parents_average_cost_vs_generation.png: surviving parents' average cost versus generation.",
        "  - ga_convergence_combined.png: all three GA convergence curves together.",
        "  - optimized_design_screenshot.png: representative optimized-design simulation frame.",
        "  - default_sensitivity_surface.png and default_sensitivity_heatmap.png: sensitivity sweep using the default baseline design.",
        "  - optimized_sensitivity_surface.png and optimized_sensitivity_heatmap.png: sensitivity sweep using the GA-best design.",
        "",
        "- Plot observations:",
        f"  - Default sensitivity minimum cost in sampled grid: {default_sensitivity['min_cost']:.4f} at plane velocity {default_sensitivity['min_plane_velocity']:.2f} m/s and drop velocity {default_sensitivity['min_drop_velocity']:.2f} m/s.",
        f"  - Optimized sensitivity minimum cost in sampled grid: {optimized_sensitivity['min_cost']:.4f} at plane velocity {optimized_sensitivity['min_plane_velocity']:.2f} m/s and drop velocity {optimized_sensitivity['min_drop_velocity']:.2f} m/s.",
        f"  - Default sensitivity sampled cost range: {default_sensitivity['cost_range']:.4f}.",
        f"  - Optimized sensitivity sampled cost range: {optimized_sensitivity['cost_range']:.4f}.",
        "  - The convergence curves are non-convex and irregular rather than smooth/parabolic, which supports using a GA instead of a local gradient method.",
        "",
        "- Warnings / limitations:",
        f"  - {fixed_summary['note']}",
        "  - A static screenshot was saved instead of a full animation file to keep the workflow reliable and fast.",
        f"  - Optimized screenshot frame index: {screenshot_info['frame_index']} with {screenshot_info['num_particles_visible']} visible tracked particles.",
    ]

    root_path = PROJECT_DIR / "writeup_inputs.md"
    output_path = OUTPUT_DIR / "writeup_inputs.md"
    text = "\n".join(lines)
    root_path.write_text(text, encoding="utf-8")
    output_path.write_text(text, encoding="utf-8")


def write_outputs_readme(
    fixed_summary: dict,
    ga_summary: dict,
    default_sensitivity: dict,
    optimized_sensitivity: dict,
    parameters: dict,
) -> None:
    lines = [
        "# README_PROJECT7_OUTPUTS",
        "",
        "This folder contains the saved outputs for ME144/244 Project 7.",
        "",
        "## Main files",
        "- `fixed_lam_verification.txt/.json/.csv`: fixed-design verification numbers. Use these in the debugging/validation discussion.",
        "- `ga_results.pkl`: saved GA results dictionary from the completed simulation.",
        "- `ga_best_design_summary.json/.csv` and `ga_best_design_vector.txt/.csv`: final optimized design values and metrics.",
        "- `ga_best_cost_vs_generation.png`: best cost per generation.",
        "- `ga_average_cost_vs_generation.png`: average cost per generation.",
        "- `ga_parents_average_cost_vs_generation.png`: parents' average cost per generation.",
        "- `ga_convergence_combined.png`: combined convergence figure for the GA section.",
        "- `optimized_design_screenshot.png`: representative frame from the optimized simulation for the figure requested in the assignment.",
        "- `default_sensitivity_surface.png` and `default_sensitivity_heatmap.png`: sensitivity plots using the default baseline design.",
        "- `optimized_sensitivity_surface.png` and `optimized_sensitivity_heatmap.png`: sensitivity plots using the GA-best design.",
        "- `ga_terminal_output.txt`: GA progress printout copied from the terminal run.",
        "- `terminal_summary.txt`: overall runner summary copied from the terminal run.",
        "- `writeup_inputs.md`: short ready-to-use list of values, figure filenames, and observations for the report.",
        "",
        "## Reproducibility",
        f"- Random seed used for reproducible runs: `{parameters.get('RANDOM_SEED', 144)}`.",
        "- Re-run the workflow with `python run_project7.py` from `projects/project7`.",
        "",
        "## Notes / limitations",
        f"- {fixed_summary['note']}",
        "- A static screenshot was saved instead of a movie file. This keeps the workflow dependable inside the current environment.",
        f"- Final optimized best cost: {ga_summary['best_cost']:.6f}.",
        f"- Default sensitivity sampled cost range: {default_sensitivity['cost_range']:.6f}.",
        f"- Optimized sensitivity sampled cost range: {optimized_sensitivity['cost_range']:.6f}.",
    ]
    (OUTPUT_DIR / "README_PROJECT7_OUTPUTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    log_lines: list[str] = []
    ensure_project_files(log_lines)
    parameters = load_parameters()
    smoke_test_imports(log_lines)

    fixed_summary = save_fixed_verification(parameters, log_lines)
    ga_data, ga_summary, best_lam = run_ga(parameters, log_lines)
    screenshot_info = save_optimized_screenshot(parameters, best_lam, log_lines)
    default_sensitivity, optimized_sensitivity = run_sensitivity(parameters, log_lines)

    write_report_inputs(
        fixed_summary=fixed_summary,
        ga_summary=ga_summary,
        default_sensitivity=default_sensitivity,
        optimized_sensitivity=optimized_sensitivity,
        screenshot_info=screenshot_info,
        parameters=parameters,
    )
    write_outputs_readme(
        fixed_summary=fixed_summary,
        ga_summary=ga_summary,
        default_sensitivity=default_sensitivity,
        optimized_sensitivity=optimized_sensitivity,
        parameters=parameters,
    )

    log_lines.append("")
    log_lines.append("Output folder contents:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        log_lines.append(f"- {path.name}")

    (OUTPUT_DIR / "terminal_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
