from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import cast

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

if __package__:
    from .config import DEFAULT_MATERIAL, OUTPUT_DIR, SimConfig
    from .design import DroneDesign
    from .geometry import build_voxel_drone
    from .optimization import format_design, optimize_design
    from .visualization import save_design_evolution_animation, save_summary_artifacts
else:
    from config import DEFAULT_MATERIAL, OUTPUT_DIR, SimConfig
    from design import DroneDesign
    from geometry import build_voxel_drone
    from optimization import format_design, optimize_design
    from visualization import save_design_evolution_animation, save_summary_artifacts


class ProgressPrinter:
    def __init__(self, total_steps: int):
        self.total_steps = max(total_steps, 1)
        self.completed = 0
        self.start = time.perf_counter()
        self.last_message = "Starting Project Y..."
        self._render()

    def add_total(self, extra_steps: int) -> None:
        self.total_steps = max(self.total_steps + max(extra_steps, 0), 1)
        self._render()

    def message(self, text: str) -> None:
        self.last_message = text
        sys.stdout.write("\n" + text + "\n")
        self._render()

    def advance(self, steps: int = 1, text: str | None = None) -> None:
        self.completed = min(self.completed + max(steps, 0), self.total_steps)
        if text is not None:
            self.last_message = text
        self._render()

    def _render(self) -> None:
        elapsed = max(time.perf_counter() - self.start, 1.0e-6)
        fraction = min(max(self.completed / self.total_steps, 0.0), 1.0)
        filled = int(round(28 * fraction))
        bar = "#" * filled + "-" * (28 - filled)
        rate = self.completed / elapsed if self.completed > 0 else 0.0
        remaining = (self.total_steps - self.completed) / rate if rate > 1.0e-9 else float("inf")
        eta = "estimating..." if not math.isfinite(remaining) else f"{int(remaining // 60):02d}:{int(remaining % 60):02d}"
        sys.stdout.write(
            "\r"
            f"[{bar}] {self.completed:>4}/{self.total_steps:<4}  "
            f"{fraction * 100:5.1f}%  ETA {eta}  {self.last_message[:70]:<70}"
        )
        sys.stdout.flush()

    def finish(self, text: str) -> None:
        self.completed = self.total_steps
        self.last_message = text
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    config = SimConfig()
    initial_total = config.optimizer_generations * config.population_size
    progress = ProgressPrinter(initial_total)

    def on_optimize(message: str, generation_index: int, candidate_index: int) -> None:
        if candidate_index >= 0:
            progress.advance(1, message)
        else:
            progress.message(message)

    progress.message("Optimizing drone designs...")
    best_design, best_result, history, generation_bests = optimize_design(config, progress_callback=on_optimize)

    animation_frames = len(best_result.state_history["position"])
    evolution_frames = max(1, 10 * len(generation_bests))
    progress.add_total(animation_frames + evolution_frames + 3)

    progress.message("Building best voxel drone...")
    drone = build_voxel_drone(best_design, DEFAULT_MATERIAL, config.voxel_resolution)
    progress.advance(1, "Best voxel drone built")

    progress.message("Writing summary plots and report artifacts...")
    save_summary_artifacts(
        drone,
        best_design,
        best_result,
        history,
        animation_progress_callback=lambda frame, total: progress.advance(
            1,
            f"Encoding final simulation GIF frame {frame + 1}/{total}",
        ),
    )
    progress.advance(1, "Summary plots and report artifacts written")

    progress.message("Encoding design evolution GIF...")
    save_design_evolution_animation(
        generation_bests,
        config,
        progress_callback=lambda frame, total: progress.advance(
            1,
            f"Encoding design evolution GIF frame {frame + 1}/{total}",
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_design": format_design(best_design),
        "best_score": best_result.score,
        "best_summary": best_result.summary,
        "survived": best_result.survived,
        "history": history,
        "generation_bests": [
            {
                "generation": row["generation"],
                "design": format_design(cast(DroneDesign, row["design"])),
                "score": row["score"],
                "progress": row["progress"],
                "max_stress": row["max_stress"],
                "survived": row["survived"],
            }
            for row in generation_bests
        ],
    }
    with open(OUTPUT_DIR / "optimization_history.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    progress.advance(1, "optimization_history.json written")

    progress.finish("Project Y complete")
    print("Project Y complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Best score: {best_result.score:.3f}")
    print(f"Progress: {best_result.progress:.3f}")
    print(f"Average speed: {best_result.avg_speed:.3f} m/s")
    print(f"Max stress: {best_result.max_stress:.3f} Pa")
    print(f"Survived: {best_result.survived}")
    print(f"Interactive snapshot archive: {OUTPUT_DIR / 'projecty_snapshot_archive.npz'}")
    print("Launch viewer:")
    print(r"python projects\projecty\interactive_viewer.py")


if __name__ == "__main__":
    main()
