"""Utilities for exporting turn-in artifacts from the rendered MP4.

Uses the system FFmpeg tools (ffprobe + ffmpeg).

Outputs:
  - Copies the MP4 into the output directory
  - Extracts N evenly spaced frames as PNGs
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _probe_duration_seconds(mp4_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mp4_path),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(res.stdout.strip())


def export_evenly_spaced_frames(
    mp4_path: str | Path,
    out_dir: str | Path,
    n_frames: int = 5,
    prefix: str = "frame",
) -> List[Path]:
    mp4_path = Path(mp4_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if not mp4_path.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4_path}")

    # Copy MP4 into output directory
    mp4_out = out_dir / mp4_path.name
    if mp4_out.resolve() != mp4_path.resolve():
        shutil.copy2(mp4_path, mp4_out)

    duration = _probe_duration_seconds(mp4_path)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid video duration from ffprobe: {duration}")

    # Evenly spaced times in [0, duration), avoiding the last instants of the
    # video where some FFmpeg builds return success but output no frame.
    if n_frames == 1:
        times = [0.0]
    else:
        # Keep a small safety margin from the end.
        safety = max(0.1, 0.02 * duration)
        max_t = max(0.0, duration - safety)
        times = [i * (max_t / (n_frames - 1)) for i in range(n_frames)]

    outputs: List[Path] = []
    for k, t in enumerate(times, start=1):
        out_path = out_dir / f"{prefix}_{k:02d}.png"

        # Some FFmpeg builds return success but produce no file if the seek time
        # is too close to the end. Retry by moving the timestamp earlier.
        max_attempts = 6
        backoff = max(0.25, 0.01 * duration)
        t_try = float(t)
        for attempt in range(max_attempts):
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{t_try:.6f}",
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                str(out_path),
            ]
            _run(cmd)
            if out_path.exists() and out_path.stat().st_size > 0:
                break
            t_try = max(0.0, t_try - backoff * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to extract frame {k} from {mp4_path}")

        outputs.append(out_path)

    return outputs
