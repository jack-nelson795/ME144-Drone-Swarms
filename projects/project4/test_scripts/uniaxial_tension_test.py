"""Uniaxial tension validation (cube/bar).

The Project 4 handout asks you to run the provided script `uniaxial_tension_test.py`.
In this codebase, the implementation lives in `uniaxial_tension_test_cube.py`.

This file is a thin compatibility wrapper so the canonical entrypoint name matches
what the assignment expects.
"""

from __future__ import annotations

# Allow running this file directly via `python test_scripts/uniaxial_tension_test.py`
# (in that case, Python's import root is not guaranteed to be the project4 folder).
if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from test_scripts.uniaxial_tension_test_cube import init_von_mises_3d_animation, main

__all__ = ["init_von_mises_3d_animation", "main"]


if __name__ == "__main__":
    main()
