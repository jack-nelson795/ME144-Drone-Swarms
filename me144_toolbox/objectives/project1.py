# me144_toolbox/objectives/project1.py
"""
ME144/244 Spring 2026 — Project 1 objectives

Defines:
  Πa(x) = x^2
  Πb(x) = (x + (π/2) sin(x))^2

and their derivatives:
  grad Π (for Newton's method f)
  Hessian of Π (for Newton's method df)

Shape convention (important for later projects):
- Accepts x as: scalar, (M,), or (M,1)
- Returns:
    Π(x)       as (M,1)
    grad Π(x)  as (M,1)
    Hessian    as (M,M)

Note: Project 1 has M = 1, but we keep the interface consistent.
"""

from __future__ import annotations

import numpy as np


def _as_col(x) -> np.ndarray:
    """
    Convert input x into a float ndarray of shape (M, 1).
    Accepts scalar, (M,), or (M,1).
    """
    x_arr = np.asarray(x, dtype=float)

    if x_arr.ndim == 0:  # scalar
        return x_arr.reshape(1, 1)

    if x_arr.ndim == 1:  # (M,)
        return x_arr.reshape(-1, 1)

    if x_arr.ndim == 2 and x_arr.shape[1] == 1:  # (M,1)
        return x_arr

    raise ValueError(
        f"x must be a scalar, shape (M,), or shape (M,1). Got shape {x_arr.shape}."
    )


def Pi_a(x) -> np.ndarray:
    """
    Objective Πa(x) = x^2
    Returns shape (M,1).
    """
    xc = _as_col(x)
    return xc**2


def Pi_b(x) -> np.ndarray:
    """
    Objective Πb(x) = (x + (π/2) sin(x))^2
    Returns shape (M,1).
    """
    xc = _as_col(x)
    g = xc + (np.pi / 2.0) * np.sin(xc)
    return g**2


def grad_Pi_a(x) -> np.ndarray:
    """
    Gradient of Πa:
      dΠa/dx = 2x
    Returns shape (M,1).
    """
    xc = _as_col(x)
    return 2.0 * xc


def grad_Pi_b(x) -> np.ndarray:
    """
    Gradient of Πb:
      Let g(x) = x + (π/2) sin(x)
      Then Πb(x) = g(x)^2
      dΠb/dx = 2 g(x) g'(x)
      g'(x) = 1 + (π/2) cos(x)

    Returns shape (M,1).
    """
    xc = _as_col(x)
    g = xc + (np.pi / 2.0) * np.sin(xc)
    gp = 1.0 + (np.pi / 2.0) * np.cos(xc)
    return 2.0 * g * gp


def hess_Pi_a(x) -> np.ndarray:
    """
    Hessian of Πa:
      d^2Πa/dx^2 = 2

    Returns shape (M,M).
    For Project 1, M = 1 => [[2]].
    """
    xc = _as_col(x)
    M = xc.shape[0]
    return 2.0 * np.eye(M, dtype=float)


def hess_Pi_b(x) -> np.ndarray:
    """
    Hessian of Πb:
      With g(x) = x + (π/2) sin(x),
           g'(x) = 1 + (π/2) cos(x),
           g''(x) = -(π/2) sin(x),

      d^2Πb/dx^2 = 2[(g')^2 + g g''].

    Returns shape (M,M) as a diagonal matrix for elementwise inputs.
    For Project 1, M = 1 => [[d^2Πb/dx^2]].
    """
    xc = _as_col(x)
    g = xc + (np.pi / 2.0) * np.sin(xc)
    gp = 1.0 + (np.pi / 2.0) * np.cos(xc)
    gpp = -(np.pi / 2.0) * np.sin(xc)

    d2 = 2.0 * ((gp**2) + g * gpp)  # shape (M,1)
    return np.diag(d2.ravel()).astype(float)
