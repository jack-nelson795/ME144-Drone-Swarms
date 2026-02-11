# me144_toolbox/optimization/newton.py
"""
ME144/244 Spring 2026 — Newton's Method (Project 1)

Implements:
    myNewton(f, df, x0, TOL, maxit) -> (sol, its, hist)

Per project spec:
- f  : function, input (M,1) -> output (M,1)   [gradient of Π]
- df : function, input (M,1) -> output (M,M)   [Hessian of Π]
- x0 : initial guess, (M,1) (Project 1 has M=1, but code supports M>=1)
- TOL: scalar, maximum allowable distance of f(sol) from 0
- maxit: scalar, max iterations

Outputs:
- sol : (M,1) final iterate
- its : number of iterations performed
- hist: (M, its+1) where hist[:, i] = x_i flattened to length M
        (hist[:, 0] corresponds to x0)
"""

from __future__ import annotations

import numpy as np


def _as_col(x) -> np.ndarray:
    """Convert x to a float ndarray of shape (M,1)."""
    x_arr = np.asarray(x, dtype=float)

    if x_arr.ndim == 0:  # scalar
        return x_arr.reshape(1, 1)
    if x_arr.ndim == 1:  # (M,)
        return x_arr.reshape(-1, 1)
    if x_arr.ndim == 2 and x_arr.shape[1] == 1:  # (M,1)
        return x_arr

    raise ValueError(
        f"x0 must be a scalar, shape (M,), or shape (M,1). Got shape {x_arr.shape}."
    )


def myNewton(f, df, x0, TOL, maxit):
    """
    Newton iteration to solve f(x) = 0.

    Stopping criterion:
        ||f(x)||_2 <= TOL

    Returns:
        sol  : (M,1)
        its  : int
        hist : (M, its+1)
    """
    if maxit < 0:
        raise ValueError("maxit must be nonnegative.")
    if TOL <= 0:
        raise ValueError("TOL must be positive.")

    x = _as_col(x0)
    M = x.shape[0]

    # Preallocate history for speed; we will trim before returning.
    hist = np.zeros((M, maxit + 1), dtype=float)
    hist[:, 0] = x.ravel()

    its = 0
    for k in range(maxit):
        fx = np.asarray(f(x), dtype=float).reshape(M, 1)

        # Check convergence based on distance of f(x) from 0.
        if np.linalg.norm(fx, ord=2) <= TOL:
            its = k
            return (x, its, hist[:, : its + 1])

        J = np.asarray(df(x), dtype=float)
        if J.shape != (M, M):
            raise ValueError(
                f"df(x) must return shape ({M},{M}); got {J.shape}."
            )

        # Solve J * delta = f(x), then x_{k+1} = x_k - delta
        try:
            delta = np.linalg.solve(J, fx)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Newton failed: Hessian/df(x) was singular or ill-conditioned."
            ) from e

        x = x - delta
        hist[:, k + 1] = x.ravel()

    # If we exit the loop, we used maxit iterations without meeting tolerance.
    its = maxit
    return (x, its, hist[:, : its + 1])
