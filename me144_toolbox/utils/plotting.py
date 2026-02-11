# me144_toolbox/utils/plotting.py
"""
Plotting utilities for ME144 projects.

Project 1 (Newton) plots:
- Objectives Πa, Πb over a domain
- Gradients dΠ/dx over a domain
- Hessians d²Π/dx² over a domain
- Convergence plots: Π(hist) vs iteration for different x0

Project 1 (GA) plots:
- Best and mean cost vs generation (semilog)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_two_curves(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    label1: str,
    label2: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
) -> None:
    """
    Plot two curves on the same axes and save.
    """
    _ensure_parent_dir(save_path)

    plt.figure()
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_convergence_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    semilogy: bool = True,
) -> None:
    """
    Plot multiple convergence curves.
    curves: dict of name -> (iters, values)
    """
    _ensure_parent_dir(save_path)

    plt.figure()
    for name, (iters, values) in curves.items():
        if semilogy:
            plt.semilogy(iters, values, marker="o", label=name)
        else:
            plt.plot(iters, values, marker="o", label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_ga_best_mean(
    best: np.ndarray,
    mean: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    semilogy: bool = True,
) -> None:
    """
    Plot GA best and mean cost per generation.

    Parameters
    ----------
    best : array-like, shape (G_run,)
        Best cost per generation.
    mean : array-like, shape (G_run,)
        Mean cost per generation.
    title, xlabel, ylabel : str
        Plot labels.
    save_path : Path
        Where to save the plot.
    semilogy : bool
        If True, semilog-y plot; otherwise linear.
    """
    _ensure_parent_dir(save_path)

    best = np.asarray(best, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()
    gens = np.arange(best.size)

    plt.figure()
    if semilogy:
        plt.semilogy(gens, best, marker="o", label="best cost")
        plt.semilogy(gens, mean, marker="o", label="mean cost")
    else:
        plt.plot(gens, best, marker="o", label="best cost")
        plt.plot(gens, mean, marker="o", label="mean cost")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
