from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Project5Config:
    """Project constants taken directly from the assignment brief."""

    # Genetic algorithm parameters
    P: int = 10
    K: int = 10
    G: int = 5000
    S: int = 200
    dv: int = 5

    # Phase 1 properties
    k1: float = 90.0e9
    mu1: float = 20.0e9
    sig1E: float = 1.1e7
    K1: float = 4.5

    # Desired effective properties
    k_effD: float = 125.0e9
    mu_effD: float = 50.0e9
    sigE_effD: float = 2.2e7
    K_effD: float = 7.5

    # Tolerances
    TOL_k: float = 0.5
    TOL_mu: float = 0.5
    TOL_K: float = 0.5
    TOL_sig: float = 0.8

    # Search bounds for phase 2
    k2min: float = 90.0e9
    k2max: float = 900.0e9
    mu2min: float = 20.0e9
    mu2max: float = 200.0e9
    K2min: float = 4.5
    K2max: float = 45.0
    sig2Emin: float = 1.1e7
    sig2Emax: float = 1.1e8
    v2min: float = 0.0
    v2max: float = 2.0 / 3.0

    # Cost weights
    W1: float = 1.0 / 3.0
    W2: float = 1.0 / 3.0
    W3: float = 1.0 / 3.0
    w1: float = 1.0
    wj: float = 0.5
    gamma: float = 0.5

    # Case C mutation interval
    mutation_a: float = -0.5
    mutation_b: float = 1.5

    # Numerical safety / reproducibility
    eps: float = 1.0e-12
    seed: int = 144


CONFIG = Project5Config()

