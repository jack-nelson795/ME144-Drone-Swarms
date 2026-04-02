from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .config import CONFIG, Project5Config


DESIGN_KEYS = ("k2", "mu2", "sig2E", "K2", "v2")


def _safe_div(numerator: float, denominator: float, eps: float) -> float:
    if abs(denominator) < eps:
        denominator = eps if denominator >= 0.0 else -eps
    return numerator / denominator


def clip_design(design: np.ndarray, cfg: Project5Config = CONFIG) -> np.ndarray:
    lower = np.array(
        [cfg.k2min, cfg.mu2min, cfg.sig2Emin, cfg.K2min, cfg.v2min],
        dtype=float,
    )
    upper = np.array(
        [cfg.k2max, cfg.mu2max, cfg.sig2Emax, cfg.K2max, cfg.v2max],
        dtype=float,
    )
    return np.clip(np.asarray(design, dtype=float), lower, upper)


def random_design(rng: np.random.Generator, cfg: Project5Config = CONFIG) -> np.ndarray:
    lower = np.array([cfg.k2min, cfg.mu2min, cfg.sig2Emin, cfg.K2min, cfg.v2min], dtype=float)
    upper = np.array([cfg.k2max, cfg.mu2max, cfg.sig2Emax, cfg.K2max, cfg.v2max], dtype=float)
    return rng.uniform(lower, upper)


def effective_from_bounds(lower: float, upper: float, cfg: Project5Config = CONFIG) -> float:
    # The brief uses gamma = 1/2, so this is the arithmetic mean of the HS bounds.
    return cfg.gamma * upper + (1.0 - cfg.gamma) * lower


def evaluate_design(design: np.ndarray, cfg: Project5Config = CONFIG) -> dict[str, Any]:
    design = clip_design(design, cfg)
    k2, mu2, sig2E, K2, v2 = design
    v1 = 1.0 - v2
    eps = cfg.eps

    # Eq. 5: electrical conductivity bounds
    sigE_min = cfg.sig1E + _safe_div(
        v2,
        _safe_div(1.0, sig2E - cfg.sig1E, eps) + _safe_div(v1, 3.0 * cfg.sig1E, eps),
        eps,
    )
    sigE_max = sig2E + _safe_div(
        v1,
        _safe_div(1.0, cfg.sig1E - sig2E, eps) + _safe_div(v2, 3.0 * sig2E, eps),
        eps,
    )

    # Eq. 6: thermal conductivity bounds
    K_min = cfg.K1 + _safe_div(
        v2,
        _safe_div(1.0, K2 - cfg.K1, eps) + _safe_div(v1, 3.0 * cfg.K1, eps),
        eps,
    )
    K_max = K2 + _safe_div(
        v1,
        _safe_div(1.0, cfg.K1 - K2, eps) + _safe_div(v2, 3.0 * K2, eps),
        eps,
    )

    # Eq. 3: bulk modulus bounds
    k_min = cfg.k1 + _safe_div(
        v2,
        _safe_div(1.0, k2 - cfg.k1, eps) + _safe_div(3.0 * v1, 3.0 * cfg.k1 + 4.0 * cfg.mu1, eps),
        eps,
    )
    k_max = k2 + _safe_div(
        v1,
        _safe_div(1.0, cfg.k1 - k2, eps) + _safe_div(3.0 * v2, 3.0 * k2 + 4.0 * mu2, eps),
        eps,
    )

    # Eq. 4: shear modulus bounds
    mu_min = cfg.mu1 + _safe_div(
        v2,
        _safe_div(1.0, mu2 - cfg.mu1, eps)
        + _safe_div(6.0 * v1 * (cfg.k1 + 2.0 * cfg.mu1), 5.0 * cfg.mu1 * (3.0 * cfg.k1 + 4.0 * cfg.mu1), eps),
        eps,
    )
    mu_max = mu2 + _safe_div(
        v1,
        _safe_div(1.0, cfg.mu1 - mu2, eps)
        + _safe_div(6.0 * v2 * (k2 + 2.0 * mu2), 5.0 * mu2 * (3.0 * k2 + 4.0 * mu2), eps),
        eps,
    )

    # Eq. 2 with gamma = 1/2, as instructed throughout the brief
    sigE_eff = effective_from_bounds(sigE_min, sigE_max, cfg)
    K_eff = effective_from_bounds(K_min, K_max, cfg)
    k_eff = effective_from_bounds(k_min, k_max, cfg)
    mu_eff = effective_from_bounds(mu_min, mu_max, cfg)

    # Eqs. 23-24: electrical load sharing
    sig_span = max(sig2E - cfg.sig1E, eps)
    CE1 = _safe_div(sig2E - sigE_eff, v1 * sig_span, eps)
    CE2 = _safe_div(sigE_eff - cfg.sig1E, v2 * sig_span, eps)
    CJ1 = _safe_div(cfg.sig1E, sigE_eff, eps) * CE1
    CJ2 = _safe_div(sig2E, sigE_eff, eps) * CE2
    CJ1CE1 = CE1 * CJ1
    CJ2CE2 = CE2 * CJ2

    # Eqs. 25-28: thermal load sharing
    Ct2 = _safe_div(K_eff - cfg.K1, v2 * (K2 - cfg.K1), eps)
    Ct1 = _safe_div(1.0 - v2 * Ct2, v1, eps)
    Cq2 = _safe_div(K2 * Ct2, K_eff, eps)
    Cq1 = _safe_div(1.0 - v2 * Cq2, v1, eps)

    # Eqs. 9 and 11: mechanical stress concentrations
    Ck2 = _safe_div(k2, k_eff, eps) * _safe_div(k_eff - cfg.k1, v2 * (k2 - cfg.k1), eps)
    Cmu2 = _safe_div(mu2, mu_eff, eps) * _safe_div(mu_eff - cfg.mu1, v2 * (mu2 - cfg.mu1), eps)
    Ck1 = _safe_div(1.0 - v2 * Ck2, v1, eps)
    Cmu1 = _safe_div(1.0 - v2 * Cmu2, v1, eps)

    # Eqs. 34-35, 37-38, 29-32: unilateral weights
    w2E_hat = cfg.wj if _safe_div(CJ1CE1 - cfg.TOL_sig, cfg.TOL_sig, eps) > 0.0 else 0.0
    w3E_hat = cfg.wj if _safe_div(CJ2CE2 - cfg.TOL_sig, cfg.TOL_sig, eps) > 0.0 else 0.0
    w2T_hat = cfg.wj if _safe_div(Cq1 - cfg.TOL_K, cfg.TOL_K, eps) > 0.0 else 0.0
    w3T_hat = cfg.wj if _safe_div(Cq2 - cfg.TOL_K, cfg.TOL_K, eps) > 0.0 else 0.0
    w3M_hat = cfg.wj if Ck2 > cfg.TOL_k else 0.0
    w4M_hat = cfg.wj if Cmu2 > cfg.TOL_mu else 0.0
    w5M_hat = cfg.wj if Ck1 > cfg.TOL_k else 0.0
    w6M_hat = cfg.wj if Cmu1 > cfg.TOL_mu else 0.0

    # Eqs. 33, 36, and mechanical objective on p.6
    electrical_match_term = cfg.w1 * (_safe_div(cfg.sigE_effD - sigE_eff, cfg.sigE_effD, eps) ** 2)
    electrical_penalty_phase1 = w2E_hat * (_safe_div(CJ1CE1 - cfg.TOL_sig, cfg.TOL_sig, eps) ** 2)
    electrical_penalty_phase2 = w3E_hat * (_safe_div(CJ2CE2 - cfg.TOL_sig, cfg.TOL_sig, eps) ** 2)
    Pi_elec = (
        electrical_match_term
        + electrical_penalty_phase1
        + electrical_penalty_phase2
    )
    thermal_match_term = cfg.w1 * (_safe_div(cfg.K_effD - K_eff, cfg.K_effD, eps) ** 2)
    thermal_penalty_phase1 = w2T_hat * (_safe_div(Cq1 - cfg.TOL_K, cfg.TOL_K, eps) ** 2)
    thermal_penalty_phase2 = w3T_hat * (_safe_div(Cq2 - cfg.TOL_K, cfg.TOL_K, eps) ** 2)
    Pi_thermo = (
        thermal_match_term
        + thermal_penalty_phase1
        + thermal_penalty_phase2
    )
    mechanical_match_bulk = cfg.w1 * (_safe_div(cfg.k_effD - k_eff, cfg.k_effD, eps) ** 2)
    mechanical_match_shear = cfg.w1 * (_safe_div(cfg.mu_effD - mu_eff, cfg.mu_effD, eps) ** 2)
    mechanical_penalty_bulk_phase2 = w3M_hat * (_safe_div(Ck2 - cfg.TOL_k, cfg.TOL_k, eps) ** 2)
    mechanical_penalty_shear_phase2 = w4M_hat * (_safe_div(Cmu2 - cfg.TOL_mu, cfg.TOL_mu, eps) ** 2)
    mechanical_penalty_bulk_phase1 = w5M_hat * (_safe_div(Ck1 - cfg.TOL_k, cfg.TOL_k, eps) ** 2)
    mechanical_penalty_shear_phase1 = w6M_hat * (_safe_div(Cmu1 - cfg.TOL_mu, cfg.TOL_mu, eps) ** 2)
    Pi_mech = (
        mechanical_match_bulk
        + mechanical_match_shear
        + mechanical_penalty_bulk_phase2
        + mechanical_penalty_shear_phase2
        + mechanical_penalty_bulk_phase1
        + mechanical_penalty_shear_phase1
    )
    cost = cfg.W1 * Pi_elec + cfg.W2 * Pi_thermo + cfg.W3 * Pi_mech

    return {
        "design": dict(zip(DESIGN_KEYS, design)),
        "bounds": {
            "sigE_min": sigE_min,
            "sigE_max": sigE_max,
            "K_min": K_min,
            "K_max": K_max,
            "k_min": k_min,
            "k_max": k_max,
            "mu_min": mu_min,
            "mu_max": mu_max,
        },
        "effective_properties": {
            "sigE_eff": sigE_eff,
            "K_eff": K_eff,
            "k_eff": k_eff,
            "mu_eff": mu_eff,
        },
        "concentration_factors": {
            "CJ1CE1": CJ1CE1,
            "CJ2CE2": CJ2CE2,
            "Ct1": Ct1,
            "Ct2": Ct2,
            "Cq1": Cq1,
            "Cq2": Cq2,
            "Ck1": Ck1,
            "Ck2": Ck2,
            "Cmu1": Cmu1,
            "Cmu2": Cmu2,
            "CE1": CE1,
            "CE2": CE2,
            "CJ1": CJ1,
            "CJ2": CJ2,
        },
        "weights": {
            "w2E_hat": w2E_hat,
            "w3E_hat": w3E_hat,
            "w2T_hat": w2T_hat,
            "w3T_hat": w3T_hat,
            "w3M_hat": w3M_hat,
            "w4M_hat": w4M_hat,
            "w5M_hat": w5M_hat,
            "w6M_hat": w6M_hat,
        },
        "cost_terms": {
            "Pi_elec": Pi_elec,
            "Pi_thermo": Pi_thermo,
            "Pi_mech": Pi_mech,
            "Pi_total": cost,
        },
        "cost_breakdown": {
            "electrical_match_term": electrical_match_term,
            "electrical_penalty_phase1": electrical_penalty_phase1,
            "electrical_penalty_phase2": electrical_penalty_phase2,
            "thermal_match_term": thermal_match_term,
            "thermal_penalty_phase1": thermal_penalty_phase1,
            "thermal_penalty_phase2": thermal_penalty_phase2,
            "mechanical_match_bulk": mechanical_match_bulk,
            "mechanical_match_shear": mechanical_match_shear,
            "mechanical_penalty_bulk_phase2": mechanical_penalty_bulk_phase2,
            "mechanical_penalty_shear_phase2": mechanical_penalty_shear_phase2,
            "mechanical_penalty_bulk_phase1": mechanical_penalty_bulk_phase1,
            "mechanical_penalty_shear_phase1": mechanical_penalty_shear_phase1,
        },
        "config": asdict(cfg),
    }


def fast_total_cost(design: np.ndarray, cfg: Project5Config = CONFIG) -> float:
    design = clip_design(design, cfg)
    k2, mu2, sig2E, K2, v2 = design
    v1 = 1.0 - v2
    eps = cfg.eps

    sigE_min = cfg.sig1E + _safe_div(
        v2,
        _safe_div(1.0, sig2E - cfg.sig1E, eps) + _safe_div(v1, 3.0 * cfg.sig1E, eps),
        eps,
    )
    sigE_max = sig2E + _safe_div(
        v1,
        _safe_div(1.0, cfg.sig1E - sig2E, eps) + _safe_div(v2, 3.0 * sig2E, eps),
        eps,
    )
    K_min = cfg.K1 + _safe_div(
        v2,
        _safe_div(1.0, K2 - cfg.K1, eps) + _safe_div(v1, 3.0 * cfg.K1, eps),
        eps,
    )
    K_max = K2 + _safe_div(
        v1,
        _safe_div(1.0, cfg.K1 - K2, eps) + _safe_div(v2, 3.0 * K2, eps),
        eps,
    )
    k_min = cfg.k1 + _safe_div(
        v2,
        _safe_div(1.0, k2 - cfg.k1, eps) + _safe_div(3.0 * v1, 3.0 * cfg.k1 + 4.0 * cfg.mu1, eps),
        eps,
    )
    k_max = k2 + _safe_div(
        v1,
        _safe_div(1.0, cfg.k1 - k2, eps) + _safe_div(3.0 * v2, 3.0 * k2 + 4.0 * mu2, eps),
        eps,
    )
    mu_min = cfg.mu1 + _safe_div(
        v2,
        _safe_div(1.0, mu2 - cfg.mu1, eps)
        + _safe_div(6.0 * v1 * (cfg.k1 + 2.0 * cfg.mu1), 5.0 * cfg.mu1 * (3.0 * cfg.k1 + 4.0 * cfg.mu1), eps),
        eps,
    )
    mu_max = mu2 + _safe_div(
        v1,
        _safe_div(1.0, cfg.mu1 - mu2, eps)
        + _safe_div(6.0 * v2 * (k2 + 2.0 * mu2), 5.0 * mu2 * (3.0 * k2 + 4.0 * mu2), eps),
        eps,
    )

    sigE_eff = effective_from_bounds(sigE_min, sigE_max, cfg)
    K_eff = effective_from_bounds(K_min, K_max, cfg)
    k_eff = effective_from_bounds(k_min, k_max, cfg)
    mu_eff = effective_from_bounds(mu_min, mu_max, cfg)

    sig_span = max(sig2E - cfg.sig1E, eps)
    CE1 = _safe_div(sig2E - sigE_eff, v1 * sig_span, eps)
    CE2 = _safe_div(sigE_eff - cfg.sig1E, v2 * sig_span, eps)
    CJ1 = _safe_div(cfg.sig1E, sigE_eff, eps) * CE1
    CJ2 = _safe_div(sig2E, sigE_eff, eps) * CE2
    CJ1CE1 = CE1 * CJ1
    CJ2CE2 = CE2 * CJ2

    Ct2 = _safe_div(K_eff - cfg.K1, v2 * (K2 - cfg.K1), eps)
    Cq2 = _safe_div(K2 * Ct2, K_eff, eps)
    Cq1 = _safe_div(1.0 - v2 * Cq2, v1, eps)

    Ck2 = _safe_div(k2, k_eff, eps) * _safe_div(k_eff - cfg.k1, v2 * (k2 - cfg.k1), eps)
    Cmu2 = _safe_div(mu2, mu_eff, eps) * _safe_div(mu_eff - cfg.mu1, v2 * (mu2 - cfg.mu1), eps)
    Ck1 = _safe_div(1.0 - v2 * Ck2, v1, eps)
    Cmu1 = _safe_div(1.0 - v2 * Cmu2, v1, eps)

    w2E_hat = cfg.wj if _safe_div(CJ1CE1 - cfg.TOL_sig, cfg.TOL_sig, eps) > 0.0 else 0.0
    w3E_hat = cfg.wj if _safe_div(CJ2CE2 - cfg.TOL_sig, cfg.TOL_sig, eps) > 0.0 else 0.0
    w2T_hat = cfg.wj if _safe_div(Cq1 - cfg.TOL_K, cfg.TOL_K, eps) > 0.0 else 0.0
    w3T_hat = cfg.wj if _safe_div(Cq2 - cfg.TOL_K, cfg.TOL_K, eps) > 0.0 else 0.0
    w3M_hat = cfg.wj if Ck2 > cfg.TOL_k else 0.0
    w4M_hat = cfg.wj if Cmu2 > cfg.TOL_mu else 0.0
    w5M_hat = cfg.wj if Ck1 > cfg.TOL_k else 0.0
    w6M_hat = cfg.wj if Cmu1 > cfg.TOL_mu else 0.0

    Pi_elec = (
        cfg.w1 * (_safe_div(cfg.sigE_effD - sigE_eff, cfg.sigE_effD, eps) ** 2)
        + w2E_hat * (_safe_div(CJ1CE1 - cfg.TOL_sig, cfg.TOL_sig, eps) ** 2)
        + w3E_hat * (_safe_div(CJ2CE2 - cfg.TOL_sig, cfg.TOL_sig, eps) ** 2)
    )
    Pi_thermo = (
        cfg.w1 * (_safe_div(cfg.K_effD - K_eff, cfg.K_effD, eps) ** 2)
        + w2T_hat * (_safe_div(Cq1 - cfg.TOL_K, cfg.TOL_K, eps) ** 2)
        + w3T_hat * (_safe_div(Cq2 - cfg.TOL_K, cfg.TOL_K, eps) ** 2)
    )
    Pi_mech = (
        cfg.w1 * (_safe_div(cfg.k_effD - k_eff, cfg.k_effD, eps) ** 2)
        + cfg.w1 * (_safe_div(cfg.mu_effD - mu_eff, cfg.mu_effD, eps) ** 2)
        + w3M_hat * (_safe_div(Ck2 - cfg.TOL_k, cfg.TOL_k, eps) ** 2)
        + w4M_hat * (_safe_div(Cmu2 - cfg.TOL_mu, cfg.TOL_mu, eps) ** 2)
        + w5M_hat * (_safe_div(Ck1 - cfg.TOL_k, cfg.TOL_k, eps) ** 2)
        + w6M_hat * (_safe_div(Cmu1 - cfg.TOL_mu, cfg.TOL_mu, eps) ** 2)
    )
    return float(cfg.W1 * Pi_elec + cfg.W2 * Pi_thermo + cfg.W3 * Pi_mech)


def total_cost(design: np.ndarray, cfg: Project5Config = CONFIG) -> float:
    return fast_total_cost(design, cfg)
