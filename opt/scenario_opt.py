"""
Scenario-based portfolio optimizer using CVaR risk control.

Instead of mean-variance (Markowitz), this optimizer works directly with the
(n_samples, 4) return matrix R and maximizes expected return subject to a
CVaR constraint, with the same turnover / transaction cost penalties as the
Markowitz solver.

Formulation
-----------
Given R ∈ R^{N×4}  (N scenarios, 4 assets),
portfolio return for scenario i:  ρ_i = R_i · w

    maximize  E[ρ] − λ · CVaR_α(−ρ) − τ · ||w − w_prev||_1

    subject to:
        w ≥ 0,  Σw = 1              (long-only, fully invested)
        w ≤ w_upper                  (per-asset caps)
        w[cash] ≥ min_cash           (cash floor)
        ||w − w_prev||_1 ≤ max_to   (turnover cap)

CVaR is linearized via the Rockafellar–Uryasev auxiliary formulation:

    CVaR_α(loss) ≤ t + (1 / N(1−α)) · Σ u_i
    where  u_i ≥ loss_i − t,  u_i ≥ 0

This is a linear program — fast and reliable with standard LP/SOCP solvers.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import cvxpy as cp
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


def solve_scenario_cvar(
    R: np.ndarray,
    w_prev: np.ndarray,
    mu_phys: np.ndarray,
    *,
    cvar_alpha: float = 0.95,
    cvar_lambda: float = 0.25,
    max_option_weight: float = 0.05,
    max_spy_weight: float = 1.0,
    min_cash_weight: float = 0.0,
    max_turnover: float = 0.25,
    tcost_rate: float = 0.0005,
) -> Tuple[np.ndarray, dict]:
    """
    Solve the scenario-based portfolio optimization.

    Expected return uses mu_phys (physical/realized); CVaR from scenarios R.

    Returns
    -------
    w_opt : (4,) optimal weights
    diag  : dict with cvar_model, cvar_empirical, mean_scenario_port,
            expected_ret_phys, solver_status, VaR_threshold
    """
    N, n_assets = R.shape
    assert n_assets == 4, f"Expected 4 assets, got {n_assets}"
    assert w_prev.shape == (4,), f"Bad w_prev shape: {w_prev.shape}"
    assert mu_phys.shape == (4,), f"Bad mu_phys shape: {mu_phys.shape}"
    assert not np.any(np.isnan(R)), "NaN in R"

    w = cp.Variable(n_assets)

    # Portfolio return per scenario: (N,) — used only for CVaR
    port_rets = R @ w  # (N,)

    # Expected return: physical mean (decoupled from risk-neutral scenarios)
    expected_ret = mu_phys @ w

    # CVaR via Rockafellar–Uryasev: loss_i = -port_rets_i
    t_var = cp.Variable()       # VaR threshold
    u = cp.Variable(N)          # auxiliary (excess loss above VaR)

    # CVaR ≤ t + mean(u) / (1 - alpha)
    cvar = t_var + cp.sum(u) / (N * (1.0 - cvar_alpha))

    # Turnover
    turnover = cp.norm(w - w_prev, 1)
    tcost = tcost_rate * turnover

    # Objective: maximize E[return] - lambda * CVaR - tcost
    objective = cp.Maximize(expected_ret - cvar_lambda * cvar - tcost)

    # Weight constraints
    w_upper = np.array([max_spy_weight, max_option_weight, max_option_weight, 1.0])
    constraints = [
        w >= 0,
        w <= w_upper,
        cp.sum(w) == 1,
        w[3] >= min_cash_weight,
        turnover <= max_turnover,
        u >= -port_rets - t_var,
        u >= 0,
    ]

    prob = cp.Problem(objective, constraints)

    # Prefer ECOS (LP), then Clarabel, then SCS
    for solver in (cp.ECOS, cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False, max_iters=5000)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                break
        except Exception as e:
            logger.debug("Solver %s failed: %s", getattr(solver, "__name__", solver), e)
            continue

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        w_opt = np.maximum(w.value, 0.0)
        w_opt /= w_opt.sum()

        # ---- Diagnostics ----
        # CVaR from model variables (Rockafellar–Uryasev t, u)
        cvar_model = float(t_var.value + np.sum(u.value) / (N * (1.0 - cvar_alpha)))

        # CVaR empirical (sanity check from portfolio losses)
        port_rets_opt = R @ w_opt
        losses = -port_rets_opt
        var_pct = np.percentile(losses, cvar_alpha * 100)
        tail_mask = losses >= var_pct
        cvar_empirical = float(np.mean(losses[tail_mask])) if tail_mask.any() else float(var_pct)

        diag = {
            "cvar_model": cvar_model,
            "cvar_empirical": cvar_empirical,
            "VaR_threshold": float(t_var.value),
            "mean_scenario_port": float(np.mean(port_rets_opt)),
            "expected_ret_phys": float(mu_phys @ w_opt),
            "solver_status": prob.status,
            "objective_value": float(prob.value) if prob.value is not None else None,
        }
        return w_opt, diag

    logger.warning("Scenario CVaR solver status: %s — returning previous weights", prob.status)
    diag = {
        "cvar_model": None, "cvar_empirical": None, "VaR_threshold": None,
        "mean_scenario_port": None, "expected_ret_phys": None,
        "solver_status": prob.status, "objective_value": None,
    }
    return w_prev.copy(), diag


# ---------------------------------------------------------------------------
# Convenience: solve with same interface as _solve_markowitz
# ---------------------------------------------------------------------------

def solve_scenario(
    R: np.ndarray,
    w_prev: np.ndarray,
    mu_phys: np.ndarray,
    cvar_alpha: float = 0.95,
    cvar_lambda: float = 0.25,
    max_option_weight: float = 0.05,
    max_spy_weight: float = 1.0,
    min_cash_weight: float = 0.0,
    max_turnover: float = 0.25,
    tcost_rate: float = 0.0005,
) -> Tuple[np.ndarray, dict]:
    """
    Thin wrapper: scenario matrix R + physical mu_phys; returns (w_opt, diag).
    """
    return solve_scenario_cvar(
        R, w_prev, mu_phys,
        cvar_alpha=cvar_alpha,
        cvar_lambda=cvar_lambda,
        max_option_weight=max_option_weight,
        max_spy_weight=max_spy_weight,
        min_cash_weight=min_cash_weight,
        max_turnover=max_turnover,
        tcost_rate=tcost_rate,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rng = np.random.default_rng(42)
    N = 5000
    R_fake = np.column_stack([
        rng.normal(0.005, 0.03, N),    # SPY
        rng.normal(0.01, 0.10, N),     # CALL
        rng.normal(-0.005, 0.10, N),   # PUT
        np.full(N, 0.001),             # CASH
    ])
    w_prev = np.array([0.6, 0.0, 0.0, 0.4])
    mu_phys = np.array([0.005, 0.0, 0.0, 0.001])

    w_opt, diag = solve_scenario_cvar(R_fake, w_prev, mu_phys)
    print(f"Optimal weights: SPY={w_opt[0]:.3f}  CALL={w_opt[1]:.3f}  "
          f"PUT={w_opt[2]:.3f}  CASH={w_opt[3]:.3f}")
    print(f"  Sum: {w_opt.sum():.6f}")
    print(f"  E[port] (physical): {mu_phys @ w_opt:.4f}")
    print(f"  cvar_model: {diag['cvar_model']:.4f}  "
          f"cvar_empirical: {diag['cvar_empirical']:.4f}")
    print(f"  mean_scenario_port: {diag['mean_scenario_port']:.4f}  "
          f"solver: {diag['solver_status']}")
