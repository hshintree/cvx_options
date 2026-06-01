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
from typing import Optional, Tuple

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
    cvar_lambda_rel: Optional[float] = None,
    cvar_lambda_abs: float = 0.08,
    max_option_weight: float = 0.05,
    max_put_weight: Optional[float] = None,
    max_spy_weight: float = 1.0,
    min_cash_weight: float = 0.0,
    max_turnover: float = 0.25,
    tcost_rate: float = 0.0005,
    use_rel_cvar: bool = True,
    tracking_error_cap: Optional[float] = None,
    min_put_weight: float = 0.0,
    track_eps: float = -0.0005,
) -> Tuple[np.ndarray, dict]:
    """
    Solve the scenario-based portfolio optimization.
    Supports single-asset (4 assets: eq, call, put, cash) or multi-asset
    (3*K+1: eq1, call1, put1, ..., eqK, callK, putK, cash).

    Expected return uses mu_phys (physical/realized); CVaR from scenarios R.

    Returns
    -------
    w_opt : (n_assets,) optimal weights
    diag  : dict with cvar_model, cvar_empirical, mean_scenario_port,
            expected_ret_phys, solver_status, VaR_loss_rel_threshold, VaR_loss_abs_threshold
    """
    N, n_assets = R.shape
    assert n_assets >= 4 and (n_assets - 1) % 3 == 0, (
        f"Expected 4 or 3*K+1 assets (e.g. 4, 7, 10), got {n_assets}"
    )
    assert w_prev.shape == (n_assets,), f"Bad w_prev shape: {w_prev.shape}"
    assert mu_phys.shape == (n_assets,), f"Bad mu_phys shape: {mu_phys.shape}"
    assert not np.any(np.isnan(R)), "NaN in R"
    assert not np.any(np.isinf(R)), "Inf in R"
    assert not np.any(np.isnan(mu_phys)), "NaN in mu_phys"
    assert not np.any(np.isinf(mu_phys)), "Inf in mu_phys"

    # Sanity check: if all scenario returns are extreme, warn
    if np.any(np.abs(R) > 10.0):
        logger.warning("Extreme returns detected in R: min=%.4f, max=%.4f", R.min(), R.max())

    w = cp.Variable(n_assets)

    # Portfolio return per scenario: (N,)
    port_rets = R @ w

    # Benchmark per scenario: single-asset = equity sleeve; multi = equal-weight equities
    n_sleeves = (n_assets - 1) // 3
    if n_assets == 4:
        bench = R[:, 0]
    else:
        eq_indices = [3 * i for i in range(n_sleeves)]
        bench = np.mean(R[:, eq_indices], axis=1)

    # Relative return: portfolio - benchmark (so "going to cash" doesn't dominate)
    rel_rets = port_rets - bench

    # Expected return: physical mean (unchanged)
    expected_ret = mu_phys @ w

    lambda_rel = cvar_lambda_rel if cvar_lambda_rel is not None else cvar_lambda
    if not use_rel_cvar:
        lambda_rel = 0.0
    lambda_abs = cvar_lambda_abs

    # CVaR on relative loss: loss_rel_i = -rel_rets_i (Rockafellar–Uryasev)
    t_var_rel = cp.Variable()
    u_rel = cp.Variable(N)
    cvar_rel = t_var_rel + cp.sum(u_rel) / (N * (1.0 - cvar_alpha))

    # CVaR on absolute loss: loss_abs_i = -port_rets_i
    t_var_abs = cp.Variable()
    u_abs = cp.Variable(N)
    cvar_abs = t_var_abs + cp.sum(u_abs) / (N * (1.0 - cvar_alpha))

    # Turnover
    turnover = cp.norm(w - w_prev, 1)
    tcost = tcost_rate * turnover

    # Mild regularization to prevent knife-edge CVaR solutions
    reg_weight = 1e-4
    regularization = reg_weight * cp.sum_squares(w)

    # Objective: expected_ret - λ_rel*CVaR(rel) - λ_abs*CVaR(abs) - tcost - reg
    objective = cp.Maximize(
        expected_ret - lambda_rel * cvar_rel - lambda_abs * cvar_abs - tcost - regularization
    )

    # Per-asset upper bounds: [eq, call, put] per underlying, then cash
    max_put = max_put_weight if max_put_weight is not None else max_option_weight
    n_sleeves = (n_assets - 1) // 3
    w_upper = np.zeros(n_assets)
    for i in range(n_sleeves):
        w_upper[3 * i] = max_spy_weight
        w_upper[3 * i + 1] = max_option_weight
        w_upper[3 * i + 2] = max_put
    w_upper[-1] = 1.0

    constraints = [
        w >= 0,
        w <= w_upper,
        cp.sum(w) == 1,
        w[-1] >= min_cash_weight,
        turnover <= max_turnover,
        u_rel >= -rel_rets - t_var_rel,
        u_rel >= 0,
        u_abs >= -port_rets - t_var_abs,
        u_abs >= 0,
    ]
    constraints.append(cp.sum(rel_rets) / N >= track_eps)
    # Per-sleeve call+put cap
    for i in range(n_sleeves):
        constraints.append(w[3 * i + 1] + w[3 * i + 2] <= max_option_weight)

    if tracking_error_cap is not None and tracking_error_cap > 0:
        constraints.append(cp.norm(rel_rets, 2) / cp.sqrt(N) <= tracking_error_cap)

    if min_put_weight > 0:
        if n_assets == 4:
            constraints.append(w[2] >= min_put_weight)
        else:
            for i in range(n_sleeves):
                constraints.append(w[3 * i + 2] >= min_put_weight)

    prob = cp.Problem(objective, constraints)

    # Prefer ECOS (LP), then Clarabel (conic), then SCS (fallback)
    # ECOS and CLARABEL are better for LP/SOCP than SCS
    solver_used = None
    solver_list = [cp.ECOS]
    
    # Try CLARABEL if available (often best for conic problems)
    try:
        # Check if CLARABEL solver is available in cvxpy
        _ = cp.CLARABEL
        solver_list.append(cp.CLARABEL)
    except (AttributeError, NameError):
        pass
    
    # SCS as last resort
    solver_list.append(cp.SCS)
    
    for solver in solver_list:
        try:
            solver_name = getattr(solver, "__name__", str(solver))
            
            # ECOS-specific settings
            if solver == cp.ECOS:
                prob.solve(solver=solver, verbose=False, max_iters=10000, abstol=1e-7, reltol=1e-7)
            # CLARABEL-specific settings
            elif solver == cp.CLARABEL:
                prob.solve(solver=solver, verbose=False, max_iter=10000, time_limit=30.0)
            # SCS settings
            else:
                prob.solve(solver=solver, verbose=False, max_iters=10000, eps=1e-6)
            
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                solver_used = solver_name
                break
            elif prob.status is None:
                # Try with verbose to see what's wrong
                logger.debug("Solver %s returned status=None, retrying with verbose", solver_name)
                try:
                    if solver == cp.ECOS:
                        prob.solve(solver=solver, verbose=True, max_iters=10000, abstol=1e-7, reltol=1e-7)
                    elif solver == cp.CLARABEL:
                        prob.solve(solver=solver, verbose=True, max_iter=10000, time_limit=30.0)
                    else:
                        prob.solve(solver=solver, verbose=True, max_iters=10000, eps=1e-6)
                    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                        solver_used = solver_name
                        break
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Solver %s exception: %s", getattr(solver, "__name__", solver), e)
            continue

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        w_opt = np.maximum(w.value, 0.0)
        w_opt /= w_opt.sum()

        # ---- Diagnostics (relative + absolute CVaR) ----
        port_rets_opt = R @ w_opt
        rel_rets_opt = port_rets_opt - bench
        losses_rel = -rel_rets_opt
        losses_abs = -port_rets_opt

        cvar_rel_model = float(t_var_rel.value + np.sum(u_rel.value) / (N * (1.0 - cvar_alpha)))
        var_rel_pct = np.percentile(losses_rel, cvar_alpha * 100)
        tail_rel = losses_rel >= var_rel_pct
        cvar_rel_empirical = float(np.mean(losses_rel[tail_rel])) if tail_rel.any() else float(var_rel_pct)

        cvar_abs_model = float(t_var_abs.value + np.sum(u_abs.value) / (N * (1.0 - cvar_alpha)))
        var_abs_pct = np.percentile(losses_abs, cvar_alpha * 100)
        tail_abs = losses_abs >= var_abs_pct
        cvar_abs_empirical = float(np.mean(losses_abs[tail_abs])) if tail_abs.any() else float(var_abs_pct)

        mean_port = float(np.mean(port_rets_opt))
        mean_bench = float(np.mean(bench))
        mean_rel = float(np.mean(rel_rets_opt))
        diag = {
            "cvar_model": cvar_rel_model,
            "cvar_empirical": cvar_rel_empirical,
            "cvar_relative_model": cvar_rel_model,
            "cvar_relative_empirical": cvar_rel_empirical,
            "cvar_abs_model": cvar_abs_model,
            "cvar_abs_empirical": cvar_abs_empirical,
            "VaR_loss_rel_threshold": float(t_var_rel.value),
            "VaR_loss_abs_threshold": float(t_var_abs.value),
            "mean_scenario_port": mean_port,
            "mean_port": mean_port,
            "bench_mean": mean_bench,
            "mean_bench": mean_bench,
            "mean_rel_scenario_port": mean_rel,
            "mean_rel": mean_rel,
            "expected_ret_phys": float(mu_phys @ w_opt),
            "solver_status": prob.status,
            "solver_used": solver_used,
            "objective_value": float(prob.value) if prob.value is not None else None,
        }
        return w_opt, diag

    # Solver failed: log diagnostics and return previous weights
    logger.warning(
        "Scenario CVaR solver failed: status=%s, solver=%s, R shape=%s, n_assets=%d",
        prob.status, solver_used, R.shape, n_assets,
    )
    
    # Check for common issues
    if prob.status == "infeasible":
        logger.warning("Problem is infeasible — constraints may be too tight (turnover=%s, min_cash=%s)",
                      max_turnover, min_cash_weight)
    elif prob.status == "unbounded":
        logger.warning("Problem is unbounded — check objective (mu_phys may have extreme values)")
    elif prob.status is None:
        logger.warning("Solver did not complete — possible numerical issues or timeout")
    
    diag = {
        "cvar_model": None, "cvar_empirical": None,
        "cvar_relative_model": None, "cvar_relative_empirical": None,
        "cvar_abs_model": None, "cvar_abs_empirical": None,
        "VaR_loss_rel_threshold": None, "VaR_loss_abs_threshold": None,
        "mean_scenario_port": None,
        "bench_mean": None, "mean_rel_scenario_port": None,
        "expected_ret_phys": None,
        "solver_status": prob.status, "solver_used": solver_used,
        "objective_value": None,
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
    cvar_lambda_rel: Optional[float] = None,
    cvar_lambda_abs: float = 0.08,
    max_option_weight: float = 0.05,
    max_put_weight: Optional[float] = None,
    max_spy_weight: float = 1.0,
    min_cash_weight: float = 0.0,
    max_turnover: float = 0.25,
    tcost_rate: float = 0.0005,
    use_rel_cvar: bool = True,
    tracking_error_cap: Optional[float] = None,
    min_put_weight: float = 0.0,
    track_eps: float = -0.0005,
) -> Tuple[np.ndarray, dict]:
    """
    Thin wrapper: scenario matrix R + physical mu_phys; returns (w_opt, diag).
    """
    return solve_scenario_cvar(
        R, w_prev, mu_phys,
        cvar_alpha=cvar_alpha,
        cvar_lambda=cvar_lambda,
        cvar_lambda_rel=cvar_lambda_rel,
        cvar_lambda_abs=cvar_lambda_abs,
        max_option_weight=max_option_weight,
        max_put_weight=max_put_weight,
        max_spy_weight=max_spy_weight,
        min_cash_weight=min_cash_weight,
        max_turnover=max_turnover,
        tcost_rate=tcost_rate,
        use_rel_cvar=use_rel_cvar,
        tracking_error_cap=tracking_error_cap,
        min_put_weight=min_put_weight,
        track_eps=track_eps,
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
    # w_prev equity-heavy so track_eps and turnover are feasible
    w_prev = np.array([0.9, 0.0, 0.0, 0.1])
    mu_phys = np.array([0.005, 0.0, 0.0, 0.001])

    w_opt, diag = solve_scenario_cvar(R_fake, w_prev, mu_phys, track_eps=-0.0005)
    print(f"Optimal weights: SPY={w_opt[0]:.3f}  CALL={w_opt[1]:.3f}  "
          f"PUT={w_opt[2]:.3f}  CASH={w_opt[3]:.3f}")
    print(f"  Sum: {w_opt.sum():.6f}")
    print(f"  E[port] (physical): {mu_phys @ w_opt:.4f}")
    if diag.get("solver_status") in ("optimal", "optimal_inaccurate"):
        print(f"  cvar_model: {diag['cvar_model']:.4f}  "
              f"cvar_empirical: {diag['cvar_empirical']:.4f}")
        print(f"  VaR_loss_rel_threshold: {diag.get('VaR_loss_rel_threshold')}  "
              f"VaR_loss_abs_threshold: {diag.get('VaR_loss_abs_threshold')}")
        print(f"  mean_scenario_port: {diag['mean_scenario_port']:.4f}  "
              f"solver: {diag['solver_status']}")
    else:
        print(f"  solver_status: {diag.get('solver_status')}  (no diag values)")
