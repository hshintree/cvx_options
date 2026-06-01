"""
Portfolio optimizer: single-period solve, efficient frontier tracing, and mu/Sigma builder.
Migrated and fixed from log_norm.ipynb Cell 44.

Fixes applied:
  1. SCS fallback on every CLARABEL failure
  2. Max-return upper bound solved explicitly (with SCS fallback) instead of using ret_max * 0.98
  3. build_sigma_mu: min_obs guard — uses rf prior when window is too short
  4. trace_frontier: n_pts=80 for dense frontier; gaps skipped gracefully
  5. Annualisation: Sig_d * 252 (daily → annual), mu already annualised
"""
import numpy as np
import pandas as pd
import cvxpy as cp

from .config import R_ANN, MAX_WT, MAX_OPT_WT, SYMS
from .covariance import build_full_cov


def build_sigma_mu(
    univ: list,
    syms: list,
    rp_idx: dict,
    rets_df: pd.DataFrame,
    snap: pd.Timestamp,
    lookback: int = 63,
    rf: float = R_ANN,
    min_obs: int = 20,
) -> tuple | None:
    """
    Build annualised (cols, mu, Sigma) at a snapshot date using DGV covariance
    and delta-leverage expected returns.

    Parameters
    ----------
    univ     : asset universe (list of column names)
    syms     : underlying stock symbols
    rp_idx   : pre-indexed research panel {sym: df}
    rets_df  : wide returns DataFrame
    snap     : snapshot date
    lookback : rolling window (trading days)
    rf       : risk-free rate (annualised) — used as prior when history is short
    min_obs  : minimum observations before using historical mu; else use rf / 0

    Returns
    -------
    (cols, mu, Sigma_annual) or None on error.
    """
    cols = [c for c in univ if c in rets_df.columns]
    try:
        Sig_d, gn = build_full_cov(snap, rp_idx, rets_df[syms], syms, cols, lookback)
    except Exception as e:
        print(f"  DGV error for {snap}: {e}")
        return None

    Sig = Sig_d * 252  # daily → annualised

    end_l   = rets_df.index.get_indexer([snap], method="ffill")[0]
    start_l = max(0, end_l - lookback)
    win     = rets_df.iloc[start_l:end_l]

    mu = np.zeros(len(cols))
    for k, c in enumerate(cols):
        sym = c.split("_")[0]
        if c in syms:
            h = win[c].dropna()
            mu[k] = float(h.mean()) * 252 if len(h) >= min_obs else rf
        elif sym in gn:
            lev = gn[sym]["lev_c"] if "_call" in c else gn[sym]["lev_p"]
            h   = win[sym].dropna()
            mu_und = float(h.mean()) * 252 if len(h) >= min_obs else rf
            mu[k]  = lev * mu_und

    return cols, mu, Sig


def solve_portfolio(
    cols: list,
    mu: np.ndarray,
    Sig: np.ndarray,
    target_ret: float | None = None,
    gamma: float | None = None,
    max_wt: float = MAX_WT,
    max_opt_wt: float = MAX_OPT_WT,
    rf: float = R_ANN,
) -> tuple[np.ndarray | None, float | None, float | None]:
    """
    Solve a single portfolio optimization.

    Modes (mutually exclusive priority):
      gamma is not None      → max (mu-rf)·w − γ·w'Σw   (mean-variance)
      target_ret is not None → min w'Σw  s.t. μ·w ≥ target_ret  (efficient frontier point)
      both None              → min w'Σw  (global minimum variance)

    Returns (weights, ann_return, ann_vol) — all None if infeasible.
    """
    n    = len(cols)
    w    = cp.Variable(n, nonneg=True)
    base = [cp.sum(w) == 1, w <= max_wt] + [
        w[k] <= max_opt_wt
        for k, c in enumerate(cols) if "_call" in c or "_put" in c
    ]

    if gamma is not None:
        prob = cp.Problem(cp.Maximize((mu - rf) @ w - gamma * cp.quad_form(w, Sig)), base)
    elif target_ret is not None:
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sig)), base + [mu @ w >= target_ret])
    else:
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sig)), base)

    prob.solve(solver=cp.CLARABEL, warm_start=True)
    if w.value is None:
        prob.solve(solver=cp.SCS)
    if w.value is None:
        return None, None, None

    wv  = np.clip(np.asarray(w.value).flatten(), 0, 1)
    wv /= wv.sum()
    vol = float(np.sqrt(max(wv @ Sig @ wv, 0.0)))
    ret = float(mu @ wv)
    return wv, ret, vol


def _max_feasible_return(cols, mu, Sig, max_wt, max_opt_wt):
    """Solve max μ·w s.t. constraints to find the upper bound for the frontier."""
    n = len(cols)
    w = cp.Variable(n, nonneg=True)
    base = [cp.sum(w) == 1, w <= max_wt] + [
        w[k] <= max_opt_wt
        for k, c in enumerate(cols) if "_call" in c or "_put" in c
    ]
    pb = cp.Problem(cp.Maximize(mu @ w), base)
    pb.solve(solver=cp.CLARABEL)
    if w.value is None:
        pb.solve(solver=cp.SCS)
    if w.value is not None:
        wv = np.clip(np.asarray(w.value).flatten(), 0, 1)
        return float(mu @ wv)
    return float(mu.max())


def trace_frontier(
    cols: list,
    mu: np.ndarray,
    Sig: np.ndarray,
    n_pts: int = 80,
    max_wt: float = MAX_WT,
    max_opt_wt: float = MAX_OPT_WT,
) -> tuple[list[float], list[float]]:
    """
    Trace the efficient frontier from the global minimum-variance point to
    the maximum feasible return.

    Returns (vols, rets) — lists of (vol, return) pairs along the frontier.
    Points where the solver fails are silently skipped (not None-padded).
    """
    _, ret_mv, vol_mv = solve_portfolio(cols, mu, Sig, max_wt=max_wt, max_opt_wt=max_opt_wt)
    if ret_mv is None:
        return [], []

    ret_max = _max_feasible_return(cols, mu, Sig, max_wt, max_opt_wt)
    # Slightly below max to stay feasible
    ret_max_safe = ret_mv + 0.99 * (ret_max - ret_mv)

    vols_ef = [vol_mv]
    rets_ef = [ret_mv]

    for tgt in np.linspace(ret_mv, ret_max_safe, n_pts - 1):
        _, r, v = solve_portfolio(cols, mu, Sig, target_ret=tgt,
                                  max_wt=max_wt, max_opt_wt=max_opt_wt)
        if r is not None:
            vols_ef.append(v)
            rets_ef.append(r)

    return vols_ef, rets_ef


def ablation_table(
    cols: list,
    mu: np.ndarray,
    Sig: np.ndarray,
    rf: float = R_ANN,
    max_wt: float = MAX_WT,
    max_opt_wt: float = MAX_OPT_WT,
) -> dict:
    """
    Solve multiple portfolio objectives for the same (cols, mu, Sig) and
    return a dict of {label: {w, ret, vol, sharpe}}.
    """
    specs = [
        ("Min-Var",             dict()),
        ("Min-Var + 5% floor",  dict(target_ret=0.05)),
        ("MV  γ=1",             dict(gamma=1.0)),
        ("MV  γ=5",             dict(gamma=5.0)),
        ("MV  γ=20",            dict(gamma=20.0)),
    ]
    results = {}
    for lbl, kw in specs:
        wv, r, v = solve_portfolio(cols, mu, Sig, rf=rf,
                                   max_wt=max_wt, max_opt_wt=max_opt_wt, **kw)
        if wv is not None:
            results[lbl] = dict(w=wv, ret=r, vol=v,
                                sharpe=(r - rf) / v if v > 1e-10 else np.nan)

    # Equal-weight
    wv_ew = np.ones(len(cols)) / len(cols)
    r_ew  = float(mu @ wv_ew)
    v_ew  = float(np.sqrt(max(wv_ew @ Sig @ wv_ew, 0.0)))
    results["Equal-Weight"] = dict(w=wv_ew, ret=r_ew, vol=v_ew,
                                   sharpe=(r_ew - rf) / v_ew if v_ew > 1e-10 else np.nan)

    # Max-Sharpe approximation: scan frontier
    vfs, rfs = trace_frontier(cols, mu, Sig, n_pts=80,
                              max_wt=max_wt, max_opt_wt=max_opt_wt)
    if vfs:
        sharpes = [(r - rf) / v if v > 1e-10 else -np.inf for r, v in zip(rfs, vfs)]
        i_ms    = int(np.argmax(sharpes))
        _, r_ms, v_ms = solve_portfolio(cols, mu, Sig, target_ret=rfs[i_ms],
                                        max_wt=max_wt, max_opt_wt=max_opt_wt)
        if r_ms is not None:
            results["Max-Sharpe*"] = dict(
                w=None, ret=r_ms, vol=v_ms,
                sharpe=(r_ms - rf) / v_ms if v_ms > 1e-10 else np.nan,
            )

    return results
