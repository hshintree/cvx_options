"""
Walk-forward portfolio backtest engine.

Supports:
  cov_method    = "sample"      — plain rolling sample covariance (original, baseline)
                  "ledoit_wolf" — Ledoit-Wolf shrinkage (recommended)
                  "ewma"        — exponentially-weighted covariance (recommended)

  mu_method     = "zero"        — zero expected returns (pure min-var; best when μ is noise)
                  "rolling_mean"— 63-day rolling mean × 252 (original naive approach)
                  "momentum"    — 12-1 month cross-sectional momentum (documented factor)

  turnover_penalty — L1 penalty on trades: adds κ·‖w−w_prev‖₁ to objective;
                     kills excessive churn in mean-var formulations

  objective     = "minvar"   — minimize w'Σw
                  "meanvar"  — maximize (μ-rf)'w - γ·w'Σw

  use_empirical_cov = True   — use _wf_empirical (all extended experiments)
  use_empirical_cov = False  — DGV covariance path (legacy options experiments)
"""
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf  # type: ignore

from .config import LOOKBACK, REBAL_FREQ, MAX_WT, MAX_OPT_WT, MV_GAMMA, R_ANN, SYMS
from .covariance import build_full_cov


# ── Covariance estimators ─────────────────────────────────────────────────────

def _sample_cov(window: pd.DataFrame) -> np.ndarray:
    """Plain rolling sample covariance (daily). Baseline / worst estimator."""
    clean = window.dropna()
    if len(clean) < 3:
        return np.eye(window.shape[1]) * window.var().mean()
    Sig = np.cov(clean.T)
    if Sig.ndim == 0:
        Sig = np.array([[float(Sig)]])
    return 0.5 * (Sig + Sig.T)


def _ledoit_wolf_cov(window: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance (daily).
    Shrinks sample covariance toward scaled identity, with analytically
    optimal shrinkage intensity α.  Guarantees PSD.

    Reference: Ledoit & Wolf (2004) "A well-conditioned estimator for
    large-dimensional covariance matrices", Journal of Multivariate Analysis.
    """
    clean = window.dropna()
    if len(clean) < 3:
        return np.eye(window.shape[1]) * window.var().mean()
    Sig = LedoitWolf().fit(clean.values).covariance_
    return 0.5 * (Sig + Sig.T)


def _ewma_cov(window: pd.DataFrame, halflife: int) -> np.ndarray:
    """
    Exponentially-weighted covariance (daily).
    Recent observations carry more weight — captures volatility clustering
    and correlation regime changes far better than equal-weight rolling windows.

    Half-life of 21–42 trading days is empirically optimal for daily equity data
    (Boyd et al., "Covariance Prediction in Finance").
    """
    clean = window.dropna()
    n = len(clean)
    if n < 3:
        return np.eye(window.shape[1]) * window.var().mean()

    decay = np.exp(-np.log(2) / max(halflife, 1))
    weights = np.array([decay ** (n - 1 - j) for j in range(n)], dtype=float)
    weights /= weights.sum()

    X = clean.values
    mu_w = (X * weights[:, None]).sum(axis=0)   # weighted mean
    X_dm = X - mu_w
    Sig = (X_dm * weights[:, None]).T @ X_dm    # weighted outer-product sum

    # Correct small-sample bias
    w2 = (weights ** 2).sum()
    Sig = Sig / (1.0 - w2)
    return 0.5 * (Sig + Sig.T)


def _estimate_sigma(window: pd.DataFrame, cov_method: str, ewma_halflife: int) -> np.ndarray:
    """Dispatch to the chosen covariance estimator. Returns regularised daily Σ."""
    if cov_method == "ledoit_wolf":
        Sig = _ledoit_wolf_cov(window)
    elif cov_method == "ewma":
        Sig = _ewma_cov(window, halflife=ewma_halflife)
    else:
        Sig = _sample_cov(window)
    return Sig + 1e-7 * np.eye(Sig.shape[0])


# ── Expected-return estimators ────────────────────────────────────────────────

def _mu_rolling_mean(window: pd.DataFrame, cols: list, rf: float) -> np.ndarray:
    """Plain rolling mean × 252 (annualised). Naive baseline; μ MAE ≈ 120%."""
    return np.array([
        float(window[c].dropna().mean()) * 252
        if window[c].notna().sum() >= 5 else rf
        for c in cols
    ])


def _mu_momentum(rets_df: pd.DataFrame, cols: list, i: int,
                 long_window: int = 252, short_window: int = 21,
                 scale: float = 0.10, clip: float = 0.30) -> np.ndarray:
    """
    12-1 month cross-sectional momentum signal.

    Computes cumulative log-return from (t-252) to (t-21), skipping the most
    recent month to avoid short-term reversal (Jegadeesh & Titman, 1993).
    Cross-sectionally z-scored, then scaled and clipped.

    Parameters
    ----------
    scale : annualised μ per unit of z-score (default 10% per σ)
    clip  : maximum annualised μ magnitude (default ±30%)

    Returns
    -------
    μ vector in annualised units
    """
    long_start  = max(0, i - long_window)
    short_start = max(0, i - short_window)

    avail = [c for c in cols if c in rets_df.columns]
    r_long  = rets_df[avail].iloc[long_start:i].sum()
    r_short = rets_df[avail].iloc[short_start:i].sum()
    mom = (r_long - r_short).reindex(cols).fillna(0.0)

    std = mom.std()
    if std > 1e-8:
        mom_z = (mom - mom.mean()) / std
    else:
        mom_z = mom * 0.0

    return np.clip(mom_z.values * scale, -clip, clip)


def _mu_zero(cols: list) -> np.ndarray:
    """Zero expected returns — optimal when μ signal has no predictive power."""
    return np.zeros(len(cols))


# ── DGV options path (legacy) ─────────────────────────────────────────────────

def _compute_mu_delta_lev(cols: list, syms: list, gn_results: dict,
                           window: pd.DataFrame, rf: float = R_ANN) -> np.ndarray:
    """Delta-leverage μ for options experiments (unchanged from original)."""
    mu = np.zeros(len(cols))
    for k, c in enumerate(cols):
        sym = c.split("_")[0]
        if c in syms:
            h = window[c].dropna()
            mu[k] = float(h.mean()) * 252 if len(h) >= 5 else rf
        elif sym in gn_results:
            gn = gn_results[sym]
            lev = gn["lev_c"] if "_call" in c else gn["lev_p"]
            h   = window[sym].dropna()
            mu_und = float(h.mean()) * 252 if len(h) >= 5 else rf
            mu[k] = lev * mu_und
    return mu


# ── Main entry point ──────────────────────────────────────────────────────────

def run_walk_forward(
    rets_df: pd.DataFrame,
    asset_cols: list,
    syms: list,
    rp_idx: dict | None,
    label: str,
    lookback: int = LOOKBACK,
    rebal_freq: int = REBAL_FREQ,
    max_wt: float = MAX_WT,
    max_opt_wt: float = MAX_OPT_WT,
    objective: str = "minvar",
    gamma: float = MV_GAMMA,
    min_ann_return: float | None = None,
    use_empirical_cov: bool = False,
    rf: float = R_ANN,
    cov_method: str = "ledoit_wolf",
    mu_method: str = "zero",
    ewma_halflife: int = 42,
    turnover_penalty: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward portfolio backtest.

    Parameters
    ----------
    rets_df           : daily returns DataFrame
    asset_cols        : assets to include
    syms              : underlying stock symbols (for DGV path)
    rp_idx            : pre-indexed research panel (None → empirical Σ)
    label             : experiment name
    lookback          : rolling window (trading days)
    rebal_freq        : rebalance every N trading days
    max_wt            : per-asset weight cap
    max_opt_wt        : per-option-leg weight cap
    objective         : "minvar" | "meanvar"
    gamma             : risk-aversion (mean-var)
    min_ann_return    : optional minimum return constraint
    use_empirical_cov : True → _wf_empirical; False → DGV
    rf                : annualised risk-free rate
    cov_method        : "sample" | "ledoit_wolf" | "ewma"
    mu_method         : "zero" | "rolling_mean" | "momentum"
    ewma_halflife     : half-life in trading days for EWMA covariance
    turnover_penalty  : κ for L1 turnover penalty (0 = disabled)

    Returns
    -------
    (port_path, wts_history, prediction_log)
    """
    if use_empirical_cov or rp_idx is None:
        return _wf_empirical(
            rets_df, asset_cols, label,
            lookback=lookback, rebal_freq=rebal_freq, max_wt=max_wt,
            max_opt_wt=max_opt_wt, objective=objective, gamma=gamma,
            min_ann_return=min_ann_return, rf=rf,
            cov_method=cov_method, mu_method=mu_method,
            ewma_halflife=ewma_halflife, turnover_penalty=turnover_penalty,
        )

    # ── DGV (options) path — unchanged ───────────────────────────────────────
    cols        = [c for c in asset_cols if c in rets_df.columns]
    rets        = rets_df[cols].copy()
    und_rets_df = rets_df[[s for s in syms if s in rets_df.columns]].copy()
    dates       = rets.index

    port_val: dict = {}
    wts_hist: dict = {}
    cur_wts = {c: 1.0 / len(cols) for c in cols}
    port_val[dates[lookback - 1]] = 1.0

    for i in range(lookback, len(dates)):
        date = dates[i]
        if (i - lookback) % rebal_freq == 0:
            window = rets.iloc[i - lookback: i]
            try:
                Sigma, gn_results = build_full_cov(
                    date, rp_idx, und_rets_df, syms, cols, lookback)
                n = len(cols)
                w = cp.Variable(n, nonneg=True)
                constraints = [cp.sum(w) == 1, w <= max_wt]
                for k, c in enumerate(cols):
                    if "_call" in c or "_put" in c:
                        constraints.append(w[k] <= max_opt_wt)
                if min_ann_return is not None:
                    mu_hist = np.array([
                        float(window[c].dropna().mean()) * 252
                        if c in window.columns and window[c].notna().sum() >= 3 else 0.0
                        for c in cols])
                    constraints.append(mu_hist @ w >= min_ann_return / 252)
                if objective == "meanvar":
                    mu = _compute_mu_delta_lev(cols, syms, gn_results, window, rf=rf)
                    rf_period = rf / 252
                    problem = cp.Problem(
                        cp.Maximize((mu / 252 - rf_period) @ w
                                    - gamma * cp.quad_form(w, Sigma)),
                        constraints)
                else:
                    problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
                problem.solve(solver=cp.CLARABEL, warm_start=True)
                if w.value is None:
                    problem.solve(solver=cp.SCS)
                if w.value is not None:
                    wv = np.clip(np.asarray(w.value).flatten(), 0, 1)
                    wv /= wv.sum()
                    cur_wts = dict(zip(cols, wv))
            except Exception:
                pass
            wts_hist[date] = cur_wts.copy()

        r_port = sum(
            cur_wts.get(c, 0.0) * float(rets.loc[date, c])
            for c in cur_wts
            if c in rets.columns and pd.notna(rets.loc[date, c]))
        prev = list(port_val.keys())[-1]
        port_val[date] = port_val[prev] * (1.0 + r_port)

    path     = pd.Series(port_val, name=label)
    wts      = pd.DataFrame(wts_hist).T.fillna(0.0)
    pred_log = pd.DataFrame()
    return path, wts, pred_log


# ── Empirical (equity+bond) path ──────────────────────────────────────────────

def _wf_empirical(
    rets_df: pd.DataFrame,
    asset_cols: list,
    label: str,
    lookback: int = LOOKBACK,
    rebal_freq: int = REBAL_FREQ,
    max_wt: float = MAX_WT,
    max_opt_wt: float = MAX_OPT_WT,
    objective: str = "minvar",
    gamma: float = MV_GAMMA,
    min_ann_return: float | None = None,
    rf: float = R_ANN,
    cov_method: str = "ledoit_wolf",
    mu_method: str = "zero",
    ewma_halflife: int = 42,
    turnover_penalty: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest using improved covariance and return estimators.

    cov_method controls how Σ is estimated:
      "sample"       — equal-weight rolling window (baseline, worst)
      "ledoit_wolf"  — Ledoit-Wolf shrinkage (recommended minimum)
      "ewma"         — exponentially-weighted (best for vol-clustering)

    mu_method controls the expected-return signal:
      "zero"         — no signal; pure min-variance (best when μ is noise)
      "rolling_mean" — naive rolling mean (original approach; μ MAE ≈ 120%)
      "momentum"     — 12-1 month cross-sectional momentum (documented factor)

    turnover_penalty — adds κ·‖w−w_prev‖₁ to objective; suppresses churn.
    """
    cols  = [c for c in asset_cols if c in rets_df.columns]
    rets  = rets_df[cols].copy()
    dates = rets.index

    # Need enough history for momentum lookback
    momentum_warmup = 252 + lookback if mu_method == "momentum" else lookback
    first_idx = momentum_warmup

    port_val: dict  = {}
    wts_hist: dict  = {}
    pred_rows: list = []
    cur_wts = {c: 1.0 / len(cols) for c in cols}
    w_prev_arr = np.ones(len(cols)) / len(cols)   # equal-weight warm start

    # Initialise portfolio value at the first valid date
    if first_idx < len(dates):
        port_val[dates[first_idx - 1]] = 1.0
    else:
        return pd.Series(name=label), pd.DataFrame(), pd.DataFrame()

    for i in range(first_idx, len(dates)):
        date = dates[i]

        if (i - first_idx) % rebal_freq == 0:
            window = rets.iloc[i - lookback: i]
            valid  = [c for c in cols if window[c].notna().sum() >= lookback // 2]
            if len(valid) >= 2:
                n   = len(valid)
                idx = [cols.index(c) for c in valid]
                Sig = _estimate_sigma(window[valid], cov_method, ewma_halflife)

                # ── Expected returns ──────────────────────────────────────────
                if mu_method == "momentum":
                    mu_pred = _mu_momentum(rets_df, valid, i)
                elif mu_method == "rolling_mean":
                    mu_pred = _mu_rolling_mean(window, valid, rf)
                else:
                    mu_pred = _mu_zero(valid)

                # ── CVXPY problem ─────────────────────────────────────────────
                # Annualise Σ so objective is in annual-return units:
                #   min-var  : minimize  w'Σ_ann w
                #   mean-var : maximize  (μ − rf)'w  −  γ·w'Σ_ann w
                # This keeps γ and turnover_penalty on an interpretable annual scale.
                # Optimal weights are identical to daily formulation (U_ann = 252·U_daily).
                Sig_ann = Sig * 252

                w       = cp.Variable(n, nonneg=True)
                w_p_vec = w_prev_arr[idx]          # previous weights for valid assets
                constraints = [cp.sum(w) == 1, w <= max_wt]
                for k, c in enumerate(valid):
                    if "_call" in c or "_put" in c:
                        constraints.append(w[k] <= max_opt_wt)

                if min_ann_return is not None:
                    constraints.append(mu_pred @ w >= min_ann_return)

                # Core objective (all in annual units)
                risk_term = cp.quad_form(w, Sig_ann)
                if objective == "meanvar":
                    obj_expr = cp.Maximize((mu_pred - rf) @ w - gamma * risk_term)
                else:
                    obj_expr = cp.Minimize(risk_term)

                # L1 turnover penalty: κ·‖w−w_prev‖₁  (κ in annual-return units)
                # Practical range: κ ∈ [0.001, 0.02]
                #   κ=0.001 ≈ 0.1% drag per unit of L1 turnover (≈ 2 bps round-trip)
                #   κ=0.010 ≈ 1.0% drag per unit of L1 turnover (≈ 10 bps round-trip)
                if turnover_penalty > 0.0:
                    to_cost = turnover_penalty * cp.norm1(w - w_p_vec)
                    if objective == "meanvar":
                        obj_expr = cp.Maximize(obj_expr.args[0] - to_cost)
                    else:
                        obj_expr = cp.Minimize(obj_expr.args[0] + to_cost)

                problem = cp.Problem(obj_expr, constraints)
                problem.solve(solver=cp.CLARABEL, warm_start=True)
                if w.value is None:
                    problem.solve(solver=cp.SCS)

                if w.value is not None:
                    wv = np.clip(np.asarray(w.value).flatten(), 0, 1)
                    wv /= wv.sum()
                    cur_wts = dict(zip(valid, wv))
                    w_prev_arr[idx] = wv
                    # zero out assets that fell out of valid set
                    for k, c in enumerate(cols):
                        if c not in valid:
                            w_prev_arr[k] = 0.0
                    if w_prev_arr.sum() > 0:
                        w_prev_arr /= w_prev_arr.sum()

                # ── Prediction accuracy log ───────────────────────────────────
                end_idx = min(i + rebal_freq, len(dates))
                fwd     = rets.iloc[i:end_idx]
                row: dict = {"date": date}
                for k, c in enumerate(valid):
                    row[f"mu_pred_{c}"]    = mu_pred[k]
                    row[f"sigma_pred_{c}"] = float(np.sqrt(max(Sig[k, k], 0.0) * 252))
                    fwd_r = fwd[c].dropna()
                    row[f"mu_real_{c}"] = (
                        float(fwd_r.mean()) * 252 if len(fwd_r) >= 2 else np.nan)
                    row[f"sigma_real_{c}"] = (
                        float(fwd_r.std()) * np.sqrt(252) if len(fwd_r) >= 2 else np.nan)
                pred_rows.append(row)

            wts_hist[date] = cur_wts.copy()

        r_port = sum(
            cur_wts.get(c, 0.0) * float(rets.loc[date, c])
            for c in cur_wts
            if c in rets.columns and pd.notna(rets.loc[date, c]))
        prev = list(port_val.keys())[-1]
        port_val[date] = port_val[prev] * (1.0 + r_port)

    path     = pd.Series(port_val, name=label)
    wts      = pd.DataFrame(wts_hist).T.fillna(0.0)
    pred_log = (pd.DataFrame(pred_rows).set_index("date")
                if pred_rows else pd.DataFrame())
    return path, wts, pred_log
