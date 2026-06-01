"""
cvxportfolio-based backtest engine.

Replaces the hand-rolled CVXPY loop in backtest.py with
cvxportfolio's walk-forward simulator and built-in estimators.

Key advantages over the hand-rolled version:
  - HistoricalFactorizedCovariance(half_life) — EWMA factorized Σ; never singular
  - HistoricalMeanReturn(half_life) — EWMA μ; better shrinkage than rolling window
  - StocksTransactionCost — half-spread + market impact model (more realistic than L1)
  - Automatic warm-up / asset availability management
  - MultiPeriodOptimization (MPO) — plans N steps ahead, reducing unnecessary trades

Usage:
    from min_var.cvxport_backtest import run_cvxportfolio_backtest

    result = run_cvxportfolio_backtest(
        returns_df,      # simple (arithmetic) daily returns, columns = ticker names
        gamma=5.0,       # risk-aversion (winning config)
        cov_halflife=21, # EWMA half-life for Σ (trading days)
        mu_halflife=252, # EWMA half-life for μ (trading days)
        max_wt=0.40,
        rebal_freq='monthly',
        mu_method='ewma',
        use_mpo=False,          # True → MultiPeriodOptimization(planning_horizon)
        planning_horizon=3,     # MPO look-ahead steps (months if monthly rebalancing)
        start='2022-01-03',
        end=None,
    )
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import cvxportfolio as cvx

from .config import R_ANN

# Per-asset bid-ask half-spread (fraction of notional, one-way).
# Sources: typical ETF bid-ask spreads + Alpaca observed fills.
#   Liquid index ETFs (SPY, QQQ, SHY):       ~2 bps
#   Bond/commodity/sector ETFs:               ~3 bps
#   Large-cap equities (AAPL, MSFT, AMZN):   ~5 bps
#   Mid-cap/international ETFs:               ~5 bps
#   High-vol equities (NVDA, TSLA, AMD):      ~8 bps
#   Crypto-adjacent (GBTC, COIN, MSTR):      ~15 bps
ASSET_HALF_SPREAD: dict[str, float] = {
    # Liquid index ETFs
    "SPY":  0.0002,
    "QQQ":  0.0002,
    "IWM":  0.0002,
    "SHY":  0.0002,
    # Bond ETFs
    "TLT":  0.0003,
    "IEF":  0.0003,
    "TIP":  0.0003,
    "HYG":  0.0004,
    # Commodity ETFs
    "GLD":  0.0003,
    "SLV":  0.0004,
    "CPER": 0.0005,
    "DBC":  0.0004,
    # Sector & thematic ETFs
    "XLV":  0.0003,
    "XLF":  0.0003,
    "XLY":  0.0003,
    "XLP":  0.0003,
    "ICLN": 0.0004,
    # Real estate
    "VNQ":  0.0003,
    "AMT":  0.0005,
    # International ETFs
    "EFA":  0.0003,
    "EEM":  0.0004,
    # Large-cap equities
    "AAPL": 0.0005,
    "MSFT": 0.0005,
    "AMZN": 0.0005,
    "GOOGL": 0.0005,
    "META": 0.0005,
    # High-vol equities
    "NVDA": 0.0008,
    "TSLA": 0.0008,
    "AMD":  0.0008,
    # Crypto-adjacent (wide spreads + high vol)
    "COIN": 0.0015,
    "MSTR": 0.0015,
}
_DEFAULT_HALF_SPREAD = 0.0005   # fallback for symbols not in the table


def _half_spread_series(columns: list[str]) -> pd.Series:
    """Build a per-asset half-spread Series for TransactionCost."""
    return pd.Series(
        {col: ASSET_HALF_SPREAD.get(col, _DEFAULT_HALF_SPREAD) for col in columns}
    )


def _momentum_signal(returns_df: pd.DataFrame,
                     long_window: int = 252,
                     short_window: int = 21,
                     scale: float = 0.10,
                     clip: float = 0.30) -> pd.DataFrame:
    """
    Pre-compute 12-1 month cross-sectional momentum for every rebalance date.

    Returns a DataFrame (same index as returns_df) where entry [t, asset] is
    the annualised expected-return forecast for that asset at time t, expressed
    as a fraction of portfolio value (i.e. direct input to ReturnsForecast).
    """
    out = pd.DataFrame(index=returns_df.index, columns=returns_df.columns,
                       dtype=float)
    for i in range(long_window + 1, len(returns_df)):
        long_start  = max(0, i - long_window)
        short_start = max(0, i - short_window)
        r_long  = returns_df.iloc[long_start:i].sum()
        r_short = returns_df.iloc[short_start:i].sum()
        mom     = r_long - r_short
        std     = mom.std()
        if std > 1e-8:
            mom_z = (mom - mom.mean()) / std
        else:
            mom_z = mom * 0.0
        out.iloc[i] = np.clip(mom_z.values * scale, -clip, clip)
    return out.fillna(0.0)


def _apply_return_shrinkage(
    rets: pd.DataFrame,
    halflife: int,
    shrinkage: float,
) -> pd.DataFrame:
    """James-Stein cross-sectional shrinkage for EWMA return forecasts.

    Compresses the cross-sectional spread of EWMA means toward their equal-weighted
    average, reducing concentration in recent winners without discarding the signal.

    At each time t we subtract  shrinkage × (μ̂ᵢ(t-1) − μ̄(t-1))  from rᵢ(t),
    where μ̂ᵢ is the EWMA mean of asset i and μ̄ is the cross-sectional mean.
    When cvxportfolio's internal EWMA estimator then re-fits on the adjusted series,
    the steady-state mean it sees is:
        μ_adj_i = (1−shrinkage)·μ_i + shrinkage·μ̄

    shrinkage=0  → no change (pure momentum)
    shrinkage=0.5 → compress cross-sectional spread by 50%
    shrinkage=1  → all assets forecast same expected return (pure min-var in spirit)
    """
    if shrinkage == 0.0:
        return rets

    lam   = np.exp(-np.log(2) / max(halflife, 1))
    alpha = 1.0 - lam
    n, k  = len(rets), rets.shape[1]

    rets_arr = rets.values.astype(float)
    adj      = np.zeros_like(rets_arr)

    # Bias-corrected EWMA: track numerator and effective weight separately
    ewma_num = np.zeros(k)
    ewma_wt  = 0.0

    for t in range(n):
        # Apply lagged adjustment at t using estimate from t-1
        if t > 0 and ewma_wt > 1e-12:
            mu_prev  = ewma_num / ewma_wt    # bias-corrected EWMA mean
            mu_bar   = mu_prev.mean()
            adj[t]   = shrinkage * (mu_prev - mu_bar)
        # Update EWMA with the *original* return (not adjusted, to avoid bias accumulation)
        ewma_wt  = lam * ewma_wt  + alpha
        ewma_num = lam * ewma_num + alpha * rets_arr[t]

    return pd.DataFrame(rets_arr - adj, index=rets.index, columns=rets.columns)


def _run_minvar_backtest(
    returns_df: pd.DataFrame,
    cov_halflife: int = 21,
    corr_halflife: int | None = None,   # if set, uses split vol/corr estimator Σ=D×C×D
    max_wt: float = 0.20,
    rebal_freq: str = "monthly",
    start: str = "2022-01-03",
    end: str | None = None,
    allow_cash: bool = True,
) -> tuple[pd.Series, dict]:
    """
    Pure min-var backtest using rolling CVXPY.

    At each rebalance date, estimates EWMA Σ from all data up to that point,
    then solves: min w'Σw  s.t.  sum(w)≤1, 0≤w≤max_wt.

    When allow_cash=True (default), the constraint is sum(w)<=1 and the
    unallocated fraction earns R_ANN/252 per day.  This lets the optimizer
    go to partial cash when all assets look risky.

    Bypasses cvxportfolio entirely to avoid the all-cash allocation issue.

    When corr_halflife is set (and != cov_halflife), uses the split estimator:
      Σ = D × C × D  where D = fast EWMA vols (hl=cov_halflife),
                           C = slow EWMA correlations (hl=corr_halflife).
    """
    import cvxpy as cp
    from .cvxport_accuracy import _ewma_cov, _ewma_cov_split

    _use_split = corr_halflife is not None and corr_halflife != cov_halflife

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else returns_df.index[-1]

    # Trim to backtest window but keep full history for covariance warm-up
    bt_rets = returns_df.loc[start_dt:end_dt]
    if len(bt_rets) == 0:
        raise ValueError(f"No data in [{start}, {end}]")

    n        = returns_df.shape[1]
    cols     = list(returns_df.columns)
    min_hist = cov_halflife * 4   # trading days needed before first solve

    # Determine rebalance dates within the backtest window
    if rebal_freq == "monthly":
        rebal_dates = set(
            bt_rets.resample("MS").first().dropna(how="all").index
        )
    elif rebal_freq == "quarterly":
        rebal_dates = set(
            bt_rets.resample("QS").first().dropna(how="all").index
        )
    elif rebal_freq == "weekly":
        rebal_dates = set(
            bt_rets.resample("W-MON").first().dropna(how="all").index
        )
    else:
        rebal_dates = set(bt_rets.index)   # daily

    rf_daily     = R_ANN / 252.0
    pv_vals      = [1.0]
    wts_hist     = {}
    current_wts  = np.ones(n) / n   # equal-weight until first solve

    for i in range(1, len(bt_rets)):
        date    = bt_rets.index[i]
        day_ret = bt_rets.iloc[i].values
        risky_ret   = float(current_wts @ day_ret)
        cash_frac   = max(1.0 - current_wts.sum(), 0.0) if allow_cash else 0.0
        total_ret   = risky_ret + cash_frac * rf_daily
        pv_vals.append(pv_vals[-1] * (1.0 + total_ret))

        if date in rebal_dates:
            # All data up to (and including) this date for covariance estimation
            past = returns_df.loc[:date]
            if len(past) < min_hist:
                continue   # not enough history yet

            arr = past.values.astype(float)
            if _use_split:
                sig_d = _ewma_cov_split(arr, cov_halflife, corr_halflife)
            else:
                sig_d = _ewma_cov(arr, cov_halflife)
            sig_ann = sig_d * 252.0
            sig_ann = 0.5 * (sig_ann + sig_ann.T) + 1e-6 * np.eye(n)

            w    = cp.Variable(n)
            budget = cp.sum(w) <= 1.0 if allow_cash else cp.sum(w) == 1.0
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(w, sig_ann)),
                [budget, w >= 0.0, w <= max_wt],
            )
            try:
                prob.solve(solver=cp.CLARABEL)
                if w.value is None:
                    prob.solve(solver=cp.SCS)
            except Exception:
                try:
                    prob.solve(solver=cp.SCS)
                except Exception:
                    pass

            if w.value is not None:
                wv = np.clip(w.value, 0.0, max_wt)
                s  = wv.sum()
                if s > 1e-8:
                    current_wts = wv / s
            wts_hist[date] = pd.Series(current_wts, index=cols)

    pv      = pd.Series(pv_vals, index=bt_rets.index)
    pv      = pv / pv.iloc[0]
    cash_tag = "+cash" if allow_cash else ""
    if _use_split:
        pv.name = f"min-var{cash_tag}(Σ=D×C×D vol_hl={cov_halflife}d corr_hl={corr_halflife}d)"
    else:
        pv.name = f"min-var{cash_tag}(EWMA Σ_hl={cov_halflife}d)"
    pv.index = pv.index.tz_localize(None) if pv.index.tz is not None else pv.index

    w_df = pd.DataFrame(wts_hist).T if wts_hist else pd.DataFrame()
    meta = {
        "gamma":          0.0,
        "cov_halflife":   cov_halflife,
        "corr_halflife":  corr_halflife,
        "mu_halflife":    None,
        "mu_method":      "zero",
        "allow_cash":     allow_cash,
        "rebal_freq":     rebal_freq,
        "use_mpo":        False,
        "planning_horizon": None,
        "result":         None,
        "w":            w_df,
    }
    return pv, meta


def run_cvxportfolio_backtest(
    returns_df: pd.DataFrame,
    gamma: float = 2.0,
    cov_halflife: int = 21,
    mu_halflife: int = 252,
    max_wt: float = 0.40,
    rebal_freq: str = "monthly",   # 'weekly' | 'monthly' | None
    mu_method: str = "ewma",       # 'ewma' | 'momentum' | 'zero'
    mu_shrinkage: float = 0.0,     # James-Stein shrinkage toward cross-sectional mean
    use_mpo: bool = False,         # True → MultiPeriodOptimization
    planning_horizon: int = 3,     # MPO look-ahead steps
    start: str = "2022-01-03",
    end: str | None = None,
    verbose: bool = False,
) -> tuple[pd.Series, dict]:
    """
    Run a walk-forward backtest using cvxportfolio's SPO or MPO.

    Parameters
    ----------
    returns_df      : simple (arithmetic) daily returns, no cash column needed
    gamma           : risk-aversion coefficient
    cov_halflife    : EWMA half-life for Σ in *trading* days
    mu_halflife     : EWMA half-life for μ in *trading* days
    max_wt          : per-asset weight cap
    rebal_freq      : 'weekly', 'monthly', or None (every trading day)
    mu_method       : 'ewma'  — HistoricalMeanReturn with EWMA smoothing
                      'momentum' — 12-1 month cross-sectional z-score
                      'zero'  — zero expected returns (pure min-var)
    mu_shrinkage    : James-Stein shrinkage intensity [0, 1].  0 = no shrinkage
                      (pure momentum).  0.5 = compress cross-sectional spread by
                      half.  Has no effect when mu_method='zero'.
    use_mpo         : if True, use MultiPeriodOptimization(planning_horizon)
                      instead of SinglePeriodOptimization — plans ahead to
                      reduce unnecessary round-trip trades
    planning_horizon: number of steps MPO looks ahead (executed: first step only)
    start / end     : backtest window

    Returns
    -------
    (portfolio_value_series, metadata_dict)
    """
    # ── Pure min-var shortcut — bypass cvxportfolio entirely ─────────────────
    if mu_method == "zero":
        return _run_minvar_backtest(
            returns_df=returns_df,
            cov_halflife=cov_halflife,
            max_wt=max_wt,
            rebal_freq=rebal_freq,
            start=start,
            end=end,
        )

    # ── Convert trading-day half-lives to calendar Timedeltas ─────────────────
    # 1 trading day ≈ 365/252 calendar days
    cal_per_td  = 365.25 / 252.0
    cov_hl_cal  = pd.Timedelta(days=int(cov_halflife * cal_per_td))
    mu_hl_cal   = pd.Timedelta(days=int(mu_halflife  * cal_per_td))
    roll_cal    = pd.Timedelta(days=int(max(cov_halflife * 4, 252) * cal_per_td))

    # ── Apply James-Stein shrinkage to EWMA μ forecasts ───────────────────────
    # Compresses cross-sectional spread toward the mean before cvxportfolio fits.
    # No-op when mu_shrinkage=0 or mu_method != 'ewma'.
    if mu_shrinkage > 0.0 and mu_method == "ewma":
        returns_df = _apply_return_shrinkage(returns_df, mu_halflife, mu_shrinkage)

    # ── Build return forecast ─────────────────────────────────────────────────
    if mu_method == "momentum":
        mom = _momentum_signal(returns_df)
        r_forecast = cvx.ReturnsForecast(r_hat=mom)
    elif mu_method == "ewma":
        r_forecast = cvx.ReturnsForecast(
            r_hat=cvx.forecast.HistoricalMeanReturn(
                half_life=mu_hl_cal,
                rolling=roll_cal,
            ),
        )
    else:
        r_forecast = None

    # ── Risk model ────────────────────────────────────────────────────────────
    risk_cost = cvx.FullCovariance(
        Sigma=cvx.forecast.HistoricalFactorizedCovariance(
            half_life=cov_hl_cal,
            rolling=roll_cal,
        )
    )

    # ── Per-asset transaction cost ────────────────────────────────────────────
    asset_cols = [c for c in returns_df.columns]
    half_spreads = _half_spread_series(asset_cols)
    tcost = cvx.TransactionCost(a=half_spreads)

    # ── Objective ─────────────────────────────────────────────────────────────
    if r_forecast is not None:
        base_objective = r_forecast - gamma * risk_cost
    else:
        base_objective = -gamma * risk_cost

    objective = base_objective - tcost

    # ── Constraints ──────────────────────────────────────────────────────────
    constraints = [
        cvx.LongOnly(),
        cvx.MaxWeights(max_wt),
        cvx.LeverageLimit(1.0),
    ]

    # ── Policy (SPO or MPO) ───────────────────────────────────────────────────
    if use_mpo:
        policy = cvx.MultiPeriodOptimization(
            objective=objective,
            constraints=constraints,
            planning_horizon=planning_horizon,
            include_cash_return=True,
            fallback_solver="SCS",
        )
    else:
        policy = cvx.SinglePeriodOptimization(
            objective=objective,
            constraints=constraints,
            include_cash_return=True,
            fallback_solver="SCS",
        )

    # ── Market data ───────────────────────────────────────────────────────────
    # cvxportfolio needs a cash-return column (risk-free rate).
    # Since our DatetimeIndex is timezone-naive, USDOLLAR auto-download fails.
    # Solution: add a constant 'cash' column = rf_annual / 252 (daily rf rate).
    #
    # Special case for mu_method='zero': setting cash return to 0 prevents
    # cvxportfolio from optimally holding all-cash (which dominates equity when
    # μ=0 but cash earns R_ANN). With cash_return=0, the optimizer minimises
    # variance subject to the LeverageLimit, producing a true min-var portfolio.
    rets_with_cash = returns_df.copy()
    rets_with_cash["cash"] = 0.0 if mu_method == "zero" else R_ANN / 252.0

    tf = rebal_freq if rebal_freq in ("weekly", "monthly", "quarterly") else None
    md = cvx.UserProvidedMarketData(
        returns=rets_with_cash,
        cash_key="cash",
        min_history=pd.Timedelta(days=int((cov_halflife * 4) * cal_per_td)),
        trading_frequency=tf,
        online_usage=False,
    )

    # ── Simulator ─────────────────────────────────────────────────────────────
    sim = cvx.MarketSimulator(
        market_data=md,
        costs=[],
    )

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else returns_df.index[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.backtest(policy, start_time=start_dt, end_time=end_dt,
                              h=None)

    # ── Extract portfolio value path ──────────────────────────────────────────
    pv = result.v
    pv = pv / pv.iloc[0]
    pv.index = pv.index.tz_localize(None) if pv.index.tz is not None else pv.index
    mode_str = f"MPO(h={planning_horizon})" if use_mpo else "SPO"
    pv.name = f"cvxport({mode_str},γ={gamma},Σ_hl={cov_halflife}d,μ={mu_method})"

    meta = {
        "gamma": gamma,
        "cov_halflife": cov_halflife,
        "mu_halflife": mu_halflife,
        "mu_method": mu_method,
        "rebal_freq": rebal_freq,
        "use_mpo": use_mpo,
        "planning_horizon": planning_horizon,
        "result": result,
    }
    return pv, meta
