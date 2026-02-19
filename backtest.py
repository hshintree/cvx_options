"""
Markowitz++ backtest: SPY, ATM call, ATM put, cash.

Forecast approach:
  Sigma  = blend of IEWMA (historical, from Johansson et al. 2023) and
           per-period RND covariance from option chain (forward-looking)
  mu     = rolling historical mean of realized returns (physical measure)

Optimizer:
  Worst-case robust Markowitz (from DeMiguel et al., MVO review):
    - Return penalty:  subtract ρ per asset weight  (hedges μ estimation error)
    - Covariance boost: Σ_wc = Σ + κ·diag(Σ)  (hedges Σ estimation error)
  Both ρ and κ are tunable (default from config).

Realized returns use ACTUAL option prices from chain snapshots:
  - Option entry price: BS-priced ATM call/put using chain IV on entry date
  - Option exit:        BS repriced at next rebalance (sticky-strike IV)
  - SPY return:         real close-to-close over the period
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    COV_UNCERTAINTY,
    IEWMA_HALFLIFE_PAIRS,
    IEWMA_LOOKBACK,
    MU_UNCERTAINTY,
    OPTION_CHAINS_DIR,
    PROCESSED_DIR,
    SIGMA_IEWMA_WEIGHT,
    SPY_DAILY_FILE,
    TARGET_IDEAL_DTE,
)
from data.covariance import CMIEWMAPredictor
from data.forecasts import (
    ASSET_ORDER,
    _bs_call_price,
    _bs_put_price,
    compute_rnd_forecasts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Backtest parameters  (tune these in demo.ipynb)
# -----------------------------------------------------------------------
REBAL_DAYS = TARGET_IDEAL_DTE
GAMMA = 10.0                   # risk-aversion (higher = less option exposure)
MAX_TURNOVER = 0.40            # per-period turnover cap
TCOST_RATE = 0.001             # 10 bps each way
INITIAL_VALUE = 1.0
ROLLING_WINDOW = 13            # ~6 months of biweekly periods
MIN_PERIODS_FOR_ROLL = 4
MAX_OPTION_WEIGHT = 0.03       # 3% per sleeve (~60% notional via 20x leverage)
MAX_SPY_WEIGHT = 1.0           # max weight in SPY equity sleeve
MIN_CASH_WEIGHT = 0.10
MU_SHRINKAGE = 0.5
MAX_PORT_VOL = 0.04            # hard cap: portfolio std per period (SOCP constraint)

# Robust optimization parameters  (patched from demo.ipynb)
ROBUST_MU_UNCERTAINTY = MU_UNCERTAINTY       # ρ: return uncertainty per asset
ROBUST_COV_UNCERTAINTY = COV_UNCERTAINTY     # κ: covariance diagonal boost
IEWMA_WEIGHT = SIGMA_IEWMA_WEIGHT           # blend weight for IEWMA vs RND Sigma


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _available_chain_dates() -> list[str]:
    """Return sorted list of dates (YYYY-MM-DD) that have chain parquets."""
    dates = set()
    for f in OPTION_CHAINS_DIR.glob("calls_*.parquet"):
        d = f.stem.replace("calls_", "")
        put_f = OPTION_CHAINS_DIR / f"puts_{d}.parquet"
        if put_f.exists():
            try:
                pd.read_parquet(f)
                dates.add(d)
            except Exception:
                pass
    return sorted(dates)


def _shrink_mu(mu_roll: np.ndarray, r_period: float) -> np.ndarray:
    prior = np.full_like(mu_roll, r_period)
    return (1 - MU_SHRINKAGE) * mu_roll + MU_SHRINKAGE * prior


def _solve_markowitz(
    mu_arr: np.ndarray,
    Sigma_arr: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = GAMMA,
    max_port_vol: float = MAX_PORT_VOL,
    mu_uncertainty: float = ROBUST_MU_UNCERTAINTY,
    cov_uncertainty: float = ROBUST_COV_UNCERTAINTY,
) -> np.ndarray:
    """Worst-case robust Markowitz optimizer.

    For long-only portfolios the worst-case simplifications from MVO_70 apply:
      - Worst-case return: μ_wc = μ − ρ  (subtract uncertainty from each asset)
      - Worst-case covariance: Σ_wc = Σ + κ·diag(Σ)  (inflate diagonal)
    """
    n = len(mu_arr)
    w = cp.Variable(n)

    # Worst-case return: penalize by per-asset uncertainty ρ
    # For long-only, |w| = w, so μ_wc'w = (μ − ρ)'w
    rho = np.full(n, mu_uncertainty)
    rho[3] = 0.0  # cash return is known (no uncertainty)
    mu_wc = mu_arr - rho
    ret = mu_wc @ w

    # Worst-case covariance: inflate diagonal by κ
    # Σ_wc = Σ + κ * diag(diag(Σ)) = Σ * (I + κ * diag(1))  on diagonal only
    Sigma_wc = Sigma_arr.copy()
    if cov_uncertainty > 0:
        Sigma_wc += cov_uncertainty * np.diag(np.diag(Sigma_arr))
    Sigma_wc = (Sigma_wc + Sigma_wc.T) / 2.0

    risk = cp.quad_form(w, Sigma_wc, assume_PSD=True)
    turnover = cp.norm(w - w_prev, 1)
    tcost = TCOST_RATE * turnover

    objective = cp.Maximize(ret - gamma / 2 * risk - tcost)
    w_upper = np.array([MAX_SPY_WEIGHT, MAX_OPTION_WEIGHT, MAX_OPTION_WEIGHT, 1.0])
    constraints = [
        w >= 0,
        w <= w_upper,
        cp.sum(w) == 1,
        w[3] >= MIN_CASH_WEIGHT,
        turnover <= MAX_TURNOVER,
    ]

    # Hard portfolio volatility cap using worst-case covariance (SOCP)
    if max_port_vol is not None and max_port_vol > 0:
        L = np.linalg.cholesky(Sigma_wc + np.eye(n) * 1e-8)
        constraints.append(cp.norm(L.T @ w, 2) <= max_port_vol)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        w_opt = np.maximum(w.value, 0.0)
        return w_opt / w_opt.sum()
    return w_prev.copy()


def _regularize_sigma(S: np.ndarray) -> np.ndarray:
    S = (S + S.T) / 2.0
    eig = np.linalg.eigvalsh(S)
    if eig.min() < 1e-8 or np.any(np.isnan(S)):
        S += np.eye(S.shape[0]) * 1e-6
    return S


# -----------------------------------------------------------------------
# Build realized returns aligned to chain dates
# -----------------------------------------------------------------------

def build_realized_returns(
    chain_dates: list[str],
    spy_df: pd.DataFrame,
    r: float = 0.05,
) -> pd.DataFrame:
    """
    For each consecutive pair of chain dates, compute realized returns
    using BS horizon repricing (sticky-strike IV).

    Entry: buy ATM options with the DTE chosen by the RND system.
    Exit:  reprice those options at the next chain date with remaining time.
    """
    rows = []
    dates_out = []

    for i in range(len(chain_dates) - 1):
        d0 = chain_dates[i]
        d1 = chain_dates[i + 1]
        ts0 = pd.Timestamp(d0)
        ts1 = pd.Timestamp(d1)

        idx0 = spy_df.index.get_indexer([ts0], method="ffill")[0]
        idx1 = spy_df.index.get_indexer([ts1], method="ffill")[0]
        if idx0 < 0 or idx1 < 0:
            continue

        s0 = float(spy_df.iloc[idx0]["close"])
        s1 = float(spy_df.iloc[idx1]["close"])
        holding_days = (ts1 - ts0).days
        T_hold = max(holding_days / 365.0, 1 / 365.0)

        # Get IV and option DTE from the chain on entry date
        try:
            _, _, diag = compute_rnd_forecasts(
                chain_date=d0, spot=s0, n_samples=1000, return_diagnostics=True,
            )
            iv = diag["atm_iv"]
            option_dte = diag["dte"]
        except Exception:
            iv = 0.15
            option_dte = 30

        T_option = max(option_dte / 365.0, 1 / 365.0)
        T_remain = max(T_option - T_hold, 0.0)
        k_atm = round(s0)

        # Entry prices (ATM, full option DTE)
        c0 = max(_bs_call_price(s0, k_atm, r, T_option, iv), 0.01)
        p0 = max(_bs_put_price(s0, k_atm, r, T_option, iv), 0.01)

        # Exit prices (BS reprice with remaining time, sticky-strike IV)
        if T_remain > 1 / 365.0:
            c1 = max(_bs_call_price(s1, k_atm, r, T_remain, iv), 0.0)
            p1 = max(_bs_put_price(s1, k_atm, r, T_remain, iv), 0.0)
        else:
            c1 = max(s1 - k_atm, 0.0)
            p1 = max(k_atm - s1, 0.0)

        r_spy = s1 / s0 - 1.0
        r_call = c1 / c0 - 1.0
        r_put = p1 / p0 - 1.0
        r_cash = np.exp(r * T_hold) - 1.0

        dates_out.append(ts1)
        rows.append([r_spy, r_call, r_put, r_cash])

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates_out), columns=ASSET_ORDER)


# -----------------------------------------------------------------------
# Run backtest
# -----------------------------------------------------------------------

def run_backtest() -> tuple:
    spy_df = pd.read_parquet(SPY_DAILY_FILE)
    spy_df.index = pd.to_datetime(spy_df.index).tz_localize(None).normalize()

    chain_dates = _available_chain_dates()
    logger.info("Available chain dates: %d", len(chain_dates))

    # Build realized returns between consecutive chain dates
    logger.info("Building realized returns from chain dates ...")
    returns = build_realized_returns(chain_dates, spy_df)
    n_periods = len(returns)
    logger.info("Periods: %d  (%s to %s)", n_periods,
                returns.index[0].date(), returns.index[-1].date())

    # Pre-compute per-period RND Sigma from each chain snapshot
    logger.info("Computing per-period RND forecasts ...")
    rnd_sigmas = {}
    rnd_mus = {}
    rnd_diags = {}
    for d in chain_dates:
        ts = pd.Timestamp(d)
        idx = spy_df.index.get_indexer([ts], method="ffill")[0]
        spot = float(spy_df.iloc[idx]["close"]) if idx >= 0 else 550.0
        try:
            mu_rnd, Sigma_rnd, diag = compute_rnd_forecasts(
                chain_date=d, spot=spot, n_samples=5000, return_diagnostics=True,
            )
            rnd_sigmas[d] = _regularize_sigma(Sigma_rnd.values.astype(float))
            rnd_mus[d] = mu_rnd.values.astype(float)
            rnd_diags[d] = diag
        except Exception as e:
            logger.warning("RND failed for %s: %s", d, e)

    logger.info("RND computed for %d / %d dates", len(rnd_sigmas), len(chain_dates))

    # Default fallback Sigma (option vol ≈ 20x SPY due to leverage)
    fallback_Sigma = _regularize_sigma(
        np.diag([0.0009, 0.36, 0.36, 1e-10])
    )

    # ---------------------------------------------------------------
    # Initialize IEWMA covariance predictor (online, uses realized returns)
    # ---------------------------------------------------------------
    n_assets = len(ASSET_ORDER)
    iewma_predictor = CMIEWMAPredictor(
        n_assets,
        halflife_pairs=IEWMA_HALFLIFE_PAIRS,
        lookback=IEWMA_LOOKBACK,
    )

    weights_history = np.zeros((n_periods + 1, n_assets))
    weights_history[0] = [0.60, 0.0, 0.0, 0.40]
    portfolio_values = np.ones(n_periods + 1) * INITIAL_VALUE

    r_period = float(returns["USDOLLAR"].iloc[0])
    mu_history = []
    method_history = []
    sigma_source_history = []

    for t in range(n_periods):
        w_prev = weights_history[t]
        entry_date = chain_dates[t]

        # --- Sigma: blend IEWMA (historical) and RND (forward-looking) ---
        Sigma_rnd = rnd_sigmas.get(entry_date, fallback_Sigma)
        Sigma_iewma = iewma_predictor.predict()

        if Sigma_iewma is not None and IEWMA_WEIGHT > 0:
            Sigma_iewma = _regularize_sigma(Sigma_iewma)
            alpha = IEWMA_WEIGHT
            Sigma_t = (1.0 - alpha) * Sigma_rnd + alpha * Sigma_iewma
            sigma_source_history.append("blend")
        else:
            Sigma_t = Sigma_rnd
            sigma_source_history.append("rnd_only")

        Sigma_t = _regularize_sigma(Sigma_t)

        # --- mu: rolling historical (physical measure) + shrinkage ---
        if t >= MIN_PERIODS_FOR_ROLL:
            lb = max(0, t - ROLLING_WINDOW)
            mu_roll = returns.iloc[lb:t].mean().values.astype(float)
            mu_t = _shrink_mu(mu_roll, r_period)
        else:
            mu_t = np.array([0.005, 0.0, 0.0, r_period])

        mu_history.append(mu_t.copy())
        method_history.append(rnd_diags.get(entry_date, {}).get("method", "fallback"))

        w_opt = _solve_markowitz(mu_t, Sigma_t, w_prev)
        weights_history[t + 1] = w_opt

        r_vec = returns.iloc[t].values.astype(float)
        port_ret = np.dot(w_opt, r_vec)
        actual_to = np.sum(np.abs(w_opt - w_prev))
        port_ret -= TCOST_RATE * actual_to

        portfolio_values[t + 1] = portfolio_values[t] * (1 + port_ret)

        # Feed realized return to IEWMA predictor (online update)
        iewma_predictor.update(r_vec)

    start_date = returns.index[0] - pd.Timedelta(days=REBAL_DAYS)
    dates_full = pd.DatetimeIndex([start_date] + list(returns.index))

    portfolio_df = pd.DataFrame({"value": portfolio_values}, index=dates_full)
    weights_df = pd.DataFrame(weights_history, index=dates_full, columns=ASSET_ORDER)
    mu_df = pd.DataFrame(mu_history, index=returns.index, columns=ASSET_ORDER)

    n_blend = sum(1 for s in sigma_source_history if s == "blend")
    diag_summary = {
        "n_periods": n_periods,
        "n_rnd_computed": len(rnd_sigmas),
        "n_bl": sum(1 for d in rnd_diags.values() if d.get("method") == "breeden_litzenberger"),
        "n_lognormal": sum(1 for d in rnd_diags.values() if d.get("method") == "lognormal_iv_fallback"),
        "iv_range": (
            min(d.get("atm_iv", 0) for d in rnd_diags.values()),
            max(d.get("atm_iv", 0) for d in rnd_diags.values()),
        ),
        "method_history": method_history,
        "n_iewma_blend": n_blend,
        "iewma_weight": IEWMA_WEIGHT,
        "mu_uncertainty": ROBUST_MU_UNCERTAINTY,
        "cov_uncertainty": ROBUST_COV_UNCERTAINTY,
    }

    return portfolio_df, weights_df, returns, mu_df, diag_summary


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def plot_results(
    portfolio_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    returns: pd.DataFrame,
    mu_df: pd.DataFrame,
    diag: dict,
    save_path: str | Path | None = None,
) -> str:
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 14),
        gridspec_kw={"height_ratios": [3, 1, 1.2, 1.2]},
    )
    fig.suptitle(
        "Robust Markowitz++ Backtest:  SPY · ATM Call · ATM Put · Cash\n"
        f"γ={GAMMA}  max_opt={MAX_OPTION_WEIGHT:.0%}"
        f"  vol_cap={MAX_PORT_VOL:.0%}"
        f"  ρ={diag.get('mu_uncertainty', 0):.3f}"
        f"  κ={diag.get('cov_uncertainty', 0):.2f}"
        f"  IEWMA={diag.get('iewma_weight', 0):.0%}"
        f"  BL:{diag['n_bl']}/{diag['n_rnd_computed']}",
        fontsize=12, fontweight="bold",
    )

    colors = {
        "SPY": "#2563eb", "SPY_CALL": "#16a34a",
        "SPY_PUT": "#dc2626", "USDOLLAR": "#f59e0b",
    }

    pv = portfolio_df["value"].values
    total_ret = pv[-1] / INITIAL_VALUE - 1
    spy_cum = (1 + returns["SPY"]).cumprod()
    spy_total = spy_cum.iloc[-1] - 1
    period_rets = np.diff(pv) / pv[:-1]
    ann_factor = np.sqrt(252 / max(REBAL_DAYS, 1))
    sharpe = np.mean(period_rets) / (np.std(period_rets) + 1e-12) * ann_factor
    running_max = np.maximum.accumulate(pv)
    drawdown = pv / running_max - 1
    max_dd = drawdown.min()

    # Panel 1: portfolio trajectory
    ax = axes[0]
    ax.plot(portfolio_df.index, pv, color="#2563eb", linewidth=2.2, label="Optimized Portfolio")
    ax.plot(returns.index, spy_cum.values, color="#94a3b8", linewidth=1.5,
            linestyle="--", label="SPY Buy & Hold")
    ax.axhline(1.0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Value ($1 initial)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    iv_lo, iv_hi = diag["iv_range"]
    stats = (
        f"Return: {total_ret:+.1%}  (SPY: {spy_total:+.1%})\n"
        f"Sharpe (ann.): {sharpe:.2f}\n"
        f"Max DD: {max_dd:.1%}\n"
        f"Periods: {diag['n_periods']}  ({returns.index[0].date()} → {returns.index[-1].date()})\n"
        f"BL density: {diag['n_bl']}/{diag['n_rnd_computed']}  "
        f"IV range: {iv_lo:.0%}–{iv_hi:.0%}\n"
        f"IEWMA blend: {diag.get('n_iewma_blend', 0)}/{diag['n_periods']} periods"
    )
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    # Panel 2: drawdown
    ax_dd = axes[1]
    ax_dd.fill_between(portfolio_df.index, drawdown, 0, color="#dc2626", alpha=0.35)
    ax_dd.plot(portfolio_df.index, drawdown, color="#dc2626", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_ylim(min(max_dd * 1.1, -0.05), 0.02)
    ax_dd.grid(True, alpha=0.3)

    # Panel 3: weights
    ax2 = axes[2]
    ax2.stackplot(weights_df.index, weights_df[ASSET_ORDER].values.T,
                  labels=ASSET_ORDER, colors=[colors[a] for a in ASSET_ORDER], alpha=0.85)
    ax2.set_ylabel("Weight")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", ncol=4, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 4: rolling mu
    ax3 = axes[3]
    for asset in ["SPY", "SPY_CALL", "SPY_PUT"]:
        ax3.plot(mu_df.index, mu_df[asset], label=asset, color=colors[asset], linewidth=1.2)
    ax3.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("μ (shrunk)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", ncol=3, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add RND method markers on the weights panel
    for i, m in enumerate(diag.get("method_history", [])):
        if m == "breeden_litzenberger":
            ax2.axvline(returns.index[i], color="green", alpha=0.15, linewidth=1)

    plt.tight_layout()
    save_path = save_path or str(PROCESSED_DIR / "backtest_results.png")
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", save_path)
    plt.close(fig)
    return str(save_path)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    portfolio_df, weights_df, returns, mu_df, diag = run_backtest()

    pv = portfolio_df["value"]
    print("\n========== Backtest Results ==========")
    print(f"Periods:  {diag['n_periods']}")
    print(f"Range:    {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Final:    ${pv.iloc[-1]:.4f}  ({pv.iloc[-1] / INITIAL_VALUE - 1:+.2%})")

    pr = np.diff(pv.values) / pv.values[:-1]
    ann = np.sqrt(252 / max(REBAL_DAYS, 1))
    print(f"Sharpe:   {np.mean(pr) / (np.std(pr) + 1e-12) * ann:.2f}")
    print(f"Max DD:   {np.min(pv.values / np.maximum.accumulate(pv.values)) - 1:.1%}")
    print(f"BL used:  {diag['n_bl']}/{diag['n_rnd_computed']} periods")
    print(f"IEWMA:    {diag.get('n_iewma_blend', 0)}/{diag['n_periods']} blended"
          f"  (weight={diag.get('iewma_weight', 0):.0%})")
    print(f"Robust:   ρ={diag.get('mu_uncertainty', 0):.4f}"
          f"  κ={diag.get('cov_uncertainty', 0):.2f}")

    print("\nFinal weights:")
    for a in ASSET_ORDER:
        print(f"  {a:10s} {weights_df[a].iloc[-1]:6.1%}")

    plot_path = plot_results(portfolio_df, weights_df, returns, mu_df, diag)
    print(f"\nPlot saved: {plot_path}")
