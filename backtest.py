"""
Markowitz++ backtest: SPY, ATM call, ATM put, cash.

Uses REAL SPY returns + synthetic option returns (BS terminal payoff)
over periodic intervals to simulate the "hold-to-expiry and roll" strategy.

Forecast approach (hybrid):
  mu    = rolling historical mean of realized returns (physical measure)
  Sigma = RND-implied covariance from option chain (risk structure)

This captures the equity risk premium (historical mu) while using the
option-implied risk structure (RND Sigma).
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

from config import PROCESSED_DIR, SPY_DAILY_FILE, TARGET_IDEAL_DTE
from data.forecasts import (
    ASSET_ORDER,
    _bs_call_price,
    _bs_put_price,
    compute_rnd_forecasts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Backtest parameters
# -----------------------------------------------------------------------
REBAL_DAYS = TARGET_IDEAL_DTE  # ~14 trading days per period
GAMMA = 5.0                    # risk-aversion
MAX_TURNOVER = 0.50            # per-period turnover cap
TCOST_RATE = 0.001             # 10 bps each way
INITIAL_VALUE = 1.0
LOOKBACK_YEARS = 4.0           # how far back for synthetic returns
ROLLING_WINDOW = 26            # periods for rolling mu (~1 yr at 14-day freq)
MIN_PERIODS_FOR_ROLL = 8       # need at least this many periods before switching to rolling

# Position limits: options have binary payoffs (can return -100%) so cap exposure
MAX_OPTION_WEIGHT = 0.15       # max 15% in any single option sleeve
MIN_CASH_WEIGHT = 0.10         # always keep some cash
MU_SHRINKAGE = 0.5             # blend rolling mu toward cash rate (0 = pure rolling, 1 = pure prior)


# -----------------------------------------------------------------------
# 1. Build synthetic period returns from real SPY
# -----------------------------------------------------------------------

def build_synthetic_returns(
    rebal_days: int = REBAL_DAYS,
    iv: float = 0.20,
    r: float = 0.05,
    lookback_years: float = LOOKBACK_YEARS,
) -> pd.DataFrame:
    """
    Period returns for [SPY, SPY_CALL, SPY_PUT, USDOLLAR] from real SPY closes.

    Each period spans *rebal_days* trading days:
      SPY  = S_end / S_start - 1
      CALL = max(S_end - K, 0) / BS_Call(S_start, K, r, T, iv) - 1
      PUT  = max(K - S_end, 0) / BS_Put(S_start, K, r, T, iv) - 1
      CASH = exp(r * T) - 1

    K = S_start (ATM at each period start).
    """
    spy = pd.read_parquet(SPY_DAILY_FILE)
    closes = spy["close"].dropna().sort_index()
    closes.index = pd.to_datetime(closes.index)

    if lookback_years:
        cutoff = closes.index[-1] - pd.DateOffset(years=int(lookback_years))
        closes = closes[closes.index >= cutoff]

    T = rebal_days / 365.0

    dates, rows = [], []
    i = 0
    while i + rebal_days < len(closes):
        s0 = float(closes.iloc[i])
        s1 = float(closes.iloc[i + rebal_days])

        c0 = max(_bs_call_price(s0, s0, r, T, iv), 0.01)
        p0 = max(_bs_put_price(s0, s0, r, T, iv), 0.01)

        r_spy = s1 / s0 - 1.0
        r_call = max(s1 - s0, 0.0) / c0 - 1.0
        r_put = max(s0 - s1, 0.0) / p0 - 1.0
        r_cash = np.exp(r * T) - 1.0

        dates.append(closes.index[i + rebal_days])
        rows.append([r_spy, r_call, r_put, r_cash])

        i += rebal_days

    returns = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=ASSET_ORDER)
    returns.index.name = "date"
    return returns


# -----------------------------------------------------------------------
# 2. Markowitz optimiser (cvxpy)
# -----------------------------------------------------------------------

def _shrink_mu(mu_roll: np.ndarray, r_period: float) -> np.ndarray:
    """Shrink rolling mu toward the cash rate to tame extreme estimates."""
    prior = np.full_like(mu_roll, r_period)
    return (1 - MU_SHRINKAGE) * mu_roll + MU_SHRINKAGE * prior


def _solve_markowitz(
    mu_arr: np.ndarray,
    Sigma_arr: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = GAMMA,
) -> np.ndarray:
    n = len(mu_arr)
    w = cp.Variable(n)
    ret = mu_arr @ w
    risk = cp.quad_form(w, Sigma_arr, assume_PSD=True)
    turnover = cp.norm(w - w_prev, 1)
    tcost = TCOST_RATE * turnover

    objective = cp.Maximize(ret - gamma / 2 * risk - tcost)
    # SPY, CALL, PUT, CASH
    w_upper = np.array([1.0, MAX_OPTION_WEIGHT, MAX_OPTION_WEIGHT, 1.0])
    constraints = [
        w >= 0,
        w <= w_upper,
        cp.sum(w) == 1,
        w[3] >= MIN_CASH_WEIGHT,   # USDOLLAR index = 3
        turnover <= MAX_TURNOVER,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        w_opt = np.maximum(w.value, 0.0)
        return w_opt / w_opt.sum()
    return w_prev.copy()


# -----------------------------------------------------------------------
# 3. Run backtest
# -----------------------------------------------------------------------

def run_backtest() -> tuple:
    # --- RND forecasts (for Sigma and fallback mu) ---
    logger.info("Computing RND forecasts ...")
    mu_rnd, Sigma_rnd, diag = compute_rnd_forecasts(
        n_samples=10_000, return_diagnostics=True,
    )
    logger.info("RND mu:\n%s", mu_rnd)

    # --- synthetic returns from real SPY ---
    atm_iv = diag["atm_iv"]
    logger.info(
        "Building synthetic returns (period=%d days, IV=%.2f, lookback=%.0f yr) ...",
        REBAL_DAYS, atm_iv, LOOKBACK_YEARS,
    )
    returns = build_synthetic_returns(rebal_days=REBAL_DAYS, iv=atm_iv)
    n_periods = len(returns)
    logger.info(
        "Periods: %d  (%s to %s)", n_periods,
        returns.index[0].date(), returns.index[-1].date(),
    )

    # Sigma: use RND for initial periods, then switch to rolling historical
    # (RND Sigma underestimates option return variance because it's from
    #  lognormal model; real hold-to-expiry returns are binary and far more volatile)
    Sigma_rnd_arr = Sigma_rnd.values.astype(float)
    Sigma_rnd_arr = (Sigma_rnd_arr + Sigma_rnd_arr.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma_rnd_arr)
    if eigvals.min() < 1e-8:
        Sigma_rnd_arr += np.eye(len(ASSET_ORDER)) * 1e-6

    n_assets = len(ASSET_ORDER)

    weights_history = np.zeros((n_periods + 1, n_assets))
    weights_history[0] = [0.0, 0.0, 0.0, 1.0]  # start in cash
    portfolio_values = np.ones(n_periods + 1) * INITIAL_VALUE

    r_period = float(returns["USDOLLAR"].iloc[0])
    mu_history = []

    for t in range(n_periods):
        w_prev = weights_history[t]

        if t >= MIN_PERIODS_FOR_ROLL:
            lb = max(0, t - ROLLING_WINDOW)
            hist_slice = returns.iloc[lb:t]
            mu_roll = hist_slice.mean().values.astype(float)
            mu_t = _shrink_mu(mu_roll, r_period)

            # Rolling historical covariance (captures real binary-payoff risk)
            Sigma_hist = hist_slice.cov().values.astype(float)
            Sigma_hist = (Sigma_hist + Sigma_hist.T) / 2
            eig = np.linalg.eigvalsh(Sigma_hist)
            if eig.min() < 1e-8 or np.any(np.isnan(Sigma_hist)):
                Sigma_t = Sigma_rnd_arr
            else:
                Sigma_t = Sigma_hist
        else:
            mu_t = mu_rnd.values.astype(float)
            Sigma_t = Sigma_rnd_arr

        mu_history.append(mu_t.copy())

        w_opt = _solve_markowitz(mu_t, Sigma_t, w_prev)
        weights_history[t + 1] = w_opt

        r_vec = returns.iloc[t].values.astype(float)
        port_ret = np.dot(w_opt, r_vec)
        actual_to = np.sum(np.abs(w_opt - w_prev))
        port_ret -= TCOST_RATE * actual_to

        portfolio_values[t + 1] = portfolio_values[t] * (1 + port_ret)

    # assemble output frames
    start_date = returns.index[0] - pd.Timedelta(days=REBAL_DAYS)
    dates_full = pd.DatetimeIndex([start_date] + list(returns.index))

    portfolio_df = pd.DataFrame({"value": portfolio_values}, index=dates_full)
    weights_df = pd.DataFrame(weights_history, index=dates_full, columns=ASSET_ORDER)
    mu_df = pd.DataFrame(mu_history, index=returns.index, columns=ASSET_ORDER)

    return portfolio_df, weights_df, returns, mu_rnd, Sigma_rnd, diag, mu_df


# -----------------------------------------------------------------------
# 4. Plot
# -----------------------------------------------------------------------

def plot_results(
    portfolio_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    returns: pd.DataFrame,
    mu_rnd: pd.Series,
    diag: dict,
    mu_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> str:
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 14),
        gridspec_kw={"height_ratios": [3, 1, 1.2, 1.2]},
    )
    fig.suptitle(
        "Markowitz++ Backtest:  SPY · ATM Call · ATM Put · Cash\n"
        f"Period = {REBAL_DAYS}d hold-to-expiry · γ = {GAMMA}"
        f" · max option wt = {MAX_OPTION_WEIGHT:.0%}"
        f" · μ shrinkage = {MU_SHRINKAGE}",
        fontsize=13, fontweight="bold",
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

    # ---- Panel 1: portfolio trajectory ----
    ax = axes[0]
    ax.plot(
        portfolio_df.index, portfolio_df["value"],
        color="#2563eb", linewidth=2.2, label="Optimized Portfolio",
    )
    ax.plot(
        returns.index, spy_cum.values,
        color="#94a3b8", linewidth=1.5, linestyle="--", label="SPY Buy & Hold",
    )
    ax.axhline(1.0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Value ($1 initial)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    n_periods = len(returns)
    stats = (
        f"Return: {total_ret:+.1%}  (SPY: {spy_total:+.1%})\n"
        f"Sharpe (ann.): {sharpe:.2f}\n"
        f"Max DD: {max_dd:.1%}\n"
        f"Periods: {n_periods}  ({returns.index[0].date()} → {returns.index[-1].date()})\n"
        f"Forecast: rolling-{ROLLING_WINDOW} μ + rolling Σ\n"
        f"ATM IV: {diag['atm_iv']:.1%}   DTE: {diag['dte']}d   Method: {diag['method']}"
    )
    ax.text(
        0.98, 0.02, stats, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )

    # ---- Panel 2: drawdown ----
    ax_dd = axes[1]
    ax_dd.fill_between(
        portfolio_df.index, drawdown, 0,
        color="#dc2626", alpha=0.35,
    )
    ax_dd.plot(portfolio_df.index, drawdown, color="#dc2626", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_ylim(min(max_dd * 1.1, -0.05), 0.02)
    ax_dd.grid(True, alpha=0.3)

    # ---- Panel 3: allocation weights (stacked) ----
    ax2 = axes[2]
    w_arr = weights_df[ASSET_ORDER].values
    ax2.stackplot(
        weights_df.index, w_arr.T,
        labels=ASSET_ORDER,
        colors=[colors[a] for a in ASSET_ORDER],
        alpha=0.85,
    )
    ax2.set_ylabel("Weight")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", ncol=4, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 4: rolling mu used by optimizer ----
    ax3 = axes[3]
    for asset in ["SPY", "SPY_CALL", "SPY_PUT"]:
        ax3.plot(
            mu_df.index, mu_df[asset],
            label=asset, color=colors[asset], linewidth=1.2,
        )
    ax3.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("Rolling μ (shrunk)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", ncol=3, fontsize=9)
    ax3.grid(True, alpha=0.3)

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
    portfolio_df, weights_df, returns, mu_rnd, Sigma_rnd, diag, mu_df = run_backtest()

    pv = portfolio_df["value"]
    print("\n========== Backtest Results ==========")
    print(f"Periods:  {len(returns)}")
    print(f"Range:    {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Final:    ${pv.iloc[-1]:.4f}  ({pv.iloc[-1] / INITIAL_VALUE - 1:+.2%})")

    # Annualized Sharpe
    pr = np.diff(pv.values) / pv.values[:-1]
    ann = np.sqrt(252 / max(REBAL_DAYS, 1))
    print(f"Sharpe:   {np.mean(pr) / (np.std(pr) + 1e-12) * ann:.2f}")
    print(f"Max DD:   {np.min(pv.values / np.maximum.accumulate(pv.values)) - 1:.1%}")

    print("\nFinal weights:")
    for a in ASSET_ORDER:
        print(f"  {a:10s} {weights_df[a].iloc[-1]:6.1%}")

    print(f"\nRND mu (static):  {mu_rnd.to_dict()}")
    print(f"Last rolling mu:  {mu_df.iloc[-1].to_dict()}")

    plot_path = plot_results(portfolio_df, weights_df, returns, mu_rnd, diag, mu_df)
    print(f"\nPlot saved: {plot_path}")
