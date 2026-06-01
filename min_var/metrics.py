"""
Portfolio performance metrics — fixed and extended from log_norm.ipynb Cell 36 _port_stats.

Fixes applied:
  1. periods_per_year derived from median index gap (correct for non-daily paths)
  2. Annualized return uses geometric exponent = periods_per_year / n_periods
  3. Added Sortino ratio (downside deviation)
  4. Added Calmar ratio (return / max drawdown)
  5. Added Information Ratio vs optional benchmark
  6. Added avg_turnover / ann_turnover (from weights history)
  7. build_stats_table() — assembles the DataFrame that was missing in Cell 37
"""
import numpy as np
import pandas as pd
from typing import Optional

from .config import R_ANN


def port_stats(
    path: pd.Series,
    label: str,
    benchmark: Optional[pd.Series] = None,
    wts_df: Optional[pd.DataFrame] = None,
    rf_annual: float = R_ANN,
) -> dict:
    """
    Compute performance statistics for a portfolio value path.

    Parameters
    ----------
    path       : pd.Series — portfolio value indexed by date
    label      : str — experiment name
    benchmark  : optional pd.Series — benchmark value path for IR calculation
    wts_df     : optional pd.DataFrame — weight history (rebalance dates × assets)
                 used to compute avg/ann turnover
    rf_annual  : annualized risk-free rate

    Returns
    -------
    dict with keys: label, ann_return, ann_vol, sharpe, sortino, calmar,
                    max_dd, total_return, periods_per_year,
                    info_ratio (if benchmark provided),
                    avg_turnover, ann_turnover (if wts_df provided)
    """
    path = path.dropna()
    if len(path) < 2:
        return {"label": label}

    # ── Detect rebalance frequency from median index gap ─────────────────────
    day_gaps = pd.Series(path.index).diff().dt.days.dropna()
    median_gap = float(day_gaps.median()) if len(day_gaps) > 0 else 1.0
    median_gap = max(median_gap, 1.0)
    periods_per_year = 365.0 / median_gap

    # ── Return series ─────────────────────────────────────────────────────────
    period_r = path.pct_change().dropna()
    n_periods = len(period_r)
    if n_periods < 1:
        return {"label": label}

    # ── Annualized metrics ────────────────────────────────────────────────────
    ann_ret = float((path.iloc[-1] / path.iloc[0]) ** (periods_per_year / n_periods) - 1)
    ann_vol = float(period_r.std() * np.sqrt(periods_per_year))
    rf_per_period = rf_annual / periods_per_year

    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 1e-10 else np.nan

    # ── Sortino (downside deviation below rf) ─────────────────────────────────
    excess = period_r - rf_per_period
    downside = excess[excess < 0]
    if len(downside) > 0:
        downside_std = float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))
        sortino = (ann_ret - rf_annual) / downside_std if downside_std > 1e-10 else np.nan
    else:
        sortino = np.nan

    # ── Max drawdown ─────────────────────────────────────────────────────────
    cum_max  = path.cummax()
    drawdown = (path / cum_max - 1)
    max_dd   = float(drawdown.min())

    # ── Calmar ───────────────────────────────────────────────────────────────
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan

    result = dict(
        label=label,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_dd=max_dd,
        total_return=float(path.iloc[-1] / path.iloc[0] - 1),
        periods_per_year=round(periods_per_year, 1),
    )

    # ── Information Ratio vs benchmark ───────────────────────────────────────
    if benchmark is not None:
        bench_r = benchmark.pct_change().dropna().reindex(period_r.index).fillna(0.0)
        active  = period_r - bench_r
        if active.std() > 1e-10:
            ir = float(active.mean() * periods_per_year / (active.std() * np.sqrt(periods_per_year)))
        else:
            ir = np.nan
        result["info_ratio"] = ir

    # ── Turnover from weight history ─────────────────────────────────────────
    if wts_df is not None and not wts_df.empty:
        wts_sorted = wts_df.sort_index()
        turn = (wts_sorted.diff().abs().sum(axis=1) / 2.0).dropna()
        avg_turn  = float(turn.mean())
        # Annualise: assume weight history rows = rebalance dates
        n_rebal   = len(wts_sorted)
        rebal_gap = max((wts_sorted.index[-1] - wts_sorted.index[0]).days / max(n_rebal - 1, 1), 1)
        rebal_per_year = 365.0 / rebal_gap
        result["avg_turnover"]  = avg_turn
        result["ann_turnover"]  = avg_turn * rebal_per_year

    return result


def build_stats_table(
    paths: list,
    labels: list | None = None,
    wts_list: list | None = None,
    benchmark: pd.Series | None = None,
    rf_annual: float = R_ANN,
) -> pd.DataFrame:
    """
    Build a summary statistics DataFrame for a list of portfolio paths.

    This is the fix for the missing `_stats_df` in Cell 37 of the notebook.

    Parameters
    ----------
    paths    : list of pd.Series
    labels   : list of str (defaults to path.name)
    wts_list : optional list of pd.DataFrame (one per path, for turnover)
    benchmark: optional pd.Series (for IR)
    """
    rows = []
    for i, path in enumerate(paths):
        lbl = (labels[i] if labels else None) or getattr(path, "name", f"Portfolio {i}")
        wts = wts_list[i] if wts_list and i < len(wts_list) else None
        rows.append(port_stats(path, lbl, benchmark=benchmark, wts_df=wts, rf_annual=rf_annual))

    df = pd.DataFrame(rows).set_index("label")

    # Format for display
    fmt_cols = {
        "ann_return": "{:.2%}", "ann_vol": "{:.2%}", "total_return": "{:.2%}",
        "max_dd": "{:.2%}", "avg_turnover": "{:.2%}", "ann_turnover": "{:.2%}",
        "sharpe": "{:.3f}", "sortino": "{:.3f}", "calmar": "{:.3f}",
        "info_ratio": "{:.3f}",
    }
    df.attrs["fmt"] = fmt_cols
    return df


def format_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a prettily formatted copy of a stats DataFrame for display."""
    fmt = df.attrs.get("fmt", {})
    out = df.copy()
    for col, fmtstr in fmt.items():
        if col in out.columns:
            out[col] = out[col].apply(lambda x: fmtstr.format(x) if pd.notna(x) else "—")
    return out
