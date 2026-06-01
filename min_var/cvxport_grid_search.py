"""
Grid search over cvxportfolio hyperparameters.

Tunes (gamma, cov_halflife, max_wt, mu_shrinkage, rebal_freq) using
rolling walk-forward cross-validation.

## Why rolling CV instead of a single IS/OOS split

A single IS=2022-2023 / OOS=2024-present split scores every strategy against a
raging tech bull market.  Any strategy with NVDA/COIN/MSTR exposure and
max_wt ≥ 10% will produce Sharpe=10-20 and ann_ret=800,000% in that single OOS
window regardless of whether the underlying μ/Σ estimators are well-calibrated.
IS Sharpe being strongly negative while OOS Sharpe is stratospheric is the
loudest possible overfitting signal.

Rolling CV builds 5-6 non-overlapping OOS windows (default: 6-month windows)
and scores by:
  - mean_oos_sharpe  : average Sharpe across all windows (primary metric)
  - stability_score  : mean_oos_sharpe / std_oos_sharpe (penalises regime-specific luck)
  - min_oos_sharpe   : worst single-window Sharpe (must survive at least one bear period)

Any config where mean_is_sharpe < –1.5 is discarded as numerically unstable.

## Parameters locked by statistical validation (not tuned here)

  mu_method  = "ewma"     — only statistically-grounded μ estimator
  use_mpo    = False       — MPO adds a free planning-horizon parameter with no
                             validated predictive advantage over SPO
  mu_halflife = 252d       — 12-month EWMA window with significant IC at 63d+

Rebalance frequency IS included because IC at 21d is not significant (p≈0.15)
but IS significant at 63d (p≈0.007). Monthly acts on noise; quarterly aligns
with the validated signal horizon.

Typical usage:
    from min_var.cvxport_grid_search import run_cvxport_rolling_cv, save_cvxport_best_params
    results = run_cvxport_rolling_cv(arith_rets)
    save_cvxport_best_params(results)
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, R_ANN
from .cvxport_backtest import run_cvxportfolio_backtest

# ── Default grids ─────────────────────────────────────────────────────────────
# Only parameters with a validated causal link to forecast accuracy are tuned.
GAMMA_GRID          = [1.0, 2.0, 5.0, 10.0]
COV_HALFLIFE_GRID   = [21, 42]          # 10d removed: below IC significance window
MAX_WT_GRID         = [0.10, 0.15, 0.20]
MU_SHRINKAGE_GRID   = [0.50, 0.75, 0.80]   # validated range — 0% leads to concentration
REBAL_FREQ_GRID     = ["monthly", "quarterly"]  # IC evidence: quarterly more grounded

# Quick grid for fast testing (~24 combos)
QUICK_GAMMA_GRID         = [2.0, 5.0]
QUICK_COV_HALFLIFE_GRID  = [21, 42]
QUICK_MAX_WT_GRID        = [0.20]
QUICK_MU_SHRINKAGE_GRID  = [0.75, 0.80]
QUICK_REBAL_FREQ_GRID    = ["monthly", "quarterly"]

BEST_PARAMS_PATH = OUTPUT_DIR / "cvxport_best_params.json"


def _sharpe(pv: pd.Series, rf_annual: float = R_ANN) -> float:
    """Annualised Sharpe from a normalised price series."""
    rets = pv.pct_change().dropna()
    if len(rets) < 5:
        return -999.0
    rf_daily = rf_annual / 252.0
    excess   = rets - rf_daily
    ann_ret  = excess.mean() * 252
    ann_vol  = excess.std()  * np.sqrt(252)
    return float(ann_ret / ann_vol) if ann_vol > 1e-8 else -999.0


def _ann_ret(pv: pd.Series) -> float:
    """
    Annualised return from a price series, capped at ±500%.

    Uses log-linear annualization to avoid numerical explosion from short
    windows (the naive formula V^(252/n) blows up to 800,000% when a
    concentrated portfolio triples in 63 trading days).
    """
    n = max(len(pv) - 1, 1)
    v_ratio = max(pv.iloc[-1] / max(pv.iloc[0], 1e-9), 1e-9)
    log_ann = np.log(v_ratio) * (252.0 / n)
    return float(np.clip(np.expm1(log_ann), -0.99, 5.0))  # cap at 500% p.a.


def _build_cv_windows(
    index: pd.DatetimeIndex,
    is_months: int,
    oos_months: int,
    step_months: int,
) -> list[tuple[str, str, str, str]]:
    """
    Build rolling IS/OOS windows.

    Returns list of (is_start, is_end, oos_start, oos_end) date strings.
    """
    windows = []
    # First IS window starts at index[0]; first OOS starts after is_months
    oos_start = index[0] + pd.DateOffset(months=is_months)

    while True:
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > index[-1]:
            break
        is_start = oos_start - pd.DateOffset(months=is_months)
        windows.append((
            is_start.strftime("%Y-%m-%d"),
            oos_start.strftime("%Y-%m-%d"),
            oos_start.strftime("%Y-%m-%d"),
            oos_end.strftime("%Y-%m-%d"),
        ))
        oos_start = oos_start + pd.DateOffset(months=step_months)

    return windows


def run_cvxport_rolling_cv(
    arith_rets: pd.DataFrame,
    gamma_grid:         list[float] = GAMMA_GRID,
    cov_halflife_grid:  list[int]   = COV_HALFLIFE_GRID,
    max_wt_grid:        list[float] = MAX_WT_GRID,
    mu_shrinkage_grid:  list[float] = MU_SHRINKAGE_GRID,
    rebal_freq_grid:    list[str]   = REBAL_FREQ_GRID,
    mu_halflife:  int = 252,
    is_months:    int = 18,   # 18-month IS window (data-limited: COIN IPO 2021)
    oos_months:   int = 6,    # 6-month OOS window per fold
    step_months:  int = 6,    # roll OOS by 6 months each fold
    min_is_sharpe: float = -1.5,   # discard numerically unstable configs
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Rolling walk-forward cross-validation over cvxportfolio hyperparameters.

    Builds multiple IS/OOS windows (typically 5-6 folds), evaluates each config
    on every window, and scores by:
      - mean_oos_sharpe  : average Sharpe across all OOS folds
      - stability_score  : mean / std of OOS Sharpe (Sharpe-of-Sharpes)
      - min_oos_sharpe   : worst fold (must survive at least one bear market)

    Configs with mean_is_sharpe < min_is_sharpe are discarded as unstable.

    Parameters
    ----------
    arith_rets         : arithmetic (simple) daily returns
    *_grid             : hyperparameter search grids
    mu_halflife        : fixed EWMA μ half-life (252d = 12-month momentum)
    is_months          : in-sample window length per fold (months)
    oos_months         : out-of-sample window length per fold (months)
    step_months        : rolling step size (months)
    min_is_sharpe      : IS Sharpe floor — configs below this are dropped
    verbose            : print per-window progress

    Returns
    -------
    pd.DataFrame sorted by stability_score descending. Columns:
        gamma, cov_halflife, max_wt, mu_shrinkage, rebal_freq,
        mean_oos_sharpe, std_oos_sharpe, min_oos_sharpe, stability_score,
        mean_is_sharpe, mean_oos_max_dd, mean_oos_ann_ret, n_windows
    """
    windows = _build_cv_windows(arith_rets.index, is_months, oos_months, step_months)
    if not windows:
        raise ValueError(
            f"Not enough data for rolling CV. Need at least {is_months + oos_months} months. "
            f"Data spans: {arith_rets.index[0].date()} → {arith_rets.index[-1].date()}"
        )

    combos = list(product(gamma_grid, cov_halflife_grid, max_wt_grid,
                          mu_shrinkage_grid, rebal_freq_grid))
    n_combos  = len(combos)
    n_windows = len(windows)

    print(f"Rolling walk-forward CV: {n_combos} combos × {n_windows} folds"
          f" = {n_combos * n_windows} runs")
    print(f"  Locked: mu_method=ewma, use_mpo=False, mu_halflife={mu_halflife}d")
    print(f"  IS={is_months}m, OOS={oos_months}m, step={step_months}m\n")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        print(f"  Fold {i+1}: IS={is_s}→{is_e}  OOS={oos_s}→{oos_e}")
    print()

    # Accumulate per-fold stats per combo
    fold_stats: list[dict] = [
        {"is_sharpes": [], "oos_sharpes": [], "oos_max_dds": [], "oos_ann_rets": []}
        for _ in combos
    ]

    for w_idx, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        fold_label = f"Fold {w_idx + 1}/{n_windows}"
        print(f"\n{fold_label}  IS={is_s}→{is_e}  OOS={oos_s}→{oos_e}")

        for c_idx, (gamma, cov_hl, max_wt, mu_shrink, rebal) in enumerate(combos):
            label = (f"γ={gamma}, cov_hl={cov_hl}d, max_wt={max_wt:.0%}, "
                     f"shrink={mu_shrink:.0%}, rebal={rebal}")
            if not verbose:
                print(f"  [{c_idx+1:3d}/{n_combos}] {label} … ", end="", flush=True)

            # IS run
            is_sharpe = -999.0
            try:
                pv_is, _ = run_cvxportfolio_backtest(
                    returns_df=arith_rets, gamma=gamma, cov_halflife=cov_hl,
                    mu_halflife=mu_halflife, max_wt=max_wt, rebal_freq=rebal,
                    mu_method="ewma", mu_shrinkage=mu_shrink, use_mpo=False,
                    start=is_s, end=is_e,
                )
                is_sharpe = _sharpe(pv_is)
            except Exception as exc:
                if verbose:
                    print(f"  IS failed: {exc}")
            fold_stats[c_idx]["is_sharpes"].append(is_sharpe)

            # OOS run
            try:
                pv_oos, _ = run_cvxportfolio_backtest(
                    returns_df=arith_rets, gamma=gamma, cov_halflife=cov_hl,
                    mu_halflife=mu_halflife, max_wt=max_wt, rebal_freq=rebal,
                    mu_method="ewma", mu_shrinkage=mu_shrink, use_mpo=False,
                    start=oos_s, end=oos_e,
                )
                oos_r    = pv_oos.pct_change().dropna()
                roll_max = pv_oos.cummax()
                max_dd   = float(((pv_oos - roll_max) / roll_max).min())

                fold_stats[c_idx]["oos_sharpes"].append(_sharpe(pv_oos))
                fold_stats[c_idx]["oos_max_dds"].append(max_dd)
                fold_stats[c_idx]["oos_ann_rets"].append(_ann_ret(pv_oos))

                if not verbose:
                    print(f"OOS Sharpe={fold_stats[c_idx]['oos_sharpes'][-1]:.3f}", flush=True)
            except Exception as exc:
                fold_stats[c_idx]["oos_sharpes"].append(-999.0)
                fold_stats[c_idx]["oos_max_dds"].append(-999.0)
                fold_stats[c_idx]["oos_ann_rets"].append(-999.0)
                if not verbose:
                    print(f"FAILED: {exc}", flush=True)
                if verbose:
                    print(f"  OOS failed: {exc}")

    # ── Aggregate across folds ──────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    for c_idx, (gamma, cov_hl, max_wt, mu_shrink, rebal) in enumerate(combos):
        fs = fold_stats[c_idx]

        # Drop failed folds
        oos_sh  = [s for s in fs["oos_sharpes"]  if s > -900]
        is_sh   = [s for s in fs["is_sharpes"]   if s > -900]
        oos_dd  = [d for d in fs["oos_max_dds"]  if d > -900]
        oos_ret = [r for r in fs["oos_ann_rets"] if r > -900]

        mean_oos = np.mean(oos_sh)  if oos_sh  else -999.0
        std_oos  = np.std(oos_sh)   if len(oos_sh) > 1 else 999.0
        min_oos  = np.min(oos_sh)   if oos_sh  else -999.0
        mean_is  = np.mean(is_sh)   if is_sh   else -999.0

        # stability = mean/std, but penalise configs with bad IS performance
        # (numerically unstable or data snooping) and those that can't survive bear markets
        stability = -999.0
        if mean_is >= min_is_sharpe and len(oos_sh) >= 2:
            stability = mean_oos / (std_oos + 0.10)   # regularised Sharpe-of-Sharpes

        rows.append({
            "gamma":          gamma,
            "cov_halflife":   cov_hl,
            "max_wt":         max_wt,
            "mu_shrinkage":   mu_shrink,
            "rebal_freq":     rebal,
            "mean_oos_sharpe": float(mean_oos),
            "std_oos_sharpe":  float(std_oos),
            "min_oos_sharpe":  float(min_oos),
            "stability_score": float(stability),
            "mean_is_sharpe":  float(mean_is),
            "mean_oos_max_dd": float(np.mean(oos_dd))  if oos_dd  else -999.0,
            "mean_oos_ann_ret": float(np.mean(oos_ret)) if oos_ret else -999.0,
            "n_windows":       len(oos_sh),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("stability_score", ascending=False).reset_index(drop=True)
    return df


def run_cvxport_grid_search(
    arith_rets: pd.DataFrame,
    train_start: str = "2022-01-01",
    train_end:   str = "2023-12-31",
    oos_start:   str = "2024-01-01",
    oos_end:     str | None = None,
    gamma_grid:         list[float] = GAMMA_GRID,
    cov_halflife_grid:  list[int]   = COV_HALFLIFE_GRID,
    max_wt_grid:        list[float] = MAX_WT_GRID,
    mu_shrinkage_grid:  list[float] = MU_SHRINKAGE_GRID,
    rebal_freq_grid:    list[str]   = REBAL_FREQ_GRID,
    mu_halflife: int = 252,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Single IS/OOS split grid search (legacy path).

    Prefer run_cvxport_rolling_cv() for statistically robust evaluation.
    This function is kept for quick single-window checks but its OOS results
    are highly sensitive to which market regime the OOS window lands on.

    Returns
    -------
    pd.DataFrame sorted by oos_sharpe descending.
    """
    combos = list(product(gamma_grid, cov_halflife_grid, max_wt_grid,
                          mu_shrinkage_grid, rebal_freq_grid))
    n = len(combos)
    print(f"[Single IS/OOS] {n} configurations")
    print(f"  WARNING: single-window scoring is regime-sensitive. Prefer --rolling-cv.")
    print(f"  Locked: mu_method=ewma, use_mpo=False, mu_halflife={mu_halflife}d")
    print(f"  IS:  {train_start} → {train_end}")
    print(f"  OOS: {oos_start} → {oos_end or 'latest'}\n")

    rows: list[dict[str, Any]] = []

    for idx, (gamma, cov_hl, max_wt, mu_shrink, rebal) in enumerate(combos, 1):
        label = (f"γ={gamma}, cov_hl={cov_hl}d, max_wt={max_wt:.0%}, "
                 f"shrink={mu_shrink:.0%}, rebal={rebal}")
        if not verbose:
            print(f"  [{idx:3d}/{n}] {label} … ", end="", flush=True)

        row: dict[str, Any] = dict(
            gamma=gamma, cov_halflife=cov_hl, max_wt=max_wt,
            mu_shrinkage=mu_shrink, rebal_freq=rebal,
            is_sharpe=np.nan, oos_sharpe=np.nan,
            oos_max_dd=np.nan, oos_ann_ret=np.nan, oos_ann_vol=np.nan,
        )

        try:
            pv_is, _ = run_cvxportfolio_backtest(
                returns_df=arith_rets, gamma=gamma, cov_halflife=cov_hl,
                mu_halflife=mu_halflife, max_wt=max_wt, rebal_freq=rebal,
                mu_method="ewma", mu_shrinkage=mu_shrink, use_mpo=False,
                start=train_start, end=train_end,
            )
            row["is_sharpe"] = _sharpe(pv_is)
        except Exception as exc:
            if verbose:
                print(f"    IS failed: {exc}")

        try:
            pv_oos, _ = run_cvxportfolio_backtest(
                returns_df=arith_rets, gamma=gamma, cov_halflife=cov_hl,
                mu_halflife=mu_halflife, max_wt=max_wt, rebal_freq=rebal,
                mu_method="ewma", mu_shrinkage=mu_shrink, use_mpo=False,
                start=oos_start, end=oos_end,
            )
            oos_rets = pv_oos.pct_change().dropna()
            roll_max = pv_oos.cummax()
            max_dd   = ((pv_oos - roll_max) / roll_max).min()

            row["oos_sharpe"]  = _sharpe(pv_oos)
            row["oos_ann_ret"] = _ann_ret(pv_oos)        # capped at 500%
            row["oos_ann_vol"] = float(oos_rets.std() * np.sqrt(252))
            row["oos_max_dd"]  = float(max_dd)

            if not verbose:
                print(f"OOS Sharpe={row['oos_sharpe']:.3f}", flush=True)
        except Exception as exc:
            if not verbose:
                print(f"FAILED: {exc}", flush=True)
            if verbose:
                print(f"    OOS failed: {exc}")

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    return df


def save_cvxport_best_params(
    results_df: pd.DataFrame,
    path: Path | str = BEST_PARAMS_PATH,
) -> dict:
    """Write the top result to cvxport_best_params.json for the live trader."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True)

    best = results_df.iloc[0]

    # Rolling CV output has stability_score; single-window has oos_sharpe
    oos_sharpe_col = "mean_oos_sharpe" if "mean_oos_sharpe" in best.index else "oos_sharpe"
    oos_ret_col    = "mean_oos_ann_ret" if "mean_oos_ann_ret" in best.index else "oos_ann_ret"
    oos_dd_col     = "mean_oos_max_dd"  if "mean_oos_max_dd"  in best.index else "oos_max_dd"
    stability_col  = "stability_score"  if "stability_score"  in best.index else "oos_sharpe"

    params = {
        "gamma":          float(best["gamma"]),
        "cov_halflife":   int(best["cov_halflife"]),
        "mu_halflife":    252,
        "max_wt":         float(best["max_wt"]),
        "mu_method":      "ewma",
        "mu_shrinkage":   float(best["mu_shrinkage"]),
        "use_mpo":        False,
        "rebal_freq":     str(best["rebal_freq"]),
        "oos_sharpe":     float(best[oos_sharpe_col]),
        "oos_ann_ret":    float(best[oos_ret_col]),
        "oos_max_dd":     float(best[oos_dd_col]),
        "stability_score": float(best[stability_col]),
    }
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nBest params saved → {path}")
    print(f"  γ={params['gamma']}, cov_hl={params['cov_halflife']}d, "
          f"max_wt={params['max_wt']:.0%}, shrink={params['mu_shrinkage']:.0%}, "
          f"rebal={params['rebal_freq']} "
          f"| stability={params['stability_score']:.3f}, "
          f"mean OOS Sharpe={params['oos_sharpe']:.3f}")
    return params


def load_cvxport_best_params(path: Path | str = BEST_PARAMS_PATH) -> dict | None:
    """Load best params from JSON. Returns None if file not found."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
