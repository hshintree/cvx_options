"""
Grid search over cvxportfolio hyperparameters.

Uses rolling walk-forward cross-validation (default) to avoid overfitting a
single IS/OOS split to a bull-market OOS window.  Each config is scored across
5-6 non-overlapping 6-month OOS periods, including bear-market folds.

Scoring metric: stability_score = mean_OOS_Sharpe / std_OOS_Sharpe
(Sharpe-of-Sharpes — prefers configs that are consistently good, not just
lucky in one 2024-bull-market window).

Locked by statistical validation (not tuned):
  mu_method=ewma, use_mpo=False, mu_halflife=252d

Usage:
    python run_cvxportfolio_grid_search.py                 # rolling CV, full grid
    python run_cvxportfolio_grid_search.py --quick         # reduced grid (~24 combos × folds)
    python run_cvxportfolio_grid_search.py --save          # save CSV + heatmap
    python run_cvxportfolio_grid_search.py --single-split  # legacy: one IS/OOS window
    python run_cvxportfolio_grid_search.py --symbols SPY TLT IEF GLD AAPL NVDA MSFT
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from min_var.config import EXPANDED_SYMBOLS, OUTPUT_DIR
from min_var.data_equity import load_equity_returns
from min_var.cvxport_grid_search import (
    run_cvxport_rolling_cv, run_cvxport_grid_search, save_cvxport_best_params,
    GAMMA_GRID, COV_HALFLIFE_GRID, MAX_WT_GRID,
    MU_SHRINKAGE_GRID, REBAL_FREQ_GRID,
    QUICK_GAMMA_GRID, QUICK_COV_HALFLIFE_GRID, QUICK_MAX_WT_GRID,
    QUICK_MU_SHRINKAGE_GRID, QUICK_REBAL_FREQ_GRID,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="cvxportfolio hyperparameter grid search (rolling walk-forward CV)"
)
parser.add_argument("--symbols",       nargs="*", default=None)
parser.add_argument("--quick",         action="store_true",
                    help="Reduced grid for fast testing (~24 combos)")
parser.add_argument("--save",          action="store_true",
                    help="Save CSV results and heatmap PNG")
parser.add_argument("--max-wt",        type=float, default=None,
                    help="Override max_wt grid (single value, e.g. 0.20)")
# Rolling CV controls
parser.add_argument("--is-months",     type=int, default=18,
                    help="IS window length per fold in months (default: 18)")
parser.add_argument("--oos-months",    type=int, default=6,
                    help="OOS window length per fold in months (default: 6)")
parser.add_argument("--step-months",   type=int, default=6,
                    help="Rolling step size in months (default: 6)")
# Legacy single-split mode
parser.add_argument("--single-split",  action="store_true",
                    help="Use legacy single IS/OOS split instead of rolling CV")
parser.add_argument("--train-start",   default="2022-01-01")
parser.add_argument("--train-end",     default="2023-12-31")
parser.add_argument("--oos-start",     default="2024-01-01")
parser.add_argument("--oos-end",       default=None)
args = parser.parse_args()

if args.save:
    matplotlib.use("Agg")

OUTPUT_DIR.mkdir(exist_ok=True)

symbols    = args.symbols or EXPANDED_SYMBOLS
data_start = "2020-01-01"   # COIN IPO 2021-04-15

# ── 1. Load data ──────────────────────────────────────────────────────────────
print(f"Loading returns data ({len(symbols)} symbols) …")
log_rets   = load_equity_returns(symbols, start=data_start,
                                  end=None if not args.single_split else args.oos_end)
import numpy as _np
arith_rets = _np.expm1(log_rets)

print(f"  {len(arith_rets)} days × {len(symbols)} assets  "
      f"({arith_rets.index.min().date()} → {arith_rets.index.max().date()})")

# ── 2. Build search grids ─────────────────────────────────────────────────────
if args.quick:
    g_grid      = QUICK_GAMMA_GRID
    ch_grid     = QUICK_COV_HALFLIFE_GRID
    mw_grid     = QUICK_MAX_WT_GRID
    shrink_grid = QUICK_MU_SHRINKAGE_GRID
    rf_grid     = QUICK_REBAL_FREQ_GRID
else:
    g_grid      = GAMMA_GRID
    ch_grid     = COV_HALFLIFE_GRID
    mw_grid     = MAX_WT_GRID
    shrink_grid = MU_SHRINKAGE_GRID
    rf_grid     = REBAL_FREQ_GRID

if args.max_wt is not None:
    mw_grid = [args.max_wt]

# ── 3. Run grid search ────────────────────────────────────────────────────────
if args.single_split:
    results = run_cvxport_grid_search(
        arith_rets=arith_rets,
        train_start=args.train_start,
        train_end=args.train_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        gamma_grid=g_grid,
        cov_halflife_grid=ch_grid,
        max_wt_grid=mw_grid,
        mu_shrinkage_grid=shrink_grid,
        rebal_freq_grid=rf_grid,
    )
    display_cols = ["gamma", "cov_halflife", "max_wt", "mu_shrinkage", "rebal_freq",
                    "is_sharpe", "oos_sharpe", "oos_ann_ret", "oos_max_dd"]
    sort_col = "oos_sharpe"
else:
    results = run_cvxport_rolling_cv(
        arith_rets=arith_rets,
        gamma_grid=g_grid,
        cov_halflife_grid=ch_grid,
        max_wt_grid=mw_grid,
        mu_shrinkage_grid=shrink_grid,
        rebal_freq_grid=rf_grid,
        is_months=args.is_months,
        oos_months=args.oos_months,
        step_months=args.step_months,
    )
    display_cols = ["gamma", "cov_halflife", "max_wt", "mu_shrinkage", "rebal_freq",
                    "mean_oos_sharpe", "std_oos_sharpe", "min_oos_sharpe",
                    "stability_score", "mean_is_sharpe", "mean_oos_max_dd"]
    sort_col = "stability_score"

# ── 4. Summary ────────────────────────────────────────────────────────────────
print(f"\n── Top 10 Configurations (by {sort_col}) ───────────────────────────────")
print(results[display_cols].head(10).to_string(index=False, float_format="{:.3f}".format))

# ── 5. Save best params ───────────────────────────────────────────────────────
best = save_cvxport_best_params(results)

# ── 6. Save CSV ───────────────────────────────────────────────────────────────
if args.save:
    csv_path = OUTPUT_DIR / "cvxport_grid_search_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nFull results saved → {csv_path}")

# ── 7. Heatmap: stability_score (γ × cov_halflife) ──────────────────────────
print("\nGenerating Sharpe heatmap …")

score_col = "stability_score" if "stability_score" in results.columns else "oos_sharpe"

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, rf in zip(axes, ["monthly", "quarterly"]):
    sub = results[results["rebal_freq"] == rf]
    if sub.empty:
        ax.text(0.5, 0.5, f"No {rf} results", transform=ax.transAxes, ha="center")
        continue

    pivot = (sub.groupby(["gamma", "cov_halflife"])[score_col]
               .max()
               .unstack("cov_halflife"))

    vals   = pivot.values
    finite = vals[np.isfinite(vals)]
    vmin   = finite.min() if len(finite) else 0
    vmax   = finite.max() if len(finite) else 1

    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}d" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"γ={g}" for g in pivot.index])
    ax.set_xlabel("Covariance half-life")
    ax.set_ylabel("Risk aversion γ")

    metric_label = "Stability (mean/std OOS Sharpe)" if score_col == "stability_score" else "OOS Sharpe"
    ax.set_title(f"{metric_label}\n{rf} rebalancing  (best shrinkage & max_wt per cell)")
    plt.colorbar(im, ax=ax)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()

if args.save:
    p = OUTPUT_DIR / "cvxport_grid_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved → {p}")
    plt.close(fig)
else:
    plt.show()

print("\nDone.")
