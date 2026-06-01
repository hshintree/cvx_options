"""
Walk-forward forecast accuracy evaluation for EWMA μ and Σ estimators.

Tests whether the EWMA return forecast (HistoricalMeanReturn) and covariance
forecast (HistoricalFactorizedCovariance) used in the cvxportfolio engine
have statistically defensible out-of-sample predictive power.

Key outputs
-----------
- Per-asset: bias, MAE, Pearson r (pred vs realized), hit rate (direction)
- Pooled IC t-test: is cross-sectional Spearman IC significantly > 0?
- Binomial hit-rate test: is direction accuracy significantly > 50%?
- Variance ratio: is the covariance forecast well-calibrated?
- Split estimator comparison: standard EWMA vs fast-vol / slow-corr Σ = D×C×D
- Overfitting flag: if IC is not significant, high Sharpe may be regime luck

Usage
-----
    python run_forecast_accuracy.py                  # default config
    python run_forecast_accuracy.py --save           # save CSV results
    python run_forecast_accuracy.py --mu-halflife 63 --cov-halflife 10
    python run_forecast_accuracy.py --horizon 5      # 5-day forecast horizon
    python run_forecast_accuracy.py --start 2022-01-01 --end 2024-01-01
    python run_forecast_accuracy.py --corr-halflife 63   # split estimator
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from min_var.config import EXTENDED_SYMBOLS, OUTPUT_DIR
from min_var.data_equity import load_equity_returns
from min_var.cvxport_accuracy import (
    run_ewma_forecast_accuracy, print_accuracy_report, summarize_forecast_accuracy,
    extract_correlation_pairs,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EWMA forecast accuracy evaluation")
parser.add_argument("--symbols",        nargs="*", default=None)
parser.add_argument("--mu-halflife",    type=int,   default=252)
parser.add_argument("--cov-halflife",   type=int,   default=21)
parser.add_argument("--corr-halflife",  type=int,   default=63,
                    help="Σ correlation half-life for split estimator (default: 63d). "
                         "Set equal to --cov-halflife to disable split.")
parser.add_argument("--horizon",        type=int,   default=21,
                    help="Forecast horizon in trading days (default: 21 = ~1 month)")
parser.add_argument("--start",          default="2021-01-01",
                    help="Start of evaluation window (default: 2021-01-01)")
parser.add_argument("--end",            default=None)
parser.add_argument("--rebal-freq",     default="monthly",
                    help="Rebalance frequency: monthly|quarterly|<int> (default: monthly)")
parser.add_argument("--save",           action="store_true",
                    help="Save CSV results and plots to min_var_output/")
args = parser.parse_args()

if args.save:
    matplotlib.use("Agg")

OUTPUT_DIR.mkdir(exist_ok=True)
symbols = args.symbols or EXTENDED_SYMBOLS

use_split = (args.corr_halflife is not None and args.corr_halflife != args.cov_halflife)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading returns data …")
log_rets   = load_equity_returns(symbols, start="2019-01-01", end=args.end)
arith_rets = np.expm1(log_rets)
print(f"  {len(arith_rets)} days × {len(symbols)} assets  "
      f"({arith_rets.index.min().date()} → {arith_rets.index.max().date()})")

# ── 2. Run accuracy evaluation ────────────────────────────────────────────────
print(f"\nRunning walk-forward evaluation …")
print(f"  μ EWMA halflife:   {args.mu_halflife} trading days")
print(f"  Σ vol halflife:    {args.cov_halflife} trading days")
if use_split:
    print(f"  Σ corr halflife:   {args.corr_halflife} trading days  [split estimator: Σ=D×C×D]")
print(f"  Forecast horizon:  {args.horizon} trading days")
print(f"  Rebalance freq:    {args.rebal_freq}")
print(f"  Eval window:       {args.start} → {args.end or 'latest'}")

mu_df, cov_df = run_ewma_forecast_accuracy(
    arith_rets=arith_rets,
    mu_halflife=args.mu_halflife,
    cov_halflife=args.cov_halflife,
    corr_halflife=args.corr_halflife if use_split else None,
    forecast_horizon=args.horizon,
    rebal_freq=args.rebal_freq,
    start=args.start,
    end=args.end,
)

# ── 3. Print report ────────────────────────────────────────────────────────────
print_accuracy_report(mu_df, cov_df)

# ── 4. Horizon sensitivity check (1-month vs 3-month) ─────────────────────────
print("\nRunning horizon sensitivity check (1m vs 3m) …")
results_by_horizon = {}
for h in [21, 63]:
    mdf, cdf = run_ewma_forecast_accuracy(
        arith_rets=arith_rets, mu_halflife=args.mu_halflife,
        cov_halflife=args.cov_halflife,
        corr_halflife=args.corr_halflife if use_split else None,
        forecast_horizon=h,
        rebal_freq=args.rebal_freq, start=args.start, end=args.end,
    )
    summ = summarize_forecast_accuracy(mdf, cdf)
    results_by_horizon[h] = summ

print(f"\n{'Horizon':>8}  {'Mean IC':>10}  {'p-value':>10}  {'Hit rate':>10}  "
      f"{'p-value':>10}  {'Var ratio':>10}")
print("─" * 65)
for h, s in results_by_horizon.items():
    ict = s["ic_test"]
    ht  = s["hit_test"]
    cs  = s["cov_summary"]
    sig_flag = " *" if ict["p_value"] < 0.05 else "  "
    print(f"{str(h)+'d':>8}  {ict['mean_ic']:>+10.4f}  {ict['p_value']:>10.4f}{sig_flag} "
          f"{ht['hit_rate']:>10.1%}  {ht['p_value']:>10.4f}  "
          f"{cs['var_ratio_mean']:>10.3f}")
print("  (* = significant at 5%)")

# ── 5. Pairwise correlation accuracy: standard vs split estimator ──────────────
print("\nExtracting pairwise correlation accuracy …")

corr_std = extract_correlation_pairs(
    arith_rets,
    cov_halflife=args.cov_halflife,
    corr_halflife=None,          # uniform EWMA
    forecast_horizon=args.horizon,
    start=args.start,
    end=args.end,
)
r_std = corr_std["pred_corr"].corr(corr_std["real_corr"]) if len(corr_std) else float("nan")

if use_split:
    corr_split = extract_correlation_pairs(
        arith_rets,
        cov_halflife=args.cov_halflife,
        corr_halflife=args.corr_halflife,   # split: slow correlations
        forecast_horizon=args.horizon,
        start=args.start,
        end=args.end,
    )
    r_split = corr_split["pred_corr"].corr(corr_split["real_corr"]) if len(corr_split) else float("nan")

print(f"\n── Pairwise ρ accuracy — off-diagonal Σ ───────────────────────────────────")
print(f"  Standard EWMA (vol_hl=corr_hl={args.cov_halflife}d)      Pearson r = {r_std:.4f}")
if use_split:
    delta = r_split - r_std
    sign  = "+" if delta >= 0 else ""
    print(f"  Split   EWMA (vol_hl={args.cov_halflife}d, corr_hl={args.corr_halflife}d)  Pearson r = {r_split:.4f}  ({sign}{delta:.4f})")
    if delta > 0:
        print(f"  → Split estimator improves pairwise ρ accuracy by {abs(delta):.4f}")
    else:
        print(f"  → No improvement from split estimator at these half-lives")

# ── 6. Plots ───────────────────────────────────────────────────────────────────
if len(mu_df) > 0:
    ncols = 3 if use_split else 2
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 10))

    # (a) IC over time
    ax = axes[0, 0]
    cov_df.set_index("date")["ic"].plot(ax=ax, color="steelblue", linewidth=0.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(cov_df["ic"].mean(), color="red", linewidth=1.5,
               linestyle="--", label=f"Mean IC = {cov_df['ic'].mean():.3f}")
    ax.set_title("Cross-sectional IC over time\n(Spearman rank, pred μ vs realized μ)")
    ax.set_ylabel("IC")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Scatter: predicted vs realized μ
    ax = axes[0, 1]
    ax.scatter(mu_df["pred_mu"], mu_df["real_mu"],
               alpha=0.25, s=12, color="steelblue")
    lim = max(abs(mu_df["pred_mu"]).quantile(0.98),
              abs(mu_df["real_mu"]).quantile(0.98))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8, label="perfect forecast")
    r = mu_df["pred_mu"].corr(mu_df["real_mu"])
    ax.set_title(f"Predicted vs Realized μ (annualised)\nPearson r = {r:.3f}")
    ax.set_xlabel("Predicted μ")
    ax.set_ylabel("Realized μ")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Variance ratio per asset (diagonal Σ)
    ax = axes[1, 0]
    var_ratio_by_asset = mu_df.groupby("asset").apply(
        lambda g: (g["pred_vol"] / (g["real_vol"] + 1e-6)).values
    )
    ax.boxplot([var_ratio_by_asset[a] for a in var_ratio_by_asset.index],
               labels=var_ratio_by_asset.index, vert=True)
    ax.axhline(1.0, color="red", linewidth=1, linestyle="--", label="perfect calibration")
    ax.set_title("Vol forecast calibration — diagonal Σ\n(pred σ / realized σ per asset)")
    ax.set_ylabel("Ratio")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # (d) Standard pairwise correlation accuracy
    ax = axes[1, 1]
    if len(corr_std):
        ax.scatter(corr_std["pred_corr"], corr_std["real_corr"],
                   alpha=0.08, s=6, color="darkorange", rasterized=True)
        lo, hi = -1.0, 1.0
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.9, label="perfect forecast")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.axhline(0, color="grey", linewidth=0.4)
        ax.axvline(0, color="grey", linewidth=0.4)
        title_std = (f"Pairwise ρ — standard EWMA (hl={args.cov_halflife}d)\n"
                     f"Pearson r = {r_std:.3f}  ({len(corr_std):,} pair×date obs)")
        ax.set_title(title_std)
        ax.set_xlabel("Predicted ρ(i,j)")
        ax.set_ylabel("Realized ρ(i,j)")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No correlation pairs", transform=ax.transAxes, ha="center")
    ax.grid(alpha=0.3)

    # (e) Split pairwise correlation accuracy (right column, only when split requested)
    if use_split:
        # Row 0 col 2: horizon IC comparison bar chart
        ax = axes[0, 2]
        hs  = list(results_by_horizon.keys())
        ics = [results_by_horizon[h]["ic_test"]["mean_ic"] for h in hs]
        pvs = [results_by_horizon[h]["ic_test"]["p_value"] for h in hs]
        colors = ["#e74c3c" if p >= 0.05 else "#2ecc71" for p in pvs]
        ax.bar([f"{h}d" for h in hs], ics, color=colors, edgecolor="black", linewidth=0.8)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title("Mean IC by horizon\n(green = p<0.05 significant)")
        ax.set_ylabel("Mean IC")
        for i, (ic, pv) in enumerate(zip(ics, pvs)):
            ax.text(i, ic + 0.002, f"p={pv:.3f}", ha="center", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        # Row 1 col 2: split estimator pairwise accuracy
        ax = axes[1, 2]
        if len(corr_split):
            ax.scatter(corr_split["pred_corr"], corr_split["real_corr"],
                       alpha=0.08, s=6, color="steelblue", rasterized=True)
            lo, hi = -1.0, 1.0
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.9, label="perfect forecast")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.axhline(0, color="grey", linewidth=0.4)
            ax.axvline(0, color="grey", linewidth=0.4)
            delta_str = f"{'+' if r_split >= r_std else ''}{r_split - r_std:.3f} vs std"
            title_split = (f"Pairwise ρ — split EWMA (vol_hl={args.cov_halflife}d, corr_hl={args.corr_halflife}d)\n"
                           f"Pearson r = {r_split:.3f}  ({delta_str})")
            ax.set_title(title_split)
            ax.set_xlabel("Predicted ρ(i,j)")
            ax.set_ylabel("Realized ρ(i,j)")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No split correlation pairs", transform=ax.transAxes, ha="center")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if args.save:
        p = OUTPUT_DIR / "forecast_accuracy.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {p}")
        plt.close(fig)
    else:
        plt.show()

# ── 7. Save CSVs ──────────────────────────────────────────────────────────────
if args.save:
    mu_path  = OUTPUT_DIR / "forecast_accuracy_mu.csv"
    cov_path = OUTPUT_DIR / "forecast_accuracy_cov.csv"
    mu_df.to_csv(mu_path, index=False)
    cov_df.to_csv(cov_path, index=False)
    print(f"μ accuracy CSV  → {mu_path}")
    print(f"Σ accuracy CSV  → {cov_path}")

print("\nDone.")
