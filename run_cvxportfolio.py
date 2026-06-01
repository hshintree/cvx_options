"""
cvxportfolio-based extended backtest (2022–present).

Compares SPO configurations (EWMA μ + factorized Σ, varying shrinkage and γ)
against SPY buy-and-hold and 1/N equal-weight benchmarks.

Usage:
    python run_cvxportfolio.py                    # default configs (31-symbol universe)
    python run_cvxportfolio.py --save             # save plots to min_var_output/
    python run_cvxportfolio.py --symbols SPY TLT IEF GLD AAPL NVDA TSLA MSFT AMZN
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from min_var.config import R_ANN, EXPANDED_SYMBOLS, OUTPUT_DIR
from min_var.data_equity import load_equity_returns
from min_var.cvxport_backtest import run_cvxportfolio_backtest
from min_var.metrics import build_stats_table, format_stats_table

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="cvxportfolio extended backtest")
parser.add_argument("--symbols",  nargs="*", default=None)
parser.add_argument("--start",    default="2022-01-03")
parser.add_argument("--end",      default=None)
parser.add_argument("--max-wt",   type=float, default=0.20,
                    help="Per-asset weight cap (default: 0.20)")
parser.add_argument("--save",     action="store_true")
args = parser.parse_args()

if args.save:
    matplotlib.use("Agg")

OUTPUT_DIR.mkdir(exist_ok=True)

symbols    = args.symbols or EXPANDED_SYMBOLS
data_start = "2020-01-01"   # COIN IPO 2021-04-15; need pre-history warm-up

start    = args.start
end      = args.end
max_wt   = args.max_wt

# ── 1. Load data ──────────────────────────────────────────────────────────────
print(f"Loading returns data ({len(symbols)} symbols) …")
log_rets = load_equity_returns(symbols, start=data_start, end=end)
# cvxportfolio expects arithmetic (simple) returns
arith_rets = np.expm1(log_rets)

print(f"  {len(arith_rets)} days × {len(symbols)} assets  "
      f"({arith_rets.index.min().date()} → {arith_rets.index.max().date()})")
print(f"  max_wt = {max_wt:.0%}   backtest start = {start}")

# SPY benchmark
spy_bt   = log_rets.loc[log_rets.index >= start, "SPY"]
spy_path = (1 + spy_bt).cumprod()
spy_path = spy_path / spy_path.iloc[0]
spy_path.name = "SPY Buy-and-Hold"

# Equal-weight benchmark (1/N across the full universe)
ew_rets  = arith_rets.loc[arith_rets.index >= start].mean(axis=1)
ew_path  = (1 + ew_rets).cumprod()
ew_path  = ew_path / ew_path.iloc[0]
ew_path.name = f"1/N Equal-Weight ({len(symbols)})"

# ── 2. cvxportfolio experiments ───────────────────────────────────────────────
# Each tuple: (label, gamma, cov_hl, mu_hl, mu_method, rebal_freq, use_mpo, mu_shrinkage)
CVX_RUNS = [
    # (label, gamma, cov_hl, mu_hl, mu_method, rebal_freq, use_mpo, mu_shrinkage)
    # ── Shrinkage variants (quarterly) ────────────────────────────────────────
    ("cvx EWMA γ=5  SPO  shrink=50% Q",  5.0, 21, 252, "ewma", "quarterly", False, 0.50),
    ("cvx EWMA γ=5  SPO  shrink=75% Q",  5.0, 21, 252, "ewma", "quarterly", False, 0.75),
    ("cvx EWMA γ=5  SPO  shrink=80% Q",  5.0, 21, 252, "ewma", "quarterly", False, 0.80),
    ("cvx EWMA γ=10 SPO  shrink=80% Q", 10.0, 21, 252, "ewma", "quarterly", False, 0.80),
    # ── Pure min-var (EWMA Σ, no μ) ───────────────────────────────────────────
    ("cvx min-var   SPO  (μ=0) Q",       5.0, 21, 252, "zero", "quarterly", False, 0.0),
]

cvx_paths  = {}
cvx_metas  = {}

for label, g, cov_hl, mu_hl, mu_m, rf_str, mpo, shrink in CVX_RUNS:
    print(f"\nRunning {label} …")
    try:
        pv, meta = run_cvxportfolio_backtest(
            returns_df=arith_rets,
            gamma=g,
            cov_halflife=cov_hl,
            mu_halflife=mu_hl,
            max_wt=max_wt,
            rebal_freq=rf_str,
            mu_method=mu_m,
            mu_shrinkage=shrink,
            use_mpo=mpo,
            planning_horizon=3,
            start=start,
            end=end,
        )
        pv.name = label
        cvx_paths[label] = pv
        cvx_metas[label] = meta
        print(f"  Final value: {pv.iloc[-1]:.4f}")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback; traceback.print_exc()

# ── 4. Performance summary ────────────────────────────────────────────────────
all_paths = [spy_path, ew_path] + list(cvx_paths.values())
all_wts   = [pd.DataFrame()] * len(all_paths)

print("\n── Performance Summary ─────────────────────────────────────────────────────")
stats = build_stats_table(all_paths, wts_list=all_wts, benchmark=spy_path)
print(format_stats_table(stats).to_string())

# ── 4b. Average weight analysis ───────────────────────────────────────────────
# Shows WHAT each strategy was actually holding. Critical for interpreting Sharpe
# numbers — a 0.60 Sharpe from a diversified portfolio is very different from the
# same number coming from 50%+ NVDA concentration during a regime-specific bull run.
print("\n── Average Weights over backtest (equity share, cash excluded) ──────────────")
cvx_avg_wts: dict[str, pd.Series] = {}
for label, meta in cvx_metas.items():
    # min-var path stores weights directly in meta["w"] (rolling CVXPY)
    # cvxportfolio paths store them in meta["result"].w
    if meta.get("result") is not None:
        wts = meta["result"].w.drop(columns=["cash"], errors="ignore")
    elif isinstance(meta.get("w"), pd.DataFrame) and not meta["w"].empty:
        wts = meta["w"]
    else:
        continue
    # Only count periods where the strategy is actually invested (past warm-up)
    wts_active = wts[wts.abs().sum(axis=1) > 0.01]
    if len(wts_active):
        # Renormalize to equity-only (exclude cash fraction)
        equity_sum = wts_active.clip(lower=0).sum(axis=1)
        wts_norm   = wts_active.clip(lower=0).divide(equity_sum.clip(lower=1e-6), axis=0)
        cvx_avg_wts[label] = wts_norm.mean()

if cvx_avg_wts:
    # Build display header
    col_width = 16
    header = f"  {'Asset':<6}"
    for label in cvx_avg_wts:
        short = (label.replace("cvx monthly ", "")
                      .replace("cvx daily ", "daily/")
                      .replace(" SPO", "")
                      .replace(" MPO(3)", " MPO"))
        header += f"  {short[:col_width]:>{col_width}}"
    print(header)
    print("  " + "─" * (8 + (col_width + 2) * len(cvx_avg_wts)))

    for asset in arith_rets.columns:
        row = f"  {asset:<6}"
        any_nonzero = False
        for label, avg in cvx_avg_wts.items():
            w = avg.get(asset, 0.0)
            row += f"  {w:>{col_width}.1%}"
            if w > 0.005:
                any_nonzero = True
        if any_nonzero:
            print(row)

    print()
    print("  Note: weights renormalized to equity-only (cash position excluded).")
    print("        Top allocations reveal whether returns are concentrated or diversified.")

    # Flag concentration risk
    for label, avg in cvx_avg_wts.items():
        top_asset = avg.idxmax()
        top_wt    = avg.max()
        if top_wt > 0.35:
            print(f"  ⚠  {label}: {top_asset} avg weight = {top_wt:.1%} — concentrated position.")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
spy_path.plot(ax=ax, color="black", linewidth=2.5, label="SPY Buy-and-Hold")
ew_path.plot(ax=ax,  color="grey",  linewidth=1.5, linestyle="--", label=ew_path.name)
for label, pv in cvx_paths.items():
    pv.plot(ax=ax, linewidth=1.5, label=label)

ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax.set_title(f"cvxportfolio  ({len(symbols)}-symbol universe, max_wt={max_wt:.0%})  2022–")
ax.set_ylabel("Portfolio Value (start = 1.0)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()

if args.save:
    p = OUTPUT_DIR / "cvxport_01_trajectories.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {p}")
    plt.close(fig)
else:
    plt.show()

# ── 6. Average weight bar chart ───────────────────────────────────────────────
if cvx_avg_wts:
    wt_df = pd.DataFrame(cvx_avg_wts).fillna(0.0)
    wt_df = wt_df.loc[(wt_df > 0.005).any(axis=1)]   # drop near-zero assets

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    x      = np.arange(len(wt_df))
    width  = 0.8 / len(wt_df.columns)
    colors = plt.cm.tab10(np.linspace(0, 1, len(wt_df.columns)))

    for i, (col, color) in enumerate(zip(wt_df.columns, colors)):
        short = (col.replace("cvx monthly ", "").replace("cvx daily ", "d/")
                    .replace(" SPO", "").replace(" MPO(3)", " MPO"))
        ax2.bar(x + i * width, wt_df[col].values, width, label=short, color=color)

    ax2.set_xticks(x + width * (len(wt_df.columns) - 1) / 2)
    ax2.set_xticklabels(wt_df.index, rotation=30, ha="right")
    ax2.set_ylabel("Average weight (equity-normalised)")
    ax2.set_title("Average Portfolio Weights — cvxportfolio strategies")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    if args.save:
        p2 = OUTPUT_DIR / "cvxport_02_avg_weights.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p2}")
        plt.close(fig2)
    else:
        plt.show()

print("\nDone.")
