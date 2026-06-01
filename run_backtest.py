"""
End-to-end backtest: fetch data → backtest → forecast accuracy report.

Supports four rebalancing frequencies:
  monthly   (default) — cvxportfolio SPO, first trading day of each calendar month
  quarterly           — cvxportfolio SPO, first trading day of each quarter
  weekly              — aim-portfolio walk-forward + TC analysis (~52 rebal/yr)
  daily               — aim-portfolio walk-forward + TC analysis (~252 rebal/yr)

For daily/weekly, the Garleanu-Pedersen aim portfolio is used:
  w_new = w_old + λ × (w_target − w_old)
  λ = 1 / (1 + TC_per_trade / α_per_trade)
This dampens each trade to the fraction that equates marginal TC against marginal alpha,
producing a smooth, TC-aware trajectory rather than discrete full rebalances.

TC viability at avg ~5 bps one-way (31-symbol universe, ~10 bps round-trip):
  Daily   (252/yr):  gross TC ≈ 30% p.a. — only viable with aim dampening (λ ≈ 0.04)
  Weekly  (52/yr):   gross TC ≈  6% p.a. — marginal; aim brings net TC to ~1.5%
  Monthly (12/yr):   gross TC ≈1.4% p.a. — manageable; full rebalance is fine
  Quarterly (4/yr):  gross TC ≈0.5% p.a. — best; IC at 63d (p=0.007) is significant

Default hyperparameters:
  gamma        = 10.0
  mu_shrinkage = 0.80  (80% James-Stein shrinkage; EWMA IC not significant at p<0.05)
  rebal_freq   = monthly
  cov_halflife = 21 td
  mu_halflife  = 252 td
  max_wt       = 0.20

Usage:
    python run_backtest.py                              # monthly (default)
    python run_backtest.py --rebal-freq quarterly
    python run_backtest.py --rebal-freq weekly
    python run_backtest.py --rebal-freq daily
    python run_backtest.py --symbols SPY QQQ TLT IEF GLD --save
    python run_backtest.py --gamma 10 --mu-shrinkage 0.5
    python run_backtest.py --start 2023-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from min_var.config import EXPANDED_SYMBOLS, OUTPUT_DIR, R_ANN
from min_var.data_equity import ensure_equity_data, load_equity_returns
from min_var.cvxport_backtest import run_cvxportfolio_backtest
from min_var.cvxport_accuracy import (
    run_ewma_forecast_accuracy, print_accuracy_report,
    extract_correlation_pairs, _ewma_cov, _ewma_mean,
)
from min_var.metrics import build_stats_table, format_stats_table

# ── Transaction cost utilities (Garleanu-Pedersen) ────────────────────────────
# Bid-ask half-spread per asset in basis points (one-way).
HALF_SPREAD_BPS: dict[str, int] = {
    "SPY": 1, "QQQ": 1, "IWM": 2, "SHY": 2,
    "TLT": 2, "IEF": 2, "TIP": 2, "GLD": 2,
    "XLV": 3, "XLF": 3, "XLY": 3, "XLP": 3, "VNQ": 3, "EFA": 3,
    "HYG": 3, "SLV": 4, "DBC": 4, "ICLN": 4, "EEM": 4, "AMT": 4,
    "AAPL": 4, "MSFT": 4, "AMZN": 4, "GOOGL": 4, "META": 4,
    "NVDA": 7, "TSLA": 7, "AMD": 7,
    "COIN": 12, "MSTR": 15, "CPER": 5,
}
_DEFAULT_HALF_SPREAD_BPS = 5

REBAL_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}


def _avg_half_spread_bps(symbols: list[str]) -> float:
    return float(np.mean([HALF_SPREAD_BPS.get(s, _DEFAULT_HALF_SPREAD_BPS) for s in symbols]))


def _aim_lambda(rt_bps: float, ann_alpha: float, rebal_per_year: float) -> float:
    """Garleanu-Pedersen trade fraction λ = 1 / (1 + TC_per_trade / α_per_trade)."""
    tc  = rt_bps / 10_000
    alp = max(ann_alpha / rebal_per_year, 1e-9)
    return 1.0 / (1.0 + tc / alp)


def _net_annual_tc(rt_bps: float, rebal_per_year: float, lam: float) -> float:
    """Effective annual TC after aim dampening: λ × round_trip × rebalances/yr."""
    return lam * (rt_bps / 10_000) * rebal_per_year


# ── Defaults ──────────────────────────────────────────────────────────────────
COV_HALFLIFE   = 21
MU_HALFLIFE    = 252
GAMMA_DEFAULT  = 10.0
SHRINK_DEFAULT = 0.80
ANN_ALPHA_EST  = 0.10   # gross alpha estimate for λ calculation (illustrative)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="End-to-end backtest")
parser.add_argument("--symbols",      nargs="*", default=None)
parser.add_argument("--start",        default="2022-01-03")
parser.add_argument("--end",          default=None)
parser.add_argument("--rebal-freq",   choices=["daily", "weekly", "monthly", "quarterly"],
                    default="monthly",
                    help="Rebalancing frequency (default: monthly). "
                         "daily/weekly use Garleanu-Pedersen aim portfolio.")
parser.add_argument("--gamma",        type=float, default=GAMMA_DEFAULT)
parser.add_argument("--mu-shrinkage", type=float, default=SHRINK_DEFAULT,
                    help="James-Stein shrinkage intensity [0,1] (default: 0.80)")
parser.add_argument("--cov-halflife", type=int, default=COV_HALFLIFE)
parser.add_argument("--mu-halflife",  type=int, default=MU_HALFLIFE)
parser.add_argument("--max-wt",       type=float, default=0.20)
parser.add_argument("--no-fetch",     action="store_true",
                    help="Skip data fetch (assume parquets already exist)")
parser.add_argument("--save",         action="store_true",
                    help="Save figures to min_var_output/")
args = parser.parse_args()

if args.save:
    matplotlib.use("Agg")

OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Ensure data ────────────────────────────────────────────────────────────
symbols    = args.symbols or EXPANDED_SYMBOLS
data_start = "2020-01-01"

if not args.no_fetch:
    print(f"Ensuring price data for {len(symbols)} symbols …")
    ensure_equity_data(symbols, start=data_start)

print("Loading returns …")
log_rets   = load_equity_returns(symbols, start=data_start, end=args.end)
arith_rets = np.expm1(log_rets)
print(f"  {len(arith_rets)} days × {len(symbols)} assets  "
      f"({arith_rets.index.min().date()} → {arith_rets.index.max().date()})")

# ── 2. TC analysis (always computed; critical for daily/weekly) ───────────────
rt_bps  = 2.0 * _avg_half_spread_bps(symbols)
n_rebal = REBAL_PER_YEAR[args.rebal_freq]
lam     = _aim_lambda(rt_bps, ANN_ALPHA_EST, n_rebal)
net_tc  = _net_annual_tc(rt_bps, n_rebal, lam)

# ── 3. Print configuration ────────────────────────────────────────────────────
_aim_note = (f"aim λ={lam:.4f} → net TC ≈{net_tc:.1%} p.a."
             if args.rebal_freq in ("daily", "weekly") else
             f"net TC ≈{net_tc:.1%} p.a. (gross {rt_bps/10_000*n_rebal:.1%}, manageable)")
print(f"""
┌─ Backtest configuration ──────────────────────────────────────────────────┐
│  Universe       : {len(symbols)} symbols
│  Objective      : max (μ − rf)' w − γ · w' Σ w  (Markowitz SPO)
│  μ method       : EWMA (hl={args.mu_halflife}d) × (1 − {args.mu_shrinkage:.0%} James-Stein shrinkage)
│  Σ method       : EWMA factorized covariance, hl={args.cov_halflife}d
│  γ (risk aver.) : {args.gamma}
│  Rebalance      : {args.rebal_freq}  ({_aim_note})
│  Max weight     : {args.max_wt:.0%} per asset
│  Backtest       : {args.start} → {args.end or "latest"}
│  Constraints    : long-only, 0 ≤ w ≤ max_wt
└────────────────────────────────────────────────────────────────────────────┘""")

if args.rebal_freq in ("daily", "weekly"):
    print(f"""
  ── TC Analysis ({args.rebal_freq}) ───────────────────────────────────────────
  Avg half-spread : {_avg_half_spread_bps(symbols):.1f} bps one-way ({rt_bps:.1f} bps round-trip)
  Rebalances/yr   : {n_rebal}
  Gross TC/yr     : {rt_bps/10_000*n_rebal:.1%}  (without dampening)
  Aim λ           : {lam:.4f}  (move {lam*100:.1f}% toward Markowitz target per period)
  Net TC/yr       : {net_tc:.1%}  (after Garleanu-Pedersen dampening)
  Assumed α/yr    : {ANN_ALPHA_EST:.0%}  (used for λ only — illustrative)
  ─────────────────────────────────────────────────────────────────────────────
  Two paths are shown:
    "Full rebalance" — snaps fully to Markowitz target each period (TC not deducted)
    "Aim portfolio"  — moves λ={lam:.4f} fraction toward target (TC not deducted)
  The aim portfolio is what you would *actually* execute.
""")

# ── 4. Run backtest ───────────────────────────────────────────────────────────
pv_aim      = None   # only set for daily/weekly
pv_full_tc  = None   # TC-adjusted full rebalance (daily/weekly)
pv_aim_tc   = None   # TC-adjusted aim portfolio  (daily/weekly)
meta        = {}

if args.rebal_freq in ("monthly", "quarterly"):
    # ── cvxportfolio SPO path ─────────────────────────────────────────────────
    print("Running backtest …")
    pv, meta = run_cvxportfolio_backtest(
        returns_df=arith_rets,
        gamma=args.gamma,
        cov_halflife=args.cov_halflife,
        mu_halflife=args.mu_halflife,
        max_wt=args.max_wt,
        rebal_freq=args.rebal_freq,
        mu_method="ewma",
        mu_shrinkage=args.mu_shrinkage,
        start=args.start,
        end=args.end,
    )
    pv.name = f"Markowitz SPO (γ={args.gamma}, shrink={args.mu_shrinkage:.0%}, {args.rebal_freq})"

else:
    # ── Aim-portfolio walk-forward loop (daily / weekly) ──────────────────────
    import cvxpy as cp

    MIN_HIST   = args.cov_halflife * 6
    N          = len(symbols)
    rf_daily   = R_ANN / 252.0

    start_dt = pd.Timestamp(args.start)
    end_dt   = pd.Timestamp(args.end) if args.end else arith_rets.index[-1]
    bt_rets  = arith_rets.loc[start_dt:end_dt]

    if args.rebal_freq == "daily":
        rebal_dates = set(bt_rets.index[1:])
    else:   # weekly
        rebal_dates = set(bt_rets.resample("W-MON").first().dropna(how="all").index)

    pv_full_vals = [1.0]
    pv_aim_vals  = [1.0]
    full_wts     = np.ones(N) / N
    aim_wts      = np.ones(N) / N
    wts_hist     = {}

    print(f"Running aim-portfolio walk-forward ({args.rebal_freq}, "
          f"{len(rebal_dates)} rebalance dates) …")

    for i in range(1, len(bt_rets)):
        date    = bt_rets.index[i]
        day_ret = bt_rets.iloc[i].values

        pv_full_vals.append(pv_full_vals[-1] * (1.0 + float(full_wts @ day_ret)))
        pv_aim_vals.append( pv_aim_vals[-1]  * (1.0 + float(aim_wts  @ day_ret)))

        if date not in rebal_dates:
            continue

        past = arith_rets.loc[:date]
        if len(past) < MIN_HIST:
            continue

        arr = past.values.astype(float)

        # EWMA Σ (annualised)
        sig_d   = _ewma_cov(arr, args.cov_halflife)
        sig_ann = sig_d * 252.0
        sig_ann = 0.5 * (sig_ann + sig_ann.T) + 1e-6 * np.eye(N)

        # EWMA μ with James-Stein shrinkage (annualised)
        mu_d   = _ewma_mean(arr, args.mu_halflife)
        mu_ann = mu_d * 252.0
        mu_bar = float(mu_ann.mean())
        mu_eff = (1.0 - args.mu_shrinkage) * mu_ann + args.mu_shrinkage * mu_bar

        # Solve Markowitz target (fully invested)
        w = cp.Variable(N)
        prob = cp.Problem(
            cp.Maximize(mu_eff @ w - args.gamma * cp.quad_form(w, sig_ann)),
            [cp.sum(w) == 1.0, w >= 0.0, w <= args.max_wt],
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
            target = np.clip(w.value, 0.0, args.max_wt)
            target /= max(target.sum(), 1e-8)

            full_wts = target.copy()

            # Aim portfolio: move λ fraction toward target
            aim_wts  = aim_wts + lam * (target - aim_wts)
            aim_wts  = np.clip(aim_wts, 0.0, 1.0)
            aim_wts /= max(aim_wts.sum(), 1e-8)

            wts_hist[date] = pd.Series(aim_wts, index=symbols)

    pv      = pd.Series(pv_full_vals, index=bt_rets.index) / pv_full_vals[0]
    pv_aim  = pd.Series(pv_aim_vals,  index=bt_rets.index) / pv_aim_vals[0]
    pv.name     = f"No-TC upper bound ({args.rebal_freq}, γ={args.gamma})"
    pv_aim.name = f"Aim λ={lam:.3f} (no TC)"

    # Theoretical gross TC as compound daily drag.
    # Assumes full portfolio turnover at each rebalance (worst-case).
    # Actual realized TC with slow EWMA signal will be lower — but this shows
    # the risk ceiling and why aim dampening matters.
    annual_full_tc = (rt_bps / 10_000) * n_rebal
    annual_aim_tc  = lam * (rt_bps / 10_000) * n_rebal
    days_arr = np.arange(len(pv), dtype=float)

    pv_full_tc = pv * (1.0 - annual_full_tc) ** (days_arr / 252.0)
    pv_full_tc.name = f"Full rebalance (−{annual_full_tc:.1%}/yr gross TC)"

    pv_aim_tc = pv_aim * (1.0 - annual_aim_tc) ** (days_arr / 252.0)
    pv_aim_tc.name = f"Aim λ={lam:.3f} (−{annual_aim_tc:.1%}/yr gross TC)"

    meta = {"w": pd.DataFrame(wts_hist).T if wts_hist else pd.DataFrame()}

# ── 5. Benchmarks ─────────────────────────────────────────────────────────────
bt_start = args.start

spy_bt   = log_rets.loc[log_rets.index >= bt_start, "SPY"]
spy_path = (1 + spy_bt).cumprod()
spy_path = spy_path / spy_path.iloc[0]
spy_path.name = "SPY Buy-and-Hold"

ew_rets  = arith_rets.loc[arith_rets.index >= bt_start].mean(axis=1)
ew_path  = (1 + ew_rets).cumprod()
ew_path  = ew_path / ew_path.iloc[0]
ew_path.name = f"1/N Equal-Weight ({len(symbols)})"

# ── 6. Performance summary ────────────────────────────────────────────────────
if pv_full_tc is not None:
    # For daily/weekly: show TC-adjusted paths as the primary comparison
    stat_paths = [spy_path, ew_path, pv, pv_aim_tc, pv_full_tc]
else:
    stat_paths = [spy_path, ew_path, pv]

stats = build_stats_table(stat_paths, wts_list=[pd.DataFrame()] * len(stat_paths),
                          benchmark=spy_path)
print("\n── Performance Summary ──────────────────────────────────────────────────────")
print(format_stats_table(stats).to_string())

if pv_full_tc is not None:
    print(f"\n  Note: TC paths use theoretical gross TC (assumes full turnover each period).")
    print(f"  Actual realized TC with slow EWMA signal will be lower.")

# ── 7. Forecast accuracy analysis ────────────────────────────────────────────
print("\n── EWMA Forecast Accuracy ───────────────────────────────────────────────────")
mu_df, cov_df = run_ewma_forecast_accuracy(
    arith_rets=arith_rets,
    mu_halflife=args.mu_halflife,
    cov_halflife=args.cov_halflife,
    forecast_horizon=21,
    start=args.start,
    end=args.end,
)
print_accuracy_report(mu_df, cov_df)

# ── 8. Figures ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

ax_traj = fig.add_subplot(gs[0, :2])
ax_ic   = fig.add_subplot(gs[0, 2])
ax_vr   = fig.add_subplot(gs[1, 0])
ax_corr = fig.add_subplot(gs[1, 1])
ax_wts  = fig.add_subplot(gs[1, 2])

# ── (a) Portfolio trajectory ──────────────────────────────────────────────────
spy_path.plot(ax=ax_traj, color="black", linewidth=2.0, label="SPY")
ew_path.plot( ax=ax_traj, color="grey",  linewidth=1.2, linestyle="--", label="1/N")

if pv_full_tc is not None:
    # daily/weekly: show no-TC upper bound (dotted) + TC-adjusted paths (solid)
    pv.plot(       ax=ax_traj, color="steelblue", linewidth=1.0, linestyle=":",
                   alpha=0.6, label=pv.name)
    pv_aim_tc.plot(ax=ax_traj, color="steelblue", linewidth=2.0, label=pv_aim_tc.name)
    pv_full_tc.plot(ax=ax_traj, color="tomato",   linewidth=2.0, label=pv_full_tc.name)
    ax_traj.annotate(
        "TC = theoretical (full turnover/period)\nActual TC with slow EWMA < shown",
        xy=(0.02, 0.04), xycoords="axes fraction", fontsize=7, color="dimgrey",
    )
else:
    pv.plot(ax=ax_traj, color="steelblue", linewidth=2.0, label=pv.name)

ax_traj.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")
ax_traj.set_title("Portfolio Value (start = 1.0)")
ax_traj.set_ylabel("Value")
ax_traj.legend(fontsize=7)
ax_traj.grid(alpha=0.3)

# ── (b) IC over time ──────────────────────────────────────────────────────────
if len(cov_df):
    cov_df.set_index("date")["ic"].plot(ax=ax_ic, color="steelblue", linewidth=0.9)
    ax_ic.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax_ic.axhline(cov_df["ic"].mean(), color="red", linewidth=1.5,
                  linestyle="--", label=f"Mean IC={cov_df['ic'].mean():.3f}")
    ax_ic.set_title("Return forecast IC\n(Spearman, monthly)")
    ax_ic.set_ylabel("IC")
    ax_ic.legend(fontsize=8)
    ax_ic.grid(alpha=0.3)

# ── (c) Variance ratio per asset ─────────────────────────────────────────────
if len(mu_df):
    var_ratio = mu_df.groupby("asset").apply(
        lambda g: (g["pred_vol"] / (g["real_vol"] + 1e-6)).values
    )
    assets_sorted = sorted(var_ratio.index)
    ax_vr.boxplot([var_ratio[a] for a in assets_sorted], labels=assets_sorted, vert=True)
    ax_vr.axhline(1.0, color="red", linewidth=1, linestyle="--", label="perfect cal.")
    ax_vr.set_title("Vol calibration — diagonal Σ\n(pred σ / realized σ)")
    ax_vr.set_ylabel("Ratio")
    ax_vr.tick_params(axis="x", rotation=60, labelsize=7)
    ax_vr.legend(fontsize=8)
    ax_vr.grid(alpha=0.3, axis="y")

# ── (d) Pairwise correlation accuracy ────────────────────────────────────────
print("\nExtracting pairwise correlation accuracy …")
corr_pairs = extract_correlation_pairs(
    arith_rets,
    cov_halflife=args.cov_halflife,
    forecast_horizon=21,
    start=args.start,
    end=args.end,
)
if len(corr_pairs):
    ax_corr.scatter(corr_pairs["pred_corr"], corr_pairs["real_corr"],
                    alpha=0.06, s=5, color="darkorange", rasterized=True)
    ax_corr.plot([-1, 1], [-1, 1], "r--", linewidth=0.9)
    r_c = corr_pairs["pred_corr"].corr(corr_pairs["real_corr"])
    ax_corr.set_title(
        f"Pairwise ρ accuracy — off-diagonal Σ\n"
        f"r={r_c:.3f}  ({len(corr_pairs):,} obs)"
    )
    ax_corr.set_xlabel("Predicted ρ(i,j)")
    ax_corr.set_ylabel("Realized ρ(i,j)")
    ax_corr.set_xlim(-1, 1); ax_corr.set_ylim(-1, 1)
    ax_corr.grid(alpha=0.3)

# ── (e) Weights heatmap over time ────────────────────────────────────────────
if meta.get("result") is not None:
    wts_df = meta["result"].w.drop(columns=["cash"], errors="ignore")
    row_sum = wts_df.clip(lower=0).sum(axis=1).clip(lower=1e-6)
    wts_df  = wts_df.clip(lower=0).divide(row_sum, axis=0)
elif isinstance(meta.get("w"), pd.DataFrame) and not meta["w"].empty:
    wts_df = meta["w"].copy()
else:
    wts_df = pd.DataFrame()

if not wts_df.empty:
    wts_df.index = pd.to_datetime(wts_df.index).tz_localize(None)
    avg_w      = wts_df.mean().sort_values(ascending=False)
    top_n      = min(15, (avg_w > 0.002).sum())
    top_assets = avg_w.head(top_n).index.tolist()

    # Matrix: rows = assets (high avg weight on top), cols = rebalance dates
    mat = wts_df[top_assets].fillna(0).T.values   # shape (top_n, n_dates)

    im = ax_wts.pcolormesh(mat, cmap="YlOrRd", vmin=0, vmax=args.max_wt)

    # Y axis: asset names
    ax_wts.set_yticks(np.arange(top_n) + 0.5)
    ax_wts.set_yticklabels(top_assets, fontsize=7)
    ax_wts.invert_yaxis()   # highest-weight asset at top

    # X axis: show 5–6 date ticks
    n_cols   = mat.shape[1]
    n_xticks = min(6, n_cols)
    tick_idx = np.linspace(0, n_cols - 1, n_xticks, dtype=int)
    ax_wts.set_xticks(tick_idx + 0.5)
    ax_wts.set_xticklabels(
        [wts_df.index[k].strftime("%b '%y") for k in tick_idx],
        rotation=40, fontsize=7,
    )

    plt.colorbar(im, ax=ax_wts, label="Weight", shrink=0.85, pad=0.02)
    ax_wts.set_title(f"Weights heatmap — top {top_n} assets")
    ax_wts.set_xlabel("Rebalance date →", fontsize=7)

fig.suptitle(
    f"Backtest  |  {len(symbols)}-symbol universe  |  {args.rebal_freq}  |  "
    f"γ={args.gamma}, shrink={args.mu_shrinkage:.0%}, "
    f"Σ_hl={args.cov_halflife}d, μ_hl={args.mu_halflife}d  |  {args.start}–",
    fontsize=10, y=1.01,
)

if args.save:
    out = OUTPUT_DIR / "backtest_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.close(fig)
else:
    plt.tight_layout()
    plt.show()

print("\nDone.")
