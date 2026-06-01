"""
Entry-point for min_var experiments and plots.
Run from the project root:
    python run_min_var.py [--exp EXP_ID] [--frontier] [--accuracy] [--alloc] [--save]

Examples:
    python run_min_var.py                     # run all 6 experiments + stats table
    python run_min_var.py --frontier          # also trace and plot the efficient frontier
    python run_min_var.py --accuracy          # also run the mu/Sigma accuracy test
    python run_min_var.py --alloc             # also print allocation diagnostics
    python run_min_var.py --save              # save all figures as PNG instead of showing
    python run_min_var.py --exp ExpA ExpC     # run only ExpA and ExpC
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on the path so `import min_var` works
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from min_var.config import (
    SYMS, UNIV_STOCKS, UNIV_A, UNIV_B,
    MV_GAMMA, LOOKBACK, REBAL_FREQ, R_ANN,
)
from min_var.data_loader import load_research_panel, build_returns_matrix, build_rp_index
from min_var.backtest import run_walk_forward
from min_var.metrics import build_stats_table, format_stats_table
from min_var.experiments import EXPERIMENTS, align_paths
from min_var.plots import (
    plot_trajectories,
    plot_weight_heatmaps,
    plot_vol_return_bars,
    build_and_plot_frontier,
    plot_accuracy,
    plot_variance_scatter,
)
from min_var.allocation import print_allocation_report, compare_allocation_experiments
from min_var.accuracy import run_accuracy_test, print_accuracy_report


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run min_var experiments")
parser.add_argument("--exp",      nargs="*", default=None,
                    help="Experiment IDs to run (e.g. Exp0 ExpA ExpC). Default: all.")
parser.add_argument("--frontier", action="store_true",
                    help="Trace and plot the efficient frontier.")
parser.add_argument("--accuracy", action="store_true",
                    help="Run mu/Sigma accuracy test (slow: ~60 chain dates).")
parser.add_argument("--alloc",    action="store_true",
                    help="Print and plot allocation diagnostics.")
parser.add_argument("--save",     action="store_true",
                    help="Save figures as PNG files instead of showing interactively.")
parser.add_argument("--snap",     default="2023-11-30",
                    help="Snapshot date for efficient frontier (default: 2023-11-30).")
args = parser.parse_args()

if args.save:
    matplotlib.use("Agg")

save_dir = Path("min_var_output")
if args.save:
    save_dir.mkdir(exist_ok=True)

def show_or_save(fig, name: str):
    if args.save:
        path = save_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)
    else:
        plt.show()


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading research panel …")
rp       = load_research_panel()
rets_all = build_returns_matrix(rp, SYMS)
rp_idx   = build_rp_index(rp, SYMS)
print(f"  {rets_all.shape[0]} dates × {rets_all.shape[1]} assets  "
      f"({rets_all.index.min().date()} → {rets_all.index.max().date()})")

# Equal-weight benchmark
ew_path  = (1 + rets_all[SYMS].mean(axis=1).iloc[LOOKBACK:]).cumprod()
ew_path  = ew_path / ew_path.iloc[0]
ew_path.name = "EW Stocks (benchmark)"


# ── 2. Run experiments ────────────────────────────────────────────────────────
all_exp_ids  = [e[0] for e in EXPERIMENTS]
selected_ids = args.exp if args.exp else all_exp_ids

results = {"benchmark": (ew_path, pd.DataFrame())}

for exp_id, label, universe, objective, use_emp, extra_kw in EXPERIMENTS:
    if exp_id not in selected_ids:
        continue
    print(f"\nRunning {exp_id}: {label} …")
    path, wts, _pred_log = run_walk_forward(
        rets_df=rets_all,
        asset_cols=universe,
        syms=SYMS,
        rp_idx=None if use_emp else rp_idx,
        label=label,
        lookback=LOOKBACK,
        rebal_freq=REBAL_FREQ,
        objective=objective,
        gamma=MV_GAMMA,
        use_empirical_cov=use_emp,
        rf=R_ANN,
        **extra_kw,
    )
    print(f"  Final value: {path.iloc[-1]:.4f}  ({len(path)} dates)")
    results[exp_id] = (path, wts)


# ── 3. Performance stats ──────────────────────────────────────────────────────
exp_keys  = [k for k in results if k != "benchmark"]
paths     = [results[k][0] for k in exp_keys]
wts_list  = [results[k][1] for k in exp_keys]
benchmark = results["benchmark"][0]

print("\n── Performance Summary ──────────────────────────────────────────────────")
stats = build_stats_table(paths, wts_list=wts_list, benchmark=benchmark)
print(format_stats_table(stats).to_string())


# ── 4. Trajectory + drawdown plot ────────────────────────────────────────────
all_paths = paths + [benchmark]
fig = plot_trajectories(
    all_paths,
    title="Walk-Forward Portfolios — Min-Var and Mean-Var Experiments (2023)",
)
show_or_save(fig, "01_trajectories")


# ── 5. Vol / return bar chart ─────────────────────────────────────────────────
fig = plot_vol_return_bars(stats)
show_or_save(fig, "02_vol_return_bars")


# ── 6. Weight heatmaps ───────────────────────────────────────────────────────
heatmap_exps = {k: results[k] for k in exp_keys if not results[k][1].empty}
if heatmap_exps:
    fig = plot_weight_heatmaps(heatmap_exps, syms=SYMS)
    show_or_save(fig, "03_weight_heatmaps")


# ── 7. Efficient frontier (optional) ─────────────────────────────────────────
if args.frontier:
    print(f"\nBuilding efficient frontier at {args.snap} …")
    fig, ef_data, ablation = build_and_plot_frontier(
        rp_idx=rp_idx,
        rets_all=rets_all,
        univ_stocks=UNIV_STOCKS,
        univ_a=UNIV_A,
        univ_b=UNIV_B,
        syms=SYMS,
        snap_date=args.snap,
        lookback=63,
        rf=R_ANN,
    )
    show_or_save(fig, "04_efficient_frontier")

    # Print ablation table
    if ablation:
        print("\n── Frontier Ablation (Exp A universe) ───────────────────────────────────")
        print(f"  {'Objective':<22s}  {'Ann Return':>10s}  {'Ann Vol':>8s}  {'Sharpe':>7s}")
        print("  " + "-" * 54)
        for lbl, d in ablation.items():
            sh = d.get("sharpe", float("nan"))
            print(f"  {lbl:<22s}  {d['ret']:>9.1%}  {d['vol']:>7.1%}  "
                  f"{sh:>7.2f}" if not __import__('math').isnan(sh) else
                  f"  {lbl:<22s}  {d['ret']:>9.1%}  {d['vol']:>7.1%}  {'—':>7s}")


# ── 8. Allocation diagnostics (optional) ─────────────────────────────────────
if args.alloc:
    print_allocation_report(results, SYMS)

    # Side-by-side average weight chart
    wts_for_plot = {k: v[1] for k, v in results.items()
                    if k != "benchmark" and not v[1].empty}
    if wts_for_plot:
        fig = compare_allocation_experiments(wts_for_plot, SYMS)
        show_or_save(fig, "05_allocation_comparison")


# ── 9. Accuracy test (optional, slow) ────────────────────────────────────────
if args.accuracy:
    print("\nRunning mu/Sigma accuracy test (this takes a few minutes) …")
    from min_var.data_loader import eligible_chain_dates, load_spy_prices, discover_chain_dates
    from min_var.config import TEST_DTE, N_SAMPLES

    mu_df, cov_pred, cov_real = run_accuracy_test(
        test_dte=TEST_DTE, n_samples=N_SAMPLES,
    )
    print_accuracy_report(mu_df, cov_pred, cov_real)

    if len(mu_df) > 0:
        fig = plot_accuracy(mu_df)
        show_or_save(fig, "06_accuracy_mu")
        fig = plot_variance_scatter(mu_df)
        show_or_save(fig, "07_accuracy_vol")


print("\nDone.")
