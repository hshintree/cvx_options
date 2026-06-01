"""
Run all walk-forward backtest experiments and the efficient frontier ablation.

Experiment matrix:
  Exp0   UNIV_STOCKS   min-var    empirical Σ     no μ in objective
  ExpA   UNIV_A        min-var    DGV Σ           no μ in objective
  ExpA+  UNIV_A        min-var    DGV Σ           5% return floor (constraint, not objective)
  ExpB   UNIV_B        min-var    DGV Σ           no μ in objective
  ExpC   UNIV_A        mean-var   DGV Σ + δ-lev μ μ IN objective (γ=MV_GAMMA)
  ExpD   UNIV_B        mean-var   DGV Σ + δ-lev μ μ IN objective (γ=MV_GAMMA)

Usage:
    from min_var.experiments import run_all_experiments
    results = run_all_experiments()
    # results["Exp0"] = (path, wts_df)
"""
import pandas as pd
import numpy as np

from .config import (
    SYMS, UNIV_STOCKS, UNIV_A, UNIV_B,
    LOOKBACK, REBAL_FREQ, MAX_WT, MAX_OPT_WT, MV_GAMMA, R_ANN,
)
from .data_loader import load_research_panel, build_returns_matrix, build_rp_index
from .backtest import run_walk_forward
from .metrics import build_stats_table, format_stats_table


# ── Experiment registry ───────────────────────────────────────────────────────
EXPERIMENTS = [
    # (id,     label,                          universe,     objective, use_emp, extra_kwargs)
    ("Exp0",   "Min-Var: Stocks (empirical Σ)", UNIV_STOCKS, "minvar",  True,   {}),
    ("ExpA",   "Min-Var: SPY opts + stocks",    UNIV_A,      "minvar",  False,  {}),
    ("ExpA+",  "Min-Var: SPY opts + 5% floor",  UNIV_A,      "minvar",  False,  {"min_ann_return": 0.05}),
    ("ExpB",   "Min-Var: All opts + stocks",    UNIV_B,      "minvar",  False,  {}),
    ("ExpC",   "Mean-Var (γ=5): SPY opts",      UNIV_A,      "meanvar", False,  {}),
    ("ExpD",   "Mean-Var (γ=5): All opts",      UNIV_B,      "meanvar", False,  {}),
]


def run_all_experiments(
    panel_path=None,
    lookback: int = LOOKBACK,
    rebal_freq: int = REBAL_FREQ,
    max_wt: float = MAX_WT,
    max_opt_wt: float = MAX_OPT_WT,
    gamma: float = MV_GAMMA,
    rf: float = R_ANN,
    verbose: bool = True,
) -> dict:
    """
    Run all 6 experiments and return a dict of:
        {exp_id: (path: pd.Series, wts_df: pd.DataFrame)}

    Also returns a 'benchmark' key with the equal-weight stocks path.
    """
    rp       = load_research_panel(panel_path)
    rets_all = build_returns_matrix(rp, SYMS)
    rp_idx   = build_rp_index(rp, SYMS)

    if verbose:
        print(f"Returns matrix: {rets_all.shape[0]} dates × {rets_all.shape[1]} assets")
        print(f"Date range: {rets_all.index.min().date()} → {rets_all.index.max().date()}")

    # Equal-weight benchmark
    ew_daily = rets_all[SYMS].mean(axis=1)
    ew_path  = (1 + ew_daily.iloc[lookback:]).cumprod()
    ew_path  = ew_path / ew_path.iloc[0]
    ew_path.name = "EW Stocks (benchmark)"

    results = {"benchmark": (ew_path, pd.DataFrame())}

    for exp_id, label, universe, objective, use_emp, extra_kw in EXPERIMENTS:
        if verbose:
            print(f"\nRunning {exp_id}: {label} …")

        path, wts, _pred_log = run_walk_forward(
            rets_df=rets_all,
            asset_cols=universe,
            syms=SYMS,
            rp_idx=None if use_emp else rp_idx,
            label=label,
            lookback=lookback,
            rebal_freq=rebal_freq,
            max_wt=max_wt,
            max_opt_wt=max_opt_wt,
            objective=objective,
            gamma=gamma,
            use_empirical_cov=use_emp,
            rf=rf,
            **extra_kw,
        )

        if verbose:
            print(f"  Final value: {path.iloc[-1]:.4f}  ({len(path)} dates)")
        results[exp_id] = (path, wts)

    return results


def align_paths(results: dict) -> list[pd.Series]:
    """
    Align all portfolio paths to a common start date (latest first date across experiments).
    Returns list of rebased paths (start = 1.0).
    """
    paths = [v[0] for v in results.values()]
    start_dt = max(p.index.min() for p in paths if not p.empty)
    aligned  = [p.loc[start_dt:] / float(p.loc[start_dt]) for p in paths if not p.empty]
    return aligned


def summarize(results: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Build and optionally print the performance statistics table for all experiments.
    """
    benchmark_path = results.get("benchmark", (None,))[0]
    exp_ids = [k for k in results if k != "benchmark"]

    paths    = [results[k][0] for k in exp_ids]
    wts_list = [results[k][1] for k in exp_ids]

    stats = build_stats_table(paths, wts_list=wts_list, benchmark=benchmark_path)

    if verbose:
        print("\n── Performance Summary ──────────────────────────────────────────────")
        print(format_stats_table(stats).to_string())

    return stats
