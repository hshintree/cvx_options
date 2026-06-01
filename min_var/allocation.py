"""
Allocation diagnostics — understanding portfolio weights and their implications.
Migrated and extended from log_norm.ipynb Cells 37-42.
"""
import numpy as np
import pandas as pd
from typing import Optional


def allocation_summary(
    wts_df: pd.DataFrame,
    asset_cols: list,
    syms: list,
) -> dict:
    """
    Compute time-averaged and per-rebalance allocation diagnostics.

    Returns dict with keys:
      avg_weights        : pd.Series — mean weight per asset
      weight_std         : pd.Series — weight standard deviation per asset
      avg_stock_weight   : float — average total weight in underlying positions
      avg_option_weight  : float — average total weight in option legs
      avg_hhi            : float — average Herfindahl-Hirschman Index (concentration)
      turnover_series    : pd.Series — one-way turnover per rebalance date
      avg_turnover       : float — mean one-way turnover
      ann_turnover       : float — annualised turnover (assumes rebalance frequency from index)
    """
    if wts_df.empty:
        return {}

    wts = wts_df.reindex(columns=asset_cols, fill_value=0.0)

    avg_w  = wts.mean()
    std_w  = wts.std()

    opt_cols  = [c for c in asset_cols if "_call" in c or "_put" in c]
    und_cols  = [c for c in asset_cols if c in syms]
    avg_opt   = float(wts[opt_cols].sum(axis=1).mean()) if opt_cols else 0.0
    avg_und   = float(wts[und_cols].sum(axis=1).mean()) if und_cols else 0.0

    # Herfindahl-Hirschman Index = sum(w_i^2); 1/n for equal-weight
    hhi = (wts**2).sum(axis=1)
    avg_hhi = float(hhi.mean())

    # One-way turnover = sum(|Δw|) / 2 at each rebalance
    turnover = (wts.diff().abs().sum(axis=1) / 2.0).dropna()
    avg_turn = float(turnover.mean()) if len(turnover) > 0 else 0.0

    # Annualise turnover using median gap between rebalance dates
    if len(wts) > 1:
        rebal_gap = float(pd.Series(wts.index).diff().dt.days.median())
        rebal_per_year = 365.0 / max(rebal_gap, 1.0)
    else:
        rebal_per_year = 52.0
    ann_turn = avg_turn * rebal_per_year

    return dict(
        avg_weights=avg_w,
        weight_std=std_w,
        avg_stock_weight=avg_und,
        avg_option_weight=avg_opt,
        avg_hhi=avg_hhi,
        turnover_series=turnover,
        avg_turnover=avg_turn,
        ann_turnover=ann_turn,
    )


def option_attribution(
    wts_df: pd.DataFrame,
    rets_df: pd.DataFrame,
    asset_cols: list,
    syms: list,
) -> pd.DataFrame:
    """
    Decompose portfolio daily returns into stock and option contributions.

    Returns pd.DataFrame with columns: r_stocks, r_options, r_total.
    """
    if wts_df.empty:
        return pd.DataFrame()

    opt_cols = [c for c in asset_cols if "_call" in c or "_put" in c]
    und_cols = [c for c in asset_cols if c in syms]

    # Forward-fill weights to daily frequency (lagged by 1 day)
    wts_fwd = wts_df.sort_index().reindex(rets_df.index, method="ffill").shift(1)
    wts_fwd = wts_fwd.reindex(columns=asset_cols, fill_value=0.0)

    def safe_sum(cols):
        valid = [c for c in cols if c in rets_df.columns and c in wts_fwd.columns]
        if not valid:
            return pd.Series(0.0, index=rets_df.index)
        return (wts_fwd[valid].fillna(0) * rets_df[valid].fillna(0)).sum(axis=1)

    r_stocks  = safe_sum(und_cols)
    r_options = safe_sum(opt_cols)
    r_total   = safe_sum(asset_cols)

    return pd.DataFrame({
        "r_stocks":  r_stocks,
        "r_options": r_options,
        "r_total":   r_total,
    }).dropna(how="all")


def weight_stability_table(
    wts_df: pd.DataFrame,
    asset_cols: list,
    large_rebal_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Per-asset weight stability summary.

    Columns:
      median_wt          : median weight over all rebalance dates
      p25, p75           : 25th and 75th percentiles
      participation_rate : fraction of periods with weight > 1%
      large_rebal_count  : number of periods with |Δw| > large_rebal_threshold
    """
    if wts_df.empty:
        return pd.DataFrame()

    wts   = wts_df.reindex(columns=asset_cols, fill_value=0.0)
    delta = wts.diff().abs()

    rows = []
    for c in asset_cols:
        w = wts[c]
        rows.append({
            "asset":               c,
            "median_wt":           float(w.median()),
            "p25":                 float(w.quantile(0.25)),
            "p75":                 float(w.quantile(0.75)),
            "participation_rate":  float((w > 0.01).mean()),
            "large_rebal_count":   int((delta[c] > large_rebal_threshold).sum()),
        })

    return pd.DataFrame(rows).set_index("asset")


def compare_allocation_experiments(
    exp_dict: dict,
    syms: list,
) -> "matplotlib.figure.Figure":
    """
    Side-by-side bar chart of average weights per asset across experiments.

    exp_dict: {exp_name: wts_df}
    """
    import matplotlib.pyplot as plt

    exp_names = list(exp_dict.keys())
    # Gather all unique asset columns across experiments
    all_cols = []
    seen = set()
    for wdf in exp_dict.values():
        for c in wdf.columns:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)

    avg_matrix = pd.DataFrame(index=all_cols, columns=exp_names, dtype=float)
    for name, wdf in exp_dict.items():
        avg_w = wdf.mean().reindex(all_cols).fillna(0.0)
        avg_matrix[name] = avg_w

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(all_cols)), 5))
    x = np.arange(len(all_cols))
    w = 0.8 / max(len(exp_names), 1)
    palette = ["#2563eb", "#16a34a", "#dc2626", "#f97316", "#7c3aed", "#0891b2"]

    for i, exp_name in enumerate(exp_names):
        offset = (i - len(exp_names) / 2 + 0.5) * w
        bars = ax.bar(x + offset, avg_matrix[exp_name].values, w,
                      label=exp_name, color=palette[i % len(palette)],
                      alpha=0.85, edgecolor="k", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(all_cols, rotation=25, fontsize=9)
    ax.set_ylabel("Average weight")
    ax.set_title("Average Allocation Across Experiments")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="k", lw=0.5)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    plt.tight_layout()
    return fig


def print_allocation_report(
    exp_results: dict,
    syms: list,
) -> None:
    """
    Print allocation summary for all experiments.
    exp_results: {exp_id: (path, wts_df)}
    """
    for exp_id, (path, wts_df) in exp_results.items():
        if exp_id == "benchmark" or wts_df.empty:
            continue
        asset_cols = list(wts_df.columns)
        summary = allocation_summary(wts_df, asset_cols, syms)
        if not summary:
            continue

        print(f"\n── {exp_id} allocation summary ─────────────────────────────────────")
        print(f"  Avg stock weight:  {summary['avg_stock_weight']:.1%}")
        print(f"  Avg option weight: {summary['avg_option_weight']:.1%}")
        print(f"  Avg HHI:           {summary['avg_hhi']:.4f}  "
              f"(equal-wt HHI={1/max(len(asset_cols),1):.4f})")
        print(f"  Avg turnover:      {summary['avg_turnover']:.1%}  "
              f"(ann. {summary['ann_turnover']:.0%})")

        stab = weight_stability_table(wts_df, asset_cols)
        top5 = stab.sort_values("median_wt", ascending=False).head(5)
        print(f"  Top-5 assets by median weight:")
        for asset, row in top5.iterrows():
            print(f"    {asset:<18s}  median={row['median_wt']:.1%}  "
                  f"participation={row['participation_rate']:.0%}")
