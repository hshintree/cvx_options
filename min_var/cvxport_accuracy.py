"""
Walk-forward forecast accuracy tests for EWMA μ and Σ estimators.

At each rebalance date we compute the EWMA μ and Σ using only past data,
then compare to what actually happened over the next forecast horizon.

Key metrics
-----------
μ accuracy
  IC (Information Coefficient) — cross-sectional Pearson correlation between
    predicted return rank and realized return rank on each rebalance date.
    A statistically significant mean IC > 0 means the signal has real predictive
    power. We test this with a t-test (H0: mean_IC = 0).
  Hit rate — fraction of (asset, date) pairs where sign(pred) == sign(real).
    Random baseline: 50%. We test with a binomial test.
  MAE / RMSE / Bias — absolute and directional errors.

Σ accuracy
  Variance ratio — predicted_var / realized_var per asset. Should be ≈ 1.
    Systematic > 1 means over-estimation; < 1 means under-estimation.
  Correlation accuracy — Pearson r between upper-triangle of predicted and
    realized correlation matrices, averaged over all rebalance dates.
  Frobenius relative error — ||Σ_pred - Σ_real|| / ||Σ_real|| per date.

Usage
-----
    from min_var.cvxport_accuracy import run_ewma_forecast_accuracy, print_accuracy_report
    mu_df, cov_df = run_ewma_forecast_accuracy(arith_rets)
    print_accuracy_report(mu_df, cov_df)
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import R_ANN


# ── EWMA helpers (standalone, no cvxportfolio dependency) ─────────────────────

def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Exponential decay weights for a window of n observations (most recent last)."""
    decay   = np.exp(-np.log(2) / max(halflife, 1.0))
    w       = decay ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum()


def _ewma_mean(rets: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA mean vector (n_assets,) from (T, n_assets) return matrix."""
    w = _ewma_weights(len(rets), halflife)
    return (w[:, None] * rets).sum(axis=0)


def _ewma_cov(rets: np.ndarray, halflife: float) -> np.ndarray:
    """
    Bias-corrected EWMA covariance (n_assets × n_assets).
    Uses the same estimator as cvxportfolio's HistoricalFactorizedCovariance.
    """
    n = len(rets)
    w = _ewma_weights(n, halflife)
    mu  = (w[:, None] * rets).sum(axis=0)
    dev = rets - mu
    # sum-of-weights-squared for bias correction (effective sample size)
    v1     = 1.0
    v2     = (w ** 2).sum()
    factor = v1 / (v1 - v2)  # Bessel-like correction
    Sig    = factor * (w[:, None] * dev).T @ dev
    return 0.5 * (Sig + Sig.T)


def _ewma_cov_split(
    rets: np.ndarray,
    vol_halflife: float,
    corr_halflife: float,
) -> np.ndarray:
    """
    Separate-timescale covariance: Σ = D × C × D

      D = diag(EWMA vols, hl=vol_halflife)   — fast-adapting (days to weeks)
      C = EWMA correlation matrix, hl=corr_halflife — slow, stable (weeks to months)

    Rationale: individual asset volatilities react to regime shifts within days;
    the cross-asset correlation *structure* evolves much more slowly.  Using a
    longer HL for C reduces noise in the off-diagonal entries of Σ without
    sacrificing vol responsiveness.  Empirically this improves pairwise ρ
    forecast accuracy (higher Pearson r between predicted and realized ρ(i,j)).
    """
    # Fast EWMA vols (diagonal)
    sig_fast = _ewma_cov(rets, vol_halflife)
    vols = np.sqrt(np.maximum(np.diag(sig_fast), 1e-12))

    # Slow EWMA correlation matrix
    sig_slow = _ewma_cov(rets, corr_halflife)
    var_slow = np.diag(sig_slow)
    denom = (np.sqrt(var_slow[:, None]) * np.sqrt(var_slow[None, :]) + 1e-12)
    corr  = sig_slow / denom
    np.fill_diagonal(corr, 1.0)
    corr  = np.clip(corr, -0.999, 0.999)

    # Reconstruct Σ = D × C × D
    sig = vols[:, None] * corr * vols[None, :]
    return 0.5 * (sig + sig.T)


def _monthly_rebal_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return approximately monthly rebalance dates (first trading day of each month)."""
    months  = index.to_period("M").unique()
    dates   = []
    for m in months:
        mask = index.to_period("M") == m
        candidates = index[mask]
        if len(candidates):
            dates.append(candidates[0])
    return dates


# ── Main accuracy test ─────────────────────────────────────────────────────────

def run_ewma_forecast_accuracy(
    arith_rets: pd.DataFrame,
    mu_halflife: int = 252,
    cov_halflife: int = 21,
    corr_halflife: Optional[int] = None,  # if None → use cov_halflife (no split)
    forecast_horizon: int = 21,           # trading days ahead to measure "realized"
    min_history: int = 126,               # minimum rows before first forecast
    rebal_freq: str = "monthly",          # 'monthly' | 'weekly' | int (trading days)
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward evaluation of EWMA return and covariance forecasts.

    For every rebalance date t in [start, end]:
      1. Compute EWMA μ using all rows up to (and including) t.
      2. Compute EWMA Σ using all rows up to (and including) t.
      3. Record realized μ and realized Σ over the next forecast_horizon days.

    Parameters
    ----------
    arith_rets       : arithmetic (simple) daily returns, columns = assets
    mu_halflife      : EWMA half-life for μ in trading days
    cov_halflife     : EWMA half-life for Σ in trading days
    forecast_horizon : number of trading days to evaluate predictions against
    min_history      : minimum rows of history before first evaluation
    rebal_freq       : 'monthly', 'weekly', or int (custom trading-day interval)
    start / end      : restrict the evaluation window

    Returns
    -------
    mu_df  : per-asset, per-date accuracy metrics
    cov_df : per-date covariance forecast metrics
    """
    idx = arith_rets.index
    if start:
        idx_start = idx[idx >= pd.Timestamp(start)]
        start_pos = arith_rets.index.get_loc(idx_start[0]) if len(idx_start) else len(arith_rets)
    else:
        start_pos = min_history

    if end:
        idx_end = idx[idx <= pd.Timestamp(end)]
        end_pos = arith_rets.index.get_loc(idx_end[-1]) + 1 if len(idx_end) else len(arith_rets)
    else:
        end_pos = len(arith_rets) - forecast_horizon

    # Build rebalance date positions
    if rebal_freq == "monthly":
        all_dates     = _monthly_rebal_dates(idx)
        rebal_pos     = [arith_rets.index.get_loc(d) for d in all_dates
                         if start_pos <= arith_rets.index.get_loc(d) < end_pos]
    elif rebal_freq == "weekly":
        rebal_pos = list(range(start_pos, end_pos, 5))
    else:
        step = int(rebal_freq)
        rebal_pos = list(range(start_pos, end_pos, step))

    assets = list(arith_rets.columns)
    n_assets = len(assets)

    mu_rows  = []
    cov_rows = []

    rets_arr = arith_rets.values.astype(float)

    for pos in rebal_pos:
        if pos + forecast_horizon >= len(arith_rets):
            break

        history = rets_arr[:pos + 1]
        future  = rets_arr[pos + 1 : pos + 1 + forecast_horizon]

        # ── Predicted μ (annualised) ──────────────────────────────────────────
        pred_mu_daily = _ewma_mean(history, mu_halflife)
        pred_mu_ann   = pred_mu_daily * 252

        # ── Realized μ over horizon (annualised) ─────────────────────────────
        real_mu_daily = future.mean(axis=0)
        real_mu_ann   = real_mu_daily * 252

        # ── Predicted Σ (daily units) ─────────────────────────────────────────
        if corr_halflife is not None and corr_halflife != cov_halflife:
            pred_sig = _ewma_cov_split(history, cov_halflife, corr_halflife)
        else:
            pred_sig = _ewma_cov(history, cov_halflife)
        pred_var = np.diag(pred_sig)
        pred_vol = np.sqrt(np.maximum(pred_var, 0)) * np.sqrt(252)

        # ── Realized Σ over horizon ───────────────────────────────────────────
        if len(future) >= 2:
            real_sig = np.cov(future.T)
            if real_sig.ndim == 0:
                real_sig = np.array([[float(real_sig)]])
        else:
            real_sig = np.eye(n_assets) * 1e-8

        real_var = np.diag(real_sig)
        real_vol = np.sqrt(np.maximum(real_var, 0)) * np.sqrt(252)

        t = arith_rets.index[pos]

        # ── Per-asset μ rows ──────────────────────────────────────────────────
        for j, asset in enumerate(assets):
            mu_rows.append({
                "date":       t,
                "asset":      asset,
                "pred_mu":    pred_mu_ann[j],
                "real_mu":    real_mu_ann[j],
                "pred_vol":   pred_vol[j],
                "real_vol":   real_vol[j],
                "correct_dir": int(np.sign(pred_mu_ann[j]) == np.sign(real_mu_ann[j])),
            })

        # ── Per-date covariance row ───────────────────────────────────────────
        fro_pred = np.linalg.norm(pred_sig)
        fro_real = np.linalg.norm(real_sig)
        fro_err  = np.linalg.norm(pred_sig - real_sig) / (fro_real + 1e-12)

        # Predicted vs realized correlation (upper triangle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_corr = pred_sig / (np.sqrt(pred_var[:, None]) * np.sqrt(pred_var[None, :]) + 1e-12)
            real_corr = real_sig / (np.sqrt(real_var[:, None]) * np.sqrt(real_var[None, :]) + 1e-12)

        mask   = np.triu(np.ones((n_assets, n_assets), dtype=bool), k=1)
        pc_vec = pred_corr[mask]
        rc_vec = real_corr[mask]
        corr_pearson = float(np.corrcoef(pc_vec, rc_vec)[0, 1]) if len(pc_vec) > 1 else np.nan

        # Cross-sectional IC for μ at this date (Spearman rank correlation)
        if len(pred_mu_ann) > 2:
            ic, _ = stats.spearmanr(pred_mu_ann, real_mu_ann)
        else:
            ic = np.nan

        cov_rows.append({
            "date":          t,
            "frobenius_err": fro_err,
            "corr_accuracy": corr_pearson,
            "mean_var_ratio": float(np.mean(pred_var / (real_var + 1e-12))),
            "ic":            ic,
        })

    mu_df  = pd.DataFrame(mu_rows)
    cov_df = pd.DataFrame(cov_rows)
    return mu_df, cov_df


# ── Correlation pair accuracy (off-diagonal Σ) ────────────────────────────────

def extract_correlation_pairs(
    arith_rets: pd.DataFrame,
    cov_halflife: int = 21,
    corr_halflife: Optional[int] = None,   # if set, uses split vol/corr estimator
    forecast_horizon: int = 21,
    min_history: int = 126,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    For every monthly rebalance date, extract all upper-triangle pairwise
    (pred_corr, real_corr) entries.

    Returns a DataFrame with columns:
        date, asset_i, asset_j, pred_corr, real_corr
    one row per (date × asset pair).  Use this to verify the off-diagonal
    covariance forecast — not just the idiosyncratic (diagonal) risk.
    """
    assets   = list(arith_rets.columns)
    n        = len(assets)
    rets_arr = arith_rets.values.astype(float)
    idx      = arith_rets.index

    start_pos = arith_rets.index.get_loc(
        idx[idx >= pd.Timestamp(start)][0]
    ) if start else min_history
    end_pos = (
        arith_rets.index.get_loc(idx[idx <= pd.Timestamp(end)][-1]) + 1
        if end else len(arith_rets) - forecast_horizon
    )

    rebal_dates = _monthly_rebal_dates(idx)
    rebal_pos   = [arith_rets.index.get_loc(d) for d in rebal_dates
                   if start_pos <= arith_rets.index.get_loc(d) < end_pos]

    rows = []
    for pos in rebal_pos:
        if pos + forecast_horizon >= len(arith_rets):
            break
        history = rets_arr[: pos + 1]
        future  = rets_arr[pos + 1 : pos + 1 + forecast_horizon]
        t       = idx[pos]

        if corr_halflife is not None and corr_halflife != cov_halflife:
            pred_sig = _ewma_cov_split(history, cov_halflife, corr_halflife)
        else:
            pred_sig = _ewma_cov(history, cov_halflife)
        pred_var = np.diag(pred_sig)

        real_sig = np.cov(future.T) if len(future) >= 2 else np.eye(n) * 1e-8
        if real_sig.ndim == 0:
            real_sig = np.array([[float(real_sig)]])
        real_var = np.diag(real_sig)

        pred_corr = pred_sig / (
            np.sqrt(pred_var[:, None]) * np.sqrt(pred_var[None, :]) + 1e-12
        )
        real_corr = real_sig / (
            np.sqrt(real_var[:, None]) * np.sqrt(real_var[None, :]) + 1e-12
        )

        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "date":      t,
                    "asset_i":   assets[i],
                    "asset_j":   assets[j],
                    "pred_corr": float(pred_corr[i, j]),
                    "real_corr": float(real_corr[i, j]),
                })

    return pd.DataFrame(rows)


# ── Statistical tests ─────────────────────────────────────────────────────────

def _ic_ttest(ic_series: pd.Series) -> dict:
    """t-test: H0 = mean IC = 0.  Returns t-stat, p-value, annualised IR_IC."""
    clean = ic_series.dropna()
    n     = len(clean)
    if n < 3:
        return {"n": n, "mean_ic": np.nan, "t_stat": np.nan, "p_value": np.nan, "ir_ic": np.nan}
    mean_ic = clean.mean()
    std_ic  = clean.std(ddof=1)
    t_stat  = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 1e-8 else 0.0
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    ir_ic   = mean_ic / (std_ic + 1e-8) * np.sqrt(12)  # annualised (monthly IC)
    return {"n": n, "mean_ic": mean_ic, "t_stat": t_stat,
            "p_value": p_value, "ir_ic": ir_ic}


def _hit_rate_test(correct_dir: pd.Series) -> dict:
    """Binomial test: H0 = hit rate = 50%.  One-sided (better than random)."""
    n    = len(correct_dir)
    k    = correct_dir.sum()
    rate = k / n if n > 0 else np.nan
    res  = stats.binomtest(k, n, p=0.5, alternative="greater")
    return {"n": n, "hit_rate": rate, "p_value": res.pvalue}


# ── Summary + report ──────────────────────────────────────────────────────────

def summarize_forecast_accuracy(
    mu_df: pd.DataFrame, cov_df: pd.DataFrame
) -> dict:
    """
    Compute summary statistics from walk-forward accuracy DataFrames.

    Returns a dict with:
      mu_summary   : per-asset μ accuracy
      ic_test      : IC t-test (pooled across all dates)
      hit_test     : hit-rate binomial test (pooled)
      cov_summary  : covariance accuracy
      overall      : headline numbers
    """
    # ── Per-asset μ accuracy ─────────────────────────────────────────────────
    mu_agg = mu_df.groupby("asset").agg(
        pred_mu_mean=("pred_mu", "mean"),
        real_mu_mean=("real_mu", "mean"),
        bias=("pred_mu", lambda x: (x - mu_df.loc[x.index, "real_mu"]).mean()),
        mae=("pred_mu", lambda x: (x - mu_df.loc[x.index, "real_mu"]).abs().mean()),
        rmse=("pred_mu", lambda x: np.sqrt(((x - mu_df.loc[x.index, "real_mu"]) ** 2).mean())),
        hit_rate=("correct_dir", "mean"),
        n=("pred_mu", "count"),
    ).round(4)

    # Pearson correlation (pred_mu vs real_mu) per asset
    pearson = mu_df.groupby("asset").apply(
        lambda g: float(g["pred_mu"].corr(g["real_mu"]))
    ).rename("pearson_r").round(4)
    mu_agg = mu_agg.join(pearson)

    # ── IC test (pooled, cross-sectional per date) ───────────────────────────
    ic_test  = _ic_ttest(cov_df["ic"])

    # ── Hit rate test (pooled) ────────────────────────────────────────────────
    hit_test = _hit_rate_test(mu_df["correct_dir"])

    # ── Covariance summary ────────────────────────────────────────────────────
    cov_summary = {
        "frobenius_err_mean":  float(cov_df["frobenius_err"].mean()),
        "frobenius_err_median":float(cov_df["frobenius_err"].median()),
        "corr_accuracy_mean":  float(cov_df["corr_accuracy"].mean()),
        "var_ratio_mean":      float(cov_df["mean_var_ratio"].mean()),
        "n_dates":             len(cov_df),
    }

    return {
        "mu_per_asset": mu_agg,
        "ic_test":      ic_test,
        "hit_test":     hit_test,
        "cov_summary":  cov_summary,
    }


def print_accuracy_report(mu_df: pd.DataFrame, cov_df: pd.DataFrame) -> None:
    """Print a formatted accuracy report to stdout."""
    summ = summarize_forecast_accuracy(mu_df, cov_df)

    print("\n" + "=" * 70)
    print("FORECAST ACCURACY REPORT  (EWMA walk-forward evaluation)")
    print("=" * 70)

    # ── μ per-asset ───────────────────────────────────────────────────────────
    print("\n── Return forecast accuracy (per asset) ────────────────────────────────")
    print(f"  {'Asset':<6}  {'Bias':>8}  {'MAE':>8}  {'Pearson r':>10}  "
          f"{'Hit rate':>9}  {'N':>5}")
    print("  " + "─" * 55)
    for asset, row in summ["mu_per_asset"].iterrows():
        print(f"  {asset:<6}  {row.bias:>+8.3f}  {row.mae:>8.3f}  "
              f"{row.pearson_r:>10.3f}  {row.hit_rate:>9.1%}  {row.n:>5.0f}")

    # ── IC test ───────────────────────────────────────────────────────────────
    ict = summ["ic_test"]
    sig  = "**" if ict["p_value"] < 0.01 else ("*" if ict["p_value"] < 0.05 else "")
    print(f"\n── Cross-sectional IC (Spearman, pooled across rebalance dates) ────────")
    print(f"  Mean IC   : {ict['mean_ic']:+.4f}  (t={ict['t_stat']:+.2f}, "
          f"p={ict['p_value']:.4f}{sig}, n={ict['n']})")
    print(f"  Annualised IR_IC: {ict['ir_ic']:+.2f}")
    print(f"  Interpretation: {'significant predictive power' if ict['p_value'] < 0.05 else 'NOT significant — no reliable cross-sectional alpha'}")

    # ── Hit rate ──────────────────────────────────────────────────────────────
    ht = summ["hit_test"]
    sig2 = "**" if ht["p_value"] < 0.01 else ("*" if ht["p_value"] < 0.05 else "")
    print(f"\n── Direction hit rate (pooled) ─────────────────────────────────────────")
    print(f"  Hit rate  : {ht['hit_rate']:.1%}  (vs 50% null, "
          f"p={ht['p_value']:.4f}{sig2}, n={ht['n']})")
    print(f"  Interpretation: {'significantly above chance' if ht['p_value'] < 0.05 else 'NOT significantly above chance'}")

    # ── Σ accuracy ────────────────────────────────────────────────────────────
    cs = summ["cov_summary"]
    print(f"\n── Covariance forecast accuracy ({cs['n_dates']} rebalance dates) ────────────")
    print(f"  Frobenius rel. error (mean / median): "
          f"{cs['frobenius_err_mean']:.3f} / {cs['frobenius_err_median']:.3f}")
    print(f"  Correlation accuracy (pred vs real): {cs['corr_accuracy_mean']:.3f}")
    print(f"  Variance ratio (pred / real):        {cs['var_ratio_mean']:.3f}")
    vr = cs["var_ratio_mean"]
    if vr > 1.5:
        interp = "covariance over-estimated (conservative) — allocations will be more diversified than necessary"
    elif vr < 0.67:
        interp = "covariance UNDER-estimated (aggressive) — optimizer may take more risk than intended"
    else:
        interp = "well-calibrated (ratio near 1)"
    print(f"  Interpretation: {interp}")

    print("\n" + "=" * 70)

    # ── Overfitting warning ───────────────────────────────────────────────────
    print("\n── Overfitting / regime risk assessment ────────────────────────────────")
    if ict["p_value"] >= 0.05:
        print("  ⚠  IC is NOT statistically significant (p={:.3f} >= 0.05).".format(ict["p_value"]))
        print("     EWMA return forecasts cannot reliably rank assets out-of-sample.")
        print("     High Sharpe ratios in backtest are likely driven by concentration in")
        print("     regime winners (e.g. NVDA 2024) rather than repeatable alpha.")
        print("     Recommendation: use min-var (mu_method='zero') or the validated γ=5")
        print("     EWMA config as a more defensible live-trading baseline.")
    else:
        print("  ✓  IC is statistically significant (p={:.3f} < 0.05).".format(ict["p_value"]))
        print("     EWMA return forecast has measurable cross-sectional predictive power.")
        if ict["mean_ic"] < 0.10:
            print("     Note: mean IC = {:.3f} is modest — strategy still benefits from".format(ict["mean_ic"]))
            print("     diversification; don't over-concentrate based on this signal alone.")
    print()
