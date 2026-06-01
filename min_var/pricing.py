"""
Lognormal option pricing helpers.
Migrated verbatim from log_norm.ipynb Cell 8.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.optimize import brentq

from .config import R_ANN, OPTION_CHAINS_DIR

# We need load_chain_for_expiry and BS helpers from the data layer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.forecasts import load_chain_for_expiry, _bs_call_price, _bs_put_price  # noqa: E402


def lognormal_price(K, is_call, mu_ann, sigma_ann, spot_0, T, r=R_ANN):
    """Closed-form log-normal option price (Black-Scholes with arbitrary drift)."""
    sigma_ann = max(float(sigma_ann), 1e-4)
    sigma_t   = max(sigma_ann * np.sqrt(T), 1e-6)
    m  = np.log(spot_0) + (float(mu_ann) - 0.5 * sigma_ann**2) * T
    d2 = (m - np.log(K)) / sigma_t
    d1 = d2 + sigma_t
    mean_st = np.exp(m + 0.5 * sigma_t**2)
    if is_call:
        payoff = mean_st * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        payoff = K * norm.cdf(-d2) - mean_st * norm.cdf(-d1)
    return np.exp(-r * T) * payoff


def lognormal_price_vec(df, mu_ann, sigma_ann, spot_0, T, r=R_ANN):
    """Price a DataFrame of options (columns: strike, type) under log-normal."""
    return np.array([
        lognormal_price(row["strike"], row["type"] == "call",
                        mu_ann, sigma_ann, spot_0, T, r)
        for _, row in df.iterrows()
    ])


def to_per_share(mid, K, spot_0, is_call, r, T, tol=0.01):
    """Return per-share option mid, or NaN if bounds fail."""
    mid = float(mid)
    if not np.isfinite(mid) or mid <= 0:
        return np.nan
    if is_call:
        lb, ub = 0.0, spot_0
    else:
        lb, ub = max(0.0, K - spot_0), K * np.exp(-r * T)
    if lb - tol <= mid <= ub + tol:
        return mid
    if mid > 5.0 * ub and ub > 0:
        candidate = mid / 100.0
        if lb - tol <= candidate <= ub + tol:
            return candidate
    return np.nan


def load_options_for_dte(chain_date, target_dte, spot_0, asset="SPY",
                          r=R_ANN, min_mid=0.02, max_spread=0.80):
    """
    Load calls + puts closest to target_dte, filter to valid per-share mids.
    Returns (option_df, expiry_str, T).
    """
    ref = pd.Timestamp(chain_date)
    chain_dir = OPTION_CHAINS_DIR if asset == "SPY" else OPTION_CHAINS_DIR / asset

    meta = pd.read_parquet(chain_dir / f"calls_{chain_date}.parquet")
    meta.columns = [c.replace("impliedVolatility", "impl_vol").lower() for c in meta.columns]
    expiry_arg = None
    if "expiry" in meta.columns:
        expiries  = sorted(meta["expiry"].dropna().unique())
        dte_list  = [(e, (pd.Timestamp(e) - ref).days) for e in expiries
                     if (pd.Timestamp(e) - ref).days > 0]
        if dte_list:
            expiry_arg = min(dte_list, key=lambda x: abs(x[1] - target_dte))[0]

    calls, puts, expiry_used = load_chain_for_expiry(
        chain_date=chain_date, expiry=expiry_arg, underlying=asset,
        min_mid=min_mid, max_spread=max_spread,
    )
    if len(calls) == 0 or len(puts) == 0:
        raise ValueError(f"No valid calls/puts on {chain_date} near DTE {target_dte}")

    exp_ts = pd.Timestamp(expiry_used)
    T      = max((exp_ts - ref).days / 365.0, 1e-4)

    rows = []
    for is_call, df_src in [(True, calls), (False, puts)]:
        for _, row in df_src.iterrows():
            K   = float(row["strike"])
            raw = float(row.get("mid", (row.get("bid", 0.0) + row.get("ask", 0.0)) / 2.0))
            mid = to_per_share(raw, K, spot_0, is_call, r, T)
            if np.isfinite(mid):
                rows.append({"strike": K, "type": "call" if is_call else "put", "market_mid": mid})

    if len(rows) < 10:
        raise ValueError(f"Too few valid options after filter on {chain_date} DTE~{target_dte}")

    return pd.DataFrame(rows), expiry_used, T


def _bs_call(S, K, T, r, sigma):
    """Standard Black-Scholes call price (risk-neutral drift = r)."""
    sigma = max(sigma, 1e-6)
    sqrt_T = max(np.sqrt(T), 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def compute_atm_iv(option_df, spot_0, T, r=R_ANN, iv_lo=1e-4, iv_hi=5.0):
    """
    Compute ATM implied volatility from the option chain.
    Returns (sigma_iv, strike_used).
    """
    calls = option_df[option_df["type"] == "call"].copy()
    if calls.empty:
        raise ValueError("No calls in option_df for IV computation.")

    calls = calls.assign(_dist=(calls["strike"] - spot_0).abs())
    atm_row = calls.loc[calls["_dist"].idxmin()]
    K_atm = float(atm_row["strike"])
    mid   = float(atm_row["market_mid"])

    def objective(sig):
        return _bs_call(spot_0, K_atm, T, r, sig) - mid

    try:
        lo_val, hi_val = objective(iv_lo), objective(iv_hi)
        if lo_val * hi_val > 0:
            ivs = []
            for _, row in calls.iterrows():
                try:
                    iv = brentq(
                        lambda s: _bs_call(spot_0, row["strike"], T, r, s) - row["market_mid"],
                        iv_lo, iv_hi, xtol=1e-6
                    )
                    ivs.append(iv)
                except Exception:
                    pass
            if not ivs:
                raise ValueError("IV inversion failed on all calls.")
            return float(np.median(ivs)), K_atm

        sigma_iv = brentq(objective, iv_lo, iv_hi, xtol=1e-6)
        return float(sigma_iv), K_atm
    except Exception as e:
        raise ValueError(f"ATM IV inversion failed (K={K_atm:.1f}, mid={mid:.4f}): {e}")


def iv_lognormal_pdf(S_grid, spot_0, sigma_iv, T, r=R_ANN):
    """Log-normal PDF under risk-neutral measure using σ = ATM IV, drift = r."""
    sig_t = max(sigma_iv * np.sqrt(T), 1e-6)
    m     = np.log(spot_0) + (r - 0.5 * sigma_iv**2) * T
    return lognorm.pdf(S_grid, s=sig_t, scale=np.exp(m))
