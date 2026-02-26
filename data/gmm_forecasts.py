"""
GMM-based expected returns (μ) and covariance (Σ) for Markowitz.

At each rebalance date we fit the same 4-component Gaussian mixture to option
prices (least-squares to market mid), then compute μ and Σ **deterministically**
from the mixture (no Monte Carlo): analytical moments for the Gaussian mixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

from config import OPTION_CHAINS_DIR, RAW_DIR, SPY_DAILY_FILE
from data.forecasts import ASSET_ORDER, load_chain_for_expiry

# Same bucket / GMM layout as distribution.ipynb
BUCKET_STEP = 1.0
PRICE_RANGE_STDS = 4.0
LOOKBACK_DAYS = 42  # ~2 months for realized vol


def _realized_vol_annual(price_series: pd.DataFrame, end_date: pd.Timestamp) -> float:
    """Annualized realized vol from log returns over last LOOKBACK_DAYS."""
    if price_series is None or len(price_series) < 2:
        return 0.15
    px = price_series.loc[price_series.index <= end_date].tail(LOOKBACK_DAYS + 1)
    if len(px) < 2:
        return 0.15
    log_ret = np.log(px["close"] / px["close"].shift(1)).dropna()
    if len(log_ret) < 2:
        return 0.15
    return float(log_ret.std() * np.sqrt(252))


def _gaussian_probs_on_grid(centers, left_edges, right_edges, mu, sigma):
    sig_safe = max(sigma, 1e-10)
    p = norm.cdf((right_edges - mu) / sig_safe) - norm.cdf((left_edges - mu) / sig_safe)
    p = np.maximum(p, 0.0)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = p.sum()
    return p / s if s > 1e-20 else (np.ones_like(p) / len(p))


def _payoff_call(S, K):
    return np.maximum(S - K, 0.0)


def _payoff_put(S, K):
    return np.maximum(K - S, 0.0)


def _mixture_option_price(centers, probs_list, weights, K, is_call, r, T):
    probs_mix = sum(w * p for w, p in zip(weights, probs_list))
    probs_mix = probs_mix / probs_mix.sum()
    payoffs = _payoff_call(centers, K) if is_call else _payoff_put(centers, K)
    return np.exp(-r * T) * np.sum(payoffs * probs_mix)


def _component_probs(centers, left_edges, right_edges, mus, sigmas):
    """Bucket probabilities for each Gaussian component on the fixed grid."""
    probs_list = []
    for mu_c, sig_c in zip(mus, sigmas):
        sig_safe = max(sig_c, 1e-10)
        p = norm.cdf((right_edges - mu_c) / sig_safe) - norm.cdf((left_edges - mu_c) / sig_safe)
        p = np.maximum(p, 0.0)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        s = p.sum()
        p = p / s if s > 1e-20 else (np.ones_like(p) / len(p))
        probs_list.append(p)
    return probs_list


# ----- Deterministic moments under one Gaussian N(mu, sigma) -----
def _e_payoff_call_one(mu, sigma, K):
    """E[max(S-K, 0)] for S ~ N(mu, sigma)."""
    if sigma <= 0:
        return max(0.0, mu - K)
    z = (mu - K) / sigma
    return sigma * norm.pdf(z) + (mu - K) * norm.cdf(z)


def _e_payoff_put_one(mu, sigma, K):
    """E[max(K-S, 0)] for S ~ N(mu, sigma)."""
    if sigma <= 0:
        return max(0.0, K - mu)
    z = (K - mu) / sigma
    return sigma * norm.pdf(z) + (K - mu) * norm.cdf(z)


def _e_payoff_sq_call_one(mu, sigma, K):
    """E[max(S-K,0)^2] for S ~ N(mu, sigma)."""

    def integrand(s):
        return (max(s - K, 0.0) ** 2) * norm.pdf(s, mu, sigma)

    lo, hi = K, mu + 10 * sigma
    val, _ = quad(integrand, lo, hi, limit=100)
    return max(0.0, val)


def _e_payoff_sq_put_one(mu, sigma, K):
    """E[max(K-S,0)^2] for S ~ N(mu, sigma)."""

    def integrand(s):
        return (max(K - s, 0.0) ** 2) * norm.pdf(s, mu, sigma)

    lo, hi = max(0, mu - 10 * sigma), K
    val, _ = quad(integrand, lo, hi, limit=100)
    return max(0.0, val)


def _e_s_times_payoff_call_one(mu, sigma, K):
    """E[S * max(S-K, 0)] for S ~ N(mu, sigma)."""

    def integrand(s):
        return s * max(s - K, 0.0) * norm.pdf(s, mu, sigma)

    lo, hi = K, mu + 10 * sigma
    val, _ = quad(integrand, lo, hi, limit=100)
    return max(0.0, val)


def _e_s_times_payoff_put_one(mu, sigma, K):
    """E[S * max(K-S, 0)] for S ~ N(mu, sigma)."""

    def integrand(s):
        return s * max(K - s, 0.0) * norm.pdf(s, mu, sigma)

    lo, hi = max(0, mu - 10 * sigma), K
    val, _ = quad(integrand, lo, hi, limit=100)
    return max(0.0, val)


def _build_compare_and_fit_gmm(
    chain_date: str,
    spot: float,
    spy_df: pd.DataFrame,
    r: float = 0.05,
    underlying: str = "SPY",
):
    """
    Load chain, build bucket grid, fit a flexible 6-component GMM to market mid.
    Returns (compare, w_fit, means_gmm, sigmas_gmm, T, expiry_used).
    compare has mix_price; we need ATM call/put for forecasts.
    """
    calls, puts, expiry_used = load_chain_for_expiry(
        chain_date=chain_date, expiry=None, underlying=underlying
    )
    exp_ts = pd.Timestamp(expiry_used)
    ref = pd.Timestamp(chain_date)
    T = max((exp_ts - ref).days / 365.0, 1e-4)

    end_date = ref
    realized_vol_annual = _realized_vol_annual(spy_df, end_date)
    sigma_S = spot * realized_vol_annual * np.sqrt(T)
    mu = spot
    half = PRICE_RANGE_STDS * sigma_S
    centers = np.arange(mu - half, mu + half + BUCKET_STEP * 0.5, BUCKET_STEP)
    centers = np.round(centers / BUCKET_STEP) * BUCKET_STEP
    centers = np.unique(centers)
    left_edges = centers - BUCKET_STEP / 2
    right_edges = centers + BUCKET_STEP / 2
    probs = norm.cdf((right_edges - mu) / sigma_S) - norm.cdf((left_edges - mu) / sigma_S)
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()

    def our_option_price(K, is_call):
        payoffs = _payoff_call(centers, K) if is_call else _payoff_put(centers, K)
        return np.exp(-r * T) * np.sum(payoffs * probs)

    strike_lo = centers.min()
    strike_hi = centers.max()
    df_list = []
    for _, row in calls.iterrows():
        K = float(row["strike"])
        if K < strike_lo or K > strike_hi:
            continue
        mid = float(row.get("mid", (row.get("bid", 0) + row.get("ask", 0)) / 2))
        if np.isnan(mid) or mid <= 0:
            continue
        df_list.append({
            "strike": K, "type": "call", "market_mid": mid,
            "our_price": our_option_price(K, True),
        })
    for _, row in puts.iterrows():
        K = float(row["strike"])
        if K < strike_lo or K > strike_hi:
            continue
        mid = float(row.get("mid", (row.get("bid", 0) + row.get("ask", 0)) / 2))
        if np.isnan(mid) or mid <= 0:
            continue
        df_list.append({
            "strike": K, "type": "put", "market_mid": mid,
            "our_price": our_option_price(K, False),
        })
    compare = pd.DataFrame(df_list)
    compare = compare[~((compare["our_price"] < 1.0) & (compare["market_mid"] > 100.0))].copy()
    if len(compare) < 5:
        raise ValueError(f"Too few options after filter on {chain_date}")

    # Flexible 6-component GMM: free means/variances, IV-based initial widths
    N_COMPONENTS = 6

    # Use ATM implied volatility (if available) to set initial spot-space sigma
    atm_call = calls.iloc[(calls["strike"] - spot).abs().idxmin()] if len(calls) > 0 else None
    iv_atm = float(atm_call.get("impl_vol", np.nan)) if atm_call is not None else np.nan
    if not np.isfinite(iv_atm) or iv_atm <= 0:
        iv_atm = realized_vol_annual
    sigma_S_iv = spot * iv_atm * np.sqrt(T)

    mu_init = spot + sigma_S_iv * np.array([-2.0, -1.0, -0.3, 0.0, 0.7, 1.5])
    sigma_init = sigma_S_iv * np.array([2.0, 1.5, 1.0, 1.0, 1.2, 1.5])

    MIN_SIGMA = 10.0  # minimum terminal-price sigma to avoid narrow Gaussians
    MAX_SIGMA = float(np.max(sigma_init)) * 4  # cap width so no component is absurdly wide
    MIN_MU = spot * 0.85  # component means at least 15% below spot (experiment)

    def _unpack_params(x):
        mus = x[:N_COMPONENTS]
        mus = np.maximum(mus, MIN_MU)
        log_sig = np.clip(x[N_COMPONENTS:2 * N_COMPONENTS], np.log(MIN_SIGMA), np.log(MAX_SIGMA))
        sigmas_raw = np.exp(log_sig)
        sigmas = np.clip(sigmas_raw, MIN_SIGMA, MAX_SIGMA)
        logits = x[2 * N_COMPONENTS : 3 * N_COMPONENTS]
        w = np.exp(logits - logits.max())
        w /= w.sum()
        return mus, sigmas, w

    def obj(x):
        mus, sigmas, w = _unpack_params(x)
        probs_list_local = _component_probs(centers, left_edges, right_edges, mus, sigmas)
        sse = 0.0
        for _, row in compare.iterrows():
            K, is_call = row["strike"], (row["type"] == "call")
            model_p = _mixture_option_price(centers, probs_list_local, w, K, is_call, r, T)
            sse += (model_p - row["market_mid"]) ** 2
        return sse

    x0 = np.concatenate([mu_init, np.log(sigma_init), np.zeros(N_COMPONENTS)])
    res = minimize(obj, x0=x0, method="Nelder-Mead", options={"maxiter": 8000})
    means_gmm, sigmas_gmm, w_fit = _unpack_params(res.x)

    probs_list = _component_probs(centers, left_edges, right_edges, means_gmm, sigmas_gmm)
    compare["mix_price"] = [
        _mixture_option_price(centers, probs_list, w_fit, row["strike"], row["type"] == "call", r, T)
        for _, row in compare.iterrows()
    ]
    return compare, w_fit, means_gmm, sigmas_gmm, T, r, expiry_used


def compute_gmm_forecasts(
    chain_date: str,
    spot: float,
    spy_df: pd.DataFrame,
    r: float = 0.05,
    underlying: str = "SPY",
):
    """
    Fit a flexible 6-component GMM to the option chain at chain_date and return
    μ (4,) and Σ (4,4) for [SPY, SPY_CALL, SPY_PUT, USDOLLAR] from the mixture
    **deterministically** (no sampling).
    """
    compare, w_fit, means_gmm, sigmas_gmm, T, r, _ = _build_compare_and_fit_gmm(
        chain_date, spot, spy_df, r=r, underlying=underlying
    )
    # ATM call and put (strike closest to spot)
    call_df = compare[compare["type"] == "call"]
    put_df = compare[compare["type"] == "put"]
    if len(call_df) == 0 or len(put_df) == 0:
        raise ValueError("No call or put in compare")
    K_call = call_df.loc[(call_df["strike"] - spot).abs().idxmin(), "strike"]
    K_put = put_df.loc[(put_df["strike"] - spot).abs().idxmin(), "strike"]
    C0 = float(compare[(compare["type"] == "call") & (compare["strike"] == K_call)]["mix_price"].iloc[0])
    P0 = float(compare[(compare["type"] == "put") & (compare["strike"] == K_put)]["mix_price"].iloc[0])

    n_comp = len(means_gmm)

    # Mixture moments (deterministic)
    E_S = sum(w_fit[k] * means_gmm[k] for k in range(n_comp))
    E_S2 = sum(w_fit[k] * (sigmas_gmm[k] ** 2 + means_gmm[k] ** 2) for k in range(n_comp))
    Var_S = E_S2 - E_S ** 2

    E_payoff_c = sum(
        w_fit[k] * _e_payoff_call_one(means_gmm[k], sigmas_gmm[k], K_call) for k in range(n_comp)
    )
    E_payoff_p = sum(
        w_fit[k] * _e_payoff_put_one(means_gmm[k], sigmas_gmm[k], K_put) for k in range(n_comp)
    )
    E_payoff_c2 = sum(
        w_fit[k] * _e_payoff_sq_call_one(means_gmm[k], sigmas_gmm[k], K_call) for k in range(n_comp)
    )
    E_payoff_p2 = sum(
        w_fit[k] * _e_payoff_sq_put_one(means_gmm[k], sigmas_gmm[k], K_put) for k in range(n_comp)
    )
    E_S_payoff_c = sum(
        w_fit[k] * _e_s_times_payoff_call_one(means_gmm[k], sigmas_gmm[k], K_call) for k in range(n_comp)
    )
    E_S_payoff_p = sum(
        w_fit[k] * _e_s_times_payoff_put_one(means_gmm[k], sigmas_gmm[k], K_put) for k in range(n_comp)
    )

    R_cash = np.exp(r * T) - 1.0
    # Returns: R_s = (S_T - spot)/spot, R_c = (payoff_c - C0)/C0, R_p = (payoff_p - P0)/P0
    E_R_s = (E_S - spot) / spot
    E_R_c = (E_payoff_c - C0) / C0 if C0 != 0 else 0.0
    E_R_p = (E_payoff_p - P0) / P0 if P0 != 0 else 0.0
    mu = np.array([E_R_s, E_R_c, E_R_p, R_cash])

    # Variances and covariances
    Var_R_s = Var_S / (spot ** 2)
    E_R_c2 = (E_payoff_c2 - 2 * C0 * E_payoff_c + C0 ** 2) / (C0 ** 2) if C0 != 0 else 0.0
    E_R_p2 = (E_payoff_p2 - 2 * P0 * E_payoff_p + P0 ** 2) / (P0 ** 2) if P0 != 0 else 0.0
    Var_R_c = E_R_c2 - E_R_c ** 2
    Var_R_p = E_R_p2 - E_R_p ** 2

    # Cov(R_s, R_c) = E[(S-spot)(payoff_c-C0)]/(spot*C0) - E[R_s]E[R_c]
    # E[(S-spot)(payoff_c-C0)] = E[S*payoff_c] - spot*E[payoff_c] - C0*E[S] + spot*C0
    E_R_s_R_c = (E_S_payoff_c - spot * E_payoff_c - C0 * E_S + spot * C0) / (spot * C0) if (spot * C0) != 0 else 0.0
    Cov_s_c = E_R_s_R_c - E_R_s * E_R_c
    E_R_s_R_p = (E_S_payoff_p - spot * E_payoff_p - P0 * E_S + spot * P0) / (spot * P0) if (spot * P0) != 0 else 0.0
    Cov_s_p = E_R_s_R_p - E_R_s * E_R_p
    # Cov(R_c, R_p): E[R_c*R_p] - E[R_c]E[R_p]. E[(payoff_c-C0)(payoff_p-P0)]/(C0*P0). Under mixture, payoff_c and payoff_p are from same S_T.
    E_payoff_c_payoff_p = 0.0  # max(S-K_c,0)*max(K_p-S,0) is 0 when K_c >= K_p (both ATM so same strike often). Approx 0.
    E_R_c_R_p = (E_payoff_c_payoff_p - E_payoff_c * P0 - E_payoff_p * C0 + C0 * P0) / (C0 * P0) if (C0 * P0) != 0 else 0.0
    Cov_c_p = E_R_c_R_p - E_R_c * E_R_p

    Sigma = np.array([
        [Var_R_s, Cov_s_c, Cov_s_p, 0.0],
        [Cov_s_c, Var_R_c, Cov_c_p, 0.0],
        [Cov_s_p, Cov_c_p, Var_R_p, 0.0],
        [0.0, 0.0, 0.0, 1e-12],
    ])
    Sigma = (Sigma + Sigma.T) / 2.0
    Sigma += 1e-8 * np.eye(4)
    return mu, Sigma


def compute_gmm_forecasts_sampled(
    chain_date: str,
    spot: float,
    spy_df: pd.DataFrame,
    n_samples: int = 100_000,
    r: float = 0.05,
    underlying: str = "SPY",
    seed: int | None = None,
):
    """
    Fit the flexible GMM at chain_date, sample n_samples terminal spots, and compute
    return samples for [underlying, call, put, cash]. Returns (mu, Sigma) as sample
    mean and sample covariance of those returns (so predicted expected returns and
    predicted covariance from the GMM).
    """
    if seed is not None:
        np.random.seed(seed)
    compare, w_fit, means_gmm, sigmas_gmm, T, r, _ = _build_compare_and_fit_gmm(
        chain_date, spot, spy_df, r=r, underlying=underlying
    )
    call_df = compare[compare["type"] == "call"]
    put_df = compare[compare["type"] == "put"]
    if len(call_df) == 0 or len(put_df) == 0:
        raise ValueError("No call or put in compare")
    K_call = call_df.loc[(call_df["strike"] - spot).abs().idxmin(), "strike"]
    K_put = put_df.loc[(put_df["strike"] - spot).abs().idxmin(), "strike"]
    C0 = float(compare[(compare["type"] == "call") & (compare["strike"] == K_call)]["mix_price"].iloc[0])
    P0 = float(compare[(compare["type"] == "put") & (compare["strike"] == K_put)]["mix_price"].iloc[0])

    n_comp = len(means_gmm)
    comp = np.random.choice(n_comp, size=n_samples, p=w_fit)
    S_T = np.random.normal(means_gmm[comp], sigmas_gmm[comp])

    R_underlying = (S_T - spot) / spot
    payoff_call = np.maximum(S_T - K_call, 0.0)
    payoff_put = np.maximum(K_put - S_T, 0.0)
    R_call = (payoff_call - C0) / C0 if C0 != 0 else np.zeros_like(S_T)
    R_put = (payoff_put - P0) / P0 if P0 != 0 else np.zeros_like(S_T)
    R_cash = np.full(n_samples, np.exp(r * T) - 1.0)

    returns = np.column_stack([R_underlying, R_call, R_put, R_cash])
    mu = np.mean(returns, axis=0)
    Sigma = np.cov(returns, rowvar=False)
    Sigma = (Sigma + Sigma.T) / 2.0
    Sigma += 1e-8 * np.eye(4)
    return mu, Sigma
