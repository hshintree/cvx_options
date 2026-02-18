"""
Risk-neutral distribution (RND) from option chain and sampling to produce
expected return (μ) and covariance (Σ) for SPY, SPY_CALL, SPY_PUT, USDOLLAR.

Return model: horizon repricing (default)
------------------------------------------
1. Sample S_{t+1} at the next rebalance date (e.g. 7 days) from lognormal
   (or BL-implied density if the chain is clean enough).
2. Reprice call and put at S_{t+1} with remaining time T' = T_expiry - T_rebal
   using Black-Scholes and a sticky-strike IV rule (IV stays constant).
3. Compute returns:
   - SPY:  S_{t+1} / S_0 - 1
   - Call: C_{t+1} / C_0 - 1   where C_{t+1} = BS_Call(S_{t+1}, K, r, T', σ)
   - Put:  P_{t+1} / P_0 - 1   where P_{t+1} = BS_Put(S_{t+1}, K, r, T', σ)
   - Cash: exp(r * T_rebal) - 1
4. μ = sample mean; Σ = sample covariance.

This adds theta decay and vega effects, and breaks the call/put mirror
symmetry that arises from terminal-payoff-only pricing.

Fallback: if T' <= 0 (option expires before next rebalance), terminal payoff
is used instead of BS repricing.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config import (
    MAX_BID_ASK_SPREAD_PCT,
    MIN_BL_STRIKES,
    MIN_OPTION_MID,
    MU_CLIP_OPTION,
    MU_CLIP_SPY,
    OPTION_CHAINS_DIR,
    OPTION_RETURN_WINSORIZE_PCT,
    REBALANCE_DAYS,
    SPY_DAILY_FILE,
    TARGET_IDEAL_DTE,
    TARGET_MAX_DTE,
    TARGET_MIN_DTE,
)

logger = logging.getLogger(__name__)

ASSET_ORDER = ["SPY", "SPY_CALL", "SPY_PUT", "USDOLLAR"]

# Relaxed filter values used when strict filters leave too few strikes
_RELAXED_MIN_MID = 0.02
_RELAXED_MAX_SPREAD = 0.80


# ---------------------------------------------------------------------------
# Chain loading
# ---------------------------------------------------------------------------

def _latest_chain_date() -> Optional[str]:
    """Return latest date string (YYYY-MM-DD) for which we have calls/puts."""
    if not OPTION_CHAINS_DIR.exists():
        return None
    dates = set()
    for f in OPTION_CHAINS_DIR.glob("calls_*.parquet"):
        stem = f.stem.replace("calls_", "")
        if stem:
            dates.add(stem)
    return max(dates) if dates else None


def _filter_quotes(
    df: pd.DataFrame,
    min_mid: float = MIN_OPTION_MID,
    max_spread: float = MAX_BID_ASK_SPREAD_PCT,
) -> pd.DataFrame:
    """Filter option quotes by quality.  Returns sorted by strike.

    Applies spread / bid-ask checks only on rows that have live quotes
    (bid > 0 or ask > 0).  Rows with bid=ask=0 but a valid mid (from
    lastPrice) are kept — the penny-option floor (min_mid) still protects
    against tiny-denominator blowups.
    """
    df = df.dropna(subset=["mid", "strike"]).copy()
    df = df[df["mid"] >= min_mid]
    if "bid" in df.columns and "ask" in df.columns:
        has_quote = (df["bid"] > 0) | (df["ask"] > 0)
        quoted = df[has_quote]
        # For rows with live quotes: require ask >= bid and cap spread
        if len(quoted) > 0:
            ok = quoted["ask"] >= quoted["bid"]
            spread_pct = (quoted["ask"] - quoted["bid"]) / quoted["mid"].replace(0, np.nan)
            ok = ok & (spread_pct <= max_spread)
            bad_idx = quoted.index[~ok]
            df = df.drop(bad_idx)
    keep = ["strike", "mid", "expiry"]
    if "impl_vol" in df.columns:
        keep.append("impl_vol")
    return df[[c for c in keep if c in df.columns]].sort_values("strike")


def _add_mid_column(df: pd.DataFrame) -> None:
    """Compute *mid* price in-place from bid/ask, falling back to lastPrice.

    yfinance often stores bid=0, ask=0 for many contracts while lastPrice is
    valid.  We prefer (bid+ask)/2 when both are positive; otherwise we use
    lastPrice (or lastprice after lowercasing).
    """
    df["mid"] = np.nan
    if "bid" in df.columns and "ask" in df.columns:
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
        has_quote = (df["bid"] > 0) | (df["ask"] > 0)
        df.loc[has_quote, "mid"] = (df.loc[has_quote, "bid"] + df.loc[has_quote, "ask"]) / 2
    # Fall back to lastPrice / lastprice for rows still missing mid
    for col in ("lastprice", "last"):
        if col in df.columns:
            missing = df["mid"].isna() | (df["mid"] <= 0)
            df.loc[missing, "mid"] = pd.to_numeric(df.loc[missing, col], errors="coerce")
            break


def load_chain_for_expiry(
    chain_date: Optional[str] = None,
    expiry: Optional[str] = None,
    min_mid: float = MIN_OPTION_MID,
    max_spread: float = MAX_BID_ASK_SPREAD_PCT,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load calls and puts for a single expiry.

    When *expiry* is None the function picks the expiry with the **most valid
    call strikes** after filtering (instead of blindly using the front month).
    """
    date_str = chain_date or _latest_chain_date()
    if not date_str:
        raise FileNotFoundError("No option chain found in " + str(OPTION_CHAINS_DIR))
    path_calls = OPTION_CHAINS_DIR / f"calls_{date_str}.parquet"
    path_puts = OPTION_CHAINS_DIR / f"puts_{date_str}.parquet"
    if not path_calls.exists() or not path_puts.exists():
        raise FileNotFoundError("Chain files not found for date " + date_str)
    all_calls = pd.read_parquet(path_calls)
    all_puts = pd.read_parquet(path_puts)
    for df in (all_calls, all_puts):
        df.columns = [c.replace("impliedVolatility", "impl_vol").lower() for c in df.columns]
    if "expiry" not in all_calls.columns:
        raise ValueError("Option chain missing 'expiry' column")
    for df in (all_calls, all_puts):
        _add_mid_column(df)

    expiries = sorted(all_calls["expiry"].unique())
    if not expiries:
        raise ValueError("No expiries in chain")

    # Resolve spot for strike-range scoring
    _spot_for_score = None
    if SPY_DAILY_FILE.exists():
        _spy = pd.read_parquet(SPY_DAILY_FILE)
        _spot_for_score = float(_spy["close"].iloc[-1])

    ref_date = pd.Timestamp(date_str)

    if expiry is not None and expiry in expiries:
        expiry_used = expiry
    else:
        # Step 1: filter expiries by DTE band [TARGET_MIN_DTE, TARGET_MAX_DTE]
        dte_candidates = []
        for exp in expiries:
            dte = (pd.Timestamp(exp) - ref_date).days
            if TARGET_MIN_DTE <= dte <= TARGET_MAX_DTE:
                dte_candidates.append((exp, dte))
        if not dte_candidates:
            # Widen: pick the nearest expiry with DTE > TARGET_MIN_DTE
            for exp in expiries:
                dte = (pd.Timestamp(exp) - ref_date).days
                if dte >= TARGET_MIN_DTE:
                    dte_candidates.append((exp, dte))
            if dte_candidates:
                logger.warning(
                    "No expiry in [%d, %d] DTE; widened to %d..%d DTE (%d candidates)",
                    TARGET_MIN_DTE, TARGET_MAX_DTE,
                    dte_candidates[0][1], dte_candidates[-1][1],
                    len(dte_candidates),
                )
            else:
                # Last resort: use any expiry with DTE > 0
                dte_candidates = [(e, (pd.Timestamp(e) - ref_date).days)
                                  for e in expiries
                                  if (pd.Timestamp(e) - ref_date).days > 0]
                if dte_candidates:
                    logger.warning("No expiry with DTE >= %d; using all future expiries",
                                   TARGET_MIN_DTE)
                else:
                    logger.warning("No future expiries found; using closest available")
                    dte_candidates = [(expiries[-1], max((pd.Timestamp(expiries[-1]) - ref_date).days, 1))]

        # Step 2: among DTE candidates, score by strike coverage around spot
        best_expiry = dte_candidates[0][0]
        best_score = -1
        for exp, dte in dte_candidates:
            filt = _filter_quotes(
                all_calls[all_calls["expiry"] == exp],
                min_mid=min_mid, max_spread=max_spread,
            )
            if _spot_for_score is not None and len(filt) > 0:
                lo = _spot_for_score * 0.90
                hi = _spot_for_score * 1.10
                near = filt[(filt["strike"] >= lo) & (filt["strike"] <= hi)]
                has_above = (filt["strike"] > _spot_for_score).any()
                has_below = (filt["strike"] < _spot_for_score).any()
                coverage = len(near) + (10 if has_above and has_below else 0)
            else:
                coverage = len(filt)
            # Prefer closer-to-ideal DTE when coverage is comparable
            dte_penalty = abs(dte - TARGET_IDEAL_DTE) * 0.1
            score = coverage - dte_penalty
            if score > best_score:
                best_score = score
                best_expiry = exp

        expiry_used = best_expiry
        chosen_dte = (pd.Timestamp(expiry_used) - ref_date).days
        logger.info(
            "Expiry=%s  DTE=%d  score=%.1f (from %d DTE-candidates, %d total expiries)",
            expiry_used, chosen_dte, best_score, len(dte_candidates), len(expiries),
        )

    calls = _filter_quotes(
        all_calls[all_calls["expiry"] == expiry_used].copy(),
        min_mid=min_mid, max_spread=max_spread,
    )
    puts = _filter_quotes(
        all_puts[all_puts["expiry"] == expiry_used].copy(),
        min_mid=min_mid, max_spread=max_spread,
    )
    logger.info(
        "expiry=%s  calls after filter=%d  puts after filter=%d",
        expiry_used, len(calls), len(puts),
    )
    if len(calls) > 0:
        logger.info(
            "call strike range: %.1f .. %.1f",
            calls["strike"].min(), calls["strike"].max(),
        )
    return calls, puts, expiry_used


# ---------------------------------------------------------------------------
# Breeden-Litzenberger density (with spline smoothing)
# ---------------------------------------------------------------------------

def _smooth_call_prices(K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Fit a smoothing spline to call prices C(K) and enforce monotone
    non-increasing (no-arb: calls decrease in strike).
    """
    try:
        spline = UnivariateSpline(K, C, k=3, s=len(K))
        C_smooth = spline(K)
        for i in range(1, len(C_smooth)):
            C_smooth[i] = min(C_smooth[i], C_smooth[i - 1])
        C_smooth = np.maximum(C_smooth, 0.0)
        return C_smooth
    except Exception:
        logger.debug("Spline smoothing failed; using raw call prices")
        return C


def breeden_litzenberger_pdf(
    calls: pd.DataFrame,
    r: float,
    T: float,
    smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Risk-neutral density q(K) from call prices via Breeden-Litzenberger.

    Optionally smooth call prices with a spline before differentiating,
    which dramatically reduces noise from discrete quotes.

    Returns interior strikes and pdf (endpoints dropped by second derivative).
    """
    K = calls["strike"].values.astype(float)
    C = calls["mid"].values.astype(float)
    if len(K) < 3:
        return np.array([]), np.array([])
    if smooth:
        C = _smooth_call_prices(K, C)
    dK = np.diff(K)
    if np.any(dK <= 0):
        return np.array([]), np.array([])
    dK_mid = (dK[1:] + dK[:-1]) / 2
    d2C = (C[2:] - 2 * C[1:-1] + C[:-2]) / (dK_mid ** 2)
    q = np.exp(r * T) * d2C
    q = np.maximum(q, 1e-12)
    return K[1:-1].copy(), q


def sample_terminal_prices(
    strikes: np.ndarray,
    pdf: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Sample S_T from discrete RND (strikes, pdf) via inverse CDF."""
    if len(strikes) == 0 or len(pdf) == 0:
        return np.array([])
    pdf = pdf / pdf.sum()
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    u = np.random.default_rng().uniform(0, 1, size=n_samples)
    idx = np.searchsorted(cdf, u, side="right")
    idx = np.clip(idx, 1, len(cdf) - 1)
    w = (u - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1] + 1e-12)
    S_T = (1 - w) * strikes[idx - 1] + w * strikes[idx]
    return S_T


# ---------------------------------------------------------------------------
# Lognormal implied-vol fallback
# ---------------------------------------------------------------------------

def _bs_implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    is_call: bool = True,
) -> Optional[float]:
    """Newton-Raphson BS IV inversion.  Returns None if it doesn't converge."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    sigma = 0.20
    for _ in range(50):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            bs = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            bs = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        vega = S * np.sqrt(T) * norm.pdf(d1)
        if vega < 1e-12:
            break
        sigma = sigma - (bs - price) / vega
        sigma = max(sigma, 0.01)
        if abs(bs - price) < 1e-6:
            if 0.01 < sigma < 3.0:
                return float(sigma)
            break
    return None


def _get_atm_iv(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    r: float = 0.05,
    T: float = 14 / 365.0,
) -> float:
    """Extract ATM implied volatility from chain.

    1. Try the chain's impl_vol column.
    2. If garbage, invert BS from ATM mid price.
    3. Falls back to 0.20.
    """
    for df in (calls, puts):
        if "impl_vol" in df.columns and len(df) > 0:
            atm_idx = (df["strike"] - spot).abs().idxmin()
            iv = df.loc[atm_idx, "impl_vol"]
            if pd.notna(iv) and 0.01 < float(iv) < 3.0:
                return float(iv)
    # Fallback: invert BS from ATM option price
    for is_call, df in [(True, calls), (False, puts)]:
        if len(df) > 0 and "mid" in df.columns:
            atm_idx = (df["strike"] - spot).abs().idxmin()
            K = float(df.loc[atm_idx, "strike"])
            mid = float(df.loc[atm_idx, "mid"])
            iv = _bs_implied_vol(mid, spot, K, r, T, is_call=is_call)
            if iv is not None:
                logger.info("ATM IV from BS inversion: %.4f (K=%.0f, mid=%.2f)", iv, K, mid)
                return iv
    return 0.20


def _lognormal_sample(
    spot: float,
    r: float,
    T: float,
    sigma: float,
    n_samples: int,
) -> np.ndarray:
    """S_T = S_0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)."""
    rng = np.random.default_rng()
    Z = rng.standard_normal(n_samples)
    return spot * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def _bs_call_price(S: float, K: float, r: float, T: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put_price(S: float, K: float, r: float, T: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# Vectorized BS repricing (for Monte Carlo arrays)
# ---------------------------------------------------------------------------

def _bs_call_vec(S: np.ndarray, K: float, r: float, T: float, sigma: float) -> np.ndarray:
    """Vectorized BS call price for array of spot values."""
    if T <= 0:
        return np.maximum(S - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put_vec(S: np.ndarray, K: float, r: float, T: float, sigma: float) -> np.ndarray:
    """Vectorized BS put price for array of spot values."""
    if T <= 0:
        return np.maximum(K - S, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Fallback covariance (instead of identity)
# ---------------------------------------------------------------------------

def _reasonable_fallback_sigma(sig_spy: float = 0.025) -> pd.DataFrame:
    """
    Weekly SPY vol ~ 2.5 %, option vols ~ 2× SPY, call/put corr ~ −0.5.
    """
    v_spy = sig_spy ** 2
    v_opt = (2.0 * sig_spy) ** 2
    cov_spy_call = 0.6 * sig_spy * (2.0 * sig_spy)
    cov_spy_put = -0.6 * sig_spy * (2.0 * sig_spy)
    cov_call_put = -0.5 * (2.0 * sig_spy) ** 2
    v_cash = 1e-10
    return pd.DataFrame(
        [
            [v_spy, cov_spy_call, cov_spy_put, 0.0],
            [cov_spy_call, v_opt, cov_call_put, 0.0],
            [cov_spy_put, cov_call_put, v_opt, 0.0],
            [0.0, 0.0, 0.0, v_cash],
        ],
        index=ASSET_ORDER,
        columns=ASSET_ORDER,
    )


# ---------------------------------------------------------------------------
# Main forecast function
# ---------------------------------------------------------------------------

def compute_rnd_forecasts(
    chain_date: Optional[str] = None,
    spot: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    expiry: Optional[str] = None,
    n_samples: int = 10_000,
    use_risk_neutral_mu: bool = False,
    option_vol_mult: float = 2.0,
    mu_clip_spy: Optional[Tuple[float, float]] = None,
    mu_clip_option: Optional[Tuple[float, float]] = None,
    rebalance_days: Optional[int] = REBALANCE_DAYS,
    return_diagnostics: bool = False,
) -> Union[Tuple[pd.Series, pd.DataFrame], Tuple[pd.Series, pd.DataFrame, Dict]]:
    """
    Sample S_{t+1} at the rebalance horizon, reprice options at t+1 via BS
    (sticky-strike IV), and compute per-sample returns.  Falls back to
    terminal-payoff pricing when T_remain <= 0.

    Parameters
    ----------
    chain_date : snapshot date (YYYY-MM-DD); default latest
    spot : SPY price; if None, read from spy_daily
    risk_free_rate : annual rate; default 0.05
    expiry : target expiry; default = expiry with best DTE/strike coverage
    n_samples : Monte-Carlo draws
    use_risk_neutral_mu : if True, mu = r_period for all assets
    option_vol_mult : scale option vol to this × SPY vol (default 2)
    mu_clip_spy / mu_clip_option : plausible weekly return range
    rebalance_days : holding period in days.  None (default) = hold to expiry.
        Set to an integer (e.g. 7) for mid-life horizon repricing.
    return_diagnostics : return (mu, Sigma, diag_dict)
    """
    mu_clip_spy = mu_clip_spy or MU_CLIP_SPY
    mu_clip_option = mu_clip_option or MU_CLIP_OPTION

    # ------------------------------------------------------------------
    # 1. Load chain — strict filters first, relax if too few strikes
    # ------------------------------------------------------------------
    calls, puts, expiry_used = load_chain_for_expiry(
        chain_date=chain_date, expiry=expiry,
    )
    if len(calls) < MIN_BL_STRIKES:
        logger.info(
            "Strict filters left %d calls (need %d); relaxing filters",
            len(calls), MIN_BL_STRIKES,
        )
        calls, puts, expiry_used = load_chain_for_expiry(
            chain_date=chain_date, expiry=expiry,
            min_mid=_RELAXED_MIN_MID, max_spread=_RELAXED_MAX_SPREAD,
        )
        logger.info("Relaxed filters: %d calls, %d puts", len(calls), len(puts))

    # ------------------------------------------------------------------
    # 2. Resolve spot, rate, time-to-expiry
    # ------------------------------------------------------------------
    if spot is None:
        if SPY_DAILY_FILE.exists():
            spy = pd.read_parquet(SPY_DAILY_FILE)
            spot = float(spy["close"].iloc[-1])
        else:
            spot = 500.0
    r = risk_free_rate if risk_free_rate is not None else 0.05
    date_str = chain_date or _latest_chain_date()
    T = (pd.Timestamp(expiry_used) - pd.Timestamp(date_str)).days / 365.0
    T = max(T, 1 / 365.0)

    # ------------------------------------------------------------------
    # 3. Horizon repricing setup
    # ------------------------------------------------------------------
    # rebalance_days=None → hold to expiry (terminal payoff, roll at expiry)
    if rebalance_days is None:
        T_rebal = T
        use_terminal_payoff = True
    else:
        T_rebal = rebalance_days / 365.0
        use_terminal_payoff = T - T_rebal <= 1e-6
    T_remain = max(T - T_rebal, 0.0)
    logger.info(
        "T_expiry=%.4f  T_rebal=%.4f  T_remain=%.4f  mode=%s",
        T, T_rebal, T_remain,
        "hold-to-expiry" if use_terminal_payoff else "horizon-reprice",
    )

    # ------------------------------------------------------------------
    # 4. Sample S_{t+1} at rebalance horizon
    # ------------------------------------------------------------------
    # We sample spot at the rebalance date (not at expiry).
    # BL density gives S_T at expiry — not directly useful for horizon
    # repricing.  We still attempt BL to extract the risk-neutral shape,
    # but for the prototype the lognormal is the workhorse.
    T_sample = T if use_terminal_payoff else T_rebal

    strikes, pdf = breeden_litzenberger_pdf(calls, r, T, smooth=True)
    n_interior = len(strikes)

    bl_method = "lognormal_iv_fallback"
    bl_reject_reason = ""
    if n_interior >= MIN_BL_STRIKES:
        if strikes.max() < spot * 0.98 or strikes.min() > spot * 1.02:
            bl_reject_reason = (
                "strikes don't span spot: [%.0f, %.0f] vs spot=%.0f"
                % (strikes.min(), strikes.max(), spot)
            )
        else:
            S_T_candidate = sample_terminal_prices(strikes, pdf, n_samples)
            mean_ST = float(np.mean(S_T_candidate))
            tol = max(0.03, min(0.15, 0.4 * np.sqrt(T)))
            pct_off = abs(mean_ST / spot - 1.0)
            if pct_off < tol:
                bl_method = "breeden_litzenberger"
                if use_terminal_payoff:
                    S_next = S_T_candidate
                else:
                    # BL gives terminal S_T; rescale to rebalance horizon
                    # by shrinking the deviation proportionally to sqrt(T)
                    shrink = np.sqrt(T_rebal / T)
                    S_next = spot * np.exp(
                        shrink * np.log(S_T_candidate / spot)
                    )
                logger.info(
                    "BL density: %d interior strikes, mean(S_T)=%.1f vs spot=%.1f "
                    "(%.1f%% off, tol=%.1f%%), %d paths",
                    n_interior, mean_ST, spot, 100 * pct_off, 100 * tol, n_samples,
                )
            else:
                bl_reject_reason = (
                    "mean(S_T)=%.1f vs spot=%.1f (%.1f%% off > %.1f%% tol); "
                    "lastPrice data too noisy for BL"
                    % (mean_ST, spot, 100 * pct_off, 100 * tol)
                )
    else:
        bl_reject_reason = "%d interior strikes < %d required" % (n_interior, MIN_BL_STRIKES)

    if bl_reject_reason:
        logger.warning("BL rejected: %s", bl_reject_reason)

    if bl_method != "breeden_litzenberger":
        atm_iv = _get_atm_iv(calls, puts, spot, r, T)
        logger.warning(
            "Using lognormal fallback IV=%.2f  (%d BL interior strikes)",
            atm_iv, n_interior,
        )
        S_next = _lognormal_sample(spot, r, T_sample, atm_iv, n_samples)

    # ------------------------------------------------------------------
    # 4. ATM option prices (for return denominator)
    # ------------------------------------------------------------------
    atm_iv_used = _get_atm_iv(calls, puts, spot, r, T)

    # Try to get k_atm from chain; if nearest strike is > 2% from spot,
    # use spot as k_atm and BS-price the options (avoids stale/off-ATM quotes)
    k_atm = spot
    c_atm = MIN_OPTION_MID
    p_atm = MIN_OPTION_MID
    used_bs_pricing = False

    if len(calls) > 0:
        nearest_k = float(calls.iloc[(calls["strike"] - spot).abs().argmin()]["strike"])
        if abs(nearest_k / spot - 1.0) <= 0.02:
            k_atm = nearest_k
            c_row = calls.loc[calls["strike"] == k_atm]
            if len(c_row) > 0:
                c_atm = float(c_row["mid"].iloc[0])
            if len(puts) > 0:
                put_row = puts[puts["strike"] == k_atm]
                if len(put_row) > 0:
                    p_atm = float(put_row["mid"].iloc[0])
                else:
                    p_atm = float(puts.iloc[(puts["strike"] - k_atm).abs().argmin()]["mid"])

    if c_atm > 100:
        c_atm /= 100.0
    if p_atm > 100:
        p_atm /= 100.0

    # Fall back to BS pricing if chain quotes are missing or off-ATM
    if c_atm <= MIN_OPTION_MID or p_atm <= MIN_OPTION_MID or abs(k_atm / spot - 1) > 0.02:
        used_bs_pricing = True
        k_atm = round(spot)
        c_atm = max(_bs_call_price(spot, k_atm, r, T, atm_iv_used), MIN_OPTION_MID)
        p_atm = max(_bs_put_price(spot, k_atm, r, T, atm_iv_used), MIN_OPTION_MID)
        logger.info(
            "Using BS ATM pricing: k_atm=%.0f c_atm=%.4f p_atm=%.4f (IV=%.2f)",
            k_atm, c_atm, p_atm, atm_iv_used,
        )

    c_atm = max(c_atm, MIN_OPTION_MID)
    p_atm = max(p_atm, MIN_OPTION_MID)

    # ------------------------------------------------------------------
    # 5. Compute returns via horizon repricing (or terminal payoff)
    # ------------------------------------------------------------------
    r_spy = S_next / spot - 1.0

    if use_terminal_payoff:
        # Option expires before/at next rebalance → intrinsic payoff
        C_t1 = np.maximum(S_next - k_atm, 0.0)
        P_t1 = np.maximum(k_atm - S_next, 0.0)
        r_cash_arr = (np.exp(r * T) - 1.0) * np.ones_like(S_next)
    else:
        # Reprice options at t+1 with remaining time T_remain (sticky-strike IV)
        C_t1 = _bs_call_vec(S_next, k_atm, r, T_remain, atm_iv_used)
        P_t1 = _bs_put_vec(S_next, k_atm, r, T_remain, atm_iv_used)
        r_cash_arr = (np.exp(r * T_rebal) - 1.0) * np.ones_like(S_next)

    r_call = C_t1 / c_atm - 1.0
    r_put = P_t1 / p_atm - 1.0

    r_call_raw_mean = float(np.nanmean(r_call))
    r_put_raw_mean = float(np.nanmean(r_put))

    p_lo, p_hi = OPTION_RETURN_WINSORIZE_PCT
    c_lo, c_hi = np.nanpercentile(r_call, p_lo), np.nanpercentile(r_call, p_hi)
    r_call_w = np.clip(r_call, c_lo, c_hi)
    pl, ph = np.nanpercentile(r_put, p_lo), np.nanpercentile(r_put, p_hi)
    r_put_w = np.clip(r_put, pl, ph)

    n = len(r_call)
    n_call_capped = int(np.sum(r_call != r_call_w))
    n_put_capped = int(np.sum(r_put != r_put_w))

    # ------------------------------------------------------------------
    # 6. mu and Sigma
    # ------------------------------------------------------------------
    M = np.column_stack([r_spy, r_call_w, r_put_w, r_cash_arr])
    mu = pd.Series(M.mean(axis=0), index=ASSET_ORDER)
    Sigma = pd.DataFrame(np.cov(M, rowvar=False), index=ASSET_ORDER, columns=ASSET_ORDER)

    r_period = float(np.exp(r * T_rebal) - 1.0) if not use_terminal_payoff else float(np.exp(r * T) - 1.0)
    mu["USDOLLAR"] = r_period
    if use_risk_neutral_mu:
        for a in ASSET_ORDER:
            mu[a] = r_period
    else:
        mu["SPY"] = np.clip(mu["SPY"], mu_clip_spy[0], mu_clip_spy[1])
        mu["SPY_CALL"] = np.clip(mu["SPY_CALL"], mu_clip_option[0], mu_clip_option[1])
        mu["SPY_PUT"] = np.clip(mu["SPY_PUT"], mu_clip_option[0], mu_clip_option[1])

    # ------------------------------------------------------------------
    # 7. Rescale option volatilities (fix: .loc column then .loc row)
    # ------------------------------------------------------------------
    sig_spy = np.sqrt(max(Sigma.loc["SPY", "SPY"], 1e-8))
    for name in ("SPY_CALL", "SPY_PUT"):
        sig_opt = np.sqrt(max(Sigma.loc[name, name], 1e-8))
        if sig_opt > 1e-8:
            scale = (option_vol_mult * sig_spy) / sig_opt
            Sigma.loc[:, name] *= scale   # column
            Sigma.loc[name, :] *= scale   # row

    Sigma = (Sigma + Sigma.T) / 2
    for a in ASSET_ORDER:
        Sigma.loc[a, a] = max(Sigma.loc[a, a], 1e-10)

    # ------------------------------------------------------------------
    # 8. Diagnostics
    # ------------------------------------------------------------------
    dte = (pd.Timestamp(expiry_used) - pd.Timestamp(date_str)).days
    if return_diagnostics:
        diag = {
            "method": bl_method,
            "expiry": expiry_used,
            "dte": dte,
            "T_expiry_years": T,
            "T_rebal_years": T_rebal,
            "T_remain_years": max(T_remain, 0.0),
            "horizon_reprice": not use_terminal_payoff,
            "n_interior_strikes": n_interior,
            "n_calls_after_filter": len(calls),
            "n_puts_after_filter": len(puts),
            "spot": spot,
            "k_atm": k_atm,
            "c_atm": c_atm,
            "p_atm": p_atm,
            "atm_iv": atm_iv_used,
            "used_bs_pricing": used_bs_pricing,
            "n_samples": n,
            "pct_call_winsorized": 100.0 * n_call_capped / n,
            "pct_put_winsorized": 100.0 * n_put_capped / n,
            "r_call_mean_raw": r_call_raw_mean,
            "r_put_mean_raw": r_put_raw_mean,
            "r_call_mean_after": float(mu["SPY_CALL"]),
            "r_put_mean_after": float(mu["SPY_PUT"]),
        }
        return mu, Sigma, diag
    return mu, Sigma


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mu, Sigma, diag = compute_rnd_forecasts(n_samples=5000, return_diagnostics=True)

    print("\nRND mu:\n", mu)
    print("\nRND Sigma:\n", Sigma)

    print("\n--- Diagnostics ---")
    print("  method:", diag["method"])
    print("  expiry: %s  DTE=%d" % (diag["expiry"], diag["dte"]))
    print("  T_expiry=%.4f yr  T_rebal=%.4f yr  T_remain=%.4f yr  horizon_reprice=%s" % (
        diag["T_expiry_years"], diag["T_rebal_years"],
        diag["T_remain_years"], diag["horizon_reprice"]))
    if diag["method"] == "breeden_litzenberger":
        print("  BL interior strikes:", diag["n_interior_strikes"])
    else:
        print("  (BL failed; used lognormal IV=%.4f fallback)" % diag["atm_iv"])
    print("  calls after filter: %d   puts after filter: %d" % (
        diag["n_calls_after_filter"], diag["n_puts_after_filter"]))
    print("  spot=%.2f  k_atm=%.2f  c_atm=%.4f  p_atm=%.4f  BS=%s" % (
        diag["spot"], diag["k_atm"], diag["c_atm"], diag["p_atm"], diag["used_bs_pricing"]))
    print("  %% winsorized: call %.2f%%  put %.2f%%" % (
        diag["pct_call_winsorized"], diag["pct_put_winsorized"]))
    print("  r_call mean: raw %.6f -> after %.6f" % (
        diag["r_call_mean_raw"], diag["r_call_mean_after"]))
    print("  r_put  mean: raw %.6f -> after %.6f" % (
        diag["r_put_mean_raw"], diag["r_put_mean_after"]))
