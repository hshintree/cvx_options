"""
Scenario generation for the scenario-based optimizer (C+E).

C: Options are *exposures* derived from SPY scenarios, not independent assets.
E: SPY scenarios come from RND/lognormal, option returns are mechanically
   computed via Black-Scholes repricing on each scenario path.

Held-contract semantics: We set k_atm at entry and price that same contract
forward to horizon. The asset is "the contract I bought", not a rolling ATM
(no strike roll at rebalance). So over the week the position can become ITM/OTM.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    MIN_BL_STRIKES,
    MIN_OPTION_MID,
    REBALANCE_DAYS,
    SCENARIO_EQUITY_PREMIUM_ANNUAL,
    SCENARIO_SKEW_BETA,
    SCENARIO_SKEW_THRESHOLD,
    SCENARIO_WINSORIZE_PCT,
    SPY_DAILY_FILE,
)
from data.forecasts import (
    ASSET_ORDER,
    FORCE_LOGNORMAL,
    _RELAXED_MAX_SPREAD,
    _RELAXED_MIN_MID,
    _bs_call_price,
    _bs_call_vec,
    _bs_implied_vol,
    _bs_put_price,
    _bs_put_vec,
    _get_atm_iv,
    _lognormal_sample,
    _latest_chain_date,
    breeden_litzenberger_pdf,
    load_chain_for_expiry,
    sample_terminal_prices,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Generate SPY spot scenarios at rebalance horizon
# ---------------------------------------------------------------------------

def _estimate_physical_drift(
    chain_date: Optional[str] = None,
    risk_free_rate: float = 0.05,
    lookback_days: int = 252,
) -> float:
    """Estimate annualized physical drift mu_p = r + equity_premium from rolling returns.
    
    If SCENARIO_EQUITY_PREMIUM_ANNUAL is set, use r + that constant.
    Otherwise estimate from SPY history (rolling mean of log returns).
    """
    if SCENARIO_EQUITY_PREMIUM_ANNUAL is not None:
        return risk_free_rate + SCENARIO_EQUITY_PREMIUM_ANNUAL
    
    import pandas as pd
    if not SPY_DAILY_FILE.exists():
        return risk_free_rate + 0.04  # fallback: 4% premium
    
    spy = pd.read_parquet(SPY_DAILY_FILE)
    spy.index = pd.to_datetime(spy.index).tz_localize(None) if spy.index.tz else pd.to_datetime(spy.index)
    
    date_str = chain_date or _latest_chain_date()
    if date_str:
        ref_date = pd.Timestamp(date_str)
        spy = spy[spy.index <= ref_date]
    
    if len(spy) < 20:
        return risk_free_rate + 0.04
    
    spy = spy.sort_index()
    spy["log_return"] = np.log(spy["close"] / spy["close"].shift(1))
    recent = spy["log_return"].iloc[-lookback_days:].dropna()
    
    if len(recent) < 20:
        return risk_free_rate + 0.04
    
    mu_daily = float(recent.mean())
    mu_annual = mu_daily * 252.0
    return max(risk_free_rate + 0.02, mu_annual)  # floor: at least 2% premium


def generate_spy_scenarios(
    chain_date: Optional[str] = None,
    spot: Optional[float] = None,
    n_samples: int = 10_000,
    method: str = "auto",
    risk_free_rate: float = 0.05,
    rebalance_days: Optional[int] = REBALANCE_DAYS,
) -> Tuple[np.ndarray, Dict]:
    """
    Sample S_next (spot at rebalance horizon) under PHYSICAL measure (P).
    
    SPY scenarios use physical drift (r + equity premium), not risk-neutral (Q).
    Option pricing at horizon still uses Q (BS with r) conditional on realized S_next.

    Parameters
    ----------
    method : "auto" (try BL, fallback lognormal), "lognormal", "bl"
    rebalance_days : holding period; None = hold to expiry

    Returns
    -------
    S_next : array of shape (n_samples,)
    meta   : dict with keys: method, expiry, dte, T, T_rebal, T_remain,
             spot, iv, n_calls, n_puts, terminal_payoff, mu_p_annual
    """
    import pandas as pd

    calls, puts, expiry_used = load_chain_for_expiry(chain_date=chain_date)
    if len(calls) < MIN_BL_STRIKES:
        calls, puts, expiry_used = load_chain_for_expiry(
            chain_date=chain_date,
            min_mid=_RELAXED_MIN_MID,
            max_spread=_RELAXED_MAX_SPREAD,
        )

    if spot is None:
        if SPY_DAILY_FILE.exists():
            spy = pd.read_parquet(SPY_DAILY_FILE)
            spot = float(spy["close"].iloc[-1])
        else:
            spot = 500.0

    r = risk_free_rate
    mu_p_annual = _estimate_physical_drift(chain_date, r)
    date_str = chain_date or _latest_chain_date()
    T = max((pd.Timestamp(expiry_used) - pd.Timestamp(date_str)).days / 365.0,
            1 / 365.0)

    if rebalance_days is None:
        T_rebal = T
        terminal_payoff = True
    else:
        T_rebal = rebalance_days / 365.0
        terminal_payoff = (T - T_rebal) <= 1e-6
    T_remain = max(T - T_rebal, 0.0)
    T_sample = T if terminal_payoff else T_rebal

    # ATM IV (always needed for lognormal fallback; still uses r for Q pricing)
    atm_iv = _get_atm_iv(calls, puts, spot, r, T)

    # Attempt BL or lognormal
    used_method = "lognormal"
    force_ln = FORCE_LOGNORMAL or method == "lognormal"

    if not force_ln and method in ("auto", "bl"):
        strikes, pdf_q = breeden_litzenberger_pdf(
            calls, r, T, smooth=True, puts=puts, spot=spot,
        )
        if len(strikes) >= MIN_BL_STRIKES:
            if strikes.min() <= spot * 1.02 and strikes.max() >= spot * 0.98:
                # Tilt Q→P: reweight by exp(theta * log_return) to match mu_p
                log_returns = np.log(strikes / spot)
                theta = (mu_p_annual - r) / (atm_iv**2 + 1e-6)  # Esscher tilt parameter
                weights = np.exp(theta * log_returns)
                pdf_p = pdf_q * weights
                # Normalize: ensure PDF integrates to 1 (trapezoidal integration)
                integral = np.trapz(pdf_p, strikes)
                if integral > 1e-12:
                    pdf_p = pdf_p / integral
                else:
                    pdf_p = pdf_q  # fallback if tilt fails
                
                S_T = sample_terminal_prices(strikes, pdf_p, n_samples)
                mean_ST = float(np.mean(S_T))
                tol = max(0.03, min(0.15, 0.4 * np.sqrt(T)))
                if abs(mean_ST / spot - np.exp((mu_p_annual - 0.5*atm_iv**2)*T)) < tol * spot:
                    used_method = "breeden_litzenberger"
                    if terminal_payoff:
                        S_next = S_T
                    else:
                        shrink = np.sqrt(T_rebal / T)
                        S_next = spot * np.exp(shrink * np.log(S_T / spot))

    if used_method == "lognormal":
        S_next = _lognormal_sample(spot, mu_p_annual, T_sample, atm_iv, n_samples)

    assert S_next.shape == (n_samples,), f"Bad S_next shape: {S_next.shape}"
    assert not np.any(np.isnan(S_next)), "NaN in S_next"

    dte = (pd.Timestamp(expiry_used) - pd.Timestamp(date_str)).days
    meta = {
        "method": used_method,
        "expiry": expiry_used,
        "dte": dte,
        "T": T,
        "T_rebal": T_rebal,
        "T_remain": T_remain,
        "spot": spot,
        "iv": atm_iv,
        "mu_p_annual": mu_p_annual,
        "n_calls": len(calls),
        "n_puts": len(puts),
        "terminal_payoff": terminal_payoff,
    }
    return S_next, meta


# ---------------------------------------------------------------------------
# 2. Compute per-scenario returns for all 4 assets
# ---------------------------------------------------------------------------

def scenario_returns(
    S_next: np.ndarray,
    spot: float,
    k_atm: float,
    c0: float,
    p0: float,
    r: float,
    T_rebal: float,
    T_remain: float,
    iv_call: float,
    iv_put: float,
    terminal_payoff: bool = False,
    winsorize_pct: Optional[Tuple[float, float]] = None,
    skew_beta: float = 0.0,
    half_spread_call: float = 0.0,
    half_spread_put: float = 0.0,
) -> np.ndarray:
    """
    Compute (n_samples, 4) return matrix [SPY, SPY_CALL, SPY_PUT, USDOLLAR].

    Entry prices c0, p0 should be at ask (buy at ask). Exit is modeled as
    BS_fair - half_spread (sell at bid). Do NOT cap the right tail of
    option returns (protection payoff); keep SCENARIO_WINSORIZE_PCT=None.

    Parameters
    ----------
    iv_call, iv_put : per-strike implied vol at K_atm.
    skew_beta       : vol points per 100%% down move; puts get full beta, calls 0.5*beta.
    half_spread_*   : (ask - bid)/2 at entry, applied as exit haircut to BS value.
    """
    n = len(S_next)

    r_spy = S_next / spot - 1.0

    if terminal_payoff or T_remain <= 1e-6:
        C1 = np.maximum(np.maximum(S_next - k_atm, 0.0) - half_spread_call, 0.0)
        P1 = np.maximum(np.maximum(k_atm - S_next, 0.0) - half_spread_put, 0.0)
    else:
        if skew_beta > 0:
            down_move = np.maximum(0.0, (spot - S_next) / spot)
            # Only apply skew bump when down_move exceeds threshold (not constantly taxing calls)
            excess_down = np.maximum(0.0, down_move - SCENARIO_SKEW_THRESHOLD)
            bump = skew_beta * excess_down
            sigma_c = iv_call + 0.25 * bump  # calls get smaller bump
            sigma_p = iv_put + bump  # puts get full bump
        else:
            sigma_c = iv_call
            sigma_p = iv_put
        C1 = np.maximum(_bs_call_vec(S_next, k_atm, r, T_remain, sigma_c) - half_spread_call, 0.0)
        P1 = np.maximum(_bs_put_vec(S_next, k_atm, r, T_remain, sigma_p) - half_spread_put, 0.0)

    r_call = C1 / c0 - 1.0
    r_put = P1 / p0 - 1.0
    r_cash = np.full(n, np.exp(r * T_rebal) - 1.0)

    # Do NOT cap the right tail (puts/calls paying in crash) or protection is removed.
    if winsorize_pct is not None:
        plo, phi = winsorize_pct
        for arr in (r_call, r_put):
            lo, hi = np.nanpercentile(arr, plo), np.nanpercentile(arr, phi)
            np.clip(arr, lo, hi, out=arr)

    R = np.column_stack([r_spy, r_call, r_put, r_cash])

    assert R.shape == (n, 4), f"Bad R shape: {R.shape}"
    assert not np.any(np.isnan(R)), "NaN in scenario returns"
    return R


# ---------------------------------------------------------------------------
# 3. End-to-end: build full scenario matrix from a chain snapshot
# ---------------------------------------------------------------------------

def build_scenario_matrix(
    chain_date: Optional[str] = None,
    spot: Optional[float] = None,
    n_samples: int = 10_000,
    method: str = "auto",
    risk_free_rate: float = 0.05,
    rebalance_days: Optional[int] = REBALANCE_DAYS,
) -> Tuple[np.ndarray, Dict]:
    """
    Build the (n_samples, 4) return matrix R and metadata dict.

    R[:,0] = SPY returns, R[:,1] = call returns, R[:,2] = put returns,
    R[:,3] = cash returns.

    The meta dict contains everything needed by the scenario optimizer:
    iv, dte, k_atm, c0, p0, pricing_mode, spot, etc.
    """
    import pandas as pd

    S_next, meta = generate_spy_scenarios(
        chain_date=chain_date,
        spot=spot,
        n_samples=n_samples,
        method=method,
        risk_free_rate=risk_free_rate,
        rebalance_days=rebalance_days,
    )

    sp = meta["spot"]
    iv = meta["iv"]
    T = meta["T"]
    T_rebal = meta["T_rebal"]
    T_remain = meta["T_remain"]
    terminal_payoff = meta["terminal_payoff"]
    r = risk_free_rate

    # ATM strike and entry prices — same logic as compute_rnd_forecasts
    calls, puts, _ = load_chain_for_expiry(chain_date=chain_date)
    if len(calls) < MIN_BL_STRIKES:
        calls, puts, _ = load_chain_for_expiry(
            chain_date=chain_date,
            min_mid=_RELAXED_MIN_MID,
            max_spread=_RELAXED_MAX_SPREAD,
        )

    k_atm = sp
    c0 = MIN_OPTION_MID
    p0 = MIN_OPTION_MID
    used_bs = False

    if len(calls) > 0:
        nearest_k = float(
            calls.iloc[(calls["strike"] - sp).abs().argmin()]["strike"]
        )
        if abs(nearest_k / sp - 1.0) <= 0.02:
            k_atm = nearest_k
            c_row = calls.loc[calls["strike"] == k_atm]
            if len(c_row) > 0:
                c0 = float(c_row["mid"].iloc[0])
            if len(puts) > 0:
                p_row = puts[puts["strike"] == k_atm]
                if len(p_row) > 0:
                    p0 = float(p_row["mid"].iloc[0])
                else:
                    p0 = float(
                        puts.iloc[(puts["strike"] - k_atm).abs().argmin()]["mid"]
                    )

    if c0 > 100:
        c0 /= 100.0
    if p0 > 100:
        p0 /= 100.0

    if c0 <= MIN_OPTION_MID or p0 <= MIN_OPTION_MID or abs(k_atm / sp - 1) > 0.02:
        used_bs = True
        k_atm = round(sp)
        c0 = max(_bs_call_price(sp, k_atm, r, T, iv), MIN_OPTION_MID)
        p0 = max(_bs_put_price(sp, k_atm, r, T, iv), MIN_OPTION_MID)

    c0 = max(c0, MIN_OPTION_MID)
    p0 = max(p0, MIN_OPTION_MID)

    # ---- Per-strike IV (use chain IV at K_atm for each option type) ----
    import pandas as _pd
    iv_call = iv
    iv_put = iv
    if len(calls) > 0:
        c_idx = int((calls["strike"] - k_atm).abs().values.argmin())
        c_atm_row = calls.iloc[c_idx]
        if "impl_vol" in calls.columns and _pd.notna(c_atm_row.get("impl_vol")):
            _iv = float(c_atm_row["impl_vol"])
            if 0.01 < _iv < 3.0:
                iv_call = _iv
        if iv_call == iv and "mid" in calls.columns:
            _iv = _bs_implied_vol(float(c_atm_row["mid"]), sp, k_atm, r, T, is_call=True)
            if _iv is not None:
                iv_call = _iv
    if len(puts) > 0:
        p_idx = int((puts["strike"] - k_atm).abs().values.argmin())
        p_atm_row = puts.iloc[p_idx]
        if "impl_vol" in puts.columns and _pd.notna(p_atm_row.get("impl_vol")):
            _iv = float(p_atm_row["impl_vol"])
            if 0.01 < _iv < 3.0:
                iv_put = _iv
        if iv_put == iv and "mid" in puts.columns:
            _iv = _bs_implied_vol(float(p_atm_row["mid"]), sp, k_atm, r, T, is_call=False)
            if _iv is not None:
                iv_put = _iv

    # ---- Bid/ask: entry at ask (buy), exit at BS - half_spread (sell at bid) ----
    half_spread_c = 0.0
    half_spread_p = 0.0
    if len(calls) > 0 and "ask" in calls.columns and "bid" in calls.columns:
        _ask = _pd.to_numeric(c_atm_row.get("ask"), errors="coerce")
        _bid = _pd.to_numeric(c_atm_row.get("bid"), errors="coerce")
        if _pd.notna(_ask) and _pd.notna(_bid) and _ask > 0:
            _ask, _bid = float(_ask), float(_bid)
            if _ask > 100:
                _ask, _bid = _ask / 100.0, _bid / 100.0
            c0 = max(_ask, MIN_OPTION_MID)
            half_spread_c = max(0.0, (_ask - _bid) / 2.0)
    if len(puts) > 0 and "ask" in puts.columns and "bid" in puts.columns:
        _ask = _pd.to_numeric(p_atm_row.get("ask"), errors="coerce")
        _bid = _pd.to_numeric(p_atm_row.get("bid"), errors="coerce")
        if _pd.notna(_ask) and _pd.notna(_bid) and _ask > 0:
            _ask, _bid = float(_ask), float(_bid)
            if _ask > 100:
                _ask, _bid = _ask / 100.0, _bid / 100.0
            p0 = max(_ask, MIN_OPTION_MID)
            half_spread_p = max(0.0, (_ask - _bid) / 2.0)

    # Held-contract semantics: k_atm is fixed at entry; we price that same contract forward.
    R = scenario_returns(
        S_next, sp, k_atm, c0, p0, r, T_rebal, T_remain,
        iv_call, iv_put, terminal_payoff,
        winsorize_pct=SCENARIO_WINSORIZE_PCT,
        skew_beta=SCENARIO_SKEW_BETA,
        half_spread_call=half_spread_c,
        half_spread_put=half_spread_p,
    )

    dte = meta.get("dte", 0)
    mu_p_annual = meta.get("mu_p_annual", r)
    spread_c_pct = (2 * half_spread_c / c0 * 100) if c0 > 0 else 0.0
    spread_p_pct = (2 * half_spread_p / p0 * 100) if p0 > 0 else 0.0

    # Diagnostic: verify we're sampling under P (not Q)
    r_spy = R[:, 0]
    r_cash = R[:, 3]  # [SPY, CALL, PUT, CASH]
    E_r_spy = float(np.mean(r_spy))
    E_r_cash = float(np.mean(r_cash))
    cvar_alpha = 0.95
    losses_spy = -r_spy
    var_level = np.percentile(losses_spy, cvar_alpha * 100)
    cvar_spy = float(np.mean(losses_spy[losses_spy >= var_level]))
    
    # Protection check: % of scenarios where put return > 0 when SPY < -5%
    crash_mask = r_spy < -0.05
    r_put = R[:, 2]
    put_protection_pct = float(np.mean((r_put > 0) & crash_mask) * 100) if crash_mask.any() else 0.0

    meta.update({
        "k_atm": k_atm,
        "c0": c0,
        "p0": p0,
        "iv_call": iv_call,
        "iv_put": iv_put,
        "skew_beta": SCENARIO_SKEW_BETA,
        "half_spread_call": half_spread_c,
        "half_spread_put": half_spread_p,
        "used_bs_pricing": used_bs,
        "n_scenarios": n_samples,
        "pricing_mode": "terminal" if terminal_payoff else "horizon_reprice",
        "E_r_spy": E_r_spy,
        "E_r_cash": E_r_cash,
        "cvar_spy": cvar_spy,
        "put_protection_pct": put_protection_pct,
    })

    logger.info(
        "Scenarios: %d paths, method=%s | spot=%.2f k_atm=%.0f DTE=%d mu_p=%.2f%% | "
        "iv_call=%.2f iv_put=%.2f skew=%.2f | spread@K c=%.1f%% p=%.1f%% | "
        "c0=%.2f p0=%.2f mode=%s",
        n_samples, meta["method"], sp, k_atm, dte, (mu_p_annual - r) * 100,
        iv_call, iv_put, SCENARIO_SKEW_BETA,
        spread_c_pct, spread_p_pct,
        c0, p0, meta["pricing_mode"],
    )
    logger.info(
        "  Diagnostics: E[r_spy]=%.4f (vs cash %.4f) | CVaR_95(r_spy)=%.4f | "
        "Put protection: %.1f%% of -5%% crash scenarios",
        E_r_spy, E_r_cash, cvar_spy, put_protection_pct,
    )

    return R, meta


# ---------------------------------------------------------------------------
# CLI — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    R, meta = build_scenario_matrix(n_samples=5000)

    print(f"\nScenario matrix R: shape={R.shape}")
    print(f"  Method:  {meta['method']}")
    print(f"  Expiry:  {meta['expiry']}  DTE={meta['dte']}")
    print(f"  Spot:    {meta['spot']:.2f}  K_ATM={meta['k_atm']:.0f}")
    print(f"  IV:      {meta['iv']:.2%}")
    print(f"  Pricing: {meta['pricing_mode']}")
    print(f"\n  Asset means: SPY={R[:,0].mean():+.4f}  "
          f"Call={R[:,1].mean():+.4f}  Put={R[:,2].mean():+.4f}  "
          f"Cash={R[:,3].mean():+.4f}")
    print(f"  Asset stds:  SPY={R[:,0].std():.4f}  "
          f"Call={R[:,1].std():.4f}  Put={R[:,2].std():.4f}  "
          f"Cash={R[:,3].std():.6f}")
    print(f"  Call range:  [{R[:,1].min():+.2f}, {R[:,1].max():+.2f}]")
    print(f"  Put  range:  [{R[:,2].min():+.2f}, {R[:,2].max():+.2f}]")
