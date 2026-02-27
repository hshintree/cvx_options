"""
Markowitz++ backtest: SPY, ATM call, ATM put, cash.

Forecast approach:
  Sigma  = blend of IEWMA (historical, from Johansson et al. 2023) and
           per-period RND covariance from option chain (forward-looking)
  mu     = rolling historical mean of realized returns (physical measure)

Optimizer:
  Worst-case robust Markowitz (from DeMiguel et al., MVO review):
    - Return penalty:  subtract ρ per asset weight  (hedges μ estimation error)
    - Covariance boost: Σ_wc = Σ + κ·diag(Σ)  (hedges Σ estimation error)
  Both ρ and κ are tunable (default from config).

Realized returns use ACTUAL option prices from chain snapshots:
  - Option entry price: BS-priced ATM call/put using chain IV on entry date
  - Option exit:        BS repriced at next rebalance (sticky-strike IV)
  - SPY return:         real close-to-close over the period
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    COV_UNCERTAINTY,
    CVAR_ALPHA,
    CVAR_LAMBDA,
    CVAR_LAMBDA_ABS,
    CVAR_LAMBDA_REL,
    EQUITY_PREMIUM_ANNUAL_ANCHOR,
    IEWMA_HALFLIFE_PAIRS,
    IEWMA_LOOKBACK,
    MIN_BL_STRIKES,
    MU_CLIP_EQUITY,
    MU_SHRINKAGE_EQUITY,
    MU_UNCERTAINTY_ANNUAL,
    MU_UNCERTAINTY_MULTIPLIERS,
    OPTIMIZER_MODE,
    OPTION_CHAINS_DIR,
    PRINT_DIAGNOSTICS,
    PRINT_EVERY,
    PRINT_START_PERIOD,
    PROCESSED_DIR,
    RND_BLEND_WEIGHT,
    CMIEWMA_TEMPERATURE,
    REBALANCE_DAYS as CONFIG_REBALANCE_DAYS,
    SCENARIO_HEDGE_FLOOR,
    SCENARIO_MAX_OPTION_WEIGHT as SCENARIO_MAX_OPT,
    SCENARIO_MIN_CASH_WEIGHT as SCENARIO_MIN_CASH,
    SCENARIO_N_SAMPLES,
    SCENARIO_PUT_SPREAD_WIDTH,
    SCENARIO_TACTICAL_IV_LOOKBACK,
    SCENARIO_TACTICAL_IV_PCTILE,
    SCENARIO_TACTICAL_MOMENTUM_LOOKBACK,
    SCENARIO_TACTICAL_MOMENTUM_THRESHOLD,
    SCENARIO_TACTICAL_PUTS,
    SIGMA_IEWMA_WEIGHT,
    SPY_DAILY_FILE,
)
from data.covariance import CMIEWMAPredictor
from data import forecasts as _forecasts_mod
from data.derivatives import build_put_spread, price_put_spread
from data.forecasts import (
    ASSET_ORDER,
    _bs_call_price,
    compute_rnd_forecasts,
)
from data.scenarios import build_scenario_matrix, build_multi_asset_scenario_matrix
from opt.scenario_opt import solve_scenario

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Backtest parameters  (tune these in demo.ipynb)
# -----------------------------------------------------------------------
REBAL_DAYS = CONFIG_REBALANCE_DAYS  # true rebalance horizon for Sharpe / start_date
GAMMA = 1.0                    # risk-aversion (1.0 for this test; was 5.0)
MAX_TURNOVER = 0.25            # per-period turnover cap
TCOST_RATE = 0.0005             # 10 bps each way
INITIAL_VALUE = 1.0
ROLLING_WINDOW = 26            # ~6 months of biweekly periods
MIN_PERIODS_FOR_ROLL = 8
MAX_OPTION_WEIGHT = 0.05       # per-sleeve (Markowitz); scenario uses SCENARIO_MAX_OPTION_WEIGHT
MAX_SPY_WEIGHT = 1.0           # max weight in SPY equity sleeve
MIN_CASH_WEIGHT = 0.0
MU_SHRINKAGE = 0.5
MAX_PORT_VOL = 0.20            # hard cap: portfolio std per period (was 0.12; 0.20 for test)

# Robust optimization parameters  (patched from demo.ipynb)
# rho_period = MU_UNCERTAINTY_ANNUAL * (CONFIG_REBALANCE_DAYS/252); rho[-1]=0
ROBUST_COV_UNCERTAINTY = COV_UNCERTAINTY     # κ: covariance diagonal boost
IEWMA_WEIGHT = SIGMA_IEWMA_WEIGHT           # blend weight for IEWMA vs RND Sigma

# Scenario optimizer parameters  (patched from demo.ipynb)
OPT_MODE = OPTIMIZER_MODE                    # "markowitz" | "scenario"
SCENARIO_SAMPLES = SCENARIO_N_SAMPLES
CVAR_A = CVAR_ALPHA
CVAR_L = CVAR_LAMBDA
CVAR_L_REL = CVAR_LAMBDA_REL
CVAR_L_ABS = CVAR_LAMBDA_ABS


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _available_chain_dates(underlying: str = "SPY", chain_dir: Optional[Path] = None) -> list[str]:
    """Return sorted list of dates (YYYY-MM-DD) that have chain parquets.
    SPY: flat OPTION_CHAINS_DIR. Others: OPTION_CHAINS_DIR/{underlying}/.
    If chain_dir is provided (e.g. from config.chain_dir(sym)), use that.
    """
    if chain_dir is None:
        chain_dir = OPTION_CHAINS_DIR if underlying == "SPY" else OPTION_CHAINS_DIR / underlying
    if not chain_dir.exists():
        return []
    dates = set()
    for f in chain_dir.glob("calls_*.parquet"):
        d = f.stem.replace("calls_", "")
        if d and not d.startswith("_"):
            put_f = chain_dir / f"puts_{d}.parquet"
            if put_f.exists():
                try:
                    pd.read_parquet(f)
                    dates.add(d)
                except Exception:
                    pass
    return sorted(dates)


def _joint_chain_dates(underlyings: list, chain_dir_fn) -> list[str]:
    """Return sorted list of dates where every underlying has chain data."""
    if not underlyings:
        return []
    sets = []
    for sym in underlyings:
        d = chain_dir_fn(sym)
        dates = set()
        if not d.exists():
            return []
        for f in d.glob("calls_*.parquet"):
            day = f.stem.replace("calls_", "")
            if day and not day.startswith("_"):
                put_f = d / f"puts_{day}.parquet"
                if put_f.exists():
                    try:
                        pd.read_parquet(f)
                        dates.add(day)
                    except Exception:
                        pass
        sets.append(dates)
    joint = sets[0].copy()
    for s in sets[1:]:
        joint &= s
    return sorted(joint)


def _shrink_mu(mu_roll: np.ndarray, r_period: float) -> np.ndarray:
    prior = np.full_like(mu_roll, r_period)
    return (1 - MU_SHRINKAGE) * mu_roll + MU_SHRINKAGE * prior


def _build_rho(
    n_assets: int,
    rebal_days: float,
    annual_rho: float = MU_UNCERTAINTY_ANNUAL,
    multipliers: tuple = MU_UNCERTAINTY_MULTIPLIERS,
) -> np.ndarray:
    """Build per-asset rho: rho_period = annual_rho * (rebal_days/252); cash = 0."""
    rho_period = annual_rho * (rebal_days / 252.0)
    n_sleeves = (n_assets - 1) // 3
    rho = np.zeros(n_assets)
    for i in range(n_sleeves):
        for j, mult in enumerate(multipliers):
            rho[3 * i + j] = rho_period * mult
    # rho[-1] = 0 already
    return rho


def _solve_markowitz(
    mu_arr: np.ndarray,
    Sigma_arr: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = GAMMA,
    max_port_vol: float = MAX_PORT_VOL,
    rho: Optional[np.ndarray] = None,
    cov_uncertainty: float = ROBUST_COV_UNCERTAINTY,
    max_equity_weight: float = MAX_SPY_WEIGHT,
    max_option_weight: float = MAX_OPTION_WEIGHT,
    max_put_weight: Optional[float] = None,
    min_cash_weight: float = MIN_CASH_WEIGHT,
) -> np.ndarray:
    """Worst-case robust Markowitz optimizer.

    Supports n assets in [eq, call, put, ..., cash] layout (4 or 3K+1).
    rho: per-asset return uncertainty (rho[-1]=0). If None, uses zero rho.
    max_put_weight allows dynamic hedge sizing to flow through to Markowitz.
    """
    n = len(mu_arr)
    n_sleeves = (n - 1) // 3
    w = cp.Variable(n)

    if rho is not None:
        rho = np.asarray(rho, dtype=float).ravel()
        rho = np.broadcast_to(rho, n).copy()
        rho[-1] = 0.0
    else:
        rho = np.zeros(n)
    mu_wc = mu_arr - rho
    ret = mu_wc @ w

    Sigma_wc = Sigma_arr.copy()
    if cov_uncertainty > 0:
        Sigma_wc += cov_uncertainty * np.diag(np.diag(Sigma_arr))
    Sigma_wc = (Sigma_wc + Sigma_wc.T) / 2.0

    risk = cp.quad_form(w, Sigma_wc, assume_PSD=True)
    turnover = cp.norm(w - w_prev, 1)
    tcost = TCOST_RATE * turnover

    objective = cp.Maximize(ret - gamma / 2 * risk - tcost)

    _max_put = max_put_weight if max_put_weight is not None else max_option_weight
    w_upper = np.zeros(n)
    for i in range(n_sleeves):
        w_upper[3 * i] = max_equity_weight
        w_upper[3 * i + 1] = max_option_weight
        w_upper[3 * i + 2] = _max_put
    w_upper[-1] = 1.0

    constraints = [
        w >= 0,
        w <= w_upper,
        cp.sum(w) == 1,
        w[-1] >= min_cash_weight,
        turnover <= MAX_TURNOVER,
    ]
    for i in range(n_sleeves):
        constraints.append(w[3 * i + 1] + w[3 * i + 2] <= max_option_weight)

    if max_port_vol is not None and max_port_vol > 0:
        L = np.linalg.cholesky(Sigma_wc + np.eye(n) * 1e-8)
        constraints.append(cp.norm(L.T @ w, 2) <= max_port_vol)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        w_opt = np.maximum(w.value, 0.0)
        return w_opt / w_opt.sum()
    return w_prev.copy()


def _regularize_sigma(S: np.ndarray) -> np.ndarray:
    S = (S + S.T) / 2.0
    eig = np.linalg.eigvalsh(S)
    if eig.min() < 1e-8 or np.any(np.isnan(S)):
        S += np.eye(S.shape[0]) * 1e-6
    return S


def _compute_dynamic_hedge(
    R_sc: np.ndarray,
    iv_history: list,
    eq_returns_history: list,
    current_iv: float,
    max_opt: float,
    hedge_floor: float,
    hedge_mode: str,
    cvar_alpha: float,
    iv_lookback: int,
) -> tuple:
    """
    Compute put hedge weight dynamically from scenario tail risk and IV regime.

    Instead of a binary on/off switch, this produces a *continuous* hedge
    weight between hedge_floor and max_opt.  The weight scales with:
      1. Equity tail risk (CVaR of equity scenarios): higher tail → more hedge
      2. IV regime (percentile vs. history): cheaper IV → more hedge
      3. Momentum signal (recent realized equity returns): negative → more hedge
    All three are combined into a hedge_score ∈ [0, 1] then mapped to weight.

    Returns (max_put_weight, hedge_score, reason_string).
    """
    if hedge_mode == "never":
        return 0.0, 0.0, "hedge_mode=never"
    if hedge_mode == "always":
        return max_opt, 1.0, "hedge_mode=always"

    # --- Signal 1: Equity tail risk from scenario distribution ---
    eq_rets = R_sc[:, 0]
    losses = -eq_rets
    var_level = np.percentile(losses, cvar_alpha * 100)
    cvar_eq = float(np.mean(losses[losses >= var_level]))
    # Scale: CVaR 2% -> 0, CVaR 6% -> tail_score=1 (hedge less in mild regimes)
    tail_score = np.clip((cvar_eq - 0.02) / 0.04, 0.0, 1.0)

    # --- Signal 2: IV regime (cheap = good time to hedge) ---
    iv_score = 0.5
    iv_pctile = None
    if len(iv_history) >= iv_lookback:
        iv_window = iv_history[-iv_lookback:]
        iv_pctile = float(np.sum(np.array(iv_window) < current_iv)) / len(iv_window)
        # Low percentile = cheap IV = higher score
        iv_score = np.clip(1.0 - iv_pctile, 0.0, 1.0)

    # --- Signal 3: Momentum (recent drawdown = hedge more) ---
    mom_score = 0.5
    mom_sum = None
    lookback = min(5, len(eq_returns_history))
    if lookback >= 2:
        recent = eq_returns_history[-lookback:]
        mom_sum = sum(recent)
        # Negative momentum → higher score; 0 → 0.5; -5% → 1.0
        mom_score = np.clip(0.5 - mom_sum / 0.10, 0.0, 1.0)

    # --- Combine with weights: tail_risk 50%, IV 30%, momentum 20% ---
    hedge_score = 0.50 * tail_score + 0.30 * iv_score + 0.20 * mom_score
    hedge_score = float(np.clip(hedge_score, 0.0, 1.0))

    max_put_weight = hedge_floor + (max_opt - hedge_floor) * hedge_score
    max_put_weight = float(np.clip(max_put_weight, hedge_floor, max_opt))

    parts = []
    parts.append(f"tail={tail_score:.2f}(CVaR={cvar_eq:.3f})")
    iv_str = f"{iv_pctile:.2f}" if iv_pctile is not None else "N/A"
    parts.append(f"iv={iv_score:.2f}(pctile={iv_str})")
    mom_str = f"{mom_sum:.3f}" if mom_sum is not None else "N/A"
    parts.append(f"mom={mom_score:.2f}(sum={mom_str})")
    parts.append(f"score={hedge_score:.2f}→w={max_put_weight:.3f}")
    reason = " | ".join(parts)

    return max_put_weight, hedge_score, reason


# -----------------------------------------------------------------------
# Build realized returns aligned to chain dates
# -----------------------------------------------------------------------

def build_realized_returns(
    chain_dates: list[str],
    spy_df: pd.DataFrame,
    r: float = 0.05,
    asset_order: Optional[list] = None,
    underlying: str = "SPY",
) -> pd.DataFrame:
    """
    For each consecutive pair of chain dates, compute realized returns
    using BS horizon repricing (sticky-strike IV).

    Entry: buy ATM options with the DTE chosen by the RND system.
    Exit:  reprice those options at the next chain date with remaining time.
    """
    rows = []
    dates_out = []

    for i in range(len(chain_dates) - 1):
        d0 = chain_dates[i]
        d1 = chain_dates[i + 1]
        ts0 = pd.Timestamp(d0)
        ts1 = pd.Timestamp(d1)

        idx0 = spy_df.index.get_indexer([ts0], method="ffill")[0]
        idx1 = spy_df.index.get_indexer([ts1], method="ffill")[0]
        if idx0 < 0 or idx1 < 0:
            continue

        s0 = float(spy_df.iloc[idx0]["close"])
        s1 = float(spy_df.iloc[idx1]["close"])
        holding_days = (ts1 - ts0).days
        T_hold = max(holding_days / 365.0, 1 / 365.0)

        # Get IV and option DTE from the chain on entry date
        try:
            _, _, diag = compute_rnd_forecasts(
                chain_date=d0, spot=s0, n_samples=1000, return_diagnostics=True,
                underlying=underlying,
            )
            iv = diag["atm_iv"]
            option_dte = diag["dte"]
        except Exception:
            iv = 0.15
            option_dte = 30

        T_option = max(option_dte / 365.0, 1 / 365.0)
        T_remain = max(T_option - T_hold, 0.0)
        k_atm = round(s0)
        put_spread_width = float(SCENARIO_PUT_SPREAD_WIDTH)
        k_put_short, p_spread_entry = build_put_spread(
            s0, k_atm, r, T_option, iv, put_spread_width, min_mid=0.01,
        )
        p_spread_entry = max(p_spread_entry, 0.01)

        # Entry: ATM call
        c0 = max(_bs_call_price(s0, k_atm, r, T_option, iv), 0.01)
        # Exit: call and put spread (unified helper)
        if T_remain > 1 / 365.0:
            c1 = max(_bs_call_price(s1, k_atm, r, T_remain, iv), 0.0)
            spread_value_1 = price_put_spread(s1, k_atm, k_put_short, r, T_remain, iv, intrinsic_if_expired=True)
        else:
            c1 = max(s1 - k_atm, 0.0)
            spread_value_1 = price_put_spread(s1, k_atm, k_put_short, r, 0.0, iv, intrinsic_if_expired=True)

        r_spy = s1 / s0 - 1.0
        r_call = c1 / c0 - 1.0
        r_put = spread_value_1 / p_spread_entry - 1.0
        r_cash = np.exp(r * T_hold) - 1.0

        dates_out.append(ts1)
        rows.append([r_spy, r_call, r_put, r_cash])

    cols = asset_order if asset_order is not None else ASSET_ORDER
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates_out), columns=cols)


def build_realized_returns_multi(
    chain_dates: list,
    price_dfs: dict,
    underlyings: list,
    asset_order: list,
    r: float = 0.05,
) -> pd.DataFrame:
    """
    Build realized returns for multiple underlyings over consecutive chain dates.
    Each row: [r_eq1, r_c1, r_p1, r_eq2, r_c2, r_p2, ..., r_cash].
    """
    rows = []
    dates_out = []
    for i in range(len(chain_dates) - 1):
        d0, d1 = chain_dates[i], chain_dates[i + 1]
        ts0, ts1 = pd.Timestamp(d0), pd.Timestamp(d1)
        row = []
        for sym in underlyings:
            spy_df = price_dfs.get(sym)
            if spy_df is None or spy_df.empty:
                row.extend([np.nan, np.nan, np.nan])
                continue
            idx0 = spy_df.index.get_indexer([ts0], method="ffill")[0]
            idx1 = spy_df.index.get_indexer([ts1], method="ffill")[0]
            if idx0 < 0 or idx1 < 0:
                row.extend([np.nan, np.nan, np.nan])
                continue
            s0 = float(spy_df.iloc[idx0]["close"])
            s1 = float(spy_df.iloc[idx1]["close"])
            holding_days = (ts1 - ts0).days
            T_hold = max(holding_days / 365.0, 1 / 365.0)
            try:
                _, _, diag = compute_rnd_forecasts(
                    chain_date=d0, spot=s0, n_samples=1000, return_diagnostics=True,
                    underlying=sym,
                )
                iv = diag["atm_iv"]
                option_dte = diag["dte"]
            except Exception:
                iv = 0.15
                option_dte = 30
            T_option = max(option_dte / 365.0, 1 / 365.0)
            T_remain = max(T_option - T_hold, 0.0)
            k_atm = round(s0)
            put_spread_width = float(SCENARIO_PUT_SPREAD_WIDTH)
            k_put_short, p_spread_entry = build_put_spread(
                s0, k_atm, r, T_option, iv, put_spread_width, min_mid=0.01,
            )
            p_spread_entry = max(p_spread_entry, 0.01)
            c0 = max(_bs_call_price(s0, k_atm, r, T_option, iv), 0.01)
            if T_remain > 1 / 365.0:
                c1 = max(_bs_call_price(s1, k_atm, r, T_remain, iv), 0.0)
                spread_value_1 = price_put_spread(s1, k_atm, k_put_short, r, T_remain, iv, intrinsic_if_expired=True)
            else:
                c1 = max(s1 - k_atm, 0.0)
                spread_value_1 = price_put_spread(s1, k_atm, k_put_short, r, 0.0, iv, intrinsic_if_expired=True)
            r_eq = s1 / s0 - 1.0
            r_call = c1 / c0 - 1.0
            r_put = spread_value_1 / p_spread_entry - 1.0
            row.extend([r_eq, r_call, r_put])
        T_hold = (ts1 - ts0).days / 365.0 if (ts1 - ts0).days else 1 / 365.0
        r_cash = np.exp(r * max(T_hold, 1 / 365.0)) - 1.0
        row.append(r_cash)
        dates_out.append(ts1)
        rows.append(row)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates_out), columns=asset_order)


# -----------------------------------------------------------------------
# Run backtest
# -----------------------------------------------------------------------

def run_backtest(config=None) -> tuple:
    """
    Run the backtest. If config is a ScenarioBacktestConfig, use its hyperparams.
    """
    # Resolve params from config or module defaults
    if config is not None and hasattr(config, "hyperparams"):
        hp = config.hyperparams
        _n_samples = hp.n_samples
        _cvar_alpha = hp.cvar_alpha
        _cvar_lambda = hp.cvar_lambda
        _cvar_l_rel = getattr(hp, "cvar_lambda_rel", CVAR_L_REL)
        _cvar_l_abs = getattr(hp, "cvar_lambda_abs", CVAR_L_ABS)
        _min_cash = hp.min_cash_weight
        _max_opt = hp.max_option_weight
        _max_spy = hp.max_spy_weight
        _max_turnover = hp.max_turnover
        _tcost = hp.tcost_rate
        # Dynamic hedge params
        _hedge_mode = hp.hedge_mode
        _hedge_floor = hp.hedge_floor
        _hedge_iv_lookback = hp.hedge_iv_lookback
        # Patch config and scenarios (build_scenario_matrix uses these)
        import config as _cfg
        import data.scenarios as _scn
        _cfg.SCENARIO_N_SAMPLES = _n_samples
        _cfg.SCENARIO_PUT_SPREAD_WIDTH = hp.put_spread_width
        _cfg.SCENARIO_SKEW_BETA = hp.skew_beta
        _cfg.SCENARIO_SKEW_THRESHOLD = hp.skew_threshold
        _cfg.SCENARIO_EQUITY_PREMIUM_ANNUAL = hp.equity_premium_annual
        _scn.SCENARIO_PUT_SPREAD_WIDTH = hp.put_spread_width
        _scn.SCENARIO_SKEW_BETA = hp.skew_beta
        _scn.SCENARIO_SKEW_THRESHOLD = hp.skew_threshold
        _scn.SCENARIO_EQUITY_PREMIUM_ANNUAL = hp.equity_premium_annual
    else:
        _n_samples = SCENARIO_SAMPLES
        _cvar_alpha = CVAR_A
        _cvar_lambda = CVAR_L
        _cvar_l_rel = CVAR_L_REL
        _cvar_l_abs = CVAR_L_ABS
        _min_cash = SCENARIO_MIN_CASH
        _max_opt = SCENARIO_MAX_OPT
        _max_spy = MAX_SPY_WEIGHT
        _max_turnover = MAX_TURNOVER
        _tcost = TCOST_RATE
        _hedge_mode = "dynamic"
        _hedge_floor = SCENARIO_HEDGE_FLOOR
        _hedge_iv_lookback = 20

    # Resolve underlyings and data (single- or multi-asset)
    _underlyings = (config.underlyings if config and config.underlyings else ["SPY"]) or ["SPY"]
    _multi = len(_underlyings) > 1
    if _multi:
        _asset_order = config.asset_order
        _forecasts_mod.ASSET_ORDER = _asset_order
        for sym in _underlyings:
            pf = config.price_file(sym)
            if not pf.exists():
                raise FileNotFoundError(
                    f"Price file not found for {sym}: {pf}. "
                    f"Run: python -m data.fetch_alpaca --symbols {sym}"
                )
        price_dfs = {}
        for sym in _underlyings:
            df = pd.read_parquet(config.price_file(sym))
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            price_dfs[sym] = df
        chain_dates = _joint_chain_dates(_underlyings, config.chain_dir)
        logger.info("Multi-asset %s: joint chain dates %d", _underlyings, len(chain_dates))
        if not chain_dates:
            raise FileNotFoundError(
                f"No joint chain dates for {_underlyings}. "
                f"Ensure option_chains/ has data for each symbol on overlapping dates."
            )
        returns = build_realized_returns_multi(
            chain_dates, price_dfs, _underlyings, _asset_order,
            r=getattr(config, "risk_free_rate", 0.05),
        )
        spy_df = price_dfs[_underlyings[0]]
        _underlying = "+".join(_underlyings)
    else:
        _underlying = _underlyings[0]
        _price_file = config.price_file(_underlying) if config else SPY_DAILY_FILE
        if not _price_file.exists():
            raise FileNotFoundError(
                f"Price file not found: {_price_file}. "
                f"For {_underlying}, fetch data to data/raw/{_underlying.lower()}_daily.parquet"
            )
        if _underlying != "SPY":
            _forecasts_mod.ASSET_ORDER = [ _underlying, f"{_underlying}_CALL", f"{_underlying}_PUT", "USDOLLAR" ]
        _asset_order = _forecasts_mod.ASSET_ORDER
        spy_df = pd.read_parquet(_price_file)
        spy_df.index = pd.to_datetime(spy_df.index).tz_localize(None).normalize()
        chain_dates = _available_chain_dates(_underlying)
        logger.info("Available chain dates: %d", len(chain_dates))
        logger.info("Building realized returns from chain dates ...")
        returns = build_realized_returns(
            chain_dates, spy_df,
            asset_order=_asset_order,
            underlying=_underlying,
        )

    # Resolve optimizer mode: config.optimizer_mode overrides global OPT_MODE
    _opt_mode = getattr(config, "optimizer_mode", None) or OPT_MODE
    if _multi and _opt_mode != "scenario":
        raise NotImplementedError("Multi-asset backtest is only supported in scenario (CVaR) optimizer mode.")
    n_periods = len(returns)
    logger.info("Periods: %d  (%s to %s)", n_periods,
                returns.index[0].date(), returns.index[-1].date())

    use_scenario = (_opt_mode == "scenario")
    logger.info("Optimizer mode: %s", _opt_mode)

    # Pre-compute per-period RND forecasts (mu, Sigma) for Markowitz mode
    # and scenario matrices for scenario mode.
    rnd_sigmas = {}
    rnd_mus = {}
    rnd_diags = {}
    scenario_matrices = {}

    logger.info("Computing per-period forecasts ...")
    for d in chain_dates:
        ts = pd.Timestamp(d)
        if _multi:
            spots = {}
            for sym in _underlyings:
                idx = price_dfs[sym].index.get_indexer([ts], method="ffill")[0]
                spots[sym] = float(price_dfs[sym].iloc[idx]["close"]) if idx >= 0 else None
            if any(v is None for v in spots.values()):
                logger.warning("Missing spot for %s on %s; skipping", d, spots)
                continue
        else:
            idx = spy_df.index.get_indexer([ts], method="ffill")[0]
            spot = float(spy_df.iloc[idx]["close"]) if idx >= 0 else 550.0

        if use_scenario:
            try:
                if _multi:
                    R_sc, sc_meta = build_multi_asset_scenario_matrix(
                        _underlyings, d, spots, n_samples=_n_samples,
                        risk_free_rate=getattr(config, "risk_free_rate", 0.05),
                    )
                else:
                    R_sc, sc_meta = build_scenario_matrix(
                        chain_date=d, spot=spot, n_samples=_n_samples,
                        underlying=_underlying,
                    )
                scenario_matrices[d] = R_sc
                rnd_diags[d] = sc_meta
                logger.info("Scenario matrix built for %s: R shape=%s", d, R_sc.shape)
            except Exception as e:
                logger.warning("Scenario build failed for %s: %s", d, e)
        else:
            try:
                # Use a larger MC sample to stabilize both mu and Sigma for Markowitz
                mu_rnd, Sigma_rnd, diag = compute_rnd_forecasts(
                    chain_date=d,
                    spot=spot,
                    n_samples=20_000,
                    return_diagnostics=True,
                    underlying=_underlying if not _multi else _underlyings[0],
                )
                rnd_sigmas[d] = _regularize_sigma(Sigma_rnd.values.astype(float))
                rnd_mus[d] = mu_rnd.values.astype(float)
                rnd_diags[d] = diag
            except Exception as e:
                logger.warning("RND failed for %s: %s", d, e)
            # Also build scenario matrix for dynamic hedge sizing in Markowitz mode
            if _hedge_mode == "dynamic" and not _multi:
                try:
                    R_sc_mw, sc_meta_mw = build_scenario_matrix(
                        chain_date=d, spot=spot, n_samples=min(_n_samples, 5000),
                        underlying=_underlying if not _multi else _underlyings[0],
                    )
                    scenario_matrices[d] = R_sc_mw
                    if d not in rnd_diags:
                        rnd_diags[d] = sc_meta_mw
                    else:
                        rnd_diags[d].update({k: v for k, v in sc_meta_mw.items() if k not in rnd_diags[d]})
                except Exception as e:
                    logger.debug("Scenario matrix for Markowitz hedge failed for %s: %s", d, e)

    n_forecasts = len(scenario_matrices) if use_scenario else len(rnd_sigmas)
    logger.info("Forecasts computed for %d / %d dates", n_forecasts, len(chain_dates))
    if use_scenario:
        logger.info("Scenario matrices stored: %d", len(scenario_matrices))

    # Default fallback Sigma (Markowitz mode only)
    fallback_Sigma = _regularize_sigma(
        np.diag([0.0009, 0.36, 0.36, 1e-10])
    )

    # ---------------------------------------------------------------
    # Initialize IEWMA covariance predictor (Markowitz mode only)
    # ---------------------------------------------------------------
    n_assets = len(_asset_order)
    iewma_predictor = CMIEWMAPredictor(
        n_assets,
        halflife_pairs=IEWMA_HALFLIFE_PAIRS,
        lookback=IEWMA_LOOKBACK,
        temperature=CMIEWMA_TEMPERATURE,
    )

    weights_history = np.zeros((n_periods + 1, n_assets))
    if n_assets == 4:
        weights_history[0] = [0.60, 0.0, 0.0, 0.40]
    else:
        n_sleeves = (n_assets - 1) // 3
        for i in range(n_sleeves):
            weights_history[0][3 * i] = 0.60 / n_sleeves
        weights_history[0][-1] = 0.40
    portfolio_values = np.ones(n_periods + 1) * INITIAL_VALUE

    # Per-period holding days from chain_dates (used for r_period, equity prior, Sharpe)
    holding_days_list = []
    for t in range(n_periods):
        if t + 1 < len(chain_dates):
            holding_days_list.append((pd.Timestamp(chain_dates[t + 1]) - pd.Timestamp(chain_dates[t])).days)
        else:
            holding_days_list.append(holding_days_list[-1] if holding_days_list else CONFIG_REBALANCE_DAYS)
    avg_holding_days = float(np.mean(holding_days_list))

    r_period = float(returns["USDOLLAR"].iloc[0])  # fallback for first period
    mu_history = []
    method_history = []
    sigma_source_history = []
    scenario_period_diag = []
    
    # Track IV and equity returns for tactical puts
    iv_history = []
    spy_returns_history = []
    _eq_col = returns.columns[0]
    a_rnd_history = []  # per-period dynamic RND blend weight (Markowitz)

    for t in range(n_periods):
        w_prev = weights_history[t]
        entry_date = chain_dates[t]
        holding_days_t = holding_days_list[t] if t < len(holding_days_list) else avg_holding_days
        r_period_t = float(returns["USDOLLAR"].iloc[t])

        if use_scenario:
            # ---- Scenario mode ----
            # SPY mu from P-measure scenarios (equity premium).
            # Option mu = 0: they're fairly priced; CVaR alone drives allocation.
            R_sc = scenario_matrices.get(entry_date)
            if R_sc is not None:
                n_a = R_sc.shape[1]
                if n_a == 4:
                    mu_phys = np.array([
                        float(np.mean(R_sc[:, 0])), 0.0, 0.0, r_period_t
                    ])
                else:
                    mu_phys = np.zeros(n_a)
                    for i in range((n_a - 1) // 3):
                        mu_phys[3 * i] = float(np.mean(R_sc[:, 3 * i]))
                    mu_phys[-1] = r_period_t
                
                # ---- Dynamic hedge sizing: uses scenario tail risk to scale put allocation ----
                sc_meta = rnd_diags.get(entry_date, {})
                if _multi:
                    current_iv = sc_meta.get("per_underlying", {}).get(_underlyings[0], {}).get("iv_put") or \
                                 sc_meta.get("per_underlying", {}).get(_underlyings[0], {}).get("iv") or 0.15
                else:
                    current_iv = sc_meta.get("iv_put") or sc_meta.get("iv") or 0.15
                iv_history.append(current_iv)
                spy_ret = float(returns.iloc[t][_eq_col]) if t < len(returns) else 0.0
                spy_returns_history.append(spy_ret)

                max_put_weight, hedge_score, tactical_reason = _compute_dynamic_hedge(
                    R_sc, iv_history, spy_returns_history, current_iv,
                    _max_opt, _hedge_floor, _hedge_mode, _cvar_alpha,
                    _hedge_iv_lookback,
                )
                min_put_weight = 0.40 * max_put_weight if hedge_score >= 0.40 else 0.0

                w_opt, solver_diag = solve_scenario(
                    R_sc, w_prev, mu_phys,
                    cvar_alpha=_cvar_alpha,
                    cvar_lambda=_cvar_lambda,
                    cvar_lambda_rel=_cvar_l_rel,
                    cvar_lambda_abs=_cvar_l_abs,
                    max_option_weight=_max_opt,
                    max_put_weight=max_put_weight,
                    max_spy_weight=_max_spy,
                    min_cash_weight=_min_cash,
                    max_turnover=_max_turnover,
                    tcost_rate=_tcost,
                    use_rel_cvar=True,
                    min_put_weight=min_put_weight,
                    track_eps=-0.003,
                )
                
                # Portfolio CVaR vs SPY CVaR
                port_rets = R_sc @ w_opt
                losses_port = -port_rets
                var_port = np.percentile(losses_port, _cvar_alpha * 100)
                cvar_port = float(np.mean(losses_port[losses_port >= var_port]))
                losses_spy = -R_sc[:, 0]
                var_spy = np.percentile(losses_spy, _cvar_alpha * 100)
                cvar_spy_scenario = float(np.mean(losses_spy[losses_spy >= var_spy]))
                
                scenario_period_diag.append({
                    "date": entry_date,
                    "mu_phys_spy": float(mu_phys[0]),
                    "mu_phys_call": float(mu_phys[1]),
                    "mu_phys_put": float(mu_phys[2]),
                    "mean_scenario_spy": float(np.mean(R_sc[:, 0])),
                    "mean_scenario_port": solver_diag.get("mean_scenario_port"),
                    "bench_mean": solver_diag.get("bench_mean"),
                    "mean_rel_scenario_port": solver_diag.get("mean_rel_scenario_port"),
                    "cvar_model": solver_diag.get("cvar_model"),
                    "cvar_empirical": solver_diag.get("cvar_empirical"),
                    "cvar_relative_model": solver_diag.get("cvar_relative_model"),
                    "cvar_relative_empirical": solver_diag.get("cvar_relative_empirical"),
                    "cvar_abs_model": solver_diag.get("cvar_abs_model"),
                    "cvar_abs_empirical": solver_diag.get("cvar_abs_empirical"),
                    "cvar_portfolio": cvar_port,
                    "cvar_spy": cvar_spy_scenario,
                    "put_protection_pct": sc_meta.get("put_protection_pct"),
                    "expected_ret_phys": solver_diag.get("expected_ret_phys"),
                    "solver_status": solver_diag.get("solver_status"),
                    "tactical_put_weight": max_put_weight,
                    "hedge_score": hedge_score,
                    "min_put_weight": min_put_weight,
                    "tactical_reason": tactical_reason,
                    "current_iv": current_iv,
                    "weights": w_opt.copy(),
                })
            else:
                # No scenarios available: use fallback mu_phys
                mu_phys = np.zeros(n_assets)
                mu_phys[0] = r_period_t
                mu_phys[-1] = r_period_t
                w_opt = w_prev.copy()
                scenario_period_diag.append({
                    "date": entry_date,
                    "mu_phys_spy": float(mu_phys[0]),
                    "mu_phys_call": None,
                    "mu_phys_put": None,
                    "mean_scenario_spy": None,
                    "cvar_model": None,
                    "cvar_empirical": None,
                    "cvar_portfolio": None,
                    "cvar_spy": None,
                    "put_protection_pct": None,
                    "expected_ret_phys": None,
                    "solver_status": "no_scenarios",
                    "weights": w_opt.copy(),
                })

            mu_history.append(mu_phys.copy())
            sc_meta = rnd_diags.get(entry_date, {})
            method_history.append(sc_meta.get("method", "fallback"))
            sigma_source_history.append("scenario")

        else:
            # ---- Markowitz++ mode ----
            Sigma_rnd = rnd_sigmas.get(entry_date, fallback_Sigma)
            Sigma_iewma = iewma_predictor.predict()
            rnd_diag_t = rnd_diags.get(entry_date, {})

            # Dynamic a_rnd_t: BL quality + IEWMA confidence (entropy of expert weights)
            bl_good = (
                rnd_diag_t.get("method") == "breeden_litzenberger"
                and rnd_diag_t.get("n_interior_strikes", 0) >= MIN_BL_STRIKES
            )
            a_rnd_base = 0.7 if bl_good else 0.3
            ew = iewma_predictor.get_last_weights()
            if ew is not None and len(ew) > 1:
                ew = np.asarray(ew).ravel()
                ew = np.maximum(ew, 1e-12)
                entropy = -float(np.sum(ew * np.log(ew)))
                max_entropy = np.log(len(ew))
                confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
                a_rnd_t = 0.5 + (a_rnd_base - 0.5) * float(np.clip(confidence, 0, 1))
            else:
                a_rnd_t = a_rnd_base

            if Sigma_iewma is not None and (1.0 - a_rnd_t) > 0:
                Sigma_iewma = _regularize_sigma(Sigma_iewma)
                Sigma_t = (1.0 - a_rnd_t) * Sigma_iewma + a_rnd_t * Sigma_rnd
                sigma_source_history.append("blend")
            else:
                Sigma_t = Sigma_rnd
                sigma_source_history.append("rnd_only")

            Sigma_t = _regularize_sigma(Sigma_t)

            # ------------------------------------------------------------------
            # mu: take the MC RND sample means directly (SPY, CALL, PUT_SPREAD, CASH)
            # ------------------------------------------------------------------
            mu_rnd = rnd_mus.get(entry_date)
            if mu_rnd is not None and len(mu_rnd) == n_assets:
                mu_t = np.array(mu_rnd, dtype=float)
                # Align cash to the realized period rate to respect actual holding_days_t
                mu_t[-1] = r_period_t
            else:
                # Fallback: legacy rolling + anchor + clip if RND mu is unavailable
                if t >= MIN_PERIODS_FOR_ROLL:
                    lb = max(0, t - ROLLING_WINDOW)
                    mu_roll = returns.iloc[lb:t].mean().values.astype(float)
                    mu_t = _shrink_mu(mu_roll, r_period_t)
                    anchor = EQUITY_PREMIUM_ANNUAL_ANCHOR if EQUITY_PREMIUM_ANNUAL_ANCHOR is not None else 0.04
                    mu_prior_equity = r_period_t + anchor * (holding_days_t / 252.0)
                    eq_indices = [0] if n_assets == 4 else [3 * i for i in range((n_assets - 1) // 3)]
                    for i in eq_indices:
                        mu_t[i] = (1.0 - MU_SHRINKAGE_EQUITY) * mu_roll[i] + MU_SHRINKAGE_EQUITY * mu_prior_equity
                        mu_t[i] = np.clip(mu_t[i], MU_CLIP_EQUITY[0], MU_CLIP_EQUITY[1])
                    for j in range(n_assets - 1):
                        if j not in eq_indices:
                            mu_t[j] = np.clip(mu_t[j], -0.30, 0.30)
                    mu_t[-1] = r_period_t
                else:
                    mu_t = np.zeros(n_assets)
                    mu_t[0] = 0.005
                    mu_t[-1] = r_period_t

            # Dynamic hedge sizing for Markowitz (uses scenario matrix if available)
            R_sc_mw = scenario_matrices.get(entry_date)
            if R_sc_mw is not None:
                sc_meta_mw = rnd_diags.get(entry_date, {})
                current_iv_mw = sc_meta_mw.get("iv_put") or sc_meta_mw.get("iv") or 0.15
                iv_history.append(current_iv_mw)
                spy_ret_mw = float(returns.iloc[t][_eq_col]) if t < len(returns) else 0.0
                spy_returns_history.append(spy_ret_mw)
                max_put_mw, _, _ = _compute_dynamic_hedge(
                    R_sc_mw, iv_history, spy_returns_history, current_iv_mw,
                    _max_opt, _hedge_floor, _hedge_mode, _cvar_alpha,
                    _hedge_iv_lookback,
                )
            else:
                max_put_mw = _max_opt

            a_rnd_history.append(a_rnd_t)
            mu_history.append(mu_t.copy())
            method_history.append(rnd_diag_t.get("method", "fallback"))
            rho_mw = _build_rho(n_assets, holding_days_t)
            w_opt = _solve_markowitz(
                mu_t, Sigma_t, w_prev,
                rho=rho_mw,
                max_equity_weight=_max_spy,
                max_option_weight=_max_opt,
                max_put_weight=max_put_mw,
                min_cash_weight=_min_cash,
            )

        weights_history[t + 1] = w_opt

        # ---- Transparency logging (after cold start, every PRINT_EVERY periods) ----
        _do_print = (
            PRINT_DIAGNOSTICS
            and t >= PRINT_START_PERIOD
            and (t - PRINT_START_PERIOD) % PRINT_EVERY == 0
        )
        if _do_print:
            if use_scenario:
                sp = scenario_period_diag[-1] if scenario_period_diag else {}
                _safe = lambda k: sp.get(k) if sp.get(k) is not None else np.nan  # noqa: E731
                logger.info(
                    "[Scenario] date=%s hedge_score=%.2f min_put=%.3f max_put=%.3f | "
                    "mean(port)=%.4f mean(bench)=%.4f mean(rel)=%.4f | "
                    "cvar_rel=%.4f/%.4f cvar_abs=%.4f/%.4f | "
                    "solver=%s w_opt=%s | %s",
                    entry_date,
                    sp.get("hedge_score", 0) or 0,
                    sp.get("min_put_weight", 0) or 0,
                    sp.get("tactical_put_weight", 0) or 0,
                    _safe("mean_scenario_port"),
                    _safe("bench_mean"),
                    _safe("mean_rel_scenario_port"),
                    _safe("cvar_relative_model"), _safe("cvar_relative_empirical"),
                    _safe("cvar_abs_model"), _safe("cvar_abs_empirical"),
                    sp.get("solver_status", "N/A"),
                    np.round(w_opt, 4).tolist(),
                    sp.get("tactical_reason", "N/A"),
                )
            else:
                rho_period = MU_UNCERTAINTY_ANNUAL * (holding_days_t / 252.0)
                rho = _build_rho(n_assets, holding_days_t)
                mu_wc = mu_t - rho
                Sigma_wc = Sigma_t + ROBUST_COV_UNCERTAINTY * np.diag(np.diag(Sigma_t))
                Sigma_wc = (Sigma_wc + Sigma_wc.T) / 2.0
                diag_s = np.diag(Sigma_wc)
                diag_s = np.maximum(diag_s, 1e-12)
                sharpe_like = (mu_wc - r_period_t) / np.sqrt(diag_s)
                equity_excess = float(mu_wc[0] - r_period_t)
                ret_term = float(np.dot(mu_wc, w_opt))
                risk_term = 0.5 * GAMMA * float(w_opt @ Sigma_wc @ w_opt)
                tcost_term = TCOST_RATE * float(np.sum(np.abs(w_opt - w_prev)))
                n_risky = n_assets - 1
                n_neg_sharpe = int(np.sum(sharpe_like[:n_risky] < 0))
                if n_neg_sharpe > 0.5 * n_risky:
                    logger.warning(
                        "Most risk-adjusted worst-case excess returns are negative; high cash is rational."
                    )
                a_rnd_t_log = a_rnd_history[-1] if a_rnd_history else 0.0
                logger.info(
                    "[Markowitz] date=%s rho_period=%.4f a_rnd_t=%.2f mu_t=%s rho=%s mu_wc=%s r_cash=%.6f | "
                    "equity_excess=%.4f ret_term=%.4f risk_term=%.4f tcost_term=%.4f diag(Sigma_wc)=%s sharpe_like=%s | w_opt=%s max_put_weight=%.3f",
                    entry_date,
                    rho_period,
                    a_rnd_t_log,
                    np.round(mu_t, 4).tolist(),
                    np.round(rho, 4).tolist(),
                    np.round(mu_wc, 4).tolist(),
                    r_period_t,
                    equity_excess,
                    ret_term,
                    risk_term,
                    tcost_term,
                    np.round(np.sqrt(diag_s), 4).tolist(),
                    np.round(sharpe_like, 4).tolist(),
                    np.round(w_opt, 4).tolist(),
                    max_put_mw,
                )
                ew = iewma_predictor.get_last_weights()
                if ew is not None:
                    logger.info("[CM-IEWMA] expert_weights=%s", np.round(ew, 4).tolist())

        r_vec = returns.iloc[t].values.astype(float)
        port_ret = np.dot(w_opt, r_vec)
        actual_to = np.sum(np.abs(w_opt - w_prev))
        port_ret -= _tcost * actual_to

        portfolio_values[t + 1] = portfolio_values[t] * (1 + port_ret)

        # Feed realized return to IEWMA (useful even in scenario mode for future blending)
        iewma_predictor.update(r_vec)

    start_date = returns.index[0] - pd.Timedelta(days=avg_holding_days)
    dates_full = pd.DatetimeIndex([start_date] + list(returns.index))

    portfolio_df = pd.DataFrame({"value": portfolio_values}, index=dates_full)
    weights_df = pd.DataFrame(weights_history, index=dates_full, columns=_asset_order)
    mu_df = pd.DataFrame(mu_history, index=returns.index, columns=_asset_order)

    n_blend = sum(1 for s in sigma_source_history if s == "blend")
    n_scenario = sum(1 for s in sigma_source_history if s == "scenario")
    diag_summary = {
        "n_periods": n_periods,
        "optimizer_mode": _opt_mode,
        "n_rnd_computed": n_forecasts,
        "n_bl": sum(1 for d in rnd_diags.values()
                    if d.get("method") == "breeden_litzenberger"),
        "n_lognormal": sum(1 for d in rnd_diags.values()
                          if d.get("method") in ("lognormal_iv_fallback", "lognormal")),
        "iv_range": (
            min((d.get("atm_iv", d.get("iv", 0)) for d in rnd_diags.values()), default=0),
            max((d.get("atm_iv", d.get("iv", 0)) for d in rnd_diags.values()), default=0),
        ),
        "method_history": method_history,
        "n_iewma_blend": n_blend,
        "n_scenario_periods": n_scenario,
        "iewma_weight": 1.0 - RND_BLEND_WEIGHT,
        "rnd_blend_weight": RND_BLEND_WEIGHT,
        "mu_uncertainty_annual": MU_UNCERTAINTY_ANNUAL,
        "rho_period": MU_UNCERTAINTY_ANNUAL * (CONFIG_REBALANCE_DAYS / 252.0),
        "cov_uncertainty": ROBUST_COV_UNCERTAINTY,
        "cvar_alpha": _cvar_alpha if use_scenario else None,
        "cvar_lambda": _cvar_lambda if use_scenario else None,
        "max_option_weight": _max_opt if use_scenario else None,
        "scenario_period_diag": scenario_period_diag if use_scenario else None,
        "asset_order": _asset_order,
        "underlying": _underlying,
        "avg_holding_days": avg_holding_days,
        "holding_days_list": holding_days_list,
        "a_rnd_history": a_rnd_history,
    }

    return portfolio_df, weights_df, returns, mu_df, diag_summary


# -----------------------------------------------------------------------
# Benchmark (single-asset vs multi-asset)
# -----------------------------------------------------------------------

def get_benchmark_stats(returns: pd.DataFrame) -> tuple:
    """
    Return (benchmark_label, benchmark_total_return) for the same benchmark
    used in plot_results: single-asset = that equity buy & hold,
    multi-asset = equal-weight equities.
    """
    cols = list(returns.columns)
    n_a = len(cols)
    if n_a == 4:
        eq_col = cols[0]
        cum = (1 + returns[eq_col]).cumprod()
        return f"{eq_col} Buy & Hold", float(cum.iloc[-1] - 1)
    n_sleeves = (n_a - 1) // 3
    eq_cols = [cols[3 * i] for i in range(n_sleeves)]
    bench_ret = returns[eq_cols].mean(axis=1)
    cum = (1 + bench_ret).cumprod()
    return "Equal-Weight Equities", float(cum.iloc[-1] - 1)


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def plot_results(
    portfolio_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    returns: pd.DataFrame,
    mu_df: pd.DataFrame,
    diag: dict,
    save_path: str | Path | None = None,
    show: bool = False,
) -> str:
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 14),
        gridspec_kw={"height_ratios": [3, 1, 1.2, 1.2]},
    )
    mode_label = diag.get("optimizer_mode", "markowitz")
    if mode_label == "scenario":
        param_str = (
            f"CVaR(α={diag.get('cvar_alpha', 0.95):.2f}, λ={diag.get('cvar_lambda', 2):.1f})"
            f"  max_opt={diag.get('max_option_weight', SCENARIO_MAX_OPT):.0%}"
            f"  BL:{diag['n_bl']}/{diag['n_rnd_computed']}"
        )
    else:
        param_str = (
            f"γ={GAMMA}  max_opt={MAX_OPTION_WEIGHT:.0%}"
            f"  vol_cap={MAX_PORT_VOL:.0%}"
            f"  ρ_ann={diag.get('mu_uncertainty_annual', 0):.3f} ρ_per={diag.get('rho_period', 0):.4f}"
            f"  κ={diag.get('cov_uncertainty', 0):.2f}"
            f"  IEWMA={diag.get('iewma_weight', 0):.0%}"
            f"  BL:{diag['n_bl']}/{diag['n_rnd_computed']}"
        )
    fig.suptitle(
        f"{mode_label.title()} Backtest:  {diag.get('underlying', 'SPY')} · ATM Call · Put Spread · Cash\n"
        + param_str,
        fontsize=12, fontweight="bold",
    )

    asset_list = list(weights_df.columns)
    colors = {
        "SPY": "#2563eb", "SPY_CALL": "#16a34a", "SPY_PUT": "#dc2626",
        "AAPL": "#2563eb", "AAPL_CALL": "#16a34a", "AAPL_PUT": "#dc2626",
        "CRWD": "#7c3aed", "CRWD_CALL": "#16a34a", "CRWD_PUT": "#dc2626",
        "USDOLLAR": "#f59e0b",
    }
    for a in asset_list:
        if a not in colors:
            colors[a] = "#6b7280"  # gray fallback

    pv = portfolio_df["value"].values
    total_ret = pv[-1] / INITIAL_VALUE - 1
    asset_list_for_bench = list(returns.columns)
    n_a = len(asset_list_for_bench)
    # Benchmark: single-asset = that equity buy & hold; multi-asset = equal-weight equities
    if n_a == 4:
        eq_col = asset_list_for_bench[0]
        benchmark_cum = (1 + returns[eq_col]).cumprod()
        benchmark_label = f"{eq_col} Buy & Hold"
    else:
        n_sleeves = (n_a - 1) // 3
        eq_cols = [asset_list_for_bench[3 * i] for i in range(n_sleeves)]
        benchmark_ret = returns[eq_cols].mean(axis=1)
        benchmark_cum = (1 + benchmark_ret).cumprod()
        benchmark_label = "Equal-Weight Equities"
    benchmark_total = benchmark_cum.iloc[-1] - 1
    period_rets = np.diff(pv) / pv[:-1]
    avg_hold = max(diag.get("avg_holding_days", REBAL_DAYS), 1)
    ann_factor = np.sqrt(252 / avg_hold)
    sharpe = np.mean(period_rets) / (np.std(period_rets) + 1e-12) * ann_factor
    running_max = np.maximum.accumulate(pv)
    drawdown = pv / running_max - 1
    max_dd = drawdown.min()

    # Panel 1: portfolio trajectory
    ax = axes[0]
    ax.plot(portfolio_df.index, pv, color="#2563eb", linewidth=2.2, label="Optimized Portfolio")
    ax.plot(returns.index, benchmark_cum.values, color="#94a3b8", linewidth=1.5,
            linestyle="--", label=benchmark_label)
    ax.axhline(1.0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Value ($1 initial)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    iv_lo, iv_hi = diag["iv_range"]
    avg_hold = max(diag.get("avg_holding_days", REBAL_DAYS), 1)
    stats = (
        f"Return: {total_ret:+.1%}  ({benchmark_label}: {benchmark_total:+.1%})\n"
        f"Sharpe (ann.): {sharpe:.2f}  avg_holding_days: {avg_hold:.0f}\n"
        f"Max DD: {max_dd:.1%}\n"
        f"Periods: {diag['n_periods']}  ({returns.index[0].date()} → {returns.index[-1].date()})\n"
        f"BL density: {diag['n_bl']}/{diag['n_rnd_computed']}  "
        f"IV range: {iv_lo:.0%}–{iv_hi:.0%}\n"
        f"IEWMA blend: {diag.get('n_iewma_blend', 0)}/{diag['n_periods']} periods"
    )
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    # Panel 2: drawdown
    ax_dd = axes[1]
    ax_dd.fill_between(portfolio_df.index, drawdown, 0, color="#dc2626", alpha=0.35)
    ax_dd.plot(portfolio_df.index, drawdown, color="#dc2626", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_ylim(min(max_dd * 1.1, -0.05), 0.02)
    ax_dd.grid(True, alpha=0.3)

    # Panel 3: weights
    ax2 = axes[2]
    ax2.stackplot(weights_df.index, weights_df[asset_list].values.T,
                  labels=asset_list, colors=[colors[a] for a in asset_list], alpha=0.85)
    ax2.set_ylabel("Weight")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", ncol=4, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 4: rolling mu
    ax3 = axes[3]
    risky = [a for a in asset_list if a != "USDOLLAR"]
    for asset in risky:
        ax3.plot(mu_df.index, mu_df[asset], label=asset, color=colors.get(asset, "#6b7280"), linewidth=1.2)
    ax3.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("μ (shrunk)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", ncol=3, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add RND method markers on the weights panel
    for i, m in enumerate(diag.get("method_history", [])):
        if m == "breeden_litzenberger":
            ax2.axvline(returns.index[i], color="green", alpha=0.15, linewidth=1)

    plt.tight_layout()
    save_path = save_path or str(PROCESSED_DIR / "backtest_results.png")
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return str(save_path)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "markowitz":
        OPT_MODE = "markowitz"  # noqa: F811  override for this run
        globals()["OPT_MODE"] = "markowitz"
    portfolio_df, weights_df, returns, mu_df, diag = run_backtest()

    pv = portfolio_df["value"]
    mode = diag.get("optimizer_mode", "markowitz")
    print(f"\n========== Backtest Results ({mode}) ==========")
    print(f"Periods:  {diag['n_periods']}")
    print(f"Range:    {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Final:    ${pv.iloc[-1]:.4f}  ({pv.iloc[-1] / INITIAL_VALUE - 1:+.2%})")

    pr = np.diff(pv.values) / pv.values[:-1]
    avg_hold = max(diag.get("avg_holding_days", REBAL_DAYS), 1)
    ann = np.sqrt(252 / avg_hold)
    print(f"Sharpe:   {np.mean(pr) / (np.std(pr) + 1e-12) * ann:.2f}  (avg_holding_days={avg_hold:.0f})")
    print(f"Max DD:   {np.min(pv.values / np.maximum.accumulate(pv.values)) - 1:.1%}")
    print(f"BL used:  {diag['n_bl']}/{diag['n_rnd_computed']} periods")
    if mode == "scenario":
        print(f"Scenario: {diag.get('n_scenario_periods', 0)}/{diag['n_periods']} periods"
              f"  CVaR(α={diag.get('cvar_alpha', 0.95):.2f}, λ={diag.get('cvar_lambda', 0.25):.2f})")
        if "SPY_PUT" in weights_df.columns and "USDOLLAR" in weights_df.columns:
            avg_put = float(weights_df["SPY_PUT"].iloc[1:].mean())
            avg_cash = float(weights_df["USDOLLAR"].iloc[1:].mean())
            print(f"  Average put weight: {avg_put:.1%}  Average cash weight: {avg_cash:.1%}")
        sp_diag = diag.get("scenario_period_diag") or []
        if sp_diag:
            last = sp_diag[-1]
            mu_spy = last.get("mu_phys_spy")
            mu_call = last.get("mu_phys_call")
            mu_put = last.get("mu_phys_put")
            mean_scn = last.get("mean_scenario_spy")
            cvar_m = last.get("cvar_model")
            cvar_e = last.get("cvar_empirical")
            cvar_port = last.get("cvar_portfolio")
            cvar_spy = last.get("cvar_spy")
            put_prot = last.get("put_protection_pct")
            mu_val = 0.0 if mu_spy is None else float(mu_spy)
            mu_c_val = 0.0 if mu_call is None else float(mu_call)
            mu_p_val = 0.0 if mu_put is None else float(mu_put)
            mean_val = 0.0 if mean_scn is None else float(mean_scn)
            print(f"  Last period: μ_spy={mu_val:.4f} μ_call={mu_c_val:.4f} μ_put={mu_p_val:.4f}")
            cvar_port_v = 0.0 if cvar_port is None else float(cvar_port)
            cvar_spy_v = 0.0 if cvar_spy is None else float(cvar_spy)
            put_prot_v = 0.0 if put_prot is None else float(put_prot)
            print(f"    CVaR: port={cvar_port_v:.4f}  SPY={cvar_spy_v:.4f}  put_protection={put_prot_v:.1f}%")
            if SCENARIO_TACTICAL_PUTS:
                tactical_wt = last.get("tactical_put_weight")
                tactical_reason = last.get("tactical_reason", "N/A")
                current_iv = last.get("current_iv")
                if tactical_wt is not None:
                    print(f"    Tactical puts: max_weight={tactical_wt:.1%}  reason={tactical_reason}  IV={current_iv:.3f}")
    else:
        print(f"IEWMA:    {diag.get('n_iewma_blend', 0)}/{diag['n_periods']} blended"
              f"  (weight={diag.get('iewma_weight', 0):.0%})")
        print(f"Robust:   ρ_annual={diag.get('mu_uncertainty_annual', 0):.4f} ρ_period={diag.get('rho_period', 0):.4f}"
              f"  κ={diag.get('cov_uncertainty', 0):.2f}")
        eq_col = weights_df.columns[0]
        if "USDOLLAR" in weights_df.columns:
            avg_eq = float(weights_df[eq_col].iloc[1:].mean())
            avg_cash = float(weights_df["USDOLLAR"].iloc[1:].mean())
            print(f"  Average equity weight: {avg_eq:.1%}  Average cash weight: {avg_cash:.1%}")

    print("\nFinal weights:")
    for a in weights_df.columns:
        print(f"  {a:10s} {weights_df[a].iloc[-1]:6.1%}")

    plot_path = plot_results(portfolio_df, weights_df, returns, mu_df, diag)
    print(f"\nPlot saved: {plot_path}")
