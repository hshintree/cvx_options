"""
Shared put-spread definition for forecasts, scenarios, and realized returns.

Single source of truth: long ATM put, short OTM put at (1 - width)*spot.
Used by data/forecasts.py, data/scenarios.py, and backtest build_realized_returns.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

# Avoid circular import: we need BS from forecasts
def _get_bs():
    from data.forecasts import _bs_put_price, _bs_put_vec
    return _bs_put_price, _bs_put_vec


def build_put_spread(
    spot: float,
    k_atm: float,
    r: float,
    T: float,
    iv_long: float,
    width: float,
    min_mid: float = 0.10,
) -> Tuple[float, float]:
    """
    Build put spread entry: long ATM put, short OTM put at (1 - width)*spot.

    Returns
    -------
    k_short : strike of short put (0 if width <= 0 = naked put)
    entry_value : net premium paid (long - short), >= min_mid
    """
    _bs_put_price, _ = _get_bs()
    if width <= 0:
        p_long = max(_bs_put_price(spot, k_atm, r, T, iv_long), min_mid)
        return 0.0, float(p_long)
    k_short = round(k_atm * (1.0 - width))
    p_long = max(_bs_put_price(spot, k_atm, r, T, iv_long), min_mid)
    p_short = max(_bs_put_price(spot, k_short, r, T, iv_long), 0.0)
    entry_value = max(p_long - p_short, min_mid)
    return float(k_short), float(entry_value)


def price_put_spread(
    S: Union[float, np.ndarray],
    k_atm: float,
    k_short: float,
    r: float,
    T_remain: float,
    iv_long: float,
    *,
    intrinsic_if_expired: bool = True,
) -> Union[float, np.ndarray]:
    """
    Value of put spread at spot S with remaining time T_remain (sticky-strike IV).
    If T_remain <= 0 and intrinsic_if_expired, return intrinsic max(0, P_long - P_short).
    S can be scalar or array.
    """
    _bs_put_price, _bs_put_vec = _get_bs()
    if intrinsic_if_expired and T_remain <= 1e-6:
        if np.isscalar(S):
            P_long = max(k_atm - S, 0.0)
            P_short = max(k_short - S, 0.0) if k_short > 0 else 0.0
            return max(P_long - P_short, 0.0)
        P_long = np.maximum(k_atm - S, 0.0)
        P_short = np.maximum(k_short - S, 0.0) if k_short > 0 else 0.0
        return np.maximum(P_long - P_short, 0.0)
    if np.isscalar(S):
        P_long = _bs_put_price(S, k_atm, r, T_remain, iv_long)
        P_short = _bs_put_price(S, k_short, r, T_remain, iv_long) if k_short > 0 else 0.0
        return max(P_long - P_short, 0.0)
    P_long = _bs_put_vec(S, k_atm, r, T_remain, iv_long)
    if k_short > 0:
        P_short = _bs_put_vec(S, k_short, r, T_remain, iv_long)
        out = np.maximum(P_long - P_short, 0.0)
    else:
        out = P_long
    return out
