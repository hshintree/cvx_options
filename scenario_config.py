"""
Runtime configuration for scenario/CVaR backtests.

Override these in a notebook or script to change assets and hyperparameters
without editing config.py. Supports single-asset (SPY, AAPL, CRWD) and
multi-asset portfolios.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# -----------------------------------------------------------------------------
# Default paths (from config)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OPTION_CHAINS_DIR = PROJECT_ROOT / "data" / "raw" / "option_chains"
CASH_SYMBOL = "USDOLLAR"


def asset_order_for_underlyings(underlyings: List[str]) -> List[str]:
    """Build asset order: [EQ1, CALL1, PUT1, EQ2, CALL2, PUT2, ..., USDOLLAR]."""
    order = []
    for sym in underlyings:
        order.extend([sym, f"{sym}_CALL", f"{sym}_PUT"])
    order.append(CASH_SYMBOL)
    return order


@dataclass
class ScenarioHyperparams:
    """Scenario/CVaR hyperparameters — easy to tweak in a notebook."""
    n_samples: int = 20_000
    cvar_alpha: float = 0.95
    cvar_lambda: float = 0.25
    min_cash_weight: float = 0.0
    max_option_weight: float = 0.05
    max_spy_weight: float = 1.0
    put_spread_width: float = 0.08
    skew_beta: float = 2.5
    skew_threshold: float = 0.01
    equity_premium_annual: Optional[float] = None  # None = estimate from history
    max_turnover: float = 0.25
    tcost_rate: float = 0.0005
    # Dynamic hedge sizing (replaces old tactical_puts on/off switch)
    hedge_mode: str = "dynamic"  # "dynamic" | "always" | "never"
    hedge_floor: float = 0.005   # minimum put weight even in calm markets
    hedge_iv_lookback: int = 20  # periods for IV regime estimation


@dataclass
class ScenarioBacktestConfig:
    """
    Full configuration for a scenario backtest run.

    Single-asset: underlyings = ["SPY"] or ["AAPL"] or ["CRWD"]
    Multi-asset:  underlyings = ["AAPL", "CRWD"] → assets: AAPL, AAPL_CALL, AAPL_PUT, CRWD, CRWD_CALL, CRWD_PUT, USDOLLAR
    """
    underlyings: List[str] = field(default_factory=lambda: ["SPY"])
    hyperparams: ScenarioHyperparams = field(default_factory=ScenarioHyperparams)
    optimizer_mode: str = "scenario"  # "markowitz" | "scenario"
    rebalance_days: int = 7
    risk_free_rate: float = 0.05

    @property
    def asset_order(self) -> List[str]:
        return asset_order_for_underlyings(self.underlyings)

    @property
    def n_assets(self) -> int:
        return len(self.asset_order)

    def price_file(self, symbol: str) -> Path:
        """Path to daily price file for symbol. SPY uses spy_daily.parquet for backward compat."""
        if symbol == "SPY":
            return RAW_DIR / "spy_daily.parquet"
        return RAW_DIR / f"{symbol.lower()}_daily.parquet"

    def chain_dir(self, symbol: str) -> Path:
        """Directory for option chains. SPY uses flat option_chains/ for backward compat."""
        if symbol == "SPY":
            return OPTION_CHAINS_DIR
        return OPTION_CHAINS_DIR / symbol


# -----------------------------------------------------------------------------
# Multi-asset
# -----------------------------------------------------------------------------
# For UNDERLYINGS = ["AAPL", "CRWD"], asset_order = [AAPL, AAPL_CALL, AAPL_PUT,
#   CRWD, CRWD_CALL, CRWD_PUT, USDOLLAR]. Scenario matrix R is (N, 7).
# Backtest uses joint rebalance dates (intersection of chain dates) and
# build_multi_asset_scenario_matrix (per-symbol scenarios stacked).
