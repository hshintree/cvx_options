"""
Run scenario backtest with a ScenarioBacktestConfig.

Passes config to run_backtest(). Multi-asset requires data for each symbol.
"""
from __future__ import annotations

import logging

from scenario_config import ScenarioBacktestConfig

logger = logging.getLogger(__name__)


def run_scenario_backtest(config: ScenarioBacktestConfig) -> tuple:
    """
    Run the scenario backtest with the given config.

    Returns (portfolio_df, weights_df, returns, mu_df, diag) same as run_backtest.
    """
    import backtest as _bt

    return _bt.run_backtest(config=config)
