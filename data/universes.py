"""
Research universes used by the snapshot pipeline and notebooks.

The S&P 100 membership below is frozen from the public Wikipedia component list
that was marked "as of September 22, 2025" on the page revision last edited on
January 12, 2026. This keeps the data pipeline deterministic instead of
changing silently when index membership rotates.
"""
from __future__ import annotations

from typing import List


SP100_CONSTITUENTS_2025_09_22: List[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C",
    "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GILD", "GM",
    "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ",
    "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT",
    "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE",
    "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
    "XOM",
]

LIQUID_OPTION_TICKERS_2026_02: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "TSLA", "AMD", "GOOGL", "JPM", "BAC",
    "XOM", "AVGO", "QCOM", "NFLX", "PLTR",
    "UBER", "INTC", "PYPL", "WMT", "BA",
]


def sp100_plus_spy() -> List[str]:
    """Return the frozen research basket: SPY plus the S&P 100 members."""
    return ["SPY", *SP100_CONSTITUENTS_2025_09_22]


def liquid_research_plus_spy() -> List[str]:
    """Return a smaller, more liquid basket for intraday snapshot research."""
    return ["SPY", *LIQUID_OPTION_TICKERS_2026_02]


UNIVERSE_BUILDERS = {
    "liquid_research_plus_spy": liquid_research_plus_spy,
    "sp100_plus_spy": sp100_plus_spy,
}


def get_named_universe(name: str) -> List[str]:
    if name not in UNIVERSE_BUILDERS:
        raise KeyError(f"Unknown universe {name!r}")
    return UNIVERSE_BUILDERS[name]()
