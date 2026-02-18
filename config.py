"""
Project configuration: paths, date ranges, and symbols.
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data filenames
SPY_DAILY_FILE = RAW_DIR / "spy_daily.parquet"
CASH_RATE_FILE = RAW_DIR / "cash_rate_daily.parquet"
OPTION_CHAINS_DIR = RAW_DIR / "option_chains"  # one parquet per date
OPTION_CONTRACTS_DIR = RAW_DIR / "option_contracts"  # unused with Alpaca pipeline

# Processed outputs (for cvxportfolio / backtest)
RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
VOLUMES_FILE = PROCESSED_DIR / "volumes.parquet"
PRICES_FILE = PROCESSED_DIR / "prices.parquet"
CALENDAR_FILE = PROCESSED_DIR / "trading_calendar.parquet"

# -----------------------------------------------------------------------------
# Symbols and data range
# -----------------------------------------------------------------------------
UNDERLYING = "SPY"
CASH_SYMBOL = "USDOLLAR"

# SPY equity bars: Alpaca has data going back many years
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = None  # None = today

# Historical option bars are only available since ~Feb 2024 on Alpaca
OPTION_DATA_START = "2024-02-01"

# Strike range for chain reconstruction: ATM ± this percent
OPTION_ATM_STRIKE_PCT = 0.10

# -----------------------------------------------------------------------------
# RND / forecasts
# -----------------------------------------------------------------------------
# Rebalancing: horizon repricing (options repriced via BS at next rebalance).
# Set to None to match the chosen expiry (hold-to-expiry — binary returns).
# Set to an integer (e.g. 14) to rebalance mid-life via horizon repricing.
REBALANCE_DAYS = 14

# DTE targeting: pick options with substantially more time than REBALANCE_DAYS
# so they retain time-value at rebalance (continuous, non-binary payoffs).
# At rebalance, options will have ~30 days remaining → smooth BS repricing.
TARGET_MIN_DTE = 30
TARGET_MAX_DTE = 60
TARGET_IDEAL_DTE = 45

# Quote quality (avoid penny options and bad quotes)
MIN_OPTION_MID = 0.05  # drop options with mid < this (avoid tiny denominator)
MAX_BID_ASK_SPREAD_PCT = 0.50  # drop if (ask-bid)/mid > this
MIN_BL_STRIKES = 10  # minimum interior strikes for Breeden-Litzenberger density
OPTION_RETURN_WINSORIZE_PCT = (1.0, 99.0)  # winsorize at these percentiles (no fixed cap)
MU_CLIP_SPY = (-0.10, 0.10)  # plausible biweekly SPY expected return
MU_CLIP_OPTION = (-0.80, 0.80)  # option mu can be large due to ~20x leverage

# -----------------------------------------------------------------------------
# Ensure directories exist
# -----------------------------------------------------------------------------
for _d in (RAW_DIR, PROCESSED_DIR, OPTION_CHAINS_DIR, OPTION_CONTRACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
