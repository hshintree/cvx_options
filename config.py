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
OPTION_CONTRACTS_DIR = RAW_DIR / "option_contracts"  # one parquet per contract symbol (or combined)

# Processed outputs (for cvxportfolio / backtest)
RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
VOLUMES_FILE = PROCESSED_DIR / "volumes.parquet"
PRICES_FILE = PROCESSED_DIR / "prices.parquet"
CALENDAR_FILE = PROCESSED_DIR / "trading_calendar.parquet"

# -----------------------------------------------------------------------------
# Symbols and data range
# -----------------------------------------------------------------------------
UNDERLYING = "SPY"
CASH_SYMBOL = "USDOLLAR"  # for display; we use risk-free rate series

# Default date range for initial full pull (SPY can go back further; options depend on contract history)
DEFAULT_START_DATE = "2015-01-01"  # adjust as needed; options history may be shorter
DEFAULT_END_DATE = None  # None = today

# -----------------------------------------------------------------------------
# Options selection (for building representative call/put series)
# -----------------------------------------------------------------------------
# Number of expirations to fetch history for (from front month)
OPTION_EXPIRATIONS_TO_FETCH = 4
# Strike range: fetch contracts with strike within ± this percent of ATM
OPTION_ATM_STRIKE_PCT = 0.10  # e.g. 0.10 = 10% around ATM
# Max number of contracts per expiry per type (call/put) to fetch history for (to limit API load)
OPTION_MAX_CONTRACTS_PER_EXPIRY = 15

# Historical backfill: one ATM call + one ATM put per month (third Friday).
# Yahoo often 404s for expired options. Default is OFF; use --historical-options to enable.
OPTION_HISTORICAL_MONTHLY = False
OPTION_HISTORICAL_MAX_MONTHS_BACK = 24  # when enabled, only expiries in last 24 months

# -----------------------------------------------------------------------------
# RND / forecasts
# -----------------------------------------------------------------------------
# Rebalancing: hold options to expiry (rebalance period = DTE).
# Set to None to match the chosen expiry (hold-to-expiry and roll).
# Set to an integer (e.g. 7) to rebalance mid-life via horizon repricing.
REBALANCE_DAYS = None

# DTE targeting: pick options in this DTE band
TARGET_MIN_DTE = 7    # expiry must be > chain_date + this many days
TARGET_MAX_DTE = 21   # expiry must be < chain_date + this many days
TARGET_IDEAL_DTE = 14 # prefer expiry closest to this DTE

# Quote quality (avoid penny options and bad quotes)
MIN_OPTION_MID = 0.05  # drop options with mid < this (avoid tiny denominator)
MAX_BID_ASK_SPREAD_PCT = 0.50  # drop if (ask-bid)/mid > this
MIN_BL_STRIKES = 10  # minimum interior strikes for Breeden-Litzenberger density
OPTION_RETURN_WINSORIZE_PCT = (1.0, 99.0)  # winsorize at these percentiles (no fixed cap)
MU_CLIP_SPY = (-0.10, 0.10)  # plausible weekly SPY expected return
MU_CLIP_OPTION = (-0.30, 0.30)  # plausible weekly option expected return (avoid 200%)

# -----------------------------------------------------------------------------
# Ensure directories exist
# -----------------------------------------------------------------------------
for _d in (RAW_DIR, PROCESSED_DIR, OPTION_CHAINS_DIR, OPTION_CONTRACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
