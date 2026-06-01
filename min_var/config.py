"""
min_var configuration — tunables for the walk-forward min-variance backtests.
Inherits shared path/rate constants from the top-level config.py.
"""
from pathlib import Path
import sys

# Ensure project root is on path so we can import top-level config
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    RAW_DIR,
    SPY_DAILY_FILE,
    OPTION_CHAINS_DIR,
)

R_ANN = 0.05   # annualized risk-free rate (5% — matches log_norm.ipynb notebook config)

# ── Research panel (optionsdx 2023 data with live Greeks) ─────────────────────
PANEL_PATH = _ROOT / "data" / "processed" / "free_history" / "research_panel_optionsdx_csv.parquet"

# ── Walk-forward backtest tunables ────────────────────────────────────────────
LOOKBACK    = 21     # rolling-window trading days for DGV covariance
REBAL_FREQ  = 5      # rebalance every N trading days
MAX_WT      = 0.50   # max weight per single asset
MAX_OPT_WT  = 0.05   # max weight per individual option leg

# ── Mean-variance risk-aversion (for objective="meanvar") ─────────────────────
MV_GAMMA = 5.0       # γ in: max (μ−rf)·w − γ·w'Σw

# ── Universe definitions ───────────────────────────────────────────────────────
SYMS        = ["SPY", "AAPL", "NVDA", "TSLA"]
UNIV_STOCKS = SYMS
UNIV_A      = ["SPY", "SPY_call1", "SPY_put1", "AAPL", "NVDA", "TSLA"]
UNIV_B      = [f"{s}{sfx}" for s in SYMS for sfx in ["", "_call1", "_put1"]]

# ── Accuracy test (Section 4 of log_norm.ipynb) tunables ──────────────────────
TEST_DTE       = 30       # DTE horizon to evaluate (14 / 30 / 45)
N_SAMPLES      = 60       # number of historical chain dates to sample
MIN_DAYS_BACK  = 50       # chain date must be ≥ this many calendar days before last close
LOOKBACK_DAYS  = 21       # trading-day lookback for realized vol / mu guess
N_SCENARIOS    = 50_000   # Monte Carlo draws for predicted distribution
N_PARAM_DRAWS  = 2_000    # parameter-uncertainty draws

# ── Fitting ───────────────────────────────────────────────────────────────────
DTE_14 = 14
DTE_30 = 30
DTE_45 = 45
MIN_REALIZED_DAYS = 45   # chain date needs ≥ this many calendar days of realized SPY data

# ── Extended equity+bond universe (Part 2) ────────────────────────────────────
EXTENDED_SYMBOLS = [
    "SPY", "QQQ", "TLT", "IEF", "SHY", "GLD",
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN",
]
EXTENDED_DATA_START = "2019-01-01"   # 2yr before backtest start to cover momentum warmup

# ── Expanded multi-asset universe (no defense, no oil) ────────────────────────
# Covers: US broad market, tech, sectors, real estate, international,
#         fixed income, commodities, and crypto-adjacent.
# COIN and MSTR listed since 2021-04 and 2020 respectively — handled by
# load_equity_returns dropping rows where any symbol has NaN.
EXPANDED_SYMBOLS = [
    # ── US Broad market ──────────────────────────────────────────────────────
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq-100
    "IWM",   # Russell 2000 (small-cap)
    # ── Tech mega-cap ────────────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    # ── Sector ETFs (ex-defense, ex-oil) ─────────────────────────────────────
    "XLV",   # Health Care
    "XLF",   # Financials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "ICLN",  # Clean Energy
    # ── Real estate ──────────────────────────────────────────────────────────
    "VNQ",   # Vanguard REIT ETF
    "AMT",   # American Tower (cell-tower REIT)
    # ── International ────────────────────────────────────────────────────────
    "EFA",   # MSCI EAFE (developed ex-US)
    "EEM",   # MSCI Emerging Markets
    # ── Fixed income ─────────────────────────────────────────────────────────
    "TLT",   # 20yr Treasury
    "IEF",   # 7-10yr Treasury
    "SHY",   # 1-3yr Treasury (cash proxy)
    "TIP",   # TIPS (inflation-protected)
    "HYG",   # High-yield corporate bonds
    # ── Commodities ──────────────────────────────────────────────────────────
    "GLD",   # Gold
    "SLV",   # Silver
    "CPER",  # Copper
    "DBC",   # Diversified commodity basket
    # ── Crypto-adjacent ──────────────────────────────────────────────────────
    # GBTC excluded: only exists as an ETF since 2024-01-11; would truncate dataset.
    "COIN",  # Coinbase (2021-04 IPO, Bitcoin/Ethereum exchange exposure)
    "MSTR",  # MicroStrategy (Bitcoin proxy, heavy BTC since 2020)
]
EXPANDED_DATA_START = "2020-01-01"   # limited by COIN 2021-04 IPO; use 2020 for warm-up data
EXPANDED_MAX_WT     = 0.20           # 5% per asset cap keeps portfolio well-diversified

# ── Grid search grids ─────────────────────────────────────────────────────────
LOOKBACK_GRID   = [21, 42, 63, 126]
REBAL_FREQ_GRID = [5, 10, 21]
GAMMA_GRID      = [0, 5, 10, 20, 50, 100]
MAX_WT_GRID     = [0.30, 0.40, 0.50]

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = _ROOT / "min_var_output"
