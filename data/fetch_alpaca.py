"""
Alpaca-based data pipeline for SPY equity + option chain data.

Replaces the yfinance pipeline.  Uses:
  - StockHistoricalDataClient  → SPY daily OHLCV bars
  - OptionHistoricalDataClient → option chain snapshots (current),
    historical daily bars for past rebalance dates

Historical option bars are available since ~Feb 2024.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    CASH_RATE_FILE,
    OPTION_CHAINS_DIR,
    RAW_DIR,
    SPY_DAILY_FILE,
    TARGET_IDEAL_DTE,
    TARGET_MAX_DTE,
    TARGET_MIN_DTE,
)

logger = logging.getLogger(__name__)

load_dotenv(_ROOT / ".env")

_API_KEY = os.getenv("ALPACA_API_KEY")
_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Rate-limit: brief sleep between Alpaca API calls (seconds)
_API_SLEEP = 0.25
# Max option symbols per bars request (Alpaca cap)
_BATCH_SIZE = 100
# How far back historical option data is available
OPTION_DATA_START = "2024-02-01"

# Backfill defaults (wider coverage than the rebalance-date pipeline)
_BACKFILL_MAX_DTE = 90
_BACKFILL_STRIKE_PCT = 0.25
_BACKFILL_STRIKE_STEP = 1.0
SPY_EXPIRY_WEEKDAYS = (0, 2, 4)       # Mon / Wed / Fri
DEFAULT_EXPIRY_WEEKDAYS = (0, 1, 2, 3, 4)  # try all weekdays


# ---------------------------------------------------------------------------
# Alpaca SDK clients (lazy-initialised)
# ---------------------------------------------------------------------------

_stock_client = None
_option_client = None


def _get_stock_client():
    global _stock_client
    if _stock_client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        _stock_client = StockHistoricalDataClient(_API_KEY, _SECRET_KEY)
    return _stock_client


def _get_option_client():
    global _option_client
    if _option_client is None:
        from alpaca.data.historical import OptionHistoricalDataClient
        _option_client = OptionHistoricalDataClient(_API_KEY, _SECRET_KEY)
    return _option_client


# ---------------------------------------------------------------------------
# OCC option symbology helpers
# ---------------------------------------------------------------------------

def make_occ_symbol(underlying: str, expiry: date, strike: float, is_call: bool) -> str:
    """Build OCC option symbol.  e.g. SPY250228C00550000, AAPL250228C00150000"""
    t = "C" if is_call else "P"
    strike_int = int(round(strike * 1000))
    return f"{underlying}{expiry.strftime('%y%m%d')}{t}{strike_int:08d}"


def parse_occ_symbol(sym: str) -> dict:
    """Parse an OCC symbol into components. Root length varies (SPY=3, AAPL=4, etc.)."""
    # OCC: ...ROOT + YYMMDD (6) + C|P (1) + strike (8 digits)
    strike = int(sym[-8:]) / 1000.0
    is_call = sym[-9] == "C"
    yy, mm, dd = int(sym[-15:-13]), int(sym[-13:-11]), int(sym[-11:-9])
    expiry = date(2000 + yy, mm, dd)
    root = sym[:-15].strip()
    return {
        "root": root,
        "expiry": expiry,
        "is_call": is_call,
        "strike": strike,
        "symbol": sym,
    }


# ---------------------------------------------------------------------------
# Snapshot storage helpers
# ---------------------------------------------------------------------------

def _as_utc_snapshot_str(ts: pd.Timestamp | datetime | None = None) -> str:
    """Return a second-granularity UTC timestamp string for snapshot rows."""
    ts = pd.Timestamp.now(tz="UTC") if ts is None else pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("s").strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_snapshot_time_column(df: pd.DataFrame, *, fallback_ts: str) -> pd.DataFrame:
    """Normalize snapshot_time and backfill it for legacy daily-overwrite files."""
    out = df.copy()
    if "snapshot_time" not in out.columns:
        out["snapshot_time"] = fallback_ts
        return out
    parsed = pd.to_datetime(out["snapshot_time"], utc=True, errors="coerce")
    out["snapshot_time"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["snapshot_time"] = out["snapshot_time"].fillna(fallback_ts)
    return out


def _append_intraday_snapshot_parquet(path: Path, new_df: pd.DataFrame) -> tuple[int, int]:
    """
    Append the latest snapshot rows into a single same-day parquet.

    This keeps one file per day/side/symbol while preserving every intraday
    scrape as distinct rows keyed by (snapshot_time, contractSymbol).
    """
    if new_df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        return 0, 0

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.empty:
            existing_ts = _as_utc_snapshot_str(pd.Timestamp.utcfromtimestamp(path.stat().st_mtime))
            existing = _coerce_snapshot_time_column(existing, fallback_ts=existing_ts)
            frames.append(existing)

    current_ts = _as_utc_snapshot_str()
    current = _coerce_snapshot_time_column(new_df, fallback_ts=current_ts)
    frames.append(current)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if {"snapshot_time", "contractSymbol"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["snapshot_time", "contractSymbol"], keep="last")
        merged = merged.sort_values(["snapshot_time", "contractSymbol"]).reset_index(drop=True)
    merged.to_parquet(path, index=False)
    snapshot_count = int(merged["snapshot_time"].nunique()) if "snapshot_time" in merged.columns else 0
    return len(merged), snapshot_count


def _extract_symbol_payload(response, symbol: str):
    """Return a symbol-specific payload from an Alpaca multi-symbol response."""
    if response is None:
        return None
    getter = getattr(response, "get", None)
    if callable(getter):
        payload = getter(symbol)
        if payload is not None:
            return payload
    try:
        return response[symbol]
    except Exception:
        pass
    return response


def _fetch_latest_underlying_snapshot(underlying_symbol: str, fallback_spot: float) -> dict:
    """
    Capture the underlying stock at the same instant as the option snapshot.

    For intraday delta-gamma-theta tests, daily close is not enough. We store
    the latest stock bid/ask/trade and a best-effort spot proxy beside every
    option row so the notebook can compute intraday dS.
    """
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

    quote = None
    trade = None
    stock_client = _get_stock_client()
    try:
        quote_resp = stock_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[underlying_symbol]),
        )
        quote = _extract_symbol_payload(quote_resp, underlying_symbol)
    except Exception as exc:
        logger.debug("Latest quote fetch failed for %s: %s", underlying_symbol, exc)
    try:
        trade_resp = stock_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=[underlying_symbol]),
        )
        trade = _extract_symbol_payload(trade_resp, underlying_symbol)
    except Exception as exc:
        logger.debug("Latest trade fetch failed for %s: %s", underlying_symbol, exc)

    bid = getattr(quote, "bid_price", None)
    ask = getattr(quote, "ask_price", None)
    last = getattr(trade, "price", None)
    if bid is not None:
        bid = float(bid)
    if ask is not None:
        ask = float(ask)
    if last is not None:
        last = float(last)

    if bid and ask and bid > 0 and ask > 0 and ask >= bid:
        spot = 0.5 * (bid + ask)
        source = "stock_quote_mid"
    elif last and last > 0:
        spot = last
        source = "stock_trade"
    else:
        spot = float(fallback_spot)
        source = "daily_close_fallback"

    return {
        "snapshot_time": _as_utc_snapshot_str(),
        "underlying_symbol": underlying_symbol,
        "underlying_bid": bid,
        "underlying_ask": ask,
        "underlying_last": last,
        "underlying_spot": float(spot),
        "underlying_spot_source": source,
    }


# ---------------------------------------------------------------------------
# 1.  Stock daily bars  (SPY or any symbol)
# ---------------------------------------------------------------------------

def fetch_stock_bars(
    symbol: str,
    start: str = "2020-01-01",
    end: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Fetch daily bars for a stock from Alpaca and optionally save as parquet.
    Saves to RAW_DIR / {symbol.lower()}_daily.parquet (SPY → spy_daily.parquet).
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_stock_client()
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    request = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame.Day,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=end_dt,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        logger.warning("No bars returned for %s", symbol)
        return pd.DataFrame()

    # bars.df has MultiIndex (symbol, timestamp).  Flatten.
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"

    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_DIR / f"{symbol.lower()}_daily.parquet"
        df.to_parquet(out_path, index=True)
        logger.info("Saved %s bars: %d rows → %s", symbol, len(df), out_path)
    return df


def fetch_spy_bars(
    start: str = "2020-01-01",
    end: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Fetch SPY daily bars (convenience wrapper)."""
    return fetch_stock_bars("SPY", start=start, end=end, save=save)


# ---------------------------------------------------------------------------
# 2.  Cash / risk-free rate (synthetic from constant or Fed data)
# ---------------------------------------------------------------------------

def build_cash_rate(spy_df: pd.DataFrame, annual_rate: float = 0.05, save: bool = True) -> pd.DataFrame:
    """Build a daily cash-rate series aligned to the SPY calendar."""
    cal = spy_df.index
    daily_return = (1.0 + annual_rate) ** (1.0 / 252) - 1.0
    df = pd.DataFrame({"rate": annual_rate, "return": daily_return}, index=cal)
    df.index.name = "date"
    if save:
        df.to_parquet(CASH_RATE_FILE, index=True)
        logger.info("Saved cash rate: %d rows → %s", len(df), CASH_RATE_FILE)
    return df


# ---------------------------------------------------------------------------
# 3.  Current option chain snapshot  (get_option_chain)
# ---------------------------------------------------------------------------

def fetch_current_chain(
    underlying_symbol: str = "SPY",
    spot: float | None = None,
    dte_min: int = TARGET_MIN_DTE,
    dte_max: int = TARGET_MAX_DTE,
    strike_pct_range: float = 0.15,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the live option chain from Alpaca (snapshot: bid, ask, IV, greeks).
    Saves to OPTION_CHAINS_DIR for SPY (flat), or OPTION_CHAINS_DIR/{symbol}/ for others.
    """
    from alpaca.data.requests import OptionChainRequest

    underlying_symbol = underlying_symbol.upper()
    client = _get_option_client()
    today = date.today()

    # Resolve spot from saved daily bars
    if spot is None:
        price_file = SPY_DAILY_FILE if underlying_symbol == "SPY" else RAW_DIR / f"{underlying_symbol.lower()}_daily.parquet"
        if price_file.exists():
            px = pd.read_parquet(price_file)
            spot = float(px["close"].iloc[-1])
    spot = spot or (550.0 if underlying_symbol == "SPY" else 200.0)
    underlying_ctx = _fetch_latest_underlying_snapshot(underlying_symbol, fallback_spot=float(spot))
    spot = float(underlying_ctx["underlying_spot"])

    exp_gte = (today + timedelta(days=dte_min)).isoformat()
    exp_lte = (today + timedelta(days=dte_max)).isoformat()

    request = OptionChainRequest(
        underlying_symbol=underlying_symbol,
        expiration_date_gte=exp_gte,
        expiration_date_lte=exp_lte,
        strike_price_gte=spot * (1.0 - strike_pct_range),
        strike_price_lte=spot * (1.0 + strike_pct_range),
    )
    chain = client.get_option_chain(request)
    logger.info("Fetched %s option chain: %d contracts", underlying_symbol, len(chain))

    rows = []
    for sym, snap in chain.items():
        parsed = parse_occ_symbol(sym)
        bid = snap.latest_quote.bid_price if snap.latest_quote else 0.0
        ask = snap.latest_quote.ask_price if snap.latest_quote else 0.0
        last_price = snap.latest_trade.price if snap.latest_trade else 0.0
        iv = snap.implied_volatility or 0.0
        delta = snap.greeks.delta if snap.greeks else None
        gamma = snap.greeks.gamma if snap.greeks else None
        rho = snap.greeks.rho if snap.greeks else None
        theta = snap.greeks.theta if snap.greeks else None
        vega = snap.greeks.vega if snap.greeks else None

        rows.append({
            "contractSymbol": sym,
            "expiry": parsed["expiry"].isoformat(),
            "strike": parsed["strike"],
            "lastPrice": last_price,
            "bid": bid,
            "ask": ask,
            "impliedVolatility": iv,
            "is_call": parsed["is_call"],
            "delta": delta,
            "gamma": gamma,
            "rho": rho,
            "theta": theta,
            "vega": vega,
            "data_source": "alpaca_snapshot",
            **underlying_ctx,
        })

    df = pd.DataFrame(rows)
    calls = df[df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)
    puts = df[~df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)

    if save:
        chain_dir = OPTION_CHAINS_DIR if underlying_symbol == "SPY" else OPTION_CHAINS_DIR / underlying_symbol
        chain_dir.mkdir(parents=True, exist_ok=True)
        date_str = today.isoformat()
        calls_path = chain_dir / f"calls_{date_str}.parquet"
        puts_path = chain_dir / f"puts_{date_str}.parquet"
        call_rows, call_snapshots = _append_intraday_snapshot_parquet(calls_path, calls)
        put_rows, put_snapshots = _append_intraday_snapshot_parquet(puts_path, puts)
        logger.info(
            "Saved %s chain %s @ %s: %d calls, %d puts (stored rows: %d/%d, snapshots today: %d/%d)",
            underlying_symbol,
            date_str,
            underlying_ctx["snapshot_time"],
            len(calls),
            len(puts),
            call_rows,
            put_rows,
            call_snapshots,
            put_snapshots,
        )
    return calls, puts


def fetch_latest_snapshots_for_symbols(
    option_symbols: Iterable[str],
    *,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch latest option snapshots for a concrete list of contract symbols.

    This is the cleanest forward-looking path for building a research-quality
    dataset with actual bid/ask, implied volatility, and greeks for the same
    contracts over time. Alpaca's historical bars do not include these fields.
    """
    from alpaca.data.requests import OptionSnapshotRequest

    symbols = [sym for sym in dict.fromkeys(option_symbols) if sym]
    if not symbols:
        return pd.DataFrame()

    client = _get_option_client()
    request = OptionSnapshotRequest(symbol_or_symbols=symbols)
    snaps = client.get_option_snapshot(request)

    rows = []
    for sym, snap in snaps.items():
        if snap is None:
            continue
        bid = snap.latest_quote.bid_price if snap.latest_quote else 0.0
        ask = snap.latest_quote.ask_price if snap.latest_quote else 0.0
        last_price = snap.latest_trade.price if snap.latest_trade else 0.0
        rows.append({
            "contractSymbol": sym,
            "lastPrice": last_price,
            "bid": bid,
            "ask": ask,
            "impliedVolatility": snap.implied_volatility,
            "delta": snap.greeks.delta if snap.greeks else None,
            "gamma": snap.greeks.gamma if snap.greeks else None,
            "rho": snap.greeks.rho if snap.greeks else None,
            "theta": snap.greeks.theta if snap.greeks else None,
            "vega": snap.greeks.vega if snap.greeks else None,
            "data_source": "alpaca_symbol_snapshot",
            "snapshot_ts": pd.Timestamp.utcnow(),
        })

    df = pd.DataFrame(rows)
    if save_path is not None and not df.empty:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info("Saved %d symbol snapshots → %s", len(df), save_path)
    return df


# ---------------------------------------------------------------------------
# 4.  Historical chain reconstruction from option bars
# ---------------------------------------------------------------------------

def _candidate_expiries(ref_date: date, min_dte: int, max_dte: int) -> List[date]:
    """Generate candidate SPY expiry dates (Mon/Wed/Fri within DTE range)."""
    candidates = []
    for delta in range(min_dte, max_dte + 1):
        d = ref_date + timedelta(days=delta)
        if d.weekday() in (0, 2, 4):  # Mon, Wed, Fri
            candidates.append(d)
    return candidates


def _generate_symbols(
    expiries: List[date],
    spot: float,
    underlying: str = "SPY",
    pct_range: float = 0.10,
    strike_step: float = 1.0,
) -> Tuple[List[str], List[str]]:
    """Generate OCC call and put symbols for strikes near ATM."""
    underlying = underlying.upper()
    lo = int(spot * (1 - pct_range))
    hi = int(spot * (1 + pct_range)) + 1
    strikes = np.arange(lo, hi, strike_step)

    call_syms, put_syms = [], []
    for exp in expiries:
        for k in strikes:
            call_syms.append(make_occ_symbol(underlying, exp, float(k), True))
            put_syms.append(make_occ_symbol(underlying, exp, float(k), False))
    return call_syms, put_syms


def _fetch_bars_batch(
    symbols: List[str],
    bar_date: date,
) -> pd.DataFrame:
    """Fetch daily bars for a list of option symbols on a single date.

    Returns a DataFrame with columns: symbol, open, high, low, close, volume.
    Batches into groups of _BATCH_SIZE to stay within API limits.
    """
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_option_client()
    start_dt = datetime.combine(bar_date, datetime.min.time())
    end_dt = datetime.combine(bar_date + timedelta(days=3), datetime.min.time())

    frames = []
    for i in range(0, len(symbols), _BATCH_SIZE):
        batch = symbols[i : i + _BATCH_SIZE]
        try:
            request = OptionBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
            )
            bars = client.get_option_bars(request)
            df = bars.df
            if len(df) > 0:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                    if "symbol" in df.columns:
                        df = df.rename(columns={"symbol": "contract"})
                frames.append(df)
        except Exception as e:
            logger.warning("Bars batch %d failed: %s", i // _BATCH_SIZE, e)
        time.sleep(_API_SLEEP)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Keep only bars on the target date
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined[combined["timestamp"].dt.date == bar_date]

    return combined


def fetch_historical_chain(
    chain_date: date,
    spot: float,
    underlying: str = "SPY",
    min_dte: int = TARGET_MIN_DTE,
    max_dte: int = TARGET_MAX_DTE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct the option chain for a past date by querying option bars.

    1. Generate candidate expiry dates (M/W/F within DTE range).
    2. Generate OCC symbols for ATM ± 10% strikes.
    3. Fetch daily bars on chain_date for those symbols.
    4. Parse into calls/puts DataFrames matching the format forecasts.py expects.
    Saves to OPTION_CHAINS_DIR (SPY) or OPTION_CHAINS_DIR/{underlying}/ for others.
    """
    underlying = underlying.upper()
    expiries = _candidate_expiries(chain_date, min_dte, max_dte)
    if not expiries:
        logger.warning("No candidate expiries for %s (DTE %d-%d)", chain_date, min_dte, max_dte)
        return pd.DataFrame(), pd.DataFrame()

    call_syms, put_syms = _generate_symbols(expiries, spot, underlying=underlying)
    all_syms = call_syms + put_syms
    logger.info(
        "Fetching bars for %s %s: spot=%.0f, %d expiries, %d symbols",
        underlying, chain_date, spot, len(expiries), len(all_syms),
    )

    bars_df = _fetch_bars_batch(all_syms, chain_date)
    if bars_df.empty:
        logger.warning("No option bars returned for %s", chain_date)
        return pd.DataFrame(), pd.DataFrame()

    # Identify which column holds the symbol
    sym_col = "contract" if "contract" in bars_df.columns else "symbol"
    if sym_col not in bars_df.columns:
        logger.warning("No symbol column in bars response; columns=%s", bars_df.columns.tolist())
        return pd.DataFrame(), pd.DataFrame()

    # Parse each bar into chain format
    rows = []
    for _, row in bars_df.iterrows():
        sym = row[sym_col]
        parsed = parse_occ_symbol(sym)
        close_price = row.get("close", 0.0)
        open_price = row.get("open", 0.0)
        mid = (open_price + close_price) / 2 if (open_price > 0 and close_price > 0) else close_price
        rows.append({
            "contractSymbol": sym,
            "expiry": parsed["expiry"].isoformat(),
            "strike": parsed["strike"],
            "lastPrice": close_price,
            "bid": 0.0,
            "ask": 0.0,
            # Historical bars do not include snapshot greeks or IV.
            # We keep explicit null fields so downstream cleaning can distinguish
            # "missing from source" from "forgot to populate".
            "impliedVolatility": None,
            "is_call": parsed["is_call"],
            "volume": row.get("volume", 0),
            "delta": None,
            "gamma": None,
            "rho": None,
            "theta": None,
            "vega": None,
            "data_source": "alpaca_bar",
        })

    chain_df = pd.DataFrame(rows)
    calls = chain_df[chain_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)
    puts = chain_df[~chain_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)

    logger.info(
        "Chain %s: %d calls, %d puts (from %d bars)",
        chain_date, len(calls), len(puts), len(bars_df),
    )

    if save:
        chain_dir = OPTION_CHAINS_DIR if underlying == "SPY" else OPTION_CHAINS_DIR / underlying
        chain_dir.mkdir(parents=True, exist_ok=True)
        date_str = chain_date.isoformat()
        calls.to_parquet(chain_dir / f"calls_{date_str}.parquet", index=False)
        puts.to_parquet(chain_dir / f"puts_{date_str}.parquet", index=False)

    return calls, puts


# ---------------------------------------------------------------------------
# 4b.  Daily backfill: every trading day, all DTE, wide strikes
# ---------------------------------------------------------------------------

def _fetch_bars_batch_safe(
    symbols: List[str],
    bar_date: date,
    sleep: float = _API_SLEEP,
) -> pd.DataFrame:
    """Like _fetch_bars_batch but with configurable sleep and 429 retry."""
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_option_client()
    start_dt = datetime.combine(bar_date, datetime.min.time())
    end_dt = datetime.combine(bar_date + timedelta(days=3), datetime.min.time())

    frames = []
    for i in range(0, len(symbols), _BATCH_SIZE):
        batch = symbols[i : i + _BATCH_SIZE]
        retries = 0
        while retries < 4:
            try:
                request = OptionBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start_dt,
                    end=end_dt,
                )
                bars = client.get_option_bars(request)
                df = bars.df
                if len(df) > 0:
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.reset_index()
                        if "symbol" in df.columns:
                            df = df.rename(columns={"symbol": "contract"})
                    frames.append(df)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "too many" in err_str.lower():
                    wait = sleep * (2 ** retries) + 1.0
                    logger.warning("Rate-limited (batch %d), retrying in %.1fs", i // _BATCH_SIZE, wait)
                    time.sleep(wait)
                    retries += 1
                else:
                    logger.warning("Bars batch %d failed: %s", i // _BATCH_SIZE, e)
                    break
        time.sleep(sleep)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined[combined["timestamp"].dt.date == bar_date]
    return combined


def _bars_to_chain_df(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw option bars into the chain DataFrame format."""
    sym_col = "contract" if "contract" in bars_df.columns else "symbol"
    if sym_col not in bars_df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in bars_df.iterrows():
        parsed = parse_occ_symbol(row[sym_col])
        close_price = row.get("close", 0.0)
        open_price = row.get("open", 0.0)
        rows.append({
            "contractSymbol": row[sym_col],
            "expiry": parsed["expiry"].isoformat(),
            "strike": parsed["strike"],
            "lastPrice": close_price,
            "bid": 0.0,
            "ask": 0.0,
            "impliedVolatility": None,
            "is_call": parsed["is_call"],
            "volume": row.get("volume", 0),
            "delta": None,
            "gamma": None,
            "rho": None,
            "theta": None,
            "vega": None,
            "data_source": "alpaca_bar",
        })
    return pd.DataFrame(rows)


def backfill_daily_chains(
    symbol: str = "SPY",
    start_date: str = OPTION_DATA_START,
    end_date: str | None = None,
    max_dte: int = _BACKFILL_MAX_DTE,
    min_dte: int = 0,
    strike_pct_range: float = _BACKFILL_STRIKE_PCT,
    strike_step: float = _BACKFILL_STRIKE_STEP,
    expiry_weekdays: tuple | None = None,
    api_sleep: float = 0.35,
    force: bool = False,
) -> dict:
    """
    Fetch option chain bars for **every trading day** in [start_date, end_date].

    This is the comprehensive backfill function — much wider coverage than the
    rebalance-date pipeline:

    - All DTE from *min_dte* to *max_dte* (default 0–90)
    - Strikes from spot*(1 - strike_pct_range) to spot*(1 + strike_pct_range)
    - All candidate expiry dates matching *expiry_weekdays*

    Incremental: dates that already have both calls_*.parquet and puts_*.parquet
    are skipped unless *force=True*.

    Parameters
    ----------
    symbol : str
        Underlying ticker (e.g. "SPY", "AAPL").
    start_date, end_date : str
        ISO date bounds.  end_date defaults to today.
    max_dte, min_dte : int
        Calendar-day DTE range for candidate expiries.
    strike_pct_range : float
        Fraction of spot for strike bounds (0.25 = ±25 %).
    strike_step : float
        Dollar increment between strikes.
    expiry_weekdays : tuple[int,...] | None
        Which weekdays are valid expiry dates (0=Mon … 4=Fri).
        Defaults to M/W/F for SPY, all weekdays for others.
    api_sleep : float
        Seconds between API batches (increase if rate-limited).
    force : bool
        Re-fetch even if files exist.

    Returns
    -------
    dict  with keys fetched, skipped, failed, total_calls, total_puts.
    """
    symbol = symbol.upper()
    if expiry_weekdays is None:
        expiry_weekdays = SPY_EXPIRY_WEEKDAYS if symbol == "SPY" else DEFAULT_EXPIRY_WEEKDAYS

    # Load equity bars for spot prices and trading calendar
    price_file = SPY_DAILY_FILE if symbol == "SPY" else RAW_DIR / f"{symbol.lower()}_daily.parquet"
    if not price_file.exists():
        raise FileNotFoundError(
            f"No equity bars for {symbol}. Run fetch_stock_bars('{symbol}') first."
        )
    prices = pd.read_parquet(price_file)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    chain_dir = OPTION_CHAINS_DIR if symbol == "SPY" else OPTION_CHAINS_DIR / symbol
    chain_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize()
    trading_days = prices.index[(prices.index >= start_ts) & (prices.index <= end_ts)].sort_values()

    # Detect already-fetched dates (both calls + puts must exist)
    existing: set[pd.Timestamp] = set()
    for p in chain_dir.glob("calls_*.parquet"):
        ds = p.name.split("_", 1)[1].replace(".parquet", "")
        try:
            existing.add(pd.Timestamp(ds))
        except Exception:
            pass

    to_fetch = [d for d in trading_days if force or d not in existing]
    cached = len(set(trading_days) & existing)

    logger.info(
        "Backfill %s: %d trading days in [%s, %s], %d cached, %d to fetch",
        symbol, len(trading_days), start_ts.date(), end_ts.date(), cached, len(to_fetch),
    )

    stats = {"fetched": 0, "skipped": 0, "failed": 0, "total_calls": 0, "total_puts": 0}
    empty_cols = [
        "contractSymbol", "expiry", "strike", "lastPrice",
        "bid", "ask", "impliedVolatility", "volume",
        "delta", "gamma", "rho", "theta", "vega", "data_source",
    ]

    for i, day in enumerate(to_fetch):
        if day not in prices.index:
            stats["skipped"] += 1
            continue

        spot = float(prices.loc[day, "close"])
        chain_date = day.date()

        # Generate candidate expiry dates within DTE range
        expiries = _candidate_expiries(chain_date, min_dte, max_dte)
        # Widen: also include expiry_weekdays not covered by original M/W/F helper
        all_expiries = []
        for delta in range(min_dte, max_dte + 1):
            d = chain_date + timedelta(days=delta)
            if d.weekday() in expiry_weekdays:
                all_expiries.append(d)
        if not all_expiries:
            stats["skipped"] += 1
            continue

        call_syms, put_syms = _generate_symbols(
            all_expiries, spot, underlying=symbol,
            pct_range=strike_pct_range, strike_step=strike_step,
        )
        all_syms = call_syms + put_syms

        logger.info(
            "[%d/%d] %s %s  spot=%.0f  %d expiries  %d symbols",
            i + 1, len(to_fetch), symbol, chain_date,
            spot, len(all_expiries), len(all_syms),
        )

        bars_df = _fetch_bars_batch_safe(all_syms, chain_date, sleep=api_sleep)

        if bars_df.empty:
            logger.warning("  No bars returned — saving empty chain files")
            stats["failed"] += 1
            pd.DataFrame(columns=empty_cols).to_parquet(
                chain_dir / f"calls_{chain_date.isoformat()}.parquet", index=False,
            )
            pd.DataFrame(columns=empty_cols).to_parquet(
                chain_dir / f"puts_{chain_date.isoformat()}.parquet", index=False,
            )
            continue

        chain_df = _bars_to_chain_df(bars_df)
        if chain_df.empty:
            stats["failed"] += 1
            continue

        calls = chain_df[chain_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)
        puts = chain_df[~chain_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)

        calls.to_parquet(chain_dir / f"calls_{chain_date.isoformat()}.parquet", index=False)
        puts.to_parquet(chain_dir / f"puts_{chain_date.isoformat()}.parquet", index=False)

        stats["fetched"] += 1
        stats["total_calls"] += len(calls)
        stats["total_puts"] += len(puts)
        logger.info("  Saved: %d calls, %d puts", len(calls), len(puts))

    logger.info("Backfill complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# 5.  Compute rebalance dates from SPY calendar
# ---------------------------------------------------------------------------

def compute_rebalance_dates(
    spy_df: pd.DataFrame,
    period_days: int = TARGET_IDEAL_DTE,
    start_date: str = OPTION_DATA_START,
) -> List[date]:
    """Pick rebalance dates every `period_days` trading days, starting from
    start_date or the first available date with option data."""
    cal = spy_df.index.sort_values()
    cal = cal[cal >= pd.Timestamp(start_date)]
    dates = []
    i = 0
    while i < len(cal):
        dates.append(cal[i].date())
        i += period_days
    return dates


# ---------------------------------------------------------------------------
# 6.  Full pipeline
# ---------------------------------------------------------------------------

def _extend_chain(
    chain_date: date,
    spot: float,
    new_min_dte: int,
    new_max_dte: int,
    underlying: str = "SPY",
):
    """Fetch additional expiries and merge into existing chain parquets."""
    underlying = underlying.upper()
    chain_dir = OPTION_CHAINS_DIR if underlying == "SPY" else OPTION_CHAINS_DIR / underlying
    date_str = chain_date.isoformat()
    calls_path = chain_dir / f"calls_{date_str}.parquet"
    puts_path = chain_dir / f"puts_{date_str}.parquet"

    old_calls = pd.read_parquet(calls_path) if calls_path.exists() else pd.DataFrame()
    old_puts = pd.read_parquet(puts_path) if puts_path.exists() else pd.DataFrame()

    # Determine which DTE ranges are already covered
    ref = pd.Timestamp(date_str)
    existing_dtes = set()
    for df in (old_calls, old_puts):
        if "expiry" in df.columns and len(df) > 0:
            for exp in df["expiry"].unique():
                existing_dtes.add((pd.Timestamp(exp) - ref).days)

    # Only fetch expiries not already present
    needed_expiries = []
    for exp in _candidate_expiries(chain_date, new_min_dte, new_max_dte):
        dte = (exp - chain_date).days
        if dte not in existing_dtes:
            needed_expiries.append(exp)

    if not needed_expiries:
        logger.debug("Chain %s already has DTE %d-%d covered", chain_date, new_min_dte, new_max_dte)
        return

    call_syms, put_syms = _generate_symbols(needed_expiries, spot, underlying=underlying)
    all_syms = call_syms + put_syms
    logger.info(
        "Extending chain %s: %d new expiries (%d symbols)",
        chain_date, len(needed_expiries), len(all_syms),
    )

    bars_df = _fetch_bars_batch(all_syms, chain_date)
    if bars_df.empty:
        logger.warning("No bars for extended expiries on %s", chain_date)
        return

    sym_col = "contract" if "contract" in bars_df.columns else "symbol"
    if sym_col not in bars_df.columns:
        return

    rows = []
    for _, row in bars_df.iterrows():
        parsed = parse_occ_symbol(row[sym_col])
        close_price = row.get("close", 0.0)
        open_price = row.get("open", 0.0)
        mid = (open_price + close_price) / 2 if (open_price > 0 and close_price > 0) else close_price
        rows.append({
            "contractSymbol": row[sym_col],
            "expiry": parsed["expiry"].isoformat(),
            "strike": parsed["strike"],
            "lastPrice": close_price,
            "bid": 0.0,
            "ask": 0.0,
            "impliedVolatility": 0.0,
            "is_call": parsed["is_call"],
            "volume": row.get("volume", 0),
        })

    new_df = pd.DataFrame(rows)
    new_calls = new_df[new_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)
    new_puts = new_df[~new_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)

    merged_calls = pd.concat([old_calls, new_calls], ignore_index=True).drop_duplicates(
        subset=["contractSymbol"], keep="last",
    )
    merged_puts = pd.concat([old_puts, new_puts], ignore_index=True).drop_duplicates(
        subset=["contractSymbol"], keep="last",
    )

    chain_dir.mkdir(parents=True, exist_ok=True)
    merged_calls.to_parquet(calls_path, index=False)
    merged_puts.to_parquet(puts_path, index=False)
    logger.info("Extended %s %s: now %d calls, %d puts", underlying, chain_date, len(merged_calls), len(merged_puts))


def run_pipeline_for_symbol(
    symbol: str,
    start: str = "2020-01-01",
    end: str | None = None,
    fetch_historical: bool = True,
    period_days: int = TARGET_IDEAL_DTE,
) -> None:
    """
    Fetch daily bars, current option chain, and (optionally) historical chains
    for a single underlying (e.g. AAPL, CRWD). Uses that symbol's calendar for
    rebalance dates. Does not build cash rate (use SPY pipeline for that).
    """
    symbol = symbol.upper()
    logger.info("=== Pipeline for %s ===", symbol)

    stock_df = fetch_stock_bars(symbol, start=start, end=end, save=True)
    if stock_df.empty:
        logger.warning("No bars for %s; skipping chain fetch", symbol)
        return

    spot = float(stock_df["close"].iloc[-1])
    logger.info("%s spot: %.2f", symbol, spot)
    fetch_current_chain(underlying_symbol=symbol, spot=spot, save=True)

    if fetch_historical:
        rebal_dates = compute_rebalance_dates(stock_df, period_days=period_days)
        logger.info("Rebalance dates: %d", len(rebal_dates))
        for i, rd in enumerate(rebal_dates):
            rd_ts = pd.Timestamp(rd)
            spot_row = stock_df.index.get_indexer([rd_ts], method="ffill")
            spot_rd = float(stock_df.iloc[spot_row[0]]["close"]) if spot_row[0] >= 0 else spot
            chain_dir = OPTION_CHAINS_DIR if symbol == "SPY" else OPTION_CHAINS_DIR / symbol
            chain_file = chain_dir / f"calls_{rd.isoformat()}.parquet"
            if chain_file.exists():
                logger.info("[%d/%d] %s — cached", i + 1, len(rebal_dates), rd)
                continue
            logger.info("[%d/%d] Fetching chain for %s %s (spot=%.0f)", i + 1, len(rebal_dates), symbol, rd, spot_rd)
            fetch_historical_chain(rd, spot_rd, underlying=symbol, save=True)
            time.sleep(_API_SLEEP)

    logger.info("=== %s pipeline complete ===", symbol)


def run_full_pipeline(
    spy_start: str = "2020-01-01",
    spy_end: str | None = None,
    fetch_historical: bool = True,
    period_days: int = TARGET_IDEAL_DTE,
    extend_dte: bool = False,
):
    """
    End-to-end Alpaca data fetch:
      1. SPY daily bars
      2. Cash rate series
      3. Current option chain snapshot
      4. Historical chain reconstructions for each rebalance date

    If extend_dte=True, also fetch longer-dated expiries for existing chains.
    """
    logger.info("=== Alpaca Data Pipeline ===")

    # --- SPY bars ---
    spy_df = fetch_spy_bars(start=spy_start, end=spy_end, save=True)

    # --- Cash rate ---
    build_cash_rate(spy_df, save=True)

    # --- Current chain ---
    spot = float(spy_df["close"].iloc[-1])
    logger.info("Current spot: %.2f", spot)
    fetch_current_chain(spot=spot, save=True)

    # --- Historical chains ---
    if fetch_historical:
        rebal_dates = compute_rebalance_dates(spy_df, period_days=period_days)
        logger.info("Rebalance dates: %d (from %s to %s)", len(rebal_dates), rebal_dates[0], rebal_dates[-1])

        for i, rd in enumerate(rebal_dates):
            rd_ts = pd.Timestamp(rd)
            spot_row = spy_df.index.get_indexer([rd_ts], method="ffill")
            spot_rd = float(spy_df.iloc[spot_row[0]]["close"]) if spot_row[0] >= 0 else spot

            chain_file = OPTION_CHAINS_DIR / f"calls_{rd.isoformat()}.parquet"
            if chain_file.exists() and not extend_dte:
                logger.info("[%d/%d] %s — cached, skipping", i + 1, len(rebal_dates), rd)
                continue

            if chain_file.exists() and extend_dte:
                logger.info("[%d/%d] %s — extending DTE range ...", i + 1, len(rebal_dates), rd)
                _extend_chain(rd, spot_rd, TARGET_MIN_DTE, TARGET_MAX_DTE, underlying="SPY")
                time.sleep(_API_SLEEP)
            else:
                logger.info("[%d/%d] Fetching chain for %s (spot=%.0f) ...", i + 1, len(rebal_dates), rd, spot_rd)
                fetch_historical_chain(rd, spot_rd, underlying="SPY", save=True)
                time.sleep(_API_SLEEP)

    logger.info("=== Pipeline complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch Alpaca data (SPY, AAPL, CRWD, etc.).")
    sub = parser.add_subparsers(dest="cmd")

    # --- legacy: rebalance-date pipeline ---
    p_legacy = sub.add_parser("pipeline", help="Original rebalance-date pipeline")
    p_legacy.add_argument("--symbols", nargs="+", default=["SPY"])
    p_legacy.add_argument("--start", default="2020-01-01")
    p_legacy.add_argument("--end", default=None)
    p_legacy.add_argument("--no-historical", action="store_true")
    p_legacy.add_argument("--period-days", type=int, default=TARGET_IDEAL_DTE)

    # --- backfill: every trading day ---
    p_bf = sub.add_parser("backfill", help="Backfill option chains for every trading day")
    p_bf.add_argument("--symbols", nargs="+", default=["SPY"],
                       help="Underlying symbols (e.g. SPY AAPL CRWD)")
    p_bf.add_argument("--start", default=OPTION_DATA_START,
                       help="First date to backfill (default: %(default)s)")
    p_bf.add_argument("--end", default=None,
                       help="Last date to backfill (default: today)")
    p_bf.add_argument("--max-dte", type=int, default=_BACKFILL_MAX_DTE,
                       help="Max DTE for expiry candidates (default: %(default)s)")
    p_bf.add_argument("--strike-pct", type=float, default=_BACKFILL_STRIKE_PCT,
                       help="Strike range as fraction of spot (default: %(default)s)")
    p_bf.add_argument("--strike-step", type=float, default=_BACKFILL_STRIKE_STEP,
                       help="Dollar step between strikes (default: %(default)s)")
    p_bf.add_argument("--api-sleep", type=float, default=0.35,
                       help="Seconds between API batches (default: %(default)s)")
    p_bf.add_argument("--force", action="store_true",
                       help="Re-fetch even if chain files already exist")
    p_bf.add_argument("--fetch-bars-first", action="store_true",
                       help="Fetch/update equity daily bars before backfilling chains")

    args = parser.parse_args()

    if args.cmd == "backfill":
        symbols = [s.upper() for s in args.symbols]
        for sym in symbols:
            if args.fetch_bars_first:
                logger.info("Fetching equity bars for %s ...", sym)
                fetch_stock_bars(sym, start="2020-01-01", end=args.end, save=True)
            backfill_daily_chains(
                symbol=sym,
                start_date=args.start,
                end_date=args.end,
                max_dte=args.max_dte,
                strike_pct_range=args.strike_pct,
                strike_step=args.strike_step,
                api_sleep=args.api_sleep,
                force=args.force,
            )
    else:
        # Default: legacy pipeline
        symbols = [s.upper() for s in (args.symbols if args.cmd else ["SPY"])]
        if "SPY" in symbols:
            run_full_pipeline(
                spy_start=getattr(args, "start", "2020-01-01"),
                spy_end=getattr(args, "end", None),
                fetch_historical=not getattr(args, "no_historical", False),
            )
            time.sleep(_API_SLEEP)
        for sym in symbols:
            if sym == "SPY":
                continue
            run_pipeline_for_symbol(
                sym,
                start=getattr(args, "start", "2020-01-01"),
                end=getattr(args, "end", None),
                fetch_historical=not getattr(args, "no_historical", False),
                period_days=getattr(args, "period_days", TARGET_IDEAL_DTE),
            )
            time.sleep(_API_SLEEP)
