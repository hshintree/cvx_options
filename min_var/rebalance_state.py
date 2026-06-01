"""
Fixed-period rebalance state tracker.

State is persisted to min_var_output/rebal_state.json so that consecutive
invocations of run_live.py share knowledge of the last rebalance without
requiring a long-running process.

Logic: rebalance when trading_days_since_last >= rebal_freq.
rebal_freq is a tuned hyperparameter loaded from best_params.json.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)

STATE_FILE = OUTPUT_DIR / "rebal_state.json"


def trading_days_since(last_date_str: str | None) -> int:
    """
    Count weekday business days between last_date and today.
    Returns 9999 if last_date_str is None (never rebalanced).
    Approximates NYSE trading days (does not account for holidays).
    """
    if last_date_str is None:
        return 9999

    today = pd.Timestamp.now().normalize()
    last  = pd.Timestamp(last_date_str).normalize()

    if last >= today:
        return 0

    bdays = pd.bdate_range(last + pd.Timedelta(days=1), today)
    return len(bdays)


def next_rebal_date(last_date_str: str, rebal_freq: int) -> date:
    """Return the approximate date when the next rebalance becomes due."""
    last = pd.Timestamp(last_date_str).normalize()
    # Advance rebal_freq business days from last
    bdays = pd.bdate_range(last + pd.Timedelta(days=1),
                           last + pd.Timedelta(days=rebal_freq * 2))
    if len(bdays) >= rebal_freq:
        return bdays[rebal_freq - 1].date()
    return (last + pd.Timedelta(days=rebal_freq)).date()


def load_state() -> dict:
    """Return {last_rebal_date: str | None}.  Returns defaults if file missing."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {"last_rebal_date": None}
    with open(STATE_FILE) as f:
        data = json.load(f)
    return data


def save_state(rebal_date: str | pd.Timestamp | date) -> None:
    """Persist last rebalance date after a successful rebalance."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rebal_date, (pd.Timestamp, date)):
        rebal_date = str(rebal_date)[:10]
    with open(STATE_FILE, "w") as f:
        json.dump({"last_rebal_date": rebal_date}, f, indent=2)
    logger.info("Rebalance state saved: last_rebal_date=%s", rebal_date)


def should_rebalance(rebal_freq: int) -> tuple[bool, str]:
    """
    Check whether a rebalance is due based on fixed-period logic.

    Parameters
    ----------
    rebal_freq : tuned hyperparameter (number of NYSE trading days between rebalances)

    Returns
    -------
    (bool, reason_string)
      True  → "N trading days since last rebal (threshold: rebal_freq)"
      True  → "no prior rebalance on record"
      False → "only N days since last rebal — next rebal: YYYY-MM-DD"
    """
    state     = load_state()
    last_date = state.get("last_rebal_date")
    n_days    = trading_days_since(last_date)

    if last_date is None:
        return True, "no prior rebalance on record"

    if n_days >= rebal_freq:
        return True, f"{n_days} trading days since last rebal (threshold: {rebal_freq})"

    nrd = next_rebal_date(last_date, rebal_freq)
    return False, f"only {n_days} day(s) since last rebal — next rebal: {nrd}"
