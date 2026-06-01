"""
Helpers for ingesting free historical option-chain datasets into a
research-clean daily panel.

The intended primary source is an OptionsDX-style historical chain export with
observed bid/ask, implied volatility, and Greeks. The ingestion layer is
source-agnostic so that other CSV/Parquet sources can be normalized into the
same internal schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from glob import glob
import logging
from pathlib import Path
import re
from typing import Any, Collection, Mapping, Optional, Sequence
import warnings

import numpy as np
import pandas as pd

from data.forecasts import (
    _bs_delta,
    _bs_gamma,
    _bs_implied_vol,
    _bs_rho,
    _bs_theta,
    _bs_vega,
)

logger = logging.getLogger(__name__)

DEFAULT_RF = 0.05
TRADING_DAYS_PER_YEAR = 252.0
_GREEK_COLUMNS = ["delta", "gamma", "theta", "vega", "rho"]
_CONTRACT_BUCKETS = ["call_1", "call_2", "put_1", "put_2"]
DEFAULT_ROLL_DTE = 7
_CANONICAL_COLUMNS = [
    "symbol",
    "quote_time",
    "quote_date",
    "expiry",
    "option_side",
    "contract_symbol",
    "strike",
    "dte_calendar",
    "underlying_spot",
    "bid",
    "ask",
    "mid",
    "last_price",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv_source",
    "greeks_source",
    "price_source",
    "data_source",
    "row_quality_flag",
    "ingested_at",
]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    aliases: Mapping[str, tuple[str, ...]]
    side_map: Mapping[str, str]


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "optionsdx_csv": ProviderSpec(
        name="optionsdx_csv",
        aliases={
            "symbol": ("underlying_symbol", "symbol", "underlying"),
            "quote_time": ("quote_unixtime", "quote_readtime", "quote_time", "datetime", "timestamp", "date"),
            "expiry": ("expiration", "expiry", "expiration_date", "expire_date", "expire_unix"),
            "option_side": ("option_type", "type", "side"),
            "contract_symbol": ("option_symbol", "contract_symbol", "symbol_option", "option"),
            "strike": ("strike", "strike_price"),
            "underlying_spot": ("underlying_price", "active_underlying_price", "underlying_last"),
            "bid": ("bid",),
            "ask": ("ask",),
            "last_price": ("last", "last_price", "close"),
            "volume": ("volume",),
            "open_interest": ("open_interest", "openinterest", "oi"),
            "implied_volatility": ("implied_volatility", "iv", "impl_vol"),
            "delta": ("delta",),
            "gamma": ("gamma",),
            "theta": ("theta",),
            "vega": ("vega",),
            "rho": ("rho",),
        },
        side_map={
            "c": "call",
            "call": "call",
            "calls": "call",
            "p": "put",
            "put": "put",
            "puts": "put",
        },
    ),
}


def _get_provider_spec(source: str) -> ProviderSpec:
    if source not in PROVIDER_SPECS:
        raise KeyError(f"Unsupported source {source!r}. Available sources: {sorted(PROVIDER_SPECS)}")
    return PROVIDER_SPECS[source]


def _infer_symbol_from_path(path: Path) -> Optional[str]:
    match = re.match(r"^([A-Za-z0-9.\-]+)_eod_", path.name)
    if not match:
        return None
    return match.group(1).upper()


def _resolve_input_paths(path_or_query: str, *, target_symbols: Optional[Collection[str]] = None) -> list[Path]:
    raw = Path(path_or_query).expanduser()
    if raw.exists():
        if raw.is_dir():
            paths = sorted(
                p for p in raw.rglob("*")
                if p.is_file() and (
                    p.suffix.lower() in {".csv", ".parquet", ".txt"}
                    or p.name.lower().endswith((".csv.gz", ".txt.gz"))
                )
            )
        else:
            paths = [raw]
    else:
        paths = [Path(p) for p in sorted(glob(path_or_query))]
    if target_symbols:
        allowed = {str(sym).upper() for sym in target_symbols}
        filtered: list[Path] = []
        for path in paths:
            inferred = _infer_symbol_from_path(path)
            if inferred is None or inferred in allowed:
                filtered.append(path)
        paths = filtered
    if not paths:
        raise FileNotFoundError(f"No files matched {path_or_query!r}")
    return paths


def _read_source_file(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes[-1:] == [".parquet"]:
        frame = pd.read_parquet(path)
    elif suffixes[-2:] == [".csv", ".gz"] or suffixes[-1:] == [".csv"]:
        frame = pd.read_csv(path, skipinitialspace=True)
    elif suffixes[-2:] == [".txt", ".gz"] or suffixes[-1:] == [".txt"]:
        frame = pd.read_csv(path, skipinitialspace=True)
    else:
        raise ValueError(f"Unsupported file type for {path}")
    frame.columns = [str(col).strip().strip("[]").lower() for col in frame.columns]
    return frame


def _infer_symbol_from_source_file(df: pd.DataFrame) -> pd.Series:
    source_series = pd.Series("", index=df.index, dtype="string")
    if "_source_file" in df.columns:
        source_series = df["_source_file"].astype("string")
    inferred = source_series.str.extract(r"^([A-Za-z0-9.\-]+)_eod_", expand=False)
    return inferred.str.upper()


def _extract_first(df: pd.DataFrame, aliases: Sequence[str], default: Any = np.nan) -> pd.Series:
    for alias in aliases:
        if alias in df.columns:
            return df[alias]
    return pd.Series(default, index=df.index)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _winsorize_series(series: pd.Series, *, lower_q: float = 0.10, upper_q: float = 0.90) -> pd.Series:
    if series.empty:
        return series.astype("float64")
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    if numeric.empty:
        return numeric
    if len(numeric) < 5:
        return numeric
    lo = float(numeric.quantile(lower_q))
    hi = float(numeric.quantile(upper_q))
    return numeric.clip(lower=lo, upper=hi)


def _parse_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        magnitude = float(numeric.abs().median()) if len(numeric) else 0.0
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        ts = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def _normalize_side(series: pd.Series, mapping: Mapping[str, str]) -> pd.Series:
    clean = series.astype(str).str.strip().str.lower()
    return clean.map(mapping)


def _contract_symbol_fallback(
    symbol: pd.Series,
    expiry: pd.Series,
    side: pd.Series,
    strike: pd.Series,
) -> pd.Series:
    expiry_txt = expiry.dt.strftime("%Y%m%d").fillna("00000000")
    strike_txt = strike.fillna(0.0).map(lambda x: f"{float(x):.4f}")
    return (
        symbol.fillna("UNKNOWN").astype(str)
        + "_"
        + expiry_txt
        + "_"
        + side.fillna("na").astype(str)
        + "_"
        + strike_txt
    )


def _is_optionsdx_paired_format(df: pd.DataFrame) -> bool:
    return {"c_bid", "c_ask", "p_bid", "p_ask", "strike"}.issubset(df.columns) and "option_type" not in df.columns


def _expand_optionsdx_paired_chain(df: pd.DataFrame) -> pd.DataFrame:
    symbol = _extract_first(df, ("underlying_symbol", "symbol", "underlying")).astype("string").str.strip().str.upper()
    inferred_symbol = _infer_symbol_from_source_file(df)
    symbol = symbol.where(symbol.notna() & (symbol != ""), inferred_symbol)

    quote_time = _extract_first(df, ("quote_readtime", "quote_unixtime", "quote_time", "quote_date"))
    expiry = _extract_first(df, ("expire_date", "expiration", "expiry", "expire_unix"))
    shared = {
        "underlying_symbol": symbol,
        "quote_time": quote_time,
        "expiration": expiry,
        "strike": _extract_first(df, ("strike",)),
        "underlying_price": _extract_first(df, ("underlying_last", "underlying_price")),
        "dte": _extract_first(df, ("dte",)),
        "_source_file": _extract_first(df, ("_source_file",), default=""),
    }

    calls = pd.DataFrame(
        {
            **shared,
            "option_type": "call",
            "bid": _extract_first(df, ("c_bid",)),
            "ask": _extract_first(df, ("c_ask",)),
            "last": _extract_first(df, ("c_last",)),
            "volume": _extract_first(df, ("c_volume",)),
            "open_interest": np.nan,
            "implied_volatility": _extract_first(df, ("c_iv",)),
            "delta": _extract_first(df, ("c_delta",)),
            "gamma": _extract_first(df, ("c_gamma",)),
            "theta": _extract_first(df, ("c_theta",)),
            "vega": _extract_first(df, ("c_vega",)),
            "rho": _extract_first(df, ("c_rho",)),
        }
    )
    puts = pd.DataFrame(
        {
            **shared,
            "option_type": "put",
            "bid": _extract_first(df, ("p_bid",)),
            "ask": _extract_first(df, ("p_ask",)),
            "last": _extract_first(df, ("p_last",)),
            "volume": _extract_first(df, ("p_volume",)),
            "open_interest": np.nan,
            "implied_volatility": _extract_first(df, ("p_iv",)),
            "delta": _extract_first(df, ("p_delta",)),
            "gamma": _extract_first(df, ("p_gamma",)),
            "theta": _extract_first(df, ("p_theta",)),
            "vega": _extract_first(df, ("p_vega",)),
            "rho": _extract_first(df, ("p_rho",)),
        }
    )
    return pd.concat([calls, puts], ignore_index=True, sort=False)


def _compute_row_quality(df: pd.DataFrame) -> pd.Series:
    quality = pd.Series("ok", index=df.index, dtype="object")
    quality = quality.mask(df["quote_time"].isna(), "missing_quote_time")
    quality = quality.mask((quality == "ok") & df["expiry"].isna(), "missing_expiry")
    quality = quality.mask(
        (quality == "ok") & ((df["underlying_spot"].isna()) | (df["underlying_spot"] <= 0)),
        "missing_underlying",
    )
    quality = quality.mask(
        (quality == "ok")
        & (
            df["bid"].isna()
            | df["ask"].isna()
            | (df["bid"] <= 0)
            | (df["ask"] <= 0)
            | (df["ask"] < df["bid"])
        ),
        "missing_bid_ask",
    )
    return quality


def _latest_daily_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ordered = df.sort_values(["symbol", "quote_date", "contract_symbol", "quote_time"])
    return ordered.drop_duplicates(["symbol", "quote_date", "contract_symbol"], keep="last").reset_index(drop=True)


def _is_greek_complete(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return np.isfinite(df[_GREEK_COLUMNS]).all(axis=1)


def _screen_contracts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["rel_spread"] = np.where(
        (out["mid"] > 0) & out["bid"].notna() & out["ask"].notna(),
        (out["ask"] - out["bid"]) / out["mid"],
        np.nan,
    )
    out["moneyness"] = np.where(
        out["underlying_spot"] > 0,
        out["strike"] / out["underlying_spot"] - 1.0,
        np.nan,
    )
    out["screen_pass"] = (
        out["dte_calendar"].between(7, 35)
        & (out["mid"] > 0.25)
        & (out["bid"] > 0)
        & (out["ask"] > 0)
        & (out["ask"] >= out["bid"])
        & (out["rel_spread"] <= 0.15)
        & (out["implied_volatility"] > 0)
        & np.isfinite(out["moneyness"])
        & (out["moneyness"].abs() <= 0.05)
        & _is_greek_complete(out)
    )
    return out


def _normalize_theta(theta: float, *, theta_is_per_year: bool, trading_days: float = 1.0) -> float:
    if pd.isna(theta):
        return np.nan
    return float(theta / TRADING_DAYS_PER_YEAR * trading_days) if theta_is_per_year else float(theta * trading_days)


def _resolve_risk_free(df: pd.DataFrame, rf_series: Optional[pd.Series]) -> pd.Series:
    if rf_series is None:
        return pd.Series(DEFAULT_RF, index=df.index, dtype="float64")
    if np.isscalar(rf_series):
        return pd.Series(float(rf_series), index=df.index, dtype="float64")
    if not isinstance(rf_series, pd.Series):
        raise TypeError("rf_series must be None, a scalar, or a pandas Series")
    rate = rf_series.copy()
    rate.index = pd.to_datetime(rate.index, utc=True, errors="coerce").tz_convert("UTC").tz_localize(None).normalize()
    date_key = pd.to_datetime(df["quote_date"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    values = date_key.map(rate)
    return values.fillna(method="ffill").fillna(method="bfill").fillna(DEFAULT_RF)


def _select_nearest_atm(df: pd.DataFrame, spot: float) -> pd.Series:
    ranked = df.assign(
        _atm_abs=(df["strike"] - spot).abs(),
        _oi=df["open_interest"].fillna(0.0),
    ).sort_values(
        ["_atm_abs", "rel_spread", "_oi", "contract_symbol"],
        ascending=[True, True, False, True],
    )
    return ranked.iloc[0]


def _select_next_call(call_df: pd.DataFrame, anchor_strike: float) -> Optional[pd.Series]:
    wing = call_df[call_df["strike"] > anchor_strike]
    if wing.empty:
        return None
    ranked = wing.assign(
        _wing_gap=wing["strike"] - anchor_strike,
        _oi=wing["open_interest"].fillna(0.0),
    ).sort_values(
        ["_wing_gap", "rel_spread", "_oi", "contract_symbol"],
        ascending=[True, True, False, True],
    )
    return ranked.iloc[0]


def _select_next_put(put_df: pd.DataFrame, anchor_strike: float) -> Optional[pd.Series]:
    wing = put_df[put_df["strike"] < anchor_strike]
    if wing.empty:
        return None
    ranked = wing.assign(
        _wing_gap=anchor_strike - wing["strike"],
        _oi=wing["open_interest"].fillna(0.0),
    ).sort_values(
        ["_wing_gap", "rel_spread", "_oi", "contract_symbol"],
        ascending=[True, True, False, True],
    )
    return ranked.iloc[0]


def _contract_key(symbol: str, right: str, expiry: Any, strike: float) -> tuple[Any, ...]:
    return (
        str(symbol).upper(),
        str(right).lower(),
        pd.Timestamp(expiry).normalize(),
        float(strike),
    )


def _contract_key_to_string(key: Any) -> Any:
    if key is None or (isinstance(key, float) and np.isnan(key)):
        return None
    if isinstance(key, tuple):
        if len(key) != 4:
            return repr(key)
        symbol, right, expiry, strike = key
        expiry_str = "" if expiry is None or pd.isna(expiry) else pd.Timestamp(expiry).normalize().strftime("%Y-%m-%d")
        strike_str = "" if strike is None or pd.isna(strike) else f"{float(strike):.4f}"
        return f"{str(symbol).upper()}|{str(right).lower()}|{expiry_str}|{strike_str}"
    return key


def serialize_contract_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if "contract_key" in col:
            out[col] = out[col].map(_contract_key_to_string)
    return out


def _contract_key_from_row(row: pd.Series) -> tuple[Any, ...]:
    return _contract_key(str(row["symbol"]), str(row["option_side"]), row["expiry"], float(row["strike"]))


def _is_hold_row_valid(row: pd.Series) -> bool:
    return bool(
        pd.notna(row.get("mid"))
        and float(row["mid"]) > 0
        and pd.notna(row.get("implied_volatility"))
        and float(row["implied_volatility"]) > 0
        and np.isfinite(pd.Series([row.get(col) for col in _GREEK_COLUMNS], dtype="float64")).all()
    )


def _select_quartet_for_group(group: pd.DataFrame, *, target_dte: int) -> tuple[Optional[dict[str, pd.Series]], str]:
    eligible = group[group["screen_pass"]].copy()
    if eligible.empty:
        return None, "no_valid_expiry"

    expiry_stats: list[dict[str, Any]] = []
    for expiry, expiry_df in eligible.groupby("expiry", sort=True):
        calls = expiry_df[expiry_df["option_side"] == "call"]
        puts = expiry_df[expiry_df["option_side"] == "put"]
        if len(calls) < 1 or len(puts) < 1:
            continue
        median_dte = float(expiry_df["dte_calendar"].median())
        expiry_stats.append(
            {
                "expiry": expiry,
                "median_dte": median_dte,
                "contract_count": int(len(calls) + len(puts)),
                "avg_spread": float(expiry_df["rel_spread"].mean()),
            }
        )

    if not expiry_stats:
        return None, "no_valid_expiry"
    if len(expiry_stats) < 2:
        return None, "no_second_expiry"

    expiry_rank = pd.DataFrame(expiry_stats).sort_values(
        ["median_dte", "contract_count", "avg_spread", "expiry"],
        ascending=[False, False, True, True],
    )
    chosen_expiries = expiry_rank.iloc[:2]["expiry"].tolist()
    spot = float(eligible["underlying_spot"].median())

    chosen_1 = eligible[eligible["expiry"] == chosen_expiries[0]].copy()
    chosen_2 = eligible[eligible["expiry"] == chosen_expiries[1]].copy()

    calls_1 = chosen_1[chosen_1["option_side"] == "call"].copy()
    puts_1 = chosen_1[chosen_1["option_side"] == "put"].copy()
    calls_2 = chosen_2[chosen_2["option_side"] == "call"].copy()
    puts_2 = chosen_2[chosen_2["option_side"] == "put"].copy()

    return {
        "call_1": _select_nearest_atm(calls_1, spot),
        "put_1": _select_nearest_atm(puts_1, spot),
        "call_2": _select_nearest_atm(calls_2, spot),
        "put_2": _select_nearest_atm(puts_2, spot),
    }, "ok"


def load_raw_historical_chain(
    source: str,
    path_or_query: str,
    *,
    symbol_map: Optional[dict[str, str]] = None,
    target_symbols: Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """
    Load one or more local files for a supported free historical chain source.

    The loader is intentionally file-based: free datasets tend to be distributed
    as downloadable archives rather than stable authenticated APIs.
    """
    _get_provider_spec(source)
    frames: list[pd.DataFrame] = []
    for path in _resolve_input_paths(path_or_query, target_symbols=target_symbols):
        frame = _read_source_file(path)
        frame["_source_file"] = path.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if symbol_map:
        for key in ("underlying_symbol", "symbol", "underlying"):
            if key in raw.columns:
                raw[key] = raw[key].replace(symbol_map)
    return raw


def normalize_historical_chain(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """
    Normalize a provider-specific historical chain file into the canonical schema.
    """
    spec = _get_provider_spec(source)
    if df.empty:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    if source == "optionsdx_csv" and _is_optionsdx_paired_format(df):
        df = _expand_optionsdx_paired_chain(df)

    symbol_raw = _extract_first(df, spec.aliases["symbol"])
    symbol = symbol_raw.astype("string").str.strip().str.upper()
    inferred_symbol = _infer_symbol_from_source_file(df)
    symbol = symbol.where(symbol.notna() & (symbol != ""), inferred_symbol)
    quote_time = _parse_timestamp(_extract_first(df, spec.aliases["quote_time"]))
    quote_date = quote_time.dt.normalize()
    expiry = _parse_timestamp(_extract_first(df, spec.aliases["expiry"])).dt.normalize()
    option_side = _normalize_side(_extract_first(df, spec.aliases["option_side"]), spec.side_map)
    strike = _to_numeric(_extract_first(df, spec.aliases["strike"]))
    underlying_spot = _to_numeric(_extract_first(df, spec.aliases["underlying_spot"]))
    bid = _to_numeric(_extract_first(df, spec.aliases["bid"]))
    ask = _to_numeric(_extract_first(df, spec.aliases["ask"]))
    last_price = _to_numeric(_extract_first(df, spec.aliases["last_price"]))
    volume = _to_numeric(_extract_first(df, spec.aliases["volume"])).fillna(0.0)
    open_interest = _to_numeric(_extract_first(df, spec.aliases["open_interest"])).fillna(0.0)
    implied_volatility = _to_numeric(_extract_first(df, spec.aliases["implied_volatility"]))
    delta = _to_numeric(_extract_first(df, spec.aliases["delta"]))
    gamma = _to_numeric(_extract_first(df, spec.aliases["gamma"]))
    theta = _to_numeric(_extract_first(df, spec.aliases["theta"]))
    vega = _to_numeric(_extract_first(df, spec.aliases["vega"]))
    rho = _to_numeric(_extract_first(df, spec.aliases["rho"]))

    mid = np.where(
        bid.notna() & ask.notna() & (bid > 0) & (ask > 0) & (ask >= bid),
        (bid + ask) / 2.0,
        np.nan,
    )
    mid = pd.Series(mid, index=df.index, dtype="float64")

    contract_symbol_raw = _extract_first(df, spec.aliases["contract_symbol"])
    contract_symbol = pd.Series(contract_symbol_raw, index=df.index, dtype="object")
    contract_symbol = contract_symbol.where(contract_symbol.notna(), _contract_symbol_fallback(symbol, expiry, option_side, strike))
    contract_symbol = contract_symbol.astype(str)

    dte_calendar = (expiry - quote_date).dt.days.astype("float64")
    iv_observed = implied_volatility.where(implied_volatility > 0)
    greek_complete = np.isfinite(pd.concat([delta, gamma, theta, vega, rho], axis=1)).all(axis=1)
    price_source = np.where(mid.notna(), "mid", np.where(last_price > 0, "last", "missing"))

    normalized = pd.DataFrame(
        {
            "symbol": symbol,
            "quote_time": quote_time,
            "quote_date": quote_date,
            "expiry": expiry,
            "option_side": option_side,
            "contract_symbol": contract_symbol,
            "strike": strike,
            "dte_calendar": dte_calendar,
            "underlying_spot": underlying_spot,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "last_price": last_price,
            "volume": volume,
            "open_interest": open_interest,
            "implied_volatility": iv_observed,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "iv_source": np.where(iv_observed.notna(), "observed", "missing"),
            "greeks_source": np.where(greek_complete, "observed", "missing"),
            "price_source": price_source,
            "data_source": spec.name,
            "ingested_at": pd.Timestamp.utcnow().tz_localize(None),
        }
    )
    normalized["row_quality_flag"] = _compute_row_quality(normalized)
    normalized = normalized.dropna(subset=["symbol", "option_side", "contract_symbol", "strike"]).reset_index(drop=True)
    return normalized[_CANONICAL_COLUMNS]


def compute_missing_greeks_from_mid(
    df: pd.DataFrame,
    *,
    rf_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Fill missing implied vol and Greeks when bid/ask (preferred) or last exists.

    Bid/ask-derived rows are marked as inferred from mid. Last-price fallback is
    kept for diagnostics but remains excluded from the research-clean panel.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    rates = _resolve_risk_free(out, rf_series)
    out["mid"] = np.where(
        out["mid"].notna(),
        out["mid"],
        np.where(
            out["bid"].notna() & out["ask"].notna() & (out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"]),
            (out["bid"] + out["ask"]) / 2.0,
            np.nan,
        ),
    )

    needs_fill = (
        out["implied_volatility"].isna()
        | out[_GREEK_COLUMNS].isna().any(axis=1)
    ) & (
        ((out["mid"].notna()) & (out["mid"] > 0))
        | ((out["last_price"].notna()) & (out["last_price"] > 0))
    ) & (
        out["underlying_spot"].notna() & (out["underlying_spot"] > 0)
    ) & (
        out["strike"].notna() & (out["strike"] > 0)
    ) & (
        out["expiry"].notna() & out["quote_date"].notna()
    )

    for idx in out.index[needs_fill]:
        row = out.loc[idx]
        quote_source = None
        price = np.nan
        if pd.notna(row["mid"]) and row["mid"] > 0:
            price = float(row["mid"])
            quote_source = "mid"
        elif pd.notna(row["last_price"]) and row["last_price"] > 0:
            price = float(row["last_price"])
            quote_source = "last"

        if quote_source is None:
            continue
        if pd.isna(row["underlying_spot"]) or row["underlying_spot"] <= 0:
            continue
        if pd.isna(row["strike"]) or row["strike"] <= 0:
            continue
        if pd.isna(row["expiry"]) or pd.isna(row["quote_date"]):
            continue

        T = max((pd.Timestamp(row["expiry"]) - pd.Timestamp(row["quote_date"])).days / 365.0, 0.0)
        if T <= 0:
            continue
        is_call = row["option_side"] == "call"
        sigma = row["implied_volatility"] if pd.notna(row["implied_volatility"]) and row["implied_volatility"] > 0 else None
        if sigma is None:
            sigma = _bs_implied_vol(
                price,
                float(row["underlying_spot"]),
                float(row["strike"]),
                float(rates.loc[idx]),
                T,
                is_call=is_call,
            )
            if sigma is None:
                continue
            out.at[idx, "implied_volatility"] = sigma
            out.at[idx, "iv_source"] = "inferred_mid" if quote_source == "mid" else "inferred_last"

        computed_any = False
        if pd.isna(row["delta"]):
            out.at[idx, "delta"] = _bs_delta(float(row["underlying_spot"]), float(row["strike"]), float(rates.loc[idx]), T, float(sigma), is_call)
            computed_any = True
        if pd.isna(row["gamma"]):
            out.at[idx, "gamma"] = _bs_gamma(float(row["underlying_spot"]), float(row["strike"]), float(rates.loc[idx]), T, float(sigma))
            computed_any = True
        if pd.isna(row["theta"]):
            out.at[idx, "theta"] = _bs_theta(float(row["underlying_spot"]), float(row["strike"]), float(rates.loc[idx]), T, float(sigma), is_call)
            computed_any = True
        if pd.isna(row["vega"]):
            out.at[idx, "vega"] = _bs_vega(float(row["underlying_spot"]), float(row["strike"]), float(rates.loc[idx]), T, float(sigma))
            computed_any = True
        if pd.isna(row["rho"]):
            out.at[idx, "rho"] = _bs_rho(float(row["underlying_spot"]), float(row["strike"]), float(rates.loc[idx]), T, float(sigma), is_call)
            computed_any = True
        if computed_any:
            out.at[idx, "greeks_source"] = "inferred_mid" if quote_source == "mid" else "inferred_last"

    out["row_quality_flag"] = _compute_row_quality(out)
    return out


def summarize_source_coverage(
    df: pd.DataFrame,
    *,
    target_symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Summarize symbol coverage, quote quality, and observed/inferred field shares.
    """
    if df.empty:
        base = pd.DataFrame(columns=[
            "symbol", "available", "n_rows", "n_dates", "valid_bid_ask_share",
            "mid_share", "observed_iv_share", "observed_greek_share",
            "inferred_mid_share", "inferred_last_share", "latest_quote_date",
            "latest_quote_time",
        ])
        if target_symbols:
            base = pd.DataFrame({"symbol": [s.upper() for s in target_symbols]})
            base["available"] = False
        return base

    work = df.copy()
    work["valid_bid_ask"] = (
        work["bid"].notna()
        & work["ask"].notna()
        & (work["bid"] > 0)
        & (work["ask"] > 0)
        & (work["ask"] >= work["bid"])
    )
    work["has_mid"] = work["mid"].notna() & (work["mid"] > 0)
    work["observed_iv"] = work["iv_source"] == "observed"
    work["observed_greeks"] = work["greeks_source"] == "observed"
    work["inferred_mid"] = work["greeks_source"] == "inferred_mid"
    work["inferred_last"] = work["greeks_source"] == "inferred_last"

    summary = (
        work.groupby("symbol", as_index=False)
        .agg(
            available=("symbol", "size"),
            n_rows=("symbol", "size"),
            n_dates=("quote_date", "nunique"),
            valid_bid_ask_share=("valid_bid_ask", "mean"),
            mid_share=("has_mid", "mean"),
            observed_iv_share=("observed_iv", "mean"),
            observed_greek_share=("observed_greeks", "mean"),
            inferred_mid_share=("inferred_mid", "mean"),
            inferred_last_share=("inferred_last", "mean"),
            latest_quote_date=("quote_date", "max"),
            latest_quote_time=("quote_time", "max"),
        )
    )
    summary["available"] = summary["n_rows"] > 0

    if target_symbols:
        target_df = pd.DataFrame({"symbol": [s.upper() for s in target_symbols]})
        summary = target_df.merge(summary, on="symbol", how="left")
        summary["available"] = summary["available"].astype("boolean").fillna(False).astype(bool)
        for col in ("n_rows", "n_dates"):
            summary[col] = summary[col].fillna(0).astype(int)
        share_cols = [c for c in summary.columns if c.endswith("_share")]
        summary[share_cols] = summary[share_cols].fillna(0.0)

    return summary.sort_values("symbol").reset_index(drop=True)


def build_daily_research_candidates(
    df: pd.DataFrame,
    *,
    target_dte: int = 21,
    roll_dte: int = DEFAULT_ROLL_DTE,
) -> pd.DataFrame:
    """
    Build one four-contract (2 calls + 2 puts) candidate row per symbol-date.

    The bucket identifiers stay stable through time: ``call_1``/``put_1`` are
    the later-dated expiry pair, and ``call_2``/``put_2`` are the next-later
    expiry pair. Each symbol keeps the same selected contracts until a roll is
    required because DTE drops below ``roll_dte`` or the current contract data
    is missing.
    """
    latest = _screen_contracts(_latest_daily_rows(df))
    if latest.empty:
        out = pd.DataFrame()
        out.attrs["failure_counts"] = {}
        return out
    latest = latest.copy()
    latest["contract_key"] = latest.apply(_contract_key_from_row, axis=1)

    failure_counts = {
        "no_valid_expiry": 0,
        "no_second_expiry": 0,
        "roll_missing_data": 0,
        "roll_dte_trigger": 0,
    }
    rows: list[dict[str, Any]] = []

    for symbol, symbol_df in latest.groupby("symbol", sort=True):
        symbol_df = symbol_df.sort_values(["quote_date", "quote_time", "contract_symbol"]).reset_index(drop=True)
        current_keys: Optional[dict[str, tuple[Any, ...]]] = None

        for quote_date, group in symbol_df.groupby("quote_date", sort=True):
            group_lookup = {row["contract_key"]: row for _, row in group.iterrows()}
            bucket_rows: Optional[dict[str, pd.Series]] = None
            rolled = False
            roll_reason: Optional[str] = None

            if current_keys is not None:
                candidate_rows: dict[str, pd.Series] = {}
                for bucket in _CONTRACT_BUCKETS:
                    key = current_keys[bucket]
                    if key not in group_lookup:
                        roll_reason = "missing_data"
                        break
                    row = group_lookup[key]
                    if not _is_hold_row_valid(row):
                        roll_reason = "missing_data"
                        break
                    if float(row["dte_calendar"]) < float(roll_dte):
                        roll_reason = "roll_dte"
                        break
                    candidate_rows[bucket] = row
                if roll_reason is None:
                    bucket_rows = candidate_rows

            if current_keys is None or roll_reason is not None:
                if roll_reason == "missing_data":
                    failure_counts["roll_missing_data"] += 1
                elif roll_reason == "roll_dte":
                    failure_counts["roll_dte_trigger"] += 1
                selected_quartet, status = _select_quartet_for_group(group, target_dte=target_dte)
                if selected_quartet is None:
                    if status in failure_counts:
                        failure_counts[status] += 1
                    current_keys = None
                    continue
                bucket_rows = selected_quartet
                current_keys = {bucket: _contract_key_from_row(row) for bucket, row in bucket_rows.items()}
                rolled = True
                roll_reason = "initial" if roll_reason is None else roll_reason

            if bucket_rows is None or current_keys is None:
                continue

            chosen_expiry = pd.Timestamp(bucket_rows["call_1"]["expiry"]).normalize()
            chosen_dte = float(bucket_rows["call_1"]["dte_calendar"])
            chosen_expiry_2 = pd.Timestamp(bucket_rows["call_2"]["expiry"]).normalize()
            chosen_dte_2 = float(bucket_rows["call_2"]["dte_calendar"])
            spot = float(np.median([float(bucket_rows[b]["underlying_spot"]) for b in _CONTRACT_BUCKETS]))
            selected = {
                "symbol": symbol,
                "quote_date": quote_date,
                "quote_time": max(pd.Timestamp(bucket_rows[b]["quote_time"]) for b in _CONTRACT_BUCKETS),
                "chosen_expiry": chosen_expiry,
                "chosen_dte": chosen_dte,
                "chosen_expiry_2": chosen_expiry_2,
                "chosen_dte_2": chosen_dte_2,
                "spot_t": spot,
                "rolled": rolled,
                "roll_reason": roll_reason or "",
            }

            for bucket in _CONTRACT_BUCKETS:
                row = bucket_rows[bucket]
                prefix = f"{bucket}_"
                selected[prefix + "symbol"] = row["contract_symbol"]
                selected[prefix + "contract_key"] = current_keys[bucket]
                selected[prefix + "strike"] = float(row["strike"])
                selected[prefix + "right"] = row["option_side"]
                selected[prefix + "expiry"] = pd.Timestamp(row["expiry"]).normalize()
                selected[prefix + "mid_t"] = float(row["mid"])
                selected[prefix + "iv_t"] = float(row["implied_volatility"])
                selected[prefix + "delta_t"] = float(row["delta"])
                selected[prefix + "gamma_t"] = float(row["gamma"])
                selected[prefix + "theta_t"] = float(row["theta"])
                selected[prefix + "vega_t"] = float(row["vega"])
                selected[prefix + "rho_t"] = float(row["rho"])
                selected[prefix + "rel_spread"] = float(row["rel_spread"])
                selected[prefix + "moneyness"] = float(row["moneyness"])
                selected[prefix + "iv_source_t"] = row["iv_source"]
                selected[prefix + "greeks_source_t"] = row["greeks_source"]
                selected[prefix + "dte_t"] = float(row["dte_calendar"])
            rows.append(selected)

    out = pd.DataFrame(rows).sort_values(["symbol", "quote_date"]).reset_index(drop=True) if rows else pd.DataFrame()
    out.attrs["failure_counts"] = failure_counts
    return out


def build_research_clean_panel(
    df: pd.DataFrame,
    *,
    target_dte: int = 21,
    roll_dte: int = DEFAULT_ROLL_DTE,
) -> pd.DataFrame:
    """
    Match the four selected contracts forward to the next trading day.
    """
    latest = _latest_daily_rows(df)
    latest = latest.copy()
    latest["contract_key"] = latest.apply(_contract_key_from_row, axis=1)
    candidates = build_daily_research_candidates(latest, target_dte=target_dte, roll_dte=roll_dte)
    if candidates.empty:
        out = pd.DataFrame()
        out.attrs["failure_counts"] = {
            **candidates.attrs.get("failure_counts", {}),
            "no_next_trading_day": 0,
            "missing_contract_t1": 0,
            "invalid_mid_t1": 0,
        }
        return out

    lookup = latest.set_index(["symbol", "quote_date", "contract_symbol"], drop=False)
    failure_counts = {
        **candidates.attrs.get("failure_counts", {}),
        "no_next_trading_day": 0,
        "missing_contract_t1": 0,
        "invalid_mid_t1": 0,
    }
    rows: list[dict[str, Any]] = []

    for _, symbol_candidates in candidates.groupby("symbol", sort=True):
        symbol_candidates = symbol_candidates.sort_values("quote_date").reset_index(drop=True)
        for idx in range(len(symbol_candidates) - 1):
            current = symbol_candidates.iloc[idx]
            nxt = symbol_candidates.iloc[idx + 1]
            next_date = nxt["quote_date"]
            panel_row: dict[str, Any] = {
                "symbol": current["symbol"],
                "date_t": current["quote_date"],
                "date_t1": next_date,
                "spot_t": float(current["spot_t"]),
                "spot_t1": float(nxt["spot_t"]),
                "R_und_real": float(nxt["spot_t"] / current["spot_t"] - 1.0),
                "chosen_expiry_t": current["chosen_expiry"],
                "chosen_dte_t": float(current["chosen_dte"]),
                "chosen_expiry_2_t": current.get("chosen_expiry_2", pd.NaT),
                "chosen_dte_2_t": float(current.get("chosen_dte_2", np.nan)) if pd.notna(current.get("chosen_dte_2", np.nan)) else np.nan,
                "rolled_t": bool(current.get("rolled", False)),
                "roll_reason_t": current.get("roll_reason", ""),
                "rolled_t1": bool(nxt.get("rolled", False)),
                "roll_reason_t1": nxt.get("roll_reason", ""),
            }
            missing_contract = False
            invalid_mid = False
            for bucket in _CONTRACT_BUCKETS:
                prefix = f"{bucket}_"
                contract_symbol = current[prefix + "symbol"]
                current_key = current[prefix + "contract_key"]
                try:
                    next_row = lookup.loc[(current["symbol"], next_date, contract_symbol)]
                except KeyError:
                    missing_contract = True
                    break
                if isinstance(next_row, pd.DataFrame):
                    next_row = next_row.sort_values("quote_time").iloc[-1]
                if pd.isna(next_row["mid"]) or next_row["mid"] <= 0:
                    invalid_mid = True
                    break

                panel_row[prefix + "symbol"] = contract_symbol
                panel_row[prefix + "contract_key"] = current_key
                panel_row[prefix + "next_symbol"] = nxt[prefix + "symbol"]
                panel_row[prefix + "next_contract_key"] = nxt[prefix + "contract_key"]
                panel_row[prefix + "rolled_next"] = bool(current_key != nxt[prefix + "contract_key"])
                panel_row[prefix + "strike"] = float(current[prefix + "strike"])
                panel_row[prefix + "mid_t"] = float(current[prefix + "mid_t"])
                panel_row[prefix + "mid_t1"] = float(next_row["mid"])
                panel_row[prefix + "iv_t"] = float(current[prefix + "iv_t"])
                panel_row[prefix + "iv_t1"] = float(next_row["implied_volatility"]) if pd.notna(next_row["implied_volatility"]) else np.nan
                panel_row[prefix + "delta_t"] = float(current[prefix + "delta_t"])
                panel_row[prefix + "gamma_t"] = float(current[prefix + "gamma_t"])
                panel_row[prefix + "theta_t"] = float(current[prefix + "theta_t"])
                panel_row[prefix + "vega_t"] = float(current[prefix + "vega_t"])
                panel_row[prefix + "rho_t"] = float(current[prefix + "rho_t"])
                panel_row[prefix + "R_real"] = float(next_row["mid"] / current[prefix + "mid_t"] - 1.0)
                panel_row[prefix + "dIV"] = (
                    float(next_row["implied_volatility"] - current[prefix + "iv_t"])
                    if pd.notna(next_row["implied_volatility"]) and pd.notna(current[prefix + "iv_t"])
                    else np.nan
                )
                panel_row[prefix + "iv_source_t"] = current[prefix + "iv_source_t"]
                panel_row[prefix + "greeks_source_t"] = current[prefix + "greeks_source_t"]

            if missing_contract:
                failure_counts["missing_contract_t1"] += 1
                continue
            if invalid_mid:
                failure_counts["invalid_mid_t1"] += 1
                continue
            rows.append(panel_row)
        if len(symbol_candidates) >= 1:
            failure_counts["no_next_trading_day"] += 1

    out = pd.DataFrame(rows).sort_values(["symbol", "date_t"]).reset_index(drop=True) if rows else pd.DataFrame()
    out.attrs["failure_counts"] = failure_counts
    return out


def compute_ex_post_decomposition(
    panel: pd.DataFrame,
    *,
    theta_is_per_year: bool = True,
) -> pd.DataFrame:
    """
    Compute an ex post Taylor decomposition for each contract bucket.
    """
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "symbol", "date_t", "date_t1", "bucket", "contract_symbol",
                "R_explained", "R_real", "abs_error", "dS", "dIV",
            ]
        )

    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
        dS = float(row["spot_t1"] - row["spot_t"])
        for bucket in _CONTRACT_BUCKETS:
            prefix = f"{bucket}_"
            theta_term = _normalize_theta(
                float(row[prefix + "theta_t"]),
                theta_is_per_year=theta_is_per_year,
                trading_days=1.0,
            )
            dIV = float(row[prefix + "dIV"]) if pd.notna(row[prefix + "dIV"]) else 0.0
            dprice = (
                float(row[prefix + "delta_t"]) * dS
                + 0.5 * float(row[prefix + "gamma_t"]) * dS ** 2
                + theta_term
                + float(row[prefix + "vega_t"]) * dIV
            )
            r_explained = dprice / float(row[prefix + "mid_t"])
            r_real = float(row[prefix + "R_real"])
            rows.append(
                {
                    "symbol": row["symbol"],
                    "date_t": row["date_t"],
                    "date_t1": row["date_t1"],
                    "bucket": bucket,
                    "contract_symbol": row[prefix + "symbol"],
                    "R_explained": r_explained,
                    "R_real": r_real,
                    "abs_error": abs(r_explained - r_real),
                    "dS": dS,
                    "dIV": dIV,
                }
            )
    return pd.DataFrame(rows)


def _fit_explained_vs_real_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "n_obs": 0,
            "corr": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }
    x = df["R_explained"].astype(float).to_numpy()
    y = df["R_real"].astype(float).to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1]) if len(df) > 1 else np.nan
    if len(df) > 1 and np.any(np.abs(x - x.mean()) > 0.0):
        X = np.column_stack([x, np.ones(len(x), dtype="float64")])
        slope, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        slope = np.nan
        intercept = np.nan
    err = x - y
    return {
        "n_obs": int(len(df)),
        "corr": corr,
        "slope": float(slope) if np.isfinite(slope) else np.nan,
        "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
    }


def diagnose_theta_units(research_panel: pd.DataFrame) -> dict[str, Any]:
    """
    Compare theta unit conventions by running the ex post decomposition twice.

    Returns both per-year and per-day summaries and recommends the convention
    whose overall correlation is higher and whose regression slope is closer to 1.
    """
    empty_bucket = pd.DataFrame(
        columns=["bucket", "n_obs", "corr", "slope", "intercept", "mae", "rmse"]
    )
    if research_panel.empty:
        return {
            "per_year": {"overall": _fit_explained_vs_real_stats(pd.DataFrame()), "by_bucket": empty_bucket},
            "per_day": {"overall": _fit_explained_vs_real_stats(pd.DataFrame()), "by_bucket": empty_bucket},
            "recommendation": "per_year",
        }

    results: dict[str, dict[str, Any]] = {}
    for label, theta_is_per_year in (("per_year", True), ("per_day", False)):
        ex_post = compute_ex_post_decomposition(research_panel, theta_is_per_year=theta_is_per_year)
        bucket_rows: list[dict[str, Any]] = []
        for bucket in _CONTRACT_BUCKETS:
            sub = ex_post[ex_post["bucket"] == bucket].copy()
            stats = _fit_explained_vs_real_stats(sub)
            bucket_rows.append({"bucket": bucket, **stats})
        results[label] = {
            "overall": _fit_explained_vs_real_stats(ex_post),
            "by_bucket": pd.DataFrame(bucket_rows),
        }

    year_overall = results["per_year"]["overall"]
    day_overall = results["per_day"]["overall"]
    year_corr = year_overall["corr"] if np.isfinite(year_overall["corr"]) else -np.inf
    day_corr = day_overall["corr"] if np.isfinite(day_overall["corr"]) else -np.inf
    year_slope_gap = abs(year_overall["slope"] - 1.0) if np.isfinite(year_overall["slope"]) else np.inf
    day_slope_gap = abs(day_overall["slope"] - 1.0) if np.isfinite(day_overall["slope"]) else np.inf
    recommendation = "per_year" if (year_corr > day_corr and year_slope_gap < day_slope_gap) else "per_day"
    return {
        "per_year": results["per_year"],
        "per_day": results["per_day"],
        "recommendation": recommendation,
    }


def build_underlying_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one daily underlying spot series per symbol from the latest contract rows.
    """
    latest = _latest_daily_rows(df)
    if latest.empty:
        return pd.DataFrame(columns=["symbol", "quote_date", "underlying_spot"])
    history = (
        latest.groupby(["symbol", "quote_date"], as_index=False)
        .agg(
            underlying_spot=("underlying_spot", "median"),
            quote_time=("quote_time", "max"),
        )
        .sort_values(["symbol", "quote_date"])
        .reset_index(drop=True)
    )
    history["underlying_return"] = history.groupby("symbol")["underlying_spot"].pct_change()
    return history


def _apply_rolling_option_mu_calibration(
    ex_ante: pd.DataFrame,
    *,
    lookback: int = 120,
    min_obs: int = 30,
    shrink_obs: int = 60,
    initial_scale: float = 0.0,
    allow_negative: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Calibrate option mu forecasts with an expanding/rolling slope on past data.

    The raw model output is scaled by a data-driven multiplier estimated from
    past observations only:
        scale ~= E[R_real * mu_model] / E[mu_model^2]
    The scale is clipped to [-1, 1] (or [0, 1] if ``allow_negative=False``)
    and shrunk toward ``initial_scale`` when sample size is limited.
    """
    if ex_ante.empty:
        return ex_ante.copy(), {
            "enabled": True,
            "lookback": int(max(1, lookback)),
            "min_obs": int(max(2, min_obs)),
            "shrink_obs": int(max(1, shrink_obs)),
            "initial_scale": float(np.clip(initial_scale, -1.0, 1.0)),
            "allow_negative": bool(allow_negative),
            "mean_scale": np.nan,
            "median_scale": np.nan,
        }

    work = ex_ante.sort_values(["symbol", "date_t"]).reset_index(drop=True).copy()
    lookback = int(max(1, lookback))
    min_obs = int(max(2, min_obs))
    shrink_obs = int(max(1, shrink_obs))
    min_scale = -1.0 if allow_negative else 0.0
    initial_scale = float(np.clip(initial_scale, min_scale, 1.0))
    all_scales: list[float] = []

    for symbol, idx in work.groupby("symbol", sort=False).groups.items():
        idx_arr = np.asarray(sorted(idx), dtype=int)
        for bucket in _CONTRACT_BUCKETS:
            mu_col = f"{bucket}_mu_pred"
            real_col = f"{bucket}_R_real"
            if mu_col not in work.columns or real_col not in work.columns:
                continue
            mu_vals = work.loc[idx_arr, mu_col].astype(float).to_numpy()
            real_vals = work.loc[idx_arr, real_col].astype(float).to_numpy()
            scales = np.full(len(idx_arr), initial_scale, dtype="float64")
            for j in range(len(idx_arr)):
                start = max(0, j - lookback)
                x = mu_vals[start:j]
                y = real_vals[start:j]
                valid = np.isfinite(x) & np.isfinite(y)
                n_obs = int(valid.sum())
                if n_obs < min_obs:
                    continue
                x = x[valid]
                y = y[valid]
                denom = float(np.dot(x, x))
                if denom <= 1e-12:
                    continue
                slope = float(np.dot(x, y) / denom)
                slope = float(np.clip(slope, min_scale, 1.0))
                confidence = float(n_obs / (n_obs + shrink_obs))
                scale = (1.0 - confidence) * initial_scale + confidence * slope
                scales[j] = float(np.clip(scale, min_scale, 1.0))
            work.loc[idx_arr, f"{bucket}_mu_model"] = mu_vals
            work.loc[idx_arr, f"{bucket}_mu_scale"] = scales
            work.loc[idx_arr, mu_col] = mu_vals * scales
            all_scales.extend(scales.tolist())

    diagnostics = {
        "enabled": True,
        "lookback": lookback,
        "min_obs": min_obs,
        "shrink_obs": shrink_obs,
        "initial_scale": initial_scale,
        "allow_negative": bool(allow_negative),
        "mean_scale": float(np.mean(all_scales)) if all_scales else np.nan,
        "median_scale": float(np.median(all_scales)) if all_scales else np.nan,
    }
    return work, diagnostics


def compute_ex_ante_moments(
    panel: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    lookback: int = 20,
    theta_is_per_year: bool = True,
    mu_tstat_full_weight: float = 4.0,
    option_mu_cap: float = 0.01,
    option_mu_calibration: bool = True,
    option_mu_calibration_lookback: int = 120,
    option_mu_calibration_min_obs: int = 30,
    option_mu_calibration_shrink_obs: int = 60,
    option_mu_calibration_initial_scale: float = 0.0,
    option_mu_calibration_allow_negative: bool = True,
) -> pd.DataFrame:
    """
    Estimate next-day expected returns and approximate standard deviations.
    """
    if panel.empty:
        return pd.DataFrame()
    if price_history.empty:
        raise ValueError("price_history must contain at least one symbol/date series")

    history = price_history.sort_values(["symbol", "quote_date"]).copy()
    rows: list[dict[str, Any]] = []
    stock_mu_violations: list[str] = []
    option_mu_violations: list[str] = []
    stock_mu_violation_count = 0
    option_mu_violation_count = 0

    for _, row in panel.iterrows():
        symbol_history = history[(history["symbol"] == row["symbol"]) & (history["quote_date"] < row["date_t"])]
        returns = symbol_history["underlying_return"].dropna()
        if len(returns) < max(lookback, 2):
            continue
        window_raw = returns.tail(lookback)
        window = _winsorize_series(window_raw)
        if len(window) < 2:
            continue
        mu_raw = float(window.mean())
        sigma_d = float(window.std(ddof=1))
        if not np.isfinite(sigma_d) or sigma_d <= 0:
            continue
        se_mu = sigma_d / np.sqrt(len(window))
        t_abs = abs(mu_raw) / se_mu if np.isfinite(se_mu) and se_mu > 0 else 0.0
        mu_shrink = min(1.0, float(t_abs / max(float(mu_tstat_full_weight), 1e-6)))
        mu_d = mu_raw * mu_shrink
        var_r = sigma_d ** 2
        stock_mu_sane = abs(mu_d) < 0.02
        if not stock_mu_sane:
            stock_mu_violation_count += 1
            if len(stock_mu_violations) < 8:
                stock_mu_violations.append(f"{row['symbol']}@{pd.Timestamp(row['date_t']).date()}={mu_d:.4f}")

        out_row: dict[str, Any] = {
            "symbol": row["symbol"],
            "date_t": row["date_t"],
            "date_t1": row["date_t1"],
            "spot_t": float(row["spot_t"]),
            "mu_und_raw": mu_raw,
            "mu_und_shrink": mu_shrink,
            "mu_und_pred": mu_d,
            "mu_und_sane": stock_mu_sane,
            "sigma_und_pred": sigma_d,
            "R_und_real": row["R_und_real"],
        }
        for bucket in _CONTRACT_BUCKETS:
            prefix = f"{bucket}_"
            S = float(row["spot_t"])
            price = float(row[prefix + "mid_t"])
            delta = float(row[prefix + "delta_t"])
            gamma = float(row[prefix + "gamma_t"])
            theta_term = _normalize_theta(float(row[prefix + "theta_t"]), theta_is_per_year=theta_is_per_year)
            exp_dS = S * mu_d
            gamma_drift = 0.5 * gamma * (S ** 2) * var_r
            mean_ret_raw = (
                delta * exp_dS
                + gamma_drift
                + theta_term
            ) / price
            if option_mu_cap is not None and np.isfinite(option_mu_cap) and float(option_mu_cap) > 0:
                mean_ret = float(np.clip(mean_ret_raw, -float(option_mu_cap), float(option_mu_cap)))
            else:
                mean_ret = float(mean_ret_raw)
            option_mu_sane = abs(mean_ret) < 0.20
            if not option_mu_sane:
                option_mu_violation_count += 1
                if len(option_mu_violations) < 12:
                    option_mu_violations.append(
                        f"{row['symbol']}:{bucket}@{pd.Timestamp(row['date_t']).date()}={mean_ret:.4f}"
                    )
            var_ret = (
                ((delta * S) / price) ** 2 * var_r
                + 0.5 * ((gamma * S ** 2) / price) ** 2 * (var_r ** 2)
            )
            out_row[prefix + "mu_pred"] = mean_ret
            out_row[prefix + "mu_sane"] = option_mu_sane
            out_row[prefix + "sigma_pred"] = float(np.sqrt(max(var_ret, 0.0)))
            out_row[prefix + "R_real"] = row[prefix + "R_real"]
            out_row[prefix + "contract_symbol"] = row[prefix + "symbol"]
            out_row[prefix + "contract_key"] = row.get(prefix + "contract_key")
            out_row[prefix + "mid_t"] = price
            out_row[prefix + "delta_t"] = delta
            out_row[prefix + "gamma_t"] = gamma
            out_row[prefix + "mu_raw"] = float(mean_ret_raw)
            out_row[prefix + "mu_clipped"] = float(mean_ret)
            out_row[prefix + "gamma_drift"] = float(gamma_drift / price)
        rows.append(out_row)

    out = pd.DataFrame(rows)
    if stock_mu_violations:
        warnings.warn(
            "compute_ex_ante_moments stock mu sanity check failed (expected abs(mu_daily) < 0.02) for "
            + ", ".join(stock_mu_violations),
            RuntimeWarning,
            stacklevel=2,
        )
    if option_mu_violations:
        warnings.warn(
            "compute_ex_ante_moments option mu sanity check failed (expected abs(mu_daily) < 0.20) for "
            + ", ".join(option_mu_violations),
            RuntimeWarning,
            stacklevel=2,
        )
    if option_mu_calibration and not out.empty:
        out, mu_calibration_diag = _apply_rolling_option_mu_calibration(
            out,
            lookback=option_mu_calibration_lookback,
            min_obs=option_mu_calibration_min_obs,
            shrink_obs=option_mu_calibration_shrink_obs,
            initial_scale=option_mu_calibration_initial_scale,
            allow_negative=option_mu_calibration_allow_negative,
        )
    else:
        min_scale = -1.0 if option_mu_calibration_allow_negative else 0.0
        mu_calibration_diag = {
            "enabled": False,
            "lookback": int(option_mu_calibration_lookback),
            "min_obs": int(option_mu_calibration_min_obs),
            "shrink_obs": int(option_mu_calibration_shrink_obs),
            "initial_scale": float(np.clip(option_mu_calibration_initial_scale, min_scale, 1.0)),
            "allow_negative": bool(option_mu_calibration_allow_negative),
            "mean_scale": np.nan,
            "median_scale": np.nan,
        }

    out.attrs["mu_sanity"] = {
        "stock_violations": stock_mu_violation_count,
        "option_violations": option_mu_violation_count,
    }
    out.attrs["mu_calibration"] = mu_calibration_diag
    return out


def build_markowitz_universe(
    ex_ante: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    as_of_date: Optional[pd.Timestamp] = None,
    symbols: Optional[Sequence[str]] = None,
    lookback: int = 20,
    kappa_robust: float = 0.25,
    eps_wc: float = 0.05,
) -> dict[str, Any]:
    """
    Build a paper-style multi-asset Markowitz preview across the latest
    available symbol cross-section.

    Cash is returned separately as a risk-free sleeve and excluded from the
    risky covariance matrix to avoid numerical instability.
    """
    if ex_ante.empty or price_history.empty:
        return {}

    work = ex_ante.copy()
    if symbols is not None:
        allowed = {str(s).upper() for s in symbols}
        work = work[work["symbol"].isin(allowed)]
    if work.empty:
        return {}

    work["date_t"] = pd.to_datetime(work["date_t"])
    if as_of_date is not None:
        target_date = pd.Timestamp(as_of_date).normalize()
    else:
        counts = (
            work.groupby("date_t", as_index=False)["symbol"]
            .nunique()
            .rename(columns={"symbol": "n_symbols"})
            .sort_values(["n_symbols", "date_t"], ascending=[False, False])
        )
        target_date = pd.Timestamp(counts.iloc[0]["date_t"]).normalize()

    cross = work[work["date_t"] == target_date].copy()
    if cross.empty:
        return {}

    cross = cross.sort_values("symbol").reset_index(drop=True)
    selected_symbols = cross["symbol"].tolist()

    history = price_history.copy()
    history["quote_date"] = pd.to_datetime(history["quote_date"])
    window = (
        history[
            history["symbol"].isin(selected_symbols)
            & (history["quote_date"] < target_date)
        ]
        .pivot(index="quote_date", columns="symbol", values="underlying_return")
        .sort_index()
        .dropna(how="any")
    )
    if len(window) < max(lookback, 2):
        return {}
    window = window.tail(lookback)
    sigma_under = window.cov(ddof=1).reindex(index=selected_symbols, columns=selected_symbols)

    asset_labels: list[str] = []
    mu_values: list[float] = []
    exposures: list[np.ndarray] = []
    cost_weights: list[float] = []
    asset_contract_keys: list[Any] = []
    metadata_rows: list[dict[str, Any]] = []
    block_indices_by_symbol: list[tuple[str, list[int]]] = []

    for sym_idx, (_, row) in enumerate(cross.iterrows()):
        basis = np.zeros(len(selected_symbols), dtype="float64")
        basis[sym_idx] = 1.0
        symbol_block_indices: list[int] = []

        asset_labels.append(f"{row['symbol']}_UND")
        mu_values.append(float(row["mu_und_pred"]))
        exposures.append(basis.copy())
        symbol_block_indices.append(len(asset_labels) - 1)
        under_price = float(row["spot_t"])
        under_cost = 0.005 / max(1.0, under_price)
        cost_weights.append(under_cost)
        asset_contract_keys.append((str(row["symbol"]).upper(), "underlying", None, None))
        metadata_rows.append(
            {
                "asset": f"{row['symbol']}_UND",
                "symbol": row["symbol"],
                "asset_type": "underlying",
                "contract_symbol": row["symbol"],
                "contract_key": asset_contract_keys[-1],
                "asset_price": under_price,
                "mu_pred": float(row["mu_und_pred"]),
                "sigma_pred": float(row["sigma_und_pred"]),
                "delta_equiv_beta": 1.0,
                "transaction_cost_weight": under_cost,
            }
        )

        for bucket in _CONTRACT_BUCKETS:
            prefix = f"{bucket}_"
            beta = float(row[prefix + "delta_t"] * row["spot_t"] / row[prefix + "mid_t"])
            asset_label = f"{row['symbol']}_{bucket.upper()}"
            asset_labels.append(asset_label)
            mu_values.append(float(row[prefix + "mu_pred"]))
            exposures.append(basis * beta)
            symbol_block_indices.append(len(asset_labels) - 1)
            option_price = float(row[prefix + "mid_t"])
            option_cost = _option_transaction_cost_weight(option_price)
            cost_weights.append(option_cost)
            asset_contract_keys.append(row.get(prefix + "contract_key"))
            metadata_rows.append(
                {
                    "asset": asset_label,
                    "symbol": row["symbol"],
                    "asset_type": bucket,
                    "contract_symbol": row[prefix + "contract_symbol"],
                    "contract_key": row.get(prefix + "contract_key"),
                    "asset_price": option_price,
                    "mu_pred": float(row[prefix + "mu_pred"]),
                    "sigma_pred": float(row[prefix + "sigma_pred"]),
                    "delta_equiv_beta": beta,
                    "transaction_cost_weight": option_cost,
                }
            )
        block_indices_by_symbol.append((row["symbol"], symbol_block_indices))

    exposure_matrix = np.vstack(exposures)
    a_base = exposure_matrix @ sigma_under.to_numpy() @ exposure_matrix.T
    empirical_noise = _estimate_option_residual_noise_diag(
        work,
        cross,
        target_date=target_date,
        lookback=lookback,
        asset_labels=asset_labels,
    ).reindex(index=asset_labels, columns=asset_labels).fillna(0.0)
    structural_cov = a_base + empirical_noise.to_numpy()
    robust_bump = np.zeros_like(a_base)
    for sym_idx, (symbol, block_indices) in enumerate(block_indices_by_symbol):
        sigma_ii = max(float(sigma_under.loc[symbol, symbol]), 0.0)
        if sigma_ii <= 0.0:
            continue
        v_i = exposure_matrix[np.asarray(block_indices, dtype=int), sym_idx]
        d_i = float(kappa_robust) * sigma_ii * np.diag(np.square(v_i))
        robust_bump[np.ix_(block_indices, block_indices)] += d_i
    worst_case_bump = float(eps_wc) * (exposure_matrix @ exposure_matrix.T)
    risky_cov = structural_cov + robust_bump + worst_case_bump
    a_base = 0.5 * (a_base + a_base.T)
    structural_cov = 0.5 * (structural_cov + structural_cov.T)
    robust_bump = 0.5 * (robust_bump + robust_bump.T)
    worst_case_bump = 0.5 * (worst_case_bump + worst_case_bump.T)
    risky_cov = 0.5 * (risky_cov + risky_cov.T)

    base_eigs = np.linalg.eigvalsh(a_base)
    base_rank = int(np.linalg.matrix_rank(a_base))
    structural_eigs = np.linalg.eigvalsh(structural_cov)
    structural_rank = int(np.linalg.matrix_rank(structural_cov))
    diag_scale = float(np.nanmax(np.diag(risky_cov))) if risky_cov.size else 0.0
    diag_scale = max(diag_scale, 1.0)
    min_target = 1e-8 * diag_scale
    pre_floor_eigs = np.linalg.eigvalsh(risky_cov)
    ridge_added = max(1e-10, float(min_target - pre_floor_eigs[0])) if pre_floor_eigs[0] < min_target else 1e-10
    risky_cov += np.eye(risky_cov.shape[0]) * ridge_added
    effective_eigs = np.linalg.eigvalsh(risky_cov)
    effective_rank = int(np.linalg.matrix_rank(risky_cov))
    min_eig_effective = float(effective_eigs[0]) if len(effective_eigs) else np.nan
    max_eig_effective = float(effective_eigs[-1]) if len(effective_eigs) else np.nan
    cond_effective = (
        float(max_eig_effective / min_eig_effective)
        if np.isfinite(min_eig_effective) and min_eig_effective > 0
        else np.inf
    )
    zero_risk_fixed = effective_rank == len(asset_labels) and np.isfinite(min_eig_effective) and min_eig_effective > 0
    mu_risky = pd.Series(mu_values, index=asset_labels, name="mu_risky")
    sigma_risky = pd.DataFrame(risky_cov, index=asset_labels, columns=asset_labels)
    q_cost = pd.Series(cost_weights, index=asset_labels, name="q_cost")
    contract_key_series = pd.Series(asset_contract_keys, index=asset_labels, name="contract_key")
    asset_table = pd.DataFrame(metadata_rows)
    v_matrix = pd.DataFrame(exposure_matrix, index=asset_labels, columns=selected_symbols)

    summary_lines = [
        "Paper-style experiment design:",
        f"- Cross-section date: {target_date.date()}",
        f"- Symbols used: {', '.join(selected_symbols)}",
        f"- Risky universe size: {len(asset_labels)} = {len(selected_symbols)} underlyings + {4 * len(selected_symbols)} options",
        "- Baseline portfolio is fully invested across risky assets only (no cash sleeve)",
        "- Mean estimates use time-t Greeks with lagged historical underlying drift/variance only",
        "- Structural covariance for validation is V Sigma V^T plus a data-driven diagonal option residual-noise floor (positive unexplained variance only)",
        f"- Effective risk matrix uses A_eff = V Sigma V^T + D + eps_wc * V V^T with kappa_robust={kappa_robust:.3f}, eps_wc={eps_wc:.3f}",
        f"- Zero-risk check: delta-only rank {base_rank}/{len(asset_labels)} -> structural rank {structural_rank}/{len(asset_labels)} -> effective rank {effective_rank}/{len(asset_labels)}, "
        f"min eig {min_eig_effective:.3e}, ridge {ridge_added:.3e}, cond {cond_effective:.3e}",
        "- Transaction-cost weights q are attached per asset for the baseline turnover penalty",
    ]

    return {
        "date_t": target_date,
        "symbols": selected_symbols,
        "asset_table": asset_table,
        "mu_risky": mu_risky,
        "sigma_risky": sigma_risky,
        "cost_weights": q_cost,
        "asset_contract_keys": contract_key_series,
        "v_matrix": v_matrix,
        "a_base": pd.DataFrame(a_base, index=asset_labels, columns=asset_labels),
        "structural_cov": pd.DataFrame(structural_cov, index=asset_labels, columns=asset_labels),
        "empirical_noise": empirical_noise,
        "robust_bump": pd.DataFrame(robust_bump, index=asset_labels, columns=asset_labels),
        "worst_case_bump": pd.DataFrame(worst_case_bump, index=asset_labels, columns=asset_labels),
        "risk_diagnostics": {
            "base_rank": base_rank,
            "structural_rank": structural_rank,
            "effective_rank": effective_rank,
            "n_assets": len(asset_labels),
            "base_min_eig": float(base_eigs[0]) if len(base_eigs) else np.nan,
            "structural_min_eig": float(structural_eigs[0]) if len(structural_eigs) else np.nan,
            "effective_min_eig": min_eig_effective,
            "effective_max_eig": max_eig_effective,
            "effective_condition_number": cond_effective,
            "ridge_added": float(ridge_added),
            "zero_risk_fixed": bool(zero_risk_fixed),
        },
        "summary": "\n".join(summary_lines),
    }


def build_markowitz_preview(
    ex_ante: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backward-compatible thin wrapper for older callers.

    This single-symbol preview does not have enough information to build the
    cross-asset covariance used by the current notebook, so it is retained only
    as a compact mean-vector preview.
    """
    if ex_ante.empty:
        return {}
    work = ex_ante.copy()
    if symbol:
        work = work[work["symbol"] == symbol.upper()]
    if work.empty:
        return {}
    row = work.sort_values(["date_t", "symbol"]).iloc[-1]
    asset_labels = ["underlying", "call_1", "call_2", "put_1", "put_2", "cash"]
    mu = pd.Series(
        [
            float(row["mu_und_pred"]),
            float(row["call_1_mu_pred"]),
            float(row["call_2_mu_pred"]),
            float(row["put_1_mu_pred"]),
            float(row["put_2_mu_pred"]),
            DEFAULT_RF / TRADING_DAYS_PER_YEAR,
        ],
        index=asset_labels,
        name="mu",
    )
    return {
        "symbol": row["symbol"],
        "date_t": row["date_t"],
        "mu": mu,
    }


def _select_common_markowitz_symbols(ex_ante: pd.DataFrame, symbols: Optional[Sequence[str]] = None) -> list[str]:
    work = ex_ante.copy()
    work["date_t"] = pd.to_datetime(work["date_t"])
    if symbols is not None:
        requested = [str(s).upper() for s in symbols]
        available = set(work["symbol"].unique())
        return [s for s in requested if s in available]

    if work.empty:
        return []
    counts = (
        work.groupby("date_t", as_index=False)["symbol"]
        .nunique()
        .rename(columns={"symbol": "n_symbols"})
        .sort_values(["n_symbols", "date_t"], ascending=[False, False])
    )
    target_date = pd.Timestamp(counts.iloc[0]["date_t"]).normalize()
    return sorted(work.loc[work["date_t"].dt.normalize() == target_date, "symbol"].unique().tolist())


def _markowitz_asset_labels(symbols: Sequence[str]) -> list[str]:
    labels: list[str] = []
    for symbol in symbols:
        labels.append(f"{symbol}_UND")
        for bucket in _CONTRACT_BUCKETS:
            labels.append(f"{symbol}_{bucket.upper()}")
    return labels


def _realized_return_vector(cross: pd.DataFrame, symbols: Sequence[str]) -> pd.Series:
    by_symbol = cross.set_index("symbol")
    values: list[float] = []
    labels: list[str] = []
    for symbol in symbols:
        row = by_symbol.loc[symbol]
        labels.append(f"{symbol}_UND")
        values.append(float(row["R_und_real"]))
        for bucket in _CONTRACT_BUCKETS:
            labels.append(f"{symbol}_{bucket.upper()}")
            values.append(float(row[f"{bucket}_R_real"]))
    return pd.Series(values, index=labels, dtype="float64")


def _option_transaction_cost_weight(premium: float) -> float:
    """
    Approximate the paper's premium-bucket option transaction cost weights.

    The paper uses eta(price) / price with premium buckets. The exact Bloomberg
    bucket schedule is not encoded in the repo, so this uses a transparent
    monotone proxy that penalizes cheaper options more heavily.
    """
    p = max(float(premium), 1e-6)
    if p < 1.0:
        return 0.020
    if p < 2.5:
        return 0.015
    if p < 5.0:
        return 0.010
    return 0.005


def _rolled_asset_mask(
    current_keys: pd.Series,
    previous_keys: Optional[pd.Series],
    *,
    index: pd.Index,
) -> pd.Series:
    if previous_keys is None or previous_keys.empty:
        return pd.Series(False, index=index, dtype=bool)
    curr = current_keys.reindex(index)
    prev = previous_keys.reindex(index)
    return curr.ne(prev).fillna(False).astype(bool)


def _turnover_cost_value(
    weights: pd.Series,
    previous_weights: Optional[pd.Series],
    cost_weights: pd.Series,
    *,
    previous_cost_weights: Optional[pd.Series] = None,
    rolled_assets: Optional[pd.Series] = None,
) -> float:
    if previous_weights is None:
        return 0.0
    aligned_index = weights.index
    w = weights.reindex(aligned_index).fillna(0.0).astype(float)
    w_prev = previous_weights.reindex(aligned_index).fillna(0.0).astype(float)
    q_curr = cost_weights.reindex(aligned_index).fillna(0.0).astype(float)
    q_prev = (
        previous_cost_weights.reindex(aligned_index).fillna(0.0).astype(float)
        if previous_cost_weights is not None
        else q_curr
    )
    q_eff = np.maximum(q_curr.to_numpy(), q_prev.to_numpy())
    rolled = (
        rolled_assets.reindex(aligned_index).fillna(False).astype(bool).to_numpy()
        if rolled_assets is not None
        else np.zeros(len(aligned_index), dtype=bool)
    )
    traded = np.abs(w.to_numpy() - w_prev.to_numpy())
    if rolled.any():
        traded[rolled] = np.abs(w_prev.to_numpy()[rolled]) + np.abs(w.to_numpy()[rolled])
    return float(np.sum(q_eff * traded))


def _annualized_sharpe(returns: Sequence[float]) -> float:
    vals = np.asarray(list(returns), dtype="float64")
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return -np.inf
    std = float(vals.std(ddof=1))
    if std <= 0:
        return np.inf if float(vals.mean()) > 0 else -np.inf
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * vals.mean() / std)


def _estimate_option_residual_noise_diag(
    work: pd.DataFrame,
    cross: pd.DataFrame,
    *,
    target_date: pd.Timestamp,
    lookback: int,
    asset_labels: Sequence[str],
    shrink_obs: int = 5,
) -> pd.DataFrame:
    """
    Estimate a diagonal option residual-noise floor from past model misses.

    For each option asset, compute the positive part of:
        Var(real option return) - E[beta_t^2 * Var(underlying return)]
    over the recent lookback window for that symbol/bucket.

    This stays close to the paper's idea of adding diagonal uncertainty, while
    avoiding inflating assets whose delta-only variance already explains the
    realized history.
    """
    diag = pd.Series(0.0, index=pd.Index(asset_labels, dtype="object"), dtype="float64")
    if work.empty or cross.empty:
        return pd.DataFrame(np.diag(diag.to_numpy()), index=diag.index, columns=diag.index)

    hist = work[pd.to_datetime(work["date_t"]) < pd.Timestamp(target_date)].copy()
    if hist.empty:
        return pd.DataFrame(np.diag(diag.to_numpy()), index=diag.index, columns=diag.index)

    for _, row in cross.iterrows():
        symbol_hist = hist[hist["symbol"] == row["symbol"]].sort_values("date_t").tail(lookback)
        if symbol_hist.empty:
            continue
        for bucket in _CONTRACT_BUCKETS:
            label = f"{row['symbol']}_{bucket.upper()}"
            ret_col = f"{bucket}_R_real"
            delta_col = f"{bucket}_delta_t"
            mid_col = f"{bucket}_mid_t"
            sigma_col = "sigma_und_pred"
            spot_col = "spot_t"
            cols = [ret_col, delta_col, mid_col, sigma_col, spot_col]
            sub = symbol_hist[cols].dropna().copy()
            if len(sub) < 3:
                continue
            realized = _winsorize_series(sub[ret_col])
            aligned = sub.loc[realized.index]
            beta = (aligned[delta_col].astype(float) * aligned[spot_col].astype(float) / aligned[mid_col].astype(float)).replace([np.inf, -np.inf], np.nan)
            var_under = np.square(aligned[sigma_col].astype(float))
            model_var = float(np.nanmean(np.square(beta) * var_under))
            if not np.isfinite(model_var):
                model_var = 0.0
            realized_var = float(realized.var(ddof=1))
            if not np.isfinite(realized_var):
                continue
            idio_var = max(realized_var - model_var, 0.0)
            shrink = float(len(aligned) / (len(aligned) + max(int(shrink_obs), 1)))
            diag[label] = max(diag[label], idio_var * shrink)
    return pd.DataFrame(np.diag(diag.to_numpy()), index=diag.index, columns=diag.index)


def solve_markowitz_allocation(
    mu_risky: pd.Series,
    sigma_risky: pd.DataFrame,
    *,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 1.0,
    previous_weights: Optional[pd.Series] = None,
    cost_weights: Optional[pd.Series] = None,
    previous_cost_weights: Optional[pd.Series] = None,
    rolled_assets: Optional[pd.Series] = None,
    exposure_matrix: Optional[pd.DataFrame] = None,
    eps_wc: float = 0.05,
    long_only: bool = True,
    max_asset_weight: Optional[float] = None,
) -> pd.Series:
    """
    Solve the paper-style baseline (25): long-only, fully invested across the
    risky assets, with quadratic risk and an L1 turnover penalty.

    The passed ``sigma_risky`` is treated as the fully assembled effective risk
    matrix. The worst-case ``eps_wc`` term is folded into ``sigma_risky`` in
    ``build_markowitz_universe`` so we do not add it again here.
    """
    if mu_risky.empty or sigma_risky.empty:
        return pd.Series(dtype="float64")

    import cvxpy as cp

    mu_vec = mu_risky.astype(float).to_numpy()
    sigma = sigma_risky.astype(float).to_numpy()
    sigma = 0.5 * (sigma + sigma.T)
    sigma += np.eye(sigma.shape[0]) * 1e-10
    n = len(mu_vec)
    w_risky = cp.Variable(n)
    w_prev = (
        previous_weights.reindex(mu_risky.index).fillna(0.0).astype(float).to_numpy()
        if previous_weights is not None
        else np.zeros(n, dtype="float64")
    )
    q_vec = (
        cost_weights.reindex(mu_risky.index).fillna(0.0).astype(float).to_numpy()
        if cost_weights is not None
        else np.zeros(n, dtype="float64")
    )
    q_prev_vec = (
        previous_cost_weights.reindex(mu_risky.index).fillna(0.0).astype(float).to_numpy()
        if previous_cost_weights is not None
        else q_vec
    )
    q_eff_vec = np.maximum(q_vec, q_prev_vec)
    rolled_mask = (
        rolled_assets.reindex(mu_risky.index).fillna(False).astype(bool).to_numpy()
        if rolled_assets is not None
        else np.zeros(n, dtype=bool)
    )
    turnover_parts: list[Any] = []
    if previous_weights is not None:
        keep_idx = np.flatnonzero(~rolled_mask)
        if len(keep_idx):
            turnover_parts.append(
                cp.norm1(cp.multiply(q_eff_vec[keep_idx], w_risky[keep_idx] - w_prev[keep_idx]))
            )
        roll_idx = np.flatnonzero(rolled_mask)
        if len(roll_idx):
            turnover_parts.append(
                cp.sum(cp.multiply(q_eff_vec[roll_idx], w_risky[roll_idx] + w_prev[roll_idx]))
            )
    turnover_term = sum(turnover_parts) if turnover_parts else 0.0
    objective = cp.Maximize(
        mu_vec @ w_risky
        - float(risk_aversion) * cp.quad_form(w_risky, sigma)
        - float(turnover_penalty) * turnover_term
    )
    constraints = [cp.sum(w_risky) == 1]
    if long_only:
        constraints.append(w_risky >= 0)
    if max_asset_weight is not None:
        constraints.append(w_risky <= float(max_asset_weight))

    problem = cp.Problem(objective, constraints)
    solved = False
    for solver_name in ("CLARABEL", "OSQP", "SCS"):
        solver = getattr(cp, solver_name, None)
        if solver is None:
            continue
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
            if w_risky.value is not None:
                solved = True
                break
        except Exception:
            continue
    if not solved and w_risky.value is None:
        try:
            problem.solve(warm_start=True, verbose=False)
        except Exception:
            pass

    if w_risky.value is None:
        return pd.Series(dtype="float64")

    weights = pd.Series(np.asarray(w_risky.value).reshape(-1), index=mu_risky.index.tolist())
    weights = weights.clip(lower=0.0) if long_only else weights
    total = float(weights.sum())
    if total > 0:
        weights = weights / total
    return weights


def _prepare_markowitz_cross_section(
    work: pd.DataFrame,
    *,
    date_t: pd.Timestamp,
    selected_symbols: Sequence[str],
    asset_labels: Sequence[str],
) -> Optional[pd.Series]:
    cross = work[(work["date_t"] == pd.Timestamp(date_t)) & (work["symbol"].isin(selected_symbols))].copy()
    cross = cross.sort_values("symbol")
    if cross["symbol"].nunique() != len(selected_symbols):
        return None
    realized = _realized_return_vector(cross, selected_symbols).reindex(asset_labels)
    if realized.isna().any():
        return None
    return realized


def _walk_forward_validation_sharpe(
    val_dates: Sequence[pd.Timestamp],
    *,
    work: pd.DataFrame,
    price_history: pd.DataFrame,
    selected_symbols: Sequence[str],
    asset_labels: Sequence[str],
    lookback: int,
    risk_aversion: float,
    turnover_penalty: float,
    kappa_robust: float,
    eps_wc: float,
    preview_cache: dict[tuple[pd.Timestamp, float], Optional[dict[str, Any]]],
    realized_cache: dict[pd.Timestamp, Optional[pd.Series]],
) -> float:
    if not val_dates:
        return -np.inf

    prev_weights: Optional[pd.Series] = None
    prev_q: Optional[pd.Series] = None
    prev_keys: Optional[pd.Series] = None
    realized_returns: list[float] = []

    for date_t in val_dates:
        cache_key = (pd.Timestamp(date_t), float(kappa_robust), float(eps_wc))
        if cache_key not in preview_cache:
            preview_cache[cache_key] = build_markowitz_universe(
                work,
                price_history,
                as_of_date=pd.Timestamp(date_t),
                symbols=selected_symbols,
                lookback=lookback,
                kappa_robust=kappa_robust,
                eps_wc=eps_wc,
            )
        preview = preview_cache[cache_key]
        if not preview:
            continue

        if pd.Timestamp(date_t) not in realized_cache:
            realized_cache[pd.Timestamp(date_t)] = _prepare_markowitz_cross_section(
                work,
                date_t=pd.Timestamp(date_t),
                selected_symbols=selected_symbols,
                asset_labels=asset_labels,
            )
        realized = realized_cache[pd.Timestamp(date_t)]
        if realized is None:
            continue

        curr_keys = preview["asset_contract_keys"]
        rolled_mask = _rolled_asset_mask(curr_keys, prev_keys, index=preview["mu_risky"].index)
        weights = solve_markowitz_allocation(
            preview["mu_risky"],
            preview["sigma_risky"],
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            previous_weights=prev_weights,
            cost_weights=preview["cost_weights"],
            previous_cost_weights=prev_q,
            rolled_assets=rolled_mask,
            exposure_matrix=preview.get("v_matrix"),
            eps_wc=eps_wc,
        )
        if weights.empty:
            continue

        turnover_cost = _turnover_cost_value(
            weights,
            prev_weights,
            preview["cost_weights"],
            previous_cost_weights=prev_q,
            rolled_assets=rolled_mask,
        )
        realized_gross = float(weights.reindex(asset_labels, fill_value=0.0).to_numpy() @ realized.to_numpy())
        realized_returns.append(realized_gross - turnover_cost)
        prev_weights = weights
        prev_q = preview["cost_weights"]
        prev_keys = curr_keys

    return _annualized_sharpe(realized_returns)


def evaluate_markowitz_sample(
    ex_ante: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    symbols: Optional[Sequence[str]] = None,
    lookback: int = 20,
    daily_only: bool = True,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 1.0,
    eps_wc: float = 0.05,
    kappa_robust: float = 0.25,
    max_asset_weight: Optional[float] = None,
    cv_reopt_every: int = 10,
    lambda_grid: Optional[Sequence[float]] = None,
    xi_grid: Optional[Sequence[float]] = None,
    kappa_grid: Optional[Sequence[float]] = None,
    eps_grid: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """
    Compare predicted vs realized covariance on the common sample and run a
    rolling Markowitz allocation path over the full available period.
    """
    if ex_ante.empty or price_history.empty:
        return {}

    selected_symbols = _select_common_markowitz_symbols(ex_ante, symbols=symbols)
    if not selected_symbols:
        return {}

    work = ex_ante.copy()
    work["date_t"] = pd.to_datetime(work["date_t"]).dt.normalize()
    work["date_t1"] = pd.to_datetime(work["date_t1"]).dt.normalize()
    work = work[work["symbol"].isin(selected_symbols)]
    if daily_only and not work.empty:
        # Enforce one rebalance decision per symbol per trading day.
        work = (
            work.sort_values(["symbol", "date_t", "date_t1"])
            .drop_duplicates(["symbol", "date_t"], keep="last")
            .reset_index(drop=True)
        )
        work = work[work["date_t1"] > work["date_t"]].reset_index(drop=True)
    if work.empty:
        return {}

    counts = work.groupby("date_t")["symbol"].nunique().sort_index()
    common_dates = counts[counts == len(selected_symbols)].index.tolist()
    if not common_dates:
        return {}

    asset_labels = _markowitz_asset_labels(selected_symbols)
    structural_covs: list[np.ndarray] = []
    empirical_noise_covs: list[np.ndarray] = []
    optimizer_covs: list[np.ndarray] = []
    robust_addon_covs: list[np.ndarray] = []
    predicted_rows: list[pd.Series] = []
    realized_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    portfolio_rows: list[dict[str, Any]] = []
    used_dates: list[pd.Timestamp] = []
    previous_weights: Optional[pd.Series] = None
    previous_q: Optional[pd.Series] = None
    previous_keys: Optional[pd.Series] = None
    benchmark_label = "Equal-Weight Underlyings"
    preview_cache: dict[tuple[pd.Timestamp, float, float], Optional[dict[str, Any]]] = {}
    realized_cache: dict[pd.Timestamp, Optional[pd.Series]] = {}
    param_cache_by_date: dict[pd.Timestamp, dict[str, Any]] = {}
    risk_diag_rows: list[dict[str, Any]] = []
    lambda_grid = [1.0, 5.0] if lambda_grid is None else [float(v) for v in lambda_grid]
    xi_grid = [0.3, 1.0] if xi_grid is None else [float(v) for v in xi_grid]
    kappa_grid = [0.15, 0.25] if kappa_grid is None else [float(v) for v in kappa_grid]
    eps_grid = [0.0, 0.05] if eps_grid is None else [float(v) for v in eps_grid]
    cv_reopt_every = max(1, int(cv_reopt_every))
    cv_grid_size = len(lambda_grid) * len(xi_grid) * len(kappa_grid) * len(eps_grid)
    last_tuned_choice: Optional[dict[str, Any]] = None
    last_tuned_date: Optional[pd.Timestamp] = None

    def _default_choice() -> dict[str, Any]:
        return {
            "risk_aversion": float(risk_aversion),
            "turnover_penalty": float(turnover_penalty),
            "kappa_robust": float(kappa_robust),
            "eps_wc": float(eps_wc),
            "cv_sharpe": np.nan,
            "cv_train_n": 0,
            "cv_val_n": 0,
            "cv_ran": False,
            "cv_anchor_date": pd.NaT,
            "cv_grid_size": int(cv_grid_size),
        }

    for date_idx, date_t in enumerate(common_dates):
        date_t = pd.Timestamp(date_t)
        if date_t not in param_cache_by_date:
            should_retune = last_tuned_choice is None or (date_idx % cv_reopt_every == 0)
            if should_retune:
                history_dates = [pd.Timestamp(d) for d in common_dates[max(0, date_idx - lookback):date_idx]]
                chosen = _default_choice()
                chosen["cv_anchor_date"] = date_t
                if len(history_dates) >= 5:
                    split_idx = max(1, int(np.floor(0.8 * len(history_dates))))
                    split_idx = min(split_idx, len(history_dates) - 1)
                    train_dates = history_dates[:split_idx]
                    val_dates = history_dates[split_idx:]
                    best_sharpe = -np.inf
                    for lam in lambda_grid:
                        for xi in xi_grid:
                            for kap in kappa_grid:
                                for eps in eps_grid:
                                    sharpe = _walk_forward_validation_sharpe(
                                        val_dates,
                                        work=work,
                                        price_history=price_history,
                                        selected_symbols=selected_symbols,
                                        asset_labels=asset_labels,
                                        lookback=lookback,
                                        risk_aversion=float(lam),
                                        turnover_penalty=float(xi),
                                        kappa_robust=float(kap),
                                        eps_wc=float(eps),
                                        preview_cache=preview_cache,
                                        realized_cache=realized_cache,
                                    )
                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        chosen = {
                                            "risk_aversion": float(lam),
                                            "turnover_penalty": float(xi),
                                            "kappa_robust": float(kap),
                                            "eps_wc": float(eps),
                                            "cv_sharpe": float(sharpe),
                                            "cv_train_n": int(len(train_dates)),
                                            "cv_val_n": int(len(val_dates)),
                                            "cv_ran": True,
                                            "cv_anchor_date": date_t,
                                            "cv_grid_size": int(cv_grid_size),
                                        }
                last_tuned_choice = dict(chosen)
                last_tuned_date = pd.Timestamp(chosen["cv_anchor_date"])
            else:
                chosen = dict(last_tuned_choice)
                chosen["cv_ran"] = False
                chosen["cv_anchor_date"] = last_tuned_date
                chosen["cv_grid_size"] = int(cv_grid_size)
            param_cache_by_date[date_t] = chosen

        chosen = param_cache_by_date[date_t]
        preview_key = (date_t, float(chosen["kappa_robust"]), float(chosen["eps_wc"]))
        if preview_key not in preview_cache:
            preview_cache[preview_key] = build_markowitz_universe(
                work,
                price_history,
                as_of_date=date_t,
                symbols=selected_symbols,
                lookback=lookback,
                kappa_robust=float(chosen["kappa_robust"]),
                eps_wc=float(chosen["eps_wc"]),
            )
        preview = preview_cache[preview_key]
        if not preview:
            continue
        risk_diag = dict(preview.get("risk_diagnostics", {}))
        risk_diag["date_t"] = date_t
        risk_diag_rows.append(risk_diag)

        if date_t not in realized_cache:
            realized_cache[date_t] = _prepare_markowitz_cross_section(
                work,
                date_t=date_t,
                selected_symbols=selected_symbols,
                asset_labels=asset_labels,
            )
        realized = realized_cache[date_t]
        if realized is None:
            continue

        curr_keys = preview["asset_contract_keys"]
        rolled_mask = _rolled_asset_mask(curr_keys, previous_keys, index=preview["mu_risky"].index)
        weights = solve_markowitz_allocation(
            preview["mu_risky"],
            preview["sigma_risky"],
            risk_aversion=float(chosen["risk_aversion"]),
            turnover_penalty=float(chosen["turnover_penalty"]),
            previous_weights=previous_weights,
            cost_weights=preview.get("cost_weights"),
            previous_cost_weights=previous_q,
            rolled_assets=rolled_mask,
            exposure_matrix=preview.get("v_matrix"),
            eps_wc=float(chosen["eps_wc"]),
            max_asset_weight=max_asset_weight,
        )
        if weights.empty:
            continue

        structural_covs.append(preview["structural_cov"].reindex(index=asset_labels, columns=asset_labels).to_numpy())
        empirical_noise_covs.append(preview["empirical_noise"].reindex(index=asset_labels, columns=asset_labels).to_numpy())
        optimizer_covs.append(preview["sigma_risky"].reindex(index=asset_labels, columns=asset_labels).to_numpy())
        robust_addon_covs.append(
            (
                preview["robust_bump"].reindex(index=asset_labels, columns=asset_labels)
                + preview["worst_case_bump"].reindex(index=asset_labels, columns=asset_labels)
            ).to_numpy()
        )
        predicted_rows.append(preview["mu_risky"].reindex(asset_labels).rename(date_t))
        realized_rows.append(realized.rename(date_t))
        weight_rows.append(weights.rename(date_t))
        if "SPY_UND" in realized.index:
            benchmark_return = float(realized["SPY_UND"])
            benchmark_label = "SPY Buy & Hold"
        else:
            und_labels = [label for label in asset_labels if label.endswith("_UND")]
            benchmark_return = float(realized.reindex(und_labels).mean())
            benchmark_label = "Equal-Weight Underlyings"
        turnover_cost = _turnover_cost_value(
            weights,
            previous_weights,
            preview["cost_weights"],
            previous_cost_weights=previous_q,
            rolled_assets=rolled_mask,
        )
        realized_gross = float(weights.reindex(asset_labels, fill_value=0.0).to_numpy() @ realized.to_numpy())
        predicted_gross = float(weights.reindex(asset_labels, fill_value=0.0).to_numpy() @ preview["mu_risky"].reindex(asset_labels).to_numpy())
        realized_portfolio = realized_gross - turnover_cost
        predicted_portfolio = predicted_gross - turnover_cost
        portfolio_rows.append(
            {
                "date_t": date_t,
                "date_t1": pd.Timestamp(work.loc[work["date_t"] == date_t, "date_t1"].iloc[0]),
                "turnover_cost": turnover_cost,
                "predicted_portfolio_return_gross": predicted_gross,
                "predicted_portfolio_return": predicted_portfolio,
                "realized_portfolio_return_gross": realized_gross,
                "realized_portfolio_return": realized_portfolio,
                "benchmark_return": benchmark_return,
                "risk_aversion": float(chosen["risk_aversion"]),
                "turnover_penalty": float(chosen["turnover_penalty"]),
                "kappa_robust": float(chosen["kappa_robust"]),
                "eps_wc": float(chosen["eps_wc"]),
                "cv_sharpe": float(chosen["cv_sharpe"]) if np.isfinite(chosen["cv_sharpe"]) else np.nan,
                "cv_ran": bool(chosen.get("cv_ran", False)),
                "cv_anchor_date": chosen.get("cv_anchor_date"),
                "cv_grid_size": int(chosen.get("cv_grid_size", cv_grid_size)),
                "n_roll_assets": int(rolled_mask.sum()),
            }
        )
        used_dates.append(date_t)
        previous_weights = weights
        previous_q = preview["cost_weights"]
        previous_keys = curr_keys

    if not structural_covs:
        return {}

    structural_cov_mean = np.mean(np.stack(structural_covs, axis=0), axis=0)
    structural_cov_df = pd.DataFrame(structural_cov_mean, index=asset_labels, columns=asset_labels)
    empirical_noise_mean = np.mean(np.stack(empirical_noise_covs, axis=0), axis=0)
    empirical_noise_df = pd.DataFrame(empirical_noise_mean, index=asset_labels, columns=asset_labels)
    optimizer_cov_mean = np.mean(np.stack(optimizer_covs, axis=0), axis=0)
    optimizer_cov_df = pd.DataFrame(optimizer_cov_mean, index=asset_labels, columns=asset_labels)
    robust_addon_mean = np.mean(np.stack(robust_addon_covs, axis=0), axis=0)
    robust_addon_df = pd.DataFrame(robust_addon_mean, index=asset_labels, columns=asset_labels)
    predicted_matrix = pd.DataFrame(predicted_rows)
    realized_matrix = pd.DataFrame(realized_rows)
    predicted_matrix = predicted_matrix.reindex(index=realized_matrix.index, columns=asset_labels)
    mu_compare = pd.DataFrame(
        {
            "mean_pred": predicted_matrix.mean(axis=0),
            "mean_real": realized_matrix.mean(axis=0),
        }
    )
    mu_compare["bias"] = mu_compare["mean_pred"] - mu_compare["mean_real"]
    mu_compare["mae"] = (predicted_matrix - realized_matrix).abs().mean(axis=0)
    mu_corr = {}
    for label in asset_labels:
        series = pd.concat(
            [predicted_matrix[label].rename("pred"), realized_matrix[label].rename("real")],
            axis=1,
        ).dropna()
        mu_corr[label] = float(series["pred"].corr(series["real"])) if len(series) >= 3 else np.nan
    mu_compare["corr_pred_real"] = pd.Series(mu_corr)

    realized_cov_df = realized_matrix.cov(ddof=1) if len(realized_matrix) > 1 else pd.DataFrame(np.nan, index=asset_labels, columns=asset_labels)
    cov_gap = structural_cov_df - realized_cov_df
    optimizer_gap = optimizer_cov_df - realized_cov_df
    gap_values = cov_gap.to_numpy(dtype="float64")
    gap_values = np.nan_to_num(gap_values, nan=0.0)
    finite_gap = np.abs(cov_gap.to_numpy(dtype="float64"))
    finite_mask = np.isfinite(finite_gap)
    mean_abs_gap = float(finite_gap[finite_mask].mean()) if finite_mask.any() else np.nan
    weight_path = pd.DataFrame(weight_rows).sort_index()
    portfolio_path = pd.DataFrame(portfolio_rows).sort_values("date_t").reset_index(drop=True)
    portfolio_path["portfolio_value"] = (1.0 + portfolio_path["realized_portfolio_return"]).cumprod()
    portfolio_path["benchmark_value"] = (1.0 + portfolio_path["benchmark_return"]).cumprod()
    portfolio_path["running_peak"] = portfolio_path["portfolio_value"].cummax()
    portfolio_path["drawdown"] = portfolio_path["portfolio_value"] / portfolio_path["running_peak"] - 1.0

    aggregate_weights = pd.DataFrame(index=weight_path.index)
    aggregate_weights["UNDERLYINGS"] = weight_path[[c for c in weight_path.columns if c.endswith("_UND")]].sum(axis=1)
    aggregate_weights["CALLS"] = weight_path[[c for c in weight_path.columns if "CALL" in c]].sum(axis=1)
    aggregate_weights["PUTS"] = weight_path[[c for c in weight_path.columns if "PUT" in c]].sum(axis=1)
    mean_turnover_cost = float(portfolio_path["turnover_cost"].mean()) if not portfolio_path.empty else np.nan
    n_rebalances = max(len(portfolio_path) - 1, 0)
    mean_lambda = float(portfolio_path["risk_aversion"].mean()) if not portfolio_path.empty else np.nan
    mean_xi = float(portfolio_path["turnover_penalty"].mean()) if not portfolio_path.empty else np.nan
    mean_kappa = float(portfolio_path["kappa_robust"].mean()) if not portfolio_path.empty else np.nan
    mean_eps = float(portfolio_path["eps_wc"].mean()) if not portfolio_path.empty else np.nan
    n_cv_runs = int(portfolio_path["cv_ran"].sum()) if not portfolio_path.empty else 0
    risk_diag_df = pd.DataFrame(risk_diag_rows).sort_values("date_t").reset_index(drop=True) if risk_diag_rows else pd.DataFrame()
    zero_risk_full_rank_share = (
        float((risk_diag_df["effective_rank"] == risk_diag_df["n_assets"]).mean())
        if not risk_diag_df.empty
        else np.nan
    )
    mean_effective_min_eig = float(risk_diag_df["effective_min_eig"].mean()) if not risk_diag_df.empty else np.nan
    max_effective_cond = float(risk_diag_df["effective_condition_number"].max()) if not risk_diag_df.empty else np.nan
    mean_ridge_added = float(risk_diag_df["ridge_added"].mean()) if not risk_diag_df.empty else np.nan
    und_labels = [label for label in asset_labels if label.endswith("_UND")]
    opt_labels = [label for label in asset_labels if not label.endswith("_UND")]
    und_bias = float(mu_compare.loc[und_labels, "bias"].mean()) if und_labels else np.nan
    opt_bias = float(mu_compare.loc[opt_labels, "bias"].mean()) if opt_labels else np.nan
    und_corr = float(mu_compare.loc[und_labels, "corr_pred_real"].mean()) if und_labels else np.nan
    opt_corr = float(mu_compare.loc[opt_labels, "corr_pred_real"].mean()) if opt_labels else np.nan
    mu_scale_cols = [f"{bucket}_mu_scale" for bucket in _CONTRACT_BUCKETS if f"{bucket}_mu_scale" in work.columns]
    mu_scale_mean = float(work[mu_scale_cols].to_numpy(dtype="float64").mean()) if mu_scale_cols else np.nan

    summary_lines = [
        "Experiment setup:",
        f"- Fixed symbol universe: {', '.join(selected_symbols)}",
        f"- Risky assets: {len(asset_labels)} = {len(selected_symbols)} underlyings + {4 * len(selected_symbols)} options",
        f"- Trading-day periods used: {len(used_dates)} from {used_dates[0].date()} to {used_dates[-1].date()}",
        f"- Rebalance cadence: {'daily only (initial weights on day 0, then one rebalance per trading day)' if daily_only else 'as provided by the input panel'}",
        f"- Rebalances executed after initial day: {n_rebalances}",
        "- Portfolio is long-only and fully invested across risky assets only (no cash sleeve, no Black-Litterman)",
        f"- Stable bucket ids are maintained with contract rolls when DTE < {DEFAULT_ROLL_DTE} or data is missing",
        "- Means are predicted from lagged underlying moments plus time-t delta-gamma-theta (no ex-ante vega drift)",
        f"- Parameters are re-tuned every {cv_reopt_every} rebalance dates using rolling cross-validation on the previous lookback window (80% train / 20% validation), then reused in between",
        f"- Reduced CV grid size: {cv_grid_size} combinations ({len(lambda_grid)} lambda x {len(xi_grid)} xi x {len(kappa_grid)} kappa x {len(eps_grid)} eps)",
        f"- Actual CV re-tunes executed: {n_cv_runs}",
        f"- Mean selected lambda={mean_lambda:.3f}, xi={mean_xi:.3f}, kappa={mean_kappa:.3f}, eps_wc={mean_eps:.3f}",
        f"- Option-mu rolling calibration mean scale: {mu_scale_mean:.3f} (scale is signed in [-1, 1], where 0.0 is neutralized)",
        "- Structural covariance estimate for validation is V Sigma V^T plus an empirical diagonal option-noise floor",
        "- Robust optimizer matrix uses A_eff = V Sigma V^T + D + eps_wc * V V^T (used for allocation, not as a direct realized-covariance forecast)",
        f"- Zero-risk check across the backtest: full-rank share={zero_risk_full_rank_share:.1%}, mean min eig={mean_effective_min_eig:.3e}, "
        f"max cond={max_effective_cond:.3e}, mean ridge={mean_ridge_added:.3e}",
        "- Turnover is penalized in the objective and roll events are charged as sell+buy turnover on the replaced bucket ids",
        "- Stock transaction-cost weights use 0.5% / max(1, S); option transaction-cost weights use a premium-bucket proxy because the paper's exact eta(price) bucket table is not encoded in the repo",
        f"- Mu diagnostics: underlying mean bias={und_bias:.3%}, option mean bias={opt_bias:.3%}, "
        f"underlying mean corr={und_corr:.3f}, option mean corr={opt_corr:.3f}",
        f"- Mean turnover cost per rebalance: {mean_turnover_cost:.6f}",
        f"- Covariance gap mean abs entry: {mean_abs_gap:.6f}",
        f"- Covariance gap Frobenius norm: {float(np.linalg.norm(gap_values)):.6f}",
    ]

    return {
        "symbols": selected_symbols,
        "asset_labels": asset_labels,
        "used_dates": used_dates,
        "predicted_cov_mean": structural_cov_df,
        "structural_cov_mean": structural_cov_df,
        "empirical_noise_mean": empirical_noise_df,
        "optimizer_cov_mean": optimizer_cov_df,
        "robust_addon_mean": robust_addon_df,
        "realized_cov": realized_cov_df.reindex(index=asset_labels, columns=asset_labels),
        "cov_gap": cov_gap.reindex(index=asset_labels, columns=asset_labels),
        "optimizer_cov_gap": optimizer_gap.reindex(index=asset_labels, columns=asset_labels),
        "realized_return_panel": realized_matrix.reindex(columns=asset_labels),
        "predicted_return_panel": predicted_matrix.reindex(columns=asset_labels),
        "mu_diagnostics": mu_compare.reindex(index=asset_labels),
        "weight_path": weight_path,
        "aggregate_weight_path": aggregate_weights,
        "portfolio_path": portfolio_path,
        "latest_allocation": weight_path.iloc[-1].sort_values(ascending=False),
        "param_by_date": pd.DataFrame.from_dict(param_cache_by_date, orient="index").sort_index(),
        "risk_diagnostics": risk_diag_df,
        "benchmark_label": benchmark_label,
        "summary": "\n".join(summary_lines),
    }
