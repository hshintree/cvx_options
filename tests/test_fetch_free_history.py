from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd

from data.fetch_free_history import (
    build_daily_research_candidates,
    build_markowitz_universe,
    build_research_clean_panel,
    compute_ex_ante_moments,
    compute_missing_greeks_from_mid,
    build_underlying_history,
    diagnose_theta_units,
    evaluate_markowitz_sample,
    normalize_historical_chain,
    serialize_contract_key_columns,
)


def _quote_ts(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def _raw_optionsdx_frame() -> pd.DataFrame:
    rows = [
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-02 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "call",
            "option_symbol": "SPY_C_100",
            "strike": 100.0,
            "underlying_price": 100.0,
            "bid": 2.00,
            "ask": 2.10,
            "last": 2.05,
            "open_interest": 1000,
            "volume": 150,
            "implied_volatility": 0.20,
            "delta": 0.55,
            "gamma": 0.08,
            "theta": -12.0,
            "vega": 8.0,
            "rho": 1.2,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-02 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "call",
            "option_symbol": "SPY_C_101",
            "strike": 101.0,
            "underlying_price": 100.0,
            "bid": 1.55,
            "ask": 1.65,
            "last": 1.60,
            "open_interest": 900,
            "volume": 125,
            "implied_volatility": 0.21,
            "delta": 0.48,
            "gamma": 0.07,
            "theta": -11.0,
            "vega": 7.5,
            "rho": 1.1,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-02 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "put",
            "option_symbol": "SPY_P_100",
            "strike": 100.0,
            "underlying_price": 100.0,
            "bid": 1.95,
            "ask": 2.05,
            "last": 2.00,
            "open_interest": 980,
            "volume": 140,
            "implied_volatility": 0.20,
            "delta": -0.45,
            "gamma": 0.08,
            "theta": -12.5,
            "vega": 8.0,
            "rho": -1.1,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-02 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "put",
            "option_symbol": "SPY_P_99",
            "strike": 99.0,
            "underlying_price": 100.0,
            "bid": 1.35,
            "ask": 1.45,
            "last": 1.40,
            "open_interest": 870,
            "volume": 115,
            "implied_volatility": 0.19,
            "delta": -0.38,
            "gamma": 0.07,
            "theta": -10.0,
            "vega": 7.2,
            "rho": -0.9,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-03 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "call",
            "option_symbol": "SPY_C_100",
            "strike": 100.0,
            "underlying_price": 101.0,
            "bid": 2.60,
            "ask": 2.70,
            "last": 2.65,
            "open_interest": 1020,
            "volume": 175,
            "implied_volatility": 0.205,
            "delta": 0.60,
            "gamma": 0.075,
            "theta": -11.8,
            "vega": 7.9,
            "rho": 1.25,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-03 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "call",
            "option_symbol": "SPY_C_101",
            "strike": 101.0,
            "underlying_price": 101.0,
            "bid": 2.00,
            "ask": 2.10,
            "last": 2.05,
            "open_interest": 940,
            "volume": 135,
            "implied_volatility": 0.215,
            "delta": 0.53,
            "gamma": 0.073,
            "theta": -11.1,
            "vega": 7.4,
            "rho": 1.12,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-03 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "call",
            "option_symbol": "SPY_C_102",
            "strike": 102.0,
            "underlying_price": 101.0,
            "bid": 1.55,
            "ask": 1.65,
            "last": 1.60,
            "open_interest": 810,
            "volume": 110,
            "implied_volatility": 0.218,
            "delta": 0.45,
            "gamma": 0.068,
            "theta": -10.5,
            "vega": 7.1,
            "rho": 1.02,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-03 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "put",
            "option_symbol": "SPY_P_100",
            "strike": 100.0,
            "underlying_price": 101.0,
            "bid": 1.55,
            "ask": 1.65,
            "last": 1.60,
            "open_interest": 995,
            "volume": 145,
            "implied_volatility": 0.205,
            "delta": -0.40,
            "gamma": 0.075,
            "theta": -12.2,
            "vega": 7.9,
            "rho": -1.0,
        },
        {
            "underlying_symbol": "SPY",
            "quote_unixtime": _quote_ts("2026-02-03 21:00:00"),
            "expiration": "2026-02-20",
            "option_type": "put",
            "option_symbol": "SPY_P_99",
            "strike": 99.0,
            "underlying_price": 101.0,
            "bid": 1.05,
            "ask": 1.15,
            "last": 1.10,
            "open_interest": 880,
            "volume": 120,
            "implied_volatility": 0.195,
            "delta": -0.32,
            "gamma": 0.068,
            "theta": -9.8,
            "vega": 7.0,
            "rho": -0.82,
        },
    ]
    return pd.DataFrame(rows)


class FetchFreeHistoryTests(unittest.TestCase):
    def test_normalize_optionsdx_schema(self) -> None:
        normalized = normalize_historical_chain(_raw_optionsdx_frame(), source="optionsdx_csv")
        for column in ["symbol", "quote_time", "contract_symbol", "iv_source", "greeks_source"]:
            self.assertIn(column, normalized.columns)
        self.assertTrue((normalized["symbol"] == "SPY").all())
        self.assertTrue((normalized["iv_source"] == "observed").all())
        self.assertTrue((normalized["greeks_source"] == "observed").all())
        self.assertTrue((normalized["price_source"] == "mid").all())

    def test_compute_missing_greeks_from_mid_fills_fields(self) -> None:
        normalized = normalize_historical_chain(_raw_optionsdx_frame().iloc[:1], source="optionsdx_csv")
        normalized.loc[:, ["implied_volatility", "delta", "gamma", "theta", "vega", "rho"]] = np.nan
        normalized.loc[:, ["iv_source", "greeks_source"]] = "missing"
        computed = compute_missing_greeks_from_mid(normalized)
        row = computed.iloc[0]
        self.assertGreater(row["implied_volatility"], 0.0)
        values = row[["delta", "gamma", "theta", "vega", "rho"]].astype(float).to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(row["iv_source"], "inferred_mid")
        self.assertEqual(row["greeks_source"], "inferred_mid")

    def test_normalize_paired_optionsdx_eod_row(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "quote_unixtime": _quote_ts("2026-02-02 21:00:00"),
                    "quote_readtime": "2026-02-02 16:00",
                    "quote_date": "2026-02-02",
                    "underlying_last": 100.0,
                    "expire_date": "2026-02-20",
                    "dte": 18.0,
                    "c_delta": 0.55,
                    "c_gamma": 0.08,
                    "c_vega": 8.0,
                    "c_theta": -12.0,
                    "c_rho": 1.2,
                    "c_iv": 0.20,
                    "c_volume": 150,
                    "c_last": 2.05,
                    "c_bid": 2.00,
                    "c_ask": 2.10,
                    "strike": 100.0,
                    "p_bid": 1.95,
                    "p_ask": 2.05,
                    "p_last": 2.00,
                    "p_delta": -0.45,
                    "p_gamma": 0.08,
                    "p_vega": 8.0,
                    "p_theta": -12.5,
                    "p_rho": -1.1,
                    "p_iv": 0.20,
                    "p_volume": 140,
                    "_source_file": "spy_eod_202602.txt",
                }
            ]
        )
        normalized = normalize_historical_chain(raw, source="optionsdx_csv")
        self.assertEqual(len(normalized), 2)
        self.assertEqual(set(normalized["option_side"]), {"call", "put"})
        self.assertTrue((normalized["symbol"] == "SPY").all())

    def test_research_panel_tracks_four_contracts(self) -> None:
        normalized = normalize_historical_chain(_raw_optionsdx_frame(), source="optionsdx_csv")
        panel = build_research_clean_panel(normalized, target_dte=21)
        self.assertEqual(len(panel), 1)
        row = panel.iloc[0]
        self.assertGreater(row["call_2_strike"], row["call_1_strike"])
        self.assertLess(row["put_2_strike"], row["put_1_strike"])
        self.assertEqual(row["call_1_symbol"], "SPY_C_100")
        self.assertEqual(row["call_2_symbol"], "SPY_C_101")
        self.assertEqual(row["put_1_symbol"], "SPY_P_100")
        self.assertEqual(row["put_2_symbol"], "SPY_P_99")
        self.assertAlmostEqual(row["R_und_real"], 0.01, places=6)

    def test_daily_candidates_keep_contract_keys_stable_without_roll(self) -> None:
        normalized = normalize_historical_chain(_raw_optionsdx_frame(), source="optionsdx_csv")
        candidates = build_daily_research_candidates(normalized, target_dte=21, roll_dte=7)
        self.assertEqual(len(candidates), 2)
        for bucket in ["call_1", "call_2", "put_1", "put_2"]:
            self.assertEqual(candidates.iloc[0][f"{bucket}_contract_key"], candidates.iloc[1][f"{bucket}_contract_key"])
        self.assertTrue(bool(candidates.iloc[0]["rolled"]))
        self.assertFalse(bool(candidates.iloc[1]["rolled"]))

    def test_serialize_contract_key_columns_makes_parquet_safe_strings(self) -> None:
        normalized = normalize_historical_chain(_raw_optionsdx_frame(), source="optionsdx_csv")
        candidates = build_daily_research_candidates(normalized, target_dte=21, roll_dte=7)
        serialized = serialize_contract_key_columns(candidates)
        value = serialized.iloc[0]["call_1_contract_key"]
        self.assertIsInstance(value, str)
        self.assertIn("SPY|call|2026-02-20|100.0000", value)

    def test_diagnose_theta_units_returns_recommendation(self) -> None:
        normalized = normalize_historical_chain(_raw_optionsdx_frame(), source="optionsdx_csv")
        panel = build_research_clean_panel(normalized, target_dte=21)
        diagnosis = diagnose_theta_units(panel)

        self.assertIn(diagnosis["recommendation"], {"per_year", "per_day"})
        self.assertEqual(set(diagnosis.keys()), {"per_year", "per_day", "recommendation"})
        self.assertIn("overall", diagnosis["per_year"])
        self.assertIn("by_bucket", diagnosis["per_year"])
        self.assertEqual(len(diagnosis["per_year"]["by_bucket"]), 4)

    def test_markowitz_universe_is_risky_only(self) -> None:
        base = _raw_optionsdx_frame()
        q1 = normalize_historical_chain(base, source="optionsdx_csv")
        q2 = normalize_historical_chain(base.assign(
            underlying_symbol="AAPL",
            option_symbol=lambda df: df["option_symbol"].str.replace("SPY", "AAPL"),
            underlying_price=lambda df: df["underlying_price"] + 50.0,
            strike=lambda df: df["strike"] + 50.0,
        ), source="optionsdx_csv")
        normalized = pd.concat([q1, q2], ignore_index=True)

        panel = build_research_clean_panel(normalized, target_dte=21)
        history = build_underlying_history(normalized)
        extra_dates = pd.DataFrame(
            [
                {"symbol": "SPY", "quote_date": pd.Timestamp("2026-01-29"), "underlying_spot": 98.0},
                {"symbol": "SPY", "quote_date": pd.Timestamp("2026-01-30"), "underlying_spot": 99.0},
                {"symbol": "SPY", "quote_date": pd.Timestamp("2026-01-31"), "underlying_spot": 100.5},
                {"symbol": "AAPL", "quote_date": pd.Timestamp("2026-01-29"), "underlying_spot": 148.0},
                {"symbol": "AAPL", "quote_date": pd.Timestamp("2026-01-30"), "underlying_spot": 149.5},
                {"symbol": "AAPL", "quote_date": pd.Timestamp("2026-01-31"), "underlying_spot": 150.5},
            ]
        )
        extra_dates = extra_dates.sort_values(["symbol", "quote_date"]).reset_index(drop=True)
        extra_dates["underlying_return"] = extra_dates.groupby("symbol")["underlying_spot"].pct_change()
        history = pd.concat([history, extra_dates], ignore_index=True).sort_values(["symbol", "quote_date"]).reset_index(drop=True)
        history["underlying_return"] = history.groupby("symbol")["underlying_spot"].pct_change()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ex_ante = compute_ex_ante_moments(panel, history, lookback=2)
        preview = build_markowitz_universe(ex_ante, history, lookback=2)

        self.assertEqual(preview["mu_risky"].shape[0], 10)
        self.assertEqual(preview["sigma_risky"].shape, (10, 10))
        self.assertIn("SPY_UND", preview["cost_weights"].index)
        self.assertNotIn("CASH", preview["cost_weights"].index)
        self.assertEqual(preview["v_matrix"].shape, (10, 2))
        self.assertEqual(preview["robust_bump"].shape, (10, 10))
        self.assertEqual(preview["worst_case_bump"].shape, (10, 10))
        self.assertIn("risk_diagnostics", preview)
        self.assertTrue(bool(preview["risk_diagnostics"]["zero_risk_fixed"]))
        self.assertIn("fully invested across risky assets only", preview["summary"])

    def test_evaluate_markowitz_sample_returns_weight_path(self) -> None:
        base = _raw_optionsdx_frame()
        q1 = normalize_historical_chain(base, source="optionsdx_csv")
        q2 = normalize_historical_chain(base.assign(
            underlying_symbol="AAPL",
            option_symbol=lambda df: df["option_symbol"].str.replace("SPY", "AAPL"),
            underlying_price=lambda df: df["underlying_price"] + 50.0,
            strike=lambda df: df["strike"] + 50.0,
        ), source="optionsdx_csv")
        normalized = pd.concat([q1, q2], ignore_index=True)

        panel = build_research_clean_panel(normalized, target_dte=21)
        history = build_underlying_history(normalized)
        extra_dates = pd.DataFrame(
            [
                {"symbol": "SPY", "quote_date": pd.Timestamp("2026-01-29"), "underlying_spot": 98.0},
                {"symbol": "SPY", "quote_date": pd.Timestamp("2026-01-30"), "underlying_spot": 99.0},
                {"symbol": "SPY", "quote_date": pd.Timestamp("2026-01-31"), "underlying_spot": 100.5},
                {"symbol": "AAPL", "quote_date": pd.Timestamp("2026-01-29"), "underlying_spot": 148.0},
                {"symbol": "AAPL", "quote_date": pd.Timestamp("2026-01-30"), "underlying_spot": 149.5},
                {"symbol": "AAPL", "quote_date": pd.Timestamp("2026-01-31"), "underlying_spot": 150.5},
            ]
        )
        history = pd.concat([history, extra_dates], ignore_index=True).sort_values(["symbol", "quote_date"]).reset_index(drop=True)
        history["underlying_return"] = history.groupby("symbol")["underlying_spot"].pct_change()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ex_ante = compute_ex_ante_moments(panel, history, lookback=2)
        evaluation = evaluate_markowitz_sample(ex_ante, history, lookback=2)

        self.assertEqual(evaluation["predicted_cov_mean"].shape, (10, 10))
        self.assertEqual(len(evaluation["weight_path"]), 1)
        self.assertAlmostEqual(float(evaluation["weight_path"].iloc[0].sum()), 1.0, places=6)
        self.assertNotIn("CASH", evaluation["weight_path"].columns)
        self.assertIn("risk_diagnostics", evaluation)
        self.assertIn("Experiment setup:", evaluation["summary"])


if __name__ == "__main__":
    unittest.main()
