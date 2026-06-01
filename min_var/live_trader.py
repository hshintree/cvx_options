"""
AlpacaPortfolioController — run-once live/paper trading for the min-var strategy.

Uses cvxportfolio's SinglePeriodOptimization (winning config: EWMA γ=5, monthly)
by loading cvxport_best_params.json written by run_cvxportfolio_grid_search.py.

Falls back to the hand-rolled solver (best_params.json) if the cvxportfolio
params file does not exist.

Workflow on each invocation:
  1. Load cvxport_best_params.json (or best_params.json as fallback)
  2. Check if rebalance is due (fixed-period from rebal_state.json)
  3. Fetch recent price bars; run a warm-up cvxportfolio backtest ending today
  4. Extract final weights from backtest result
  5. Print summary; submit market orders if not dry_run

Usage:
    from min_var.live_trader import AlpacaPortfolioController
    ctrl = AlpacaPortfolioController(symbols=EXTENDED_SYMBOLS)
    summary = ctrl.run(dry_run=True)
    print(summary)
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from min_var.config import R_ANN, OUTPUT_DIR  # noqa: E402
from min_var.cvxport_grid_search import load_cvxport_best_params  # noqa: E402
from min_var.rebalance_state import should_rebalance, save_state  # noqa: E402

load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)

# How many trading-day equivalents of history to fetch for estimator warm-up.
# cov_halflife * 6 gives ~6 half-lives of EWMA weight before the first estimate.
_WARMUP_MULTIPLIER = 6


class AlpacaPortfolioController:
    """
    Run-once portfolio controller for equity + bond ETF strategies.

    Parameters
    ----------
    symbols      : list of ticker symbols to manage (no options)
    params_path  : path to cvxport_best_params.json (or best_params.json)
                   If None, looks for cvxport_best_params.json first, then
                   best_params.json.
    paper        : True → paper trading account; False → live account
    """

    def __init__(
        self,
        symbols: list[str],
        params_path: str | Path | None = None,
        paper: bool = True,
    ):
        self.symbols = symbols
        self.paper   = paper

        # ── Load params: prefer cvxportfolio params, fall back to hand-rolled ──
        cvxport_path = OUTPUT_DIR / "cvxport_best_params.json"
        fallback_path = OUTPUT_DIR / "best_params.json"

        if params_path:
            p = Path(params_path)
            raw = load_cvxport_best_params(p)
            if raw is None:
                raise FileNotFoundError(f"Could not load params from {p}")
        elif cvxport_path.exists():
            raw = load_cvxport_best_params(cvxport_path)
        else:
            raise FileNotFoundError(
                "No params file found. Run run_cvxportfolio_grid_search.py to create one."
            )

        self._params         = raw
        self._use_cvxport    = "cov_halflife" in raw   # cvxport params have this key

        # cvxportfolio params
        self.gamma           = float(raw.get("gamma", 5.0))
        self.max_wt          = float(raw.get("max_wt", 0.40))
        self.cov_halflife    = int(raw.get("cov_halflife", 21))
        self.mu_halflife     = int(raw.get("mu_halflife", 252))
        self.mu_method       = str(raw.get("mu_method", "ewma"))
        self.use_mpo         = bool(raw.get("use_mpo", False))
        self.planning_horizon = int(raw.get("planning_horizon", 3))
        # rebal_freq as string for cvxportfolio ('monthly') or int for hand-rolled
        _rf = raw.get("rebal_freq", "monthly")
        self.rebal_freq_str  = str(_rf) if isinstance(_rf, str) else "monthly"
        # for should_rebalance() we need an integer trading-day count
        self.rebal_freq_days = int(raw.get("rebal_freq_days", 21))

        # hand-rolled fallback param
        self.lookback = int(raw.get("lookback", 126))

        # Shrinkage: 0 = pure EWMA momentum; 0.5 = compress spread 50% toward mean
        self.mu_shrinkage = float(raw.get("mu_shrinkage", 0.0))

        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        from alpaca.trading.client import TradingClient  # type: ignore
        from alpaca.data.historical import StockHistoricalDataClient  # type: ignore

        self._trade_client = TradingClient(api_key, secret_key, paper=paper)
        self._data_client  = StockHistoricalDataClient(api_key, secret_key)

        _mode = ("cvxportfolio MPO" if (self._use_cvxport and self.use_mpo)
                 else "cvxportfolio SPO" if self._use_cvxport
                 else "hand-rolled CVXPY")
        logger.info("AlpacaPortfolioController initialised (%s, γ=%.1f)", _mode, self.gamma)

    # ── Portfolio state ───────────────────────────────────────────────────────

    def get_portfolio_state(self) -> tuple[dict[str, float], float]:
        """
        Returns (current_weights, portfolio_value).
        current_weights is {symbol: weight}; missing symbols default to 0.
        """
        account    = self._trade_client.get_account()
        port_value = float(account.portfolio_value)

        positions = self._trade_client.get_all_positions()
        cur_wts: dict[str, float] = {}
        for pos in positions:
            sym = pos.symbol
            if sym in self.symbols:
                cur_wts[sym] = float(pos.market_value) / port_value

        for sym in self.symbols:
            cur_wts.setdefault(sym, 0.0)

        return cur_wts, port_value

    # ── Price data ────────────────────────────────────────────────────────────

    def fetch_latest_returns(self, arith: bool = False) -> pd.DataFrame:
        """
        Load full price history for EWMA warm-up.

        Strategy (in order):
          1. Load from local parquet files (data/raw/{sym}_daily.parquet).
             These go back to 2019 and are fast to read.  Append the last few
             days from Alpaca so today's prices are included.
          2. If parquet files are missing, fall back to a full Alpaca fetch
             starting from 2019-01-01 (slower but complete).

        A 252-day EWMA halflife requires 4× the halflife (~1008 trading days)
        to converge.  Using 172 days (the previous approach) produced noisy,
        under-converged estimates.

        Parameters
        ----------
        arith : True → arithmetic returns; False → log-returns
        """
        try:
            return self._fetch_with_local_parquet(arith=arith)
        except Exception as exc:
            logger.warning("Local parquet unavailable (%s) — fetching full history from Alpaca", exc)
            return self._fetch_from_alpaca(start_date="2019-01-01", arith=arith)

    def _fetch_with_local_parquet(self, arith: bool = False) -> pd.DataFrame:
        """Load bulk history from local parquet, then append the freshest Alpaca bars."""
        from min_var.data_equity import load_equity_returns

        log_rets = load_equity_returns(self.symbols, start="2019-01-01")

        # Supplement with the last 10 trading days from Alpaca to capture today
        try:
            fresh = self._fetch_from_alpaca(start_date=None, n_cal_days=20, arith=False)
            # Merge: keep parquet rows, add any new dates from Alpaca fetch
            fresh_new = fresh[~fresh.index.isin(log_rets.index)]
            if len(fresh_new):
                log_rets = pd.concat([log_rets, fresh_new]).sort_index()
        except Exception:
            pass   # parquet data is good enough if Alpaca supplement fails

        if arith:
            return np.expm1(log_rets)
        return log_rets

    def _fetch_from_alpaca(
        self,
        start_date: str | None = None,
        n_cal_days: int | None = None,
        arith: bool = False,
    ) -> pd.DataFrame:
        """Fetch daily bars from Alpaca.  Either start_date or n_cal_days must be given."""
        from alpaca.data.requests import StockBarsRequest  # type: ignore
        from alpaca.data.timeframe import TimeFrame  # type: ignore

        if start_date:
            start = pd.Timestamp(start_date).to_pydatetime()
        elif n_cal_days:
            start = datetime.now() - timedelta(days=n_cal_days)
        else:
            raise ValueError("Provide start_date or n_cal_days")

        req = StockBarsRequest(
            symbol_or_symbols=self.symbols,
            timeframe=TimeFrame.Day,
            start=start,
        )
        bars = self._data_client.get_stock_bars(req)
        df   = bars.df

        if df.empty:
            raise RuntimeError("No price bars returned from Alpaca")

        if isinstance(df.index, pd.MultiIndex):
            df = df.unstack(level=0)["close"]
        else:
            df = df[["close"]]

        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.columns = [c.upper() if isinstance(c, str) else c for c in df.columns]
        df = df[[s for s in self.symbols if s in df.columns]]

        if arith:
            return (df / df.shift(1) - 1).dropna()
        return np.log(df / df.shift(1)).dropna()

    # ── Optimization ─────────────────────────────────────────────────────────

    def compute_target_weights(self, returns_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute target portfolio weights.

        If cvxport params are loaded: runs a cvxportfolio SPO/MPO warm-up
        backtest over the fetched history window and extracts the final weights.

        If hand-rolled params are loaded: uses rolling empirical Σ/μ + CVXPY.
        """
        if self._use_cvxport:
            return self._compute_weights_cvxport(returns_df)
        else:
            return self._compute_weights_handrolled(returns_df)

    def _compute_weights_cvxport(self, arith_rets: pd.DataFrame) -> dict[str, float]:
        """
        Run a warm-up cvxportfolio backtest ending at the latest available date.
        Extracts the final-period portfolio weights from result.w.

        Special case: when mu_method='zero', cvxportfolio puts all weight in cash
        (holding cash IS optimal when μ=0 and there's a risk-free alternative).
        We route that case directly to _compute_weights_minvar which uses CVXPY
        with the EWMA covariance to get the true min-variance equity portfolio.

        Bug note: cvxportfolio allocates equity weight + cash.  We must NOT
        renormalize to equity-only — that turns "40% GLD, 60% cash" into "100% GLD".
        We keep raw weights; the leftover goes to cash in the Alpaca account.
        """
        # Pure min-var: skip cvxportfolio (which would allocate all to cash)
        # and solve directly with CVXPY + EWMA Σ.
        if self.mu_method == "zero":
            return self._compute_weights_minvar(arith_rets)

        from min_var.cvxport_backtest import run_cvxportfolio_backtest

        min_needed = max(self.cov_halflife * 4, self.mu_halflife * 2)
        if len(arith_rets) < min_needed:
            logger.warning(
                "Too little history (%d rows, need %d) — using equal weights",
                len(arith_rets), min_needed,
            )
            return {s: 1.0 / len(self.symbols) for s in self.symbols}

        # Skip enough rows so both EWMA estimators are well-converged
        warmup_rows = min(max(self.cov_halflife * 4, self.mu_halflife * 2),
                          len(arith_rets) - 30)
        start_date  = str(arith_rets.index[warmup_rows].date())
        logger.info("Warm-up backtest: %d rows history, start=%s", len(arith_rets), start_date)

        try:
            _, meta = run_cvxportfolio_backtest(
                returns_df=arith_rets,
                gamma=self.gamma,
                cov_halflife=self.cov_halflife,
                mu_halflife=self.mu_halflife,
                max_wt=self.max_wt,
                rebal_freq=self.rebal_freq_str,
                mu_method=self.mu_method,
                mu_shrinkage=getattr(self, "mu_shrinkage", 0.0),
                use_mpo=self.use_mpo,
                planning_horizon=self.planning_horizon,
                start=start_date,
            )
            result = meta["result"]
            w_series = result.w.iloc[-1].drop("cash", errors="ignore")
        except Exception as exc:
            logger.error("cvxportfolio warm-up backtest failed: %s — using equal weights", exc)
            return {s: 1.0 / len(self.symbols) for s in self.symbols}

        # Align to self.symbols.  Clip negative (numerical noise) and cap at max_wt.
        # SCS solver can return weights slightly above the MaxWeights constraint due to
        # numerical tolerance — clipping post-hoc enforces it cleanly.
        # DO NOT divide by equity sum — cash is intentional, not a rounding residual.
        wts   = w_series.reindex(self.symbols, fill_value=0.0).clip(lower=0.0,
                                                                     upper=self.max_wt)
        total = wts.sum()

        if total < 1e-4:
            # Unexpected all-cash result from non-zero mu case — fall back to min-var
            logger.warning("All weights near zero after backtest — falling back to min-var")
            return self._compute_weights_minvar(arith_rets)

        # Guard only: if total somehow exceeds 1.0 (shouldn't after per-asset clip),
        # scale down proportionally to avoid over-investing.
        if total > 1.01:
            wts = wts / total

        return dict(wts)

    def _compute_weights_minvar(self, arith_rets: pd.DataFrame) -> dict[str, float]:
        """
        Pure minimum-variance portfolio using EWMA covariance + CVXPY.

        Bypasses cvxportfolio entirely to avoid the cash-allocation issue that
        occurs when mu=0 (holding cash is optimal in cvxportfolio's formulation,
        leaving all equity weights near zero).
        """
        import cvxpy as cp
        from min_var.cvxport_accuracy import _ewma_cov

        n = len(self.symbols)
        valid = [s for s in self.symbols if s in arith_rets.columns]
        if len(valid) < 2:
            return {s: 1.0 / n for s in self.symbols}

        rets_arr = arith_rets[valid].dropna().values.astype(float)
        if len(rets_arr) < self.cov_halflife * 2:
            logger.warning("Too few rows for EWMA min-var (%d) — equal weights", len(rets_arr))
            return {s: 1.0 / n for s in self.symbols}

        # EWMA Σ (annualised)
        sig_d   = _ewma_cov(rets_arr, self.cov_halflife)
        sig_ann = sig_d * 252
        # Regularise: small diagonal bump for numerical stability
        sig_ann = 0.5 * (sig_ann + sig_ann.T) + 1e-6 * np.eye(len(valid))

        # CVXPY: min w'Σw  s.t. 1'w=1, 0≤w≤max_wt
        w = cp.Variable(len(valid))
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, sig_ann)),
            [cp.sum(w) == 1.0, w >= 0.0, w <= self.max_wt],
        )
        try:
            prob.solve(solver=cp.CLARABEL, warm_start=False)
            if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
                prob.solve(solver=cp.SCS)
        except Exception:
            prob.solve(solver=cp.SCS)

        if w.value is None:
            logger.warning("Min-var CVXPY solve failed — equal weights")
            return {s: 1.0 / n for s in self.symbols}

        wv = np.clip(w.value, 0.0, self.max_wt)
        total = wv.sum()
        if total > 1e-6:
            wv = wv / total  # normalise (safe: pure equity, no cash concept)

        result = dict(zip(valid, wv.tolist()))
        # Fill missing symbols (absent from data) with zero
        for s in self.symbols:
            result.setdefault(s, 0.0)
        return result

    def compute_ewma_forecasts(self, arith_rets: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """
        Compute EWMA return forecast (μ) and covariance (Σ) using all available data.

        Returns
        -------
        mu_ann   : pd.Series  — annualised expected return per asset
        sig_ann  : pd.DataFrame — annualised covariance matrix
        """
        from min_var.cvxport_accuracy import _ewma_mean, _ewma_cov

        rets_arr = arith_rets.values.astype(float)
        mu_d     = _ewma_mean(rets_arr, self.mu_halflife)
        sig_d    = _ewma_cov(rets_arr, self.cov_halflife)

        mu_ann  = pd.Series(mu_d * 252, index=arith_rets.columns)
        sig_ann = pd.DataFrame(sig_d * 252,
                               index=arith_rets.columns,
                               columns=arith_rets.columns)
        return mu_ann, sig_ann

    def _compute_weights_handrolled(self, log_rets: pd.DataFrame) -> dict[str, float]:
        """Fallback: rolling empirical Σ/μ + CVXPY solve_portfolio."""
        from min_var.optimizer import solve_portfolio  # type: ignore

        window = log_rets.iloc[-self.lookback:]
        valid  = [c for c in self.symbols if c in window.columns
                  and window[c].notna().sum() >= self.lookback // 2]

        if len(valid) < 2:
            logger.warning("Not enough valid symbols for optimization (%d). "
                           "Returning equal weights.", len(valid))
            return {s: 1.0 / len(self.symbols) for s in self.symbols}

        clean = window[valid].dropna()
        Sig   = np.cov(clean.T)
        if Sig.ndim == 0:
            Sig = np.array([[float(Sig)]])
        Sig = 0.5 * (Sig + Sig.T) + 1e-7 * np.eye(len(valid))
        Sig = Sig * 252

        mu = np.array([
            float(window[c].dropna().mean()) * 252
            if window[c].notna().sum() >= 5 else R_ANN
            for c in valid
        ])

        gamma  = self.gamma if self.gamma > 0 else None
        wv, _, _ = solve_portfolio(cols=valid, mu=mu, Sig=Sig,
                                   gamma=gamma, max_wt=self.max_wt)

        if wv is None:
            logger.warning("Optimizer returned None — using equal weights")
            return {s: 1.0 / len(valid) for s in valid}

        return dict(zip(valid, wv.tolist()))

    # ── Orders ────────────────────────────────────────────────────────────────

    def compute_orders(
        self,
        current_weights: dict[str, float],
        target_weights:  dict[str, float],
        portfolio_value: float,
        min_trade_frac:  float = 0.01,
    ) -> list[dict]:
        """
        Compute order list to move from current_weights to target_weights.
        Skips trades whose notional is below min_trade_frac × portfolio_value.

        Returns list of {"symbol", "side", "notional", "delta_wt"}.
        """
        orders = []
        for sym in self.symbols:
            cur  = current_weights.get(sym, 0.0)
            tgt  = target_weights.get(sym, 0.0)
            delta    = tgt - cur
            notional = abs(delta) * portfolio_value
            if notional < min_trade_frac * portfolio_value:
                continue
            orders.append({
                "symbol":   sym,
                "side":     "buy" if delta > 0 else "sell",
                "notional": round(notional, 2),
                "delta_wt": delta,
            })
        return orders

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        dry_run: bool = False,
        force:   bool = False,
    ) -> pd.DataFrame:
        """
        Full pipeline: state check → weight computation → (optional) execution.

        Parameters
        ----------
        dry_run : print proposed orders but don't submit
        force   : bypass rebalance-period check

        Returns
        -------
        Summary DataFrame (symbol, current_wt, target_wt, delta_wt, notional, action)
        """
        # ── Rebalance check ───────────────────────────────────────────────────
        if not force:
            do_rebal, reason = should_rebalance(self.rebal_freq_days)
            if not do_rebal:
                print(f"\nNo rebalance needed — {reason}")
                self._print_config()
                cur_wts, port_value = self.get_portfolio_state()
                arith_rets = self.fetch_latest_returns(arith=self._use_cvxport)
                tgt_wts    = self.compute_target_weights(arith_rets)
                return _build_summary(self.symbols, cur_wts, tgt_wts, port_value,
                                      executed=False, reason=reason)
            else:
                print(f"\nRebalance due — {reason}")
        else:
            print("\nForced rebalance (--force).")
            do_rebal = True

        self._print_config()

        # ── Fetch state + compute ─────────────────────────────────────────────
        cur_wts, port_value = self.get_portfolio_state()
        print(f"  Portfolio value: ${port_value:,.2f}")

        arith = self._use_cvxport
        rets  = self.fetch_latest_returns(arith=arith)
        print(f"  Fetched {len(rets)} days of {'arithmetic' if arith else 'log'} returns")

        tgt_wts = self.compute_target_weights(rets)
        orders  = self.compute_orders(cur_wts, tgt_wts, port_value)

        # ── Execute ───────────────────────────────────────────────────────────
        if dry_run:
            print(f"\n  DRY RUN — {len(orders)} order(s) proposed (not submitted)")
        elif orders:
            from alpaca.trading.requests import MarketOrderRequest  # type: ignore
            from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore

            submitted = 0
            for o in orders:
                try:
                    req = MarketOrderRequest(
                        symbol=o["symbol"],
                        notional=o["notional"],
                        side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    self._trade_client.submit_order(req)
                    submitted += 1
                    logger.info("Submitted %s %s $%.2f",
                                o["side"], o["symbol"], o["notional"])
                except Exception as exc:
                    logger.error("Order failed for %s: %s", o["symbol"], exc)

            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            save_state(today)
            print(f"\n  {submitted}/{len(orders)} orders submitted. State saved.")
        else:
            print("\n  No orders to submit (all drifts below threshold).")
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            save_state(today)

        return _build_summary(self.symbols, cur_wts, tgt_wts, port_value,
                              executed=(not dry_run and bool(orders)),
                              reason="rebalanced" if do_rebal else "")

    def _print_config(self):
        if self._use_cvxport:
            mode = ("MPO(h=%d)" % self.planning_horizon) if self.use_mpo else "SPO"
            print(f"  Strategy: cvxportfolio {mode}, γ={self.gamma:.1f}, "
                  f"Σ_hl={self.cov_halflife}d, μ={self.mu_method}, "
                  f"rebal={self.rebal_freq_str}, max_wt={self.max_wt:.0%}")
        else:
            print(f"  Strategy: hand-rolled mean-var, γ={self.gamma:.0f}, "
                  f"lookback={self.lookback}d, rebal={self.rebal_freq_days}d, "
                  f"max_wt={self.max_wt:.0%}")


def _build_summary(
    symbols:    list[str],
    cur_wts:    dict,
    tgt_wts:    dict,
    port_value: float,
    executed:   bool,
    reason:     str,
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        cw    = cur_wts.get(sym, 0.0)
        tw    = tgt_wts.get(sym, 0.0)
        delta = tw - cw
        rows.append({
            "symbol":     sym,
            "current_wt": cw,
            "target_wt":  tw,
            "delta_wt":   delta,
            "notional":   abs(delta) * port_value,
            "action":     ("buy" if delta > 0 else "sell") if abs(delta) > 0.005 else "hold",
            "executed":   executed,
        })
    df = pd.DataFrame(rows).set_index("symbol")
    df.attrs["portfolio_value"] = port_value
    df.attrs["reason"]          = reason
    return df
