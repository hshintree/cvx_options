# SPY + Options Markowitz++ Experiment — Execution Plan

## 1. Goal Summary

- **Universe**: 4 assets — SPY shares, SPY calls (proxy), SPY puts (proxy), cash.
- **Optimization**: Long-only Markowitz-style with **Markowitz++** features: turnover, trading costs, worst-case/robust return.
- **Frequency**: Rebalance once per week.
- **Options rule**: If weight on “calls” is 10%, allocate 10% of portfolio to the call expiring by next period (e.g., buy that option with 10% of NAV). Same idea for puts.
- **Forecasts**: Expected returns and covariance for the 4 assets; optionally from options-implied (risk-neutral) distribution or fitted models (log-normal / Gaussian mixture).

---

## 2. Can You Use CVX Libraries?

**Yes.** **Cvxportfolio** is the right fit and makes the experiment much easier.

| Need | Cvxportfolio / CVX |
|------|---------------------|
| Long-only Markowitz | `SinglePeriodOptimization` + `LongOnly()` |
| Turnover | `TurnoverLimit(delta)` constraint |
| Trading costs | `TransactionCost` or `StocksTransactionCost` in the objective |
| Worst-case / robust risk | `WorstCaseRisk([...])`; robust return via `ReturnsForecastError(deltas)` |
| Custom μ, Σ | `ReturnsForecast(r_hat=...)`, `FullCovariance(Sigma=...)` with your DataFrames |
| Weekly data | `UserProvidedMarketData(..., trading_frequency='weekly')` or pre-resampled data |

You can stay in **cvxpy** only if you prefer to build the QP yourself (max μ'w − (γ/2)w'Σw − costs, s.t. long-only, turnover), but then you’d implement simulator, costs, and calendar logic by hand. **Recommendation: use Cvxportfolio for the optimizer and simulator; use cvxpy only if you need a one-off custom constraint or objective.**

---

## 3. Data: yfinance vs Alpaca

### 3.1 yfinance

- **Pros**: Free, no API key for basic use; `yf.Ticker("SPY").option_chain(date)` and history for SPY; good for a self-contained experiment.
- **Cons**: Options are snapshot (current expirations); historical options data is limited (no full history of past option prices). You can only “forward-fill” by getting options at each rebalance and computing one-period returns for the chosen contract.

**Clean approach with yfinance:**

1. **SPY**: `yf.Ticker("SPY").history(period="max")` → resample to weekly (e.g., Friday close), compute weekly open-to-open returns.
2. **Cash**: Risk-free rate (e.g., FRED or constant) → weekly return.
3. **Calls / Puts**: Each week at rebalance time:
   - Get option chain for next expiry (e.g., next Friday or next monthly).
   - Pick one “representative” call (e.g., ATM or 1–2% OTM) and one put.
   - Record mid (or last) price at \(t\) and at \(t+1\) (or expiry); compute one-period return. That gives you a **synthetic weekly return series** for “the call” and “the put” (rolling front-month/next-expiry).

So you **don’t** need a continuous options history: you build a single “call” and “put” return series by, each week, defining which option to use and computing its return over the coming week.

### 3.2 Alpaca

- **Pros**: Proper options API, historical bars for options possible (depending on plan); good for production.
- **Cons**: API key; rate limits; more setup.

**Recommendation for this experiment:** Start with **yfinance** to get the pipeline and math right (SPY + synthetic call/put returns + cash). Add **Alpaca** later if you want better options history or live trading.

---

## 4. Data Pipeline (Clean Pull and Cleanup)

Suggested layout:

1. **Module: `data/fetch.py` (or `data_fetch.py`)**
   - `fetch_spy_weekly(start, end)` → DataFrame: datetime index, columns e.g. `open`, `high`, `low`, `close`, `volume` (weekly).
   - `fetch_option_chain_spy(as_of_date)` → calls and puts with strike, expiry, bid, ask, last, impl_vol (if available).
   - `select_front_option(calls_or_puts, expiry_target)` → pick one contract (e.g., ATM by strike vs current SPY price).

2. **Module: `data/options_returns.py`**
   - For each week \(t\): get option chain at \(t\), select call and put expiring by \(t+1\), get price at \(t\). At \(t+1\) use settlement (or close) for that option → return for that week.
   - Build two series: `call_returns`, `put_returns` (same index as weekly SPY).

3. **Module: `data/clean.py`**
   - Align all to same weekly calendar (e.g., Friday).
   - Drop weeks with missing SPY or option data; forward-fill cash if needed.
   - Output: one DataFrame of **weekly open-to-open returns** with columns `['SPY', 'SPY_CALL', 'SPY_PUT', 'CASH']` (or similar names). Optionally a second DataFrame for **volumes** (SPY volume; options you can set to NaN or proxy) and a third for **prices** (SPY open; option “prices” can be last or mid).

4. **Cvxportfolio input**
   - Use `UserProvidedMarketData(returns=returns_df, volumes=volumes_df, prices=prices_df, trading_frequency='weekly')` so that the backtest is already weekly. No need to pass `trading_frequency` if the index of `returns_df` is already weekly.

---

## 5. Expected Returns and Covariance

You have two conceptual approaches; both can feed the same optimizer.

### 5.1 Risk-neutral distribution from options, then sampling

- **Steps**:
  1. At each rebalance, get option chain (calls/puts) for a given expiry.
  2. Build **risk-neutral density** (RND) over terminal SPY price (e.g., Breeden–Litzenberger from option prices, or fit a smooth IV surface and back out RND).
  3. Sample \(S_T\) from RND; convert to log-return for SPY: \(r_{\text{SPY}} = \log(S_T/S_0)\) (or simple return).
  4. For “call” and “put” assets: from same \(S_T\) sample, compute payoff of the chosen call and put, then \((payoff/price - 1)\) as one-period return.
  5. Cash: use risk-free rate (or constant).
  6. Repeat for many samples → **sample mean** \(\hat{\mu}\) and **sample covariance** \(\hat{\Sigma}\) for the 4 assets.

- **Caveat**: RND is risk-neutral, so \(\mathbb{E}[r]\) under RND is not the real-world expected return. You can:
  - Use \(\hat{\Sigma}\) from RND samples (often reasonable for volatility/correlation) and either keep \(\hat{\mu}\) as-is for a “market-implied” view or replace SPY/cash with your own expected return assumptions; or
  - Apply a simple risk-premium shift (e.g., add a constant to SPY expected return) and keep covariance from samples.

### 5.2 Parametric models (log-normal or Gaussian mixture)

- **Log-normal**: Assume \(\ln(S_T/S_0)\) is normal with mean \(\mu\), variance \(\sigma^2\); fit \(\mu,\sigma\) from IV (e.g., ATM IV) or from recent SPY returns. Then sample \(S_T\), compute SPY return and option payoffs, then sample means/covariances.
- **Gaussian mixture**: Fit a mixture to implied volatilities or to historical returns; sample from the mixture; again map to SPY and option returns and estimate \(\hat{\mu}, \hat{\Sigma}\).

**Practical recommendation**: Start with **5.1 (RND sampling)** to use options data directly; optionally compare with **5.2** (e.g., single log-normal from ATM IV) for robustness. Use the same \((\hat{\mu}, \hat{\Sigma})\) in Cvxportfolio via `ReturnsForecast(r_hat=...)` and `FullCovariance(Sigma=...)`.

---

## 6. Markowitz++ Formulation (What to Implement)

- **Objective** (maximize):
  - **Return**: `ReturnsForecast(r_hat=your_mu)`
  - **Risk**: e.g. `gamma_risk * FullCovariance(Sigma=your_Sigma)` or `WorstCaseRisk([FullCovariance(...), ...])`
  - **Trading costs**: `TransactionCost(a=..., b=...)` or `StocksTransactionCost(...)` so that turnover is penalized in the objective.
  - **Robust return (worst-case return)**: add `ReturnsForecastError(deltas=...)` to penalize positions in assets with uncertain expected returns (this approximates a robust/worst-case return formulation).

- **Constraints**:
  - `LongOnly(applies_to_cash=True)` (or `False` if you allow negative cash).
  - `TurnoverLimit(delta)` to cap turnover (e.g. 0.2 = 20% per week).

- **Policy**:
```python
policy = cvx.SinglePeriodOptimization(
    objective=cvx.ReturnsForecast(r_hat=r_hat_estimator)
              - gamma_risk * cvx.FullCovariance(Sigma=sigma_estimator)
              - cvx.TransactionCost(a=..., b=...)
              - cvx.ReturnsForecastError(deltas=...),
    constraints=[cvx.LongOnly(), cvx.TurnoverLimit(0.2)]
)
```

Use `UserProvidedMarketData` so that `r_hat` and `Sigma` can be your own DataFrames (one row per time, columns = assets).

---

## 7. Options “10% = 10% of portfolio in that option”

- The optimizer outputs **weights** \(w\) (e.g. 0.1 for calls).
- Interpretation: at each rebalance, spend **10% of portfolio value** on the chosen call (the one expiring by next period). So you buy `(0.1 * portfolio_value) / call_price` contracts (or round to lot size).
- In the **backtest**, Cvxportfolio expects **returns** for each asset. So your “SPY_CALL” asset must have a return series: each week it’s the return of that week’s chosen call. The simulator then applies that return to the weight you hold; the “10%” is already encoded as weight, and the return of that asset over the period is the return of the option. So no extra step in the optimizer—only in **data construction**: make sure the return you assign to “SPY_CALL” for that week is the return of the contract you would have held.

---

## 8. Suggested Project Structure

```
cvx_options/
├── PLAN.md                    # This file
├── README.md
├── requirements.txt           # cvxportfolio, yfinance, pandas, numpy, scipy, (optional: alpaca-py)
├── config.py                  # Start/end dates, gamma_risk, turnover cap, etc.
├── data/
│   ├── fetch.py               # yfinance (and optionally Alpaca) raw fetch
│   ├── options_returns.py     # Build call/put return series from option chain
│   ├── clean.py               # Align weekly, build returns/volumes/prices DataFrames
│   └── forecasts.py           # RND or log-normal / GMM → μ, Σ (DataFrames)
├── run_backtest.py            # Load data → UserProvidedMarketData → Policy → backtest
└── experiments/
    └── compare_forecasts.py   # (Optional) RND vs log-normal vs historical
```

---

## 9. Execution Order

1. **Data**
   - Implement `fetch.py` (SPY weekly, option chain at a date).
   - Implement `options_returns.py` (select front call/put, compute weekly returns).
   - Implement `clean.py` (align to weekly calendar, output returns + optional volumes/prices).
   - Validate: plot SPY vs SPY_CALL vs SPY_PUT weekly returns; check for NaNs and coverage.

2. **Forecasts**
   - Implement RND from option chain (Breeden–Litzenberger or IV-based).
   - Sample to get \((\hat{\mu}, \hat{\Sigma})\) per rebalance; optionally implement log-normal or GMM and compare.
   - In `forecasts.py`, output `r_hat` and `Sigma` as time-indexed DataFrames (columns = SPY, SPY_CALL, SPY_PUT, CASH).

3. **Optimization and backtest**
   - Build `UserProvidedMarketData` from cleaned returns (and volumes/prices if needed).
   - Instantiate policy: `SinglePeriodOptimization` with long-only, turnover, transaction cost, robust/forecast-error term, and worst-case risk if desired.
   - Run `MarketSimulator(...).backtest(policy)` and inspect weights, turnover, and performance.

4. **Options sizing**
   - In reporting or execution layer: map weight on “SPY_CALL” to dollar amount and then to number of contracts for the chosen expiry; same for “SPY_PUT”. The backtest already uses the synthetic return series, so no change inside Cvxportfolio.

---

## 10. Summary

- **Libraries**: Use **Cvxportfolio** for Markowitz++, turnover, costs, and robust/worst-case terms; optionally **cvxpy** for one-off custom terms.
- **Data**: **yfinance** is enough for a clean experiment; build weekly SPY returns and synthetic call/put returns from the option chain at each rebalance; use **UserProvidedMarketData** with `trading_frequency='weekly'` (or pre-weekly data).
- **Forecasts**: Either **(1)** risk-neutral distribution from options → sample → \((\hat{\mu}, \hat{\Sigma})\), or **(2)** log-normal / Gaussian mixture fitted to options or returns → same. Feed both via `ReturnsForecast(r_hat=...)` and `FullCovariance(Sigma=...)`.
- **Options 10% rule**: Implemented by defining “call” and “put” as single rolling contracts and building their return series; the optimizer’s weight is then “fraction of portfolio in that option” by construction.

This plan should be enough to implement the experiment step by step and keep the code clean and testable.
