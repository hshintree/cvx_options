# Options data: frequency, structure, and accuracy

## Will this accurately give a picture of SPY options across many years?

**Short answer:** It depends which part of the dataset you use.

| Source | Accuracy / coverage |
|--------|----------------------|
| **Real option prices (yfinance)** | Only for **recent** expirations. Yahoo Finance does **not** keep historical data for expired options (you get 404 / “delisted”). We therefore only request historical symbols for expiries in the **last 24 months** (`OPTION_HISTORICAL_MAX_MONTHS_BACK`). So *real* option price history is at best **~1–2 years**. No synthetic fill: SPY_CALL and SPY_PUT are NaN where we have no real data. |
| **Current option chain snapshot** | Accurate for **today**: full chain (all expirations and strikes) as of the day you run the pipeline. Good for RND/sampling and for knowing what’s listed now. |

For a **fully accurate** multi-year picture of SPY option contracts (strikes, prices, time to expiry), you’d need a provider that stores historical options (e.g. Polygon, OptionMetrics, CBOE, or broker APIs with history).

---

## What frequency do we have? Strike, prices, time to expiration?

### Processed returns (what the backtest uses)

- **Frequency:** **Daily** (one row per trading day).
- **Content:** One return per asset per day: SPY, SPY_CALL, SPY_PUT, USDOLLAR. No per-strike or per-contract breakdown in this file; it’s a single “representative” call and put return each day (front-month ATM from real option contract history). SPY_CALL and SPY_PUT are NaN where we have no option data (~24 months of coverage).

So in the **returns** dataset we do **not** have strike, price, or time-to-expiration **per row**; we only have one aggregate call and one aggregate put return per day.

### Raw option data (when we have it)

When we **do** have real option data, it lives in:

1. **Option chain snapshot** (`data/raw/option_chains/calls_YYYY-MM-DD.parquet`, `puts_YYYY-MM-DD.parquet`)
   - **Frequency:** Once per pipeline run (typically one date per day if you run daily).
   - **Per row:** One row per contract. Columns include: strike, expiry, bid, ask, last, volume, impliedVolatility, contractSymbol, etc. So you have **strike**, **prices** (bid/ask/last), and **time to expiration** = expiry − snapshot date.

2. **Option contract histories** (`data/raw/option_contracts/option_contracts.parquet`)
   - **Frequency:** **Daily** (one row per contract per trading day).
   - **Per row:** `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`. From `symbol` you can parse **strike** and **expiry** (and thus **time to expiration** = expiry − date). So you have **strike**, **prices** (OHLC), and **time to expiration** at **daily** frequency, but only for the contracts we fetched (recent expirations + current chain).

Summary:

| Dataset | Frequency | Strike | Prices | Time to expiration |
|--------|-----------|--------|--------|---------------------|
| Processed returns | Daily | No (one call/put series) | No (only returns) | No |
| Option chain snapshot | Per run (e.g. daily) | Yes (per contract) | Yes (bid/ask/last) | Yes (expiry − date) |
| Option contract histories | Daily (per contract) | Yes (from symbol) | Yes (OHLC) | Yes (expiry − date) |

So: **strike, prices, and time till expiration** are available at **daily** granularity only in the **raw** option contract history file, and only for the **recent** contracts we pull (last 24 months of expiries + current chain). The **processed** returns are daily but aggregate (no strike/price/TTE columns).
