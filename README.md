# SPY + Options Markowitz++ Experiment

Long-only portfolio optimization over **SPY**, **SPY calls**, **SPY puts**, and **cash**, using the Markowitz++ formulation (turnover, trading costs, worst-case/robust return) with **cvxportfolio**, rebalancing weekly (or daily).

See **[PLAN.md](PLAN.md)** for the full execution plan.

---

## Conda environment (recommended)

Use **Python 3.10** for compatibility with cvxportfolio, pandas, and yfinance.

### Create and activate

```bash
cd /path/to/cvx_options
conda env create -f environment.yml
conda activate cvx_options
```

### If you prefer to create the env manually

```bash
conda create -n cvx_options python=3.10 -y
conda activate cvx_options
pip install -r requirements.txt
```

### Verify

```bash
python -c "import cvxportfolio, yfinance, pandas; print('OK')"
```

---

## Build the dataset

From the project root with the conda env active, run:

```bash
python run_data_pipeline.py --start 2015-01-01
```

This fetches SPY daily, cash rate, option chain snapshot, and option contract histories (last 24 months + current chain), then builds `data/processed/returns.parquet`, `volumes.parquet`, and `prices.parquet`. Option data is **daily** where available (raw contract histories and processed call/put returns).

Options: use `--no-options` to skip option histories (SPY + cash only). Use `--historical-options` to also fetch historical monthly symbols (last 24 months); many may 404 once expired. Default is current chain only (~120 contracts).

---

## Data pipeline (details)

Data is stored **daily** so you can later use weekly or daily trading.

1. **Fetch raw data** (SPY daily, risk-free rate, option chain snapshot, option contract histories):

   ```bash
   python -m data.fetch
   ```

   Or run the full fetch with a date range and optional limits:

   ```python
   from data.fetch import run_full_fetch
   run_full_fetch(start_date="2015-01-01", end_date=None)
   ```

2. **Build call/put returns** (done inside clean):
   - Uses `data/options_returns.py` to select front-month ATM call and put at each date and compute daily returns from option contract history (real data only; ~24 months where Yahoo has it).

3. **Clean and align** (returns, volumes, prices for cvxportfolio):

   ```bash
   python -m data.clean
   ```

Outputs:

- `data/raw/`: `spy_daily.parquet`, `cash_rate_daily.parquet`, `option_chains/`, `option_contracts/option_contracts.parquet`
- `data/processed/`: `returns.parquet`, `volumes.parquet`, `prices.parquet`

**How often / how many options we pull**

- **Option chain**: Once per pipeline run. yfinance only gives the *current* chain, so each run saves one snapshot (all expirations × all strikes). To get a new snapshot every day, run the pipeline daily (e.g. cron).
- **Option contract histories**: Same run. **Default:** current chain only (**4** front expirations × up to **15** calls + **15** puts per expiry ⇒ **~120 symbols**). With `--historical-options`, we also request monthly symbols for the last 24 months (often 404 once expired). Option returns in processed data are **real only**. See **data/DATA_FREQUENCY_AND_ACCURACY.md** for frequency and structure.

---

## Universe

| Asset    | Description |
|----------|-------------|
| SPY      | SPDR S&P 500 ETF |
| SPY_CALL | Front-month ATM call (real option returns, ~24 months) |
| SPY_PUT  | Front-month ATM put (real option returns, ~24 months) |
| USDOLLAR | Risk-free (13-week T-bill proxy) |

Options rule: a 10% weight means 10% of portfolio value is allocated to the option that expires by the next rebalance.
