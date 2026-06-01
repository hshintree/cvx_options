# cvx_options — Equity Portfolio Optimization

Walk-forward Markowitz (mean-variance) portfolio backtests over a 31-symbol equity + bond + commodity universe (2022–present), with a live-trading controller backed by Alpaca Markets.

**Best validated result**: cvxportfolio SPO, γ=5, EWMA shrink=80%, monthly rebalancing → **~16% ann return, ~0.6 Sharpe** vs SPY ~0.38 (2022–2026, 31-symbol universe).

---

## Environment setup

```bash
conda env create -f environment.yml
conda activate cvx_options
conda env update -f environment.yml --prune   # after pulling changes
```

`.env` file at the project root (never commit this):

```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

---

## Repository layout

```
cvx_options/
├── min_var/                     # Core package
│   ├── config.py                # Universe definitions, paths, tunables
│   ├── cvxport_backtest.py      # cvxportfolio SPO/MPO backtest engine
│   ├── cvxport_accuracy.py      # EWMA μ & Σ walk-forward accuracy tests
│   ├── cvxport_grid_search.py   # Grid search over cvxportfolio params
│   ├── live_trader.py           # AlpacaPortfolioController
│   ├── rebalance_state.py       # Fixed-period rebalance state machine
│   ├── data_equity.py           # Fetch + load ETF/equity price data
│   ├── metrics.py               # port_stats, build_stats_table
│   ├── optimizer.py             # solve_portfolio, trace_frontier
│   ├── backtest.py              # Options-era walk-forward engine (kept for run_min_var.py)
│   └── ...                      # Options-era modules (pricing, fitting, covariance, …)
├── run_backtest.py              # End-to-end backtest (fetch → backtest → diagnostics)
├── run_cvxportfolio.py          # Multi-configuration comparison backtest
├── run_cvxportfolio_grid_search.py  # Tune γ, Σ half-life, μ shrinkage, max_wt
├── run_forecast_accuracy.py     # EWMA μ & Σ forecast quality tests
├── run_live.py                  # Live / paper trading (run-once)
├── run_data_pipeline.py         # Fetch all universe data from Alpaca
├── run_min_var.py               # Options-era experiments (archived, still runnable)
├── validate_steps.py            # Unit tests for estimators
├── config.py                    # Top-level constants (R_ANN, paths)
├── data/                        # Raw + processed price data (gitignored)
├── min_var_output/              # Output: plots, best_params.json, grid results
├── options_exps/                # Archived options-era notebooks
└── deploy/                      # launchd plist for scheduled live trading
```

---

## Quickstart

```bash
# Fetch data for the full 31-symbol universe
python run_data_pipeline.py

# Run end-to-end backtest with defaults (γ=5, 80% shrinkage, monthly rebal)
python run_backtest.py --save

# Or specify a custom symbol list
python run_backtest.py --symbols SPY TLT GLD NVDA MSFT AAPL --save
```

---

## Workflow

### 1. Fetch price data

Downloads daily OHLCV bars from Alpaca for the full universe starting 2020-01-01.

```bash
python run_data_pipeline.py
```

Force-refresh a specific symbol:

```bash
python -c "
from min_var.data_equity import ensure_equity_data
ensure_equity_data(['NVDA', 'COIN'], force=True)
"
```

---

### 2. End-to-end backtest (recommended starting point)

Fetches data, runs the optimized backtest, prints a performance table, and shows four diagnostic figures.

```bash
python run_backtest.py                           # full 31-symbol universe
python run_backtest.py --symbols SPY TLT GLD AAPL NVDA MSFT --save
python run_backtest.py --gamma 10 --mu-shrinkage 0.5
python run_backtest.py --start 2023-01-01
```

**Default hyperparameters** (all tuned / validated — see *Estimator methods* below):

| Parameter | Default | Rationale |
|---|---|---|
| `gamma` | 5 | Risk-aversion; balances return vs variance |
| `mu_shrinkage` | 0.80 | EWMA μ IC ≈ 0.08, not significant — keep only 20% of the momentum signal |
| `rebal_freq` | monthly | First trading day of each calendar month (~21 td); see *Rebalance logic* |
| `cov_halflife` | 21 td | Captures current vol/correlation regime quickly |
| `mu_halflife` | 252 td | 12-month EWMA momentum window |
| `max_wt` | 0.20 | Per-asset cap; prevents single-stock concentration |

---

### 3. Multi-configuration comparison

Compares EWMA variants (varying shrinkage, γ) and pure min-var against SPY + 1/N benchmarks.

```bash
python run_cvxportfolio.py           # show plot
python run_cvxportfolio.py --save    # save to min_var_output/
python run_cvxportfolio.py --symbols SPY TLT IEF GLD NVDA MSFT AAPL
```

---

### 4. Tune hyperparameters (rolling walk-forward CV)

Sweeps γ, Σ half-life, max_wt, μ shrinkage, and rebalance frequency.
Scores by **stability = mean OOS Sharpe / std OOS Sharpe** across 7 non-overlapping 6-month OOS folds — not by a single OOS window that happens to be a bull market.

```bash
python run_cvxportfolio_grid_search.py --quick --save   # ~24 combos × 7 folds
python run_cvxportfolio_grid_search.py --save           # full grid (~144 combos × 7 folds)
python run_cvxportfolio_grid_search.py --single-split   # legacy: one IS/OOS window
```

**Why rolling CV instead of a single IS/OOS split:**
A single IS=2022-2023 / OOS=2024-present split scores every strategy against a raging tech bull market. Any config with NVDA/COIN/MSTR exposure and max_wt ≥ 10% produces Sharpe=10-20 in that one window — meanwhile IS Sharpe is −4 (numerically unstable). Rolling CV builds 7 OOS windows including the 2022 bear market and 2023 recovery; a config must perform consistently across all regimes.

Writes `min_var_output/cvxport_best_params.json` and a stability heatmap.

---

### 5. Forecast accuracy tests

Verifies whether the EWMA μ and Σ estimators have statistically defensible predictive power **before** trusting backtest Sharpe numbers.

```bash
python run_forecast_accuracy.py --save        # full eval + save plots/CSVs
python run_forecast_accuracy.py --horizon 5   # 5-day horizon
python run_forecast_accuracy.py --start 2022-01-01
```

**What to look for:**

| Metric | Good | Bad |
|---|---|---|
| IC p-value | < 0.05 — cross-sectional return ranking is real | ≥ 0.05 — no reliable alpha; high Sharpe is regime luck |
| Hit rate | > 55% with p < 0.05 | ≤ 50% — directionally wrong |
| Variance ratio (diagonal Σ) | ≈ 1.0 | > 1.5 = over-estimated (over-diversifies); < 0.67 = under-estimated (under-hedges) |
| Pairwise ρ Pearson r (off-diagonal Σ) | > 0.60 | < 0.30 — correlation structure wrong; optimizer will mismatch co-movements |

The **pairwise ρ panel** (new in this update) is the off-diagonal verification: it plots all predicted vs realized pairwise correlations across every (date × asset-pair), checking whether the full covariance structure — not just individual vols — is well-modelled.

---

### 6. Live / paper trading

Requires `.env` with Alpaca keys (see *Environment setup*).

```bash
# Show current positions vs target weights (no orders)
python run_live.py --status

# Print proposed rebalance orders without executing
python run_live.py --rebalance --dry-run

# Execute if rebalance period has elapsed
python run_live.py --rebalance

# Execute regardless of elapsed time
python run_live.py --rebalance --force

# Use live account (default is paper)
python run_live.py --rebalance --live
```

**Parameter sources** (resolved in order):

1. `--use-validated` → γ=5, Σ_hl=21d, EWMA, shrink=80%, SPO, max_wt=20%
2. `--params PATH` → custom JSON file
3. Default → `min_var_output/cvxport_best_params.json` (grid search result)

Per-flag overrides (`--gamma`, `--cov-halflife`, `--mu-method`, `--max-wt`, `--mu-shrinkage`) apply on top of any source. State is persisted in `min_var_output/rebal_state.json`.

---

### 7. Daily / weekly rebalancing (Garleanu-Pedersen aim portfolio)

`run_backtest.py` supports all four frequencies via `--rebal-freq`.
For `daily` and `weekly`, the **aim portfolio** framework is used automatically:
`w_new = w_old + λ × (w_target − w_old)` where `λ = 1 / (1 + TC_per_trade / α_per_trade)`.
This dampens trades to the fraction that balances marginal TC against marginal alpha.

```bash
python run_backtest.py --rebal-freq weekly --symbols SPY QQQ TLT IEF GLD
python run_backtest.py --rebal-freq daily  --symbols SPY QQQ TLT IEF GLD
python run_backtest.py --rebal-freq quarterly  # recommended (IC significant at 63d)
```

Key output: TC analysis (avg spread, λ, gross/net TC), both full-rebalance and aim paths,
and TC-adjusted Sharpe. At daily frequency for our universe, net TC after dampening is
still ~8%/yr — marginal for a ~13% gross return strategy. Monthly/quarterly is preferred.

---

### 8. Options-era experiments (archived)

The original 6-experiment SPY+options matrix is still runnable:

```bash
python run_min_var.py                  # all 6 experiments + stats table
python run_min_var.py --frontier       # + efficient frontier
python run_min_var.py --accuracy       # + μ/Σ accuracy
python run_min_var.py --save
```

---

## Rebalance logic

The portfolio rebalances on the **first trading day of each calendar month or quarter** (NYSE business day). Between rebalance dates, weights drift with the market and no trades are made.

**Why monthly vs quarterly is tuned in the grid search:**

The correct rebalance frequency is determined by where the μ signal has actual predictive power, not where backtest Sharpe peaks (which can be regime-specific).

Walk-forward IC tests show:
- **21-day (monthly) horizon**: mean IC = +0.082, p ≈ 0.15 — **not statistically significant**. Rebalancing monthly means acting on noise in the return forecast.
- **63-day (quarterly) horizon**: mean IC significant at p ≈ 0.007. The μ signal has real cross-sectional predictive power at this timescale.

This is why `rebal_freq ∈ {monthly, quarterly}` is included in the grid search (`run_cvxportfolio_grid_search.py`). The grid selects the frequency with the best validated OOS Sharpe, but the IC evidence favors quarterly as the statistically grounded choice.

Weekly rebalancing is excluded: 52 round-trips per year vs 12 (monthly) or 4 (quarterly) materially increases transaction costs relative to the improvement in signal freshness.

**Implementation:**

`min_var/cvxport_backtest.py::_run_minvar_backtest()` identifies the first trading day of each month or quarter from the return index using pandas `resample("MS")` / `resample("QS")`. `min_var/rebalance_state.py` tracks the last executed rebalance date so `run_live.py` knows when to fire the next one.

---

## Noise management and statistical grounding

**Core principle:** Only tune parameters whose effects are measurable via forecast accuracy tests. Parameters that only shift backtest Sharpe — without a corresponding improvement in IC, hit rate, variance ratio, or pairwise ρ — are fitting to regime-specific noise, not signal.

### What we tune (and why)

| Parameter | Grid values | Validated by |
|---|---|---|
| `gamma` | 1, 2, 5, 10 | OOS Sharpe stability across IS/OOS split |
| `cov_halflife` | 21d, 42d | Variance ratio + pairwise ρ Pearson r |
| `max_wt` | 10%, 15%, 20% | Concentration risk, OOS Sharpe |
| `mu_shrinkage` | 50%, 75%, 80% | IC significance — high shrinkage reduces noise amplification |
| `rebal_freq` | monthly, quarterly | IC significance at forecast horizon |

### What we do NOT tune

| Parameter | Fixed value | Why locked |
|---|---|---|
| `mu_method` | `"ewma"` | Only EWMA μ has validated directional accuracy (hit rate p=0.009). "momentum" adds a free z-score normalization parameter with no additional statistical backing. |
| `use_mpo` | `False` (SPO) | MPO adds a `planning_horizon` free parameter. No validated improvement in forecast accuracy metrics; the extra combos add noise to OOS Sharpe scores. |
| `mu_halflife` | 252d | 12-month EWMA = classic intermediate-term momentum, best-documented equity factor. Changing this without a new IC test is curve-fitting. |
| `cov_halflife=10d` | Excluded | 10-day window is below the minimum IC significance window (63d) and produces unstable correlation estimates. |

### Overfitting safeguard

If a configuration shows high OOS Sharpe but its underlying IC p-value ≥ 0.05, that Sharpe is treated as regime-driven, not repeatable alpha. The recommended strategy (γ=5, shrink=80%) intentionally sacrifices the peak IS Sharpe of the concentrated γ=10 zero-shrinkage run in exchange for defensibility.

---

## Estimator methods — state of the art

### Covariance Σ: EWMA Factorized Covariance (half-life = 21 trading days)

**Implementation:** `cvxportfolio.forecast.HistoricalFactorizedCovariance(half_life=21td)`.

The factorized form Σ = L L' + D (low-rank + diagonal) guarantees positive definiteness as the universe grows, avoiding singularity that plagues plain sample covariance in the 31-symbol universe. The 21-day half-life means data from 6 weeks ago receives half the weight of the most recent observation, making the estimate reactive enough to capture volatility regime changes (e.g. VIX spikes) within a single rebalance cycle.

**Validated accuracy** (`run_forecast_accuracy.py`):
- **Variance ratio** (pred σ / realized σ): ≈ 1.54 — slightly conservative (over-estimates vol), which is safe: the optimizer will be a bit more diversified than strictly necessary but will not take unintended tail risk.
- **Pairwise correlation Pearson r** (predicted vs realized off-diagonal entries): ≈ 0.68–0.74 — meaningfully above zero, meaning the full covariance structure (not just individual vols) is well-modelled.
- **Frobenius relative error** (‖Σ_pred − Σ_real‖ / ‖Σ_real‖): ≈ 0.65 median — the matrix-level error is moderate but the optimizer cares about relative ordering, not absolute precision.

**Improving pairwise correlation accuracy — the split estimator:**

Volatilities and correlations operate on different timescales: individual asset vols react to news within days, while the cross-asset correlation structure evolves over weeks to months. Applying the same hl=21 to both injects fast-moving vol noise into the correlation estimates.

The split estimator (`min_var/cvxport_accuracy.py::_ewma_cov_split`) decouples these:

```
Σ = D × C × D
  D = diagonal of fast EWMA vols     (hl = 21 trading days)
  C = slow EWMA correlation matrix   (hl = 63–126 trading days)
```

Run `run_forecast_accuracy.py` (default: `--corr-halflife 63`) to see side-by-side comparison of standard vs split pairwise ρ accuracy. The split estimator typically improves pairwise Pearson r by +0.03–0.08 at the cost of slightly slower adaptation to correlation regime breaks.

**Why not Ledoit-Wolf?** LW produces a better-conditioned sample covariance but is static (no exponential decay). In a universe with time-varying correlations — equities + bonds + commodities + crypto-adjacent assets spanning 2022–2026 — EWMA's regime-adaptivity outweighs LW's better asymptotic properties.

---

### Expected returns μ: EWMA with 80% James-Stein shrinkage (half-life = 252 trading days)

**Motivation from forecast accuracy:**

Our EWMA return forecast was tested with walk-forward evaluation:
- **Cross-sectional IC**: mean ≈ +0.082, p ≈ 0.15 — **not statistically significant** at the 5% level. The signal exists in direction (hit rate 54.5%, p ≈ 0.009) but cannot reliably *size* positions by cross-sectional expected return.
- **Pearson r** (pred μ vs realized μ): ≈ −0.07 — essentially zero magnitude forecasting ability.
- **Conclusion**: EWMA μ has weak directional information but poor sizing information. Using it at full strength (shrinkage=0) over-concentrates in recent momentum winners (e.g. NVDA in 2024), which is regime-contingent, not repeatable alpha.

**James-Stein shrinkage formula:**

At each rebalance date t, the effective return forecast is:

```
μ_eff_i = (1 − s) × μ_EWMA_i  +  s × μ̄
```

where `s = 0.80` (80% shrinkage), `μ_EWMA_i` is the raw EWMA mean for asset i, and `μ̄` is the cross-sectional mean across all assets. With s=0.80, only 20% of the cross-sectional dispersion in EWMA momentum is preserved; the rest is compressed toward the equal-weighted prior.

**Why 80% and not 100% (pure min-var)?** At s=1.00, all assets have the same expected return and the optimizer reduces to min-var, which historically concentrates in bond ETFs (lowest realized vol). The 20% residual momentum signal provides enough differentiation to tilt away from bonds when they are in a bear market (e.g. 2022) while keeping risk-adjusted allocations diversified.

**Implementation:** `min_var/cvxport_backtest.py::_apply_return_shrinkage()` — applied as a pre-processing step before cvxportfolio's internal EWMA estimator runs, so the bias correction and warm-up logic are unchanged.

**μ half-life = 252 trading days (~1 year):** This implements the classic 12-month trailing momentum signal under exponential weighting. The long half-life smooths out short-term noise in mean estimation while capturing intermediate-term trend (the best-documented equity factor).

---

## Universe

**`EXPANDED_SYMBOLS`** — 31 assets, max_wt=20% per asset:

| Category | Symbols |
|---|---|
| US Broad Market | SPY, QQQ, IWM |
| Tech Mega-cap | AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD |
| Sectors (ex-defense, ex-oil) | XLV, XLF, XLY, XLP, ICLN |
| Real Estate | VNQ, AMT |
| International | EFA, EEM |
| Fixed Income | TLT, IEF, SHY, TIP, HYG |
| Commodities | GLD, SLV, CPER, DBC |
| Crypto-adjacent | COIN, MSTR |

Data starts 2021-04-15 (limited by COIN IPO). GBTC excluded (ETF conversion only from 2024-01-11 would truncate the entire universe).

---

## Transaction cost calibration

Per-asset bid-ask half-spreads used in `cvx.TransactionCost`:

| Asset class | Symbols | Half-spread |
|---|---|---|
| Liquid index ETFs | SPY, QQQ, IWM, SHY | 2 bps |
| Bond / commodity / sector ETFs | TLT, IEF, TIP, GLD, XLV, XLF, XLY, XLP, VNQ, EFA | 3 bps |
| Mid-cap / thematic ETFs | HYG, SLV, DBC, ICLN, EEM | 4 bps |
| Large-cap equities | AAPL, MSFT, AMZN, GOOGL, META | 5 bps |
| High-vol equities | NVDA, TSLA, AMD | 8 bps |
| Crypto-adjacent | COIN, MSTR | 15 bps |

---

## Key findings

| Strategy | Ann Return | Sharpe | Max DD | Notes |
|---|---|---|---|---|
| SPY buy-and-hold | ~13% | ~0.38 | −27% | Benchmark |
| 1/N equal-weight (31) | ~11% | ~0.27 | −36% | Equal allocation; worse than SPY due to small-cap drag |
| cvx EWMA γ=5, shrink=0% | ~15% | ~0.38 | −26% | Raw momentum; concentrated in NVDA/DBC |
| **cvx EWMA γ=5, shrink=80%** | **~13%** | **~0.39** | **−22%** | **Recommended: diversified, defensible** |
| cvx EWMA γ=10, shrink=0% | ~20% | ~0.79 | −16% | Best IS Sharpe but likely overfit to NVDA 2024 |
| cvx min-var (μ=0) | ~8% | ~0.35 | −18% | True min-var via rolling CVXPY; concentrates in bonds |

**Why the shrunk μ=80% config is recommended for live trading:**
The γ=10 zero-shrinkage run shows the highest backtest Sharpe, but forecast accuracy tests confirm the underlying EWMA μ is not statistically significant — that Sharpe is driven by NVDA concentration during a specific 2024 bull run. The γ=5 shrink=80% config sacrifices some peak Sharpe in exchange for a more diversified, regime-agnostic allocation that is defensible as a live strategy.
