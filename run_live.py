"""
Run-once live/paper portfolio rebalancer.

Loads cvxport_best_params.json (from run_cvxportfolio_grid_search.py) by default,
with full CLI override support for every hyperparameter.

Usage:
    python run_live.py --status              # positions + target weights + days since last rebal
    python run_live.py --rebalance --dry-run # print proposed orders, don't execute
    python run_live.py --rebalance           # execute if rebal_freq days have passed
    python run_live.py --rebalance --force   # execute regardless of period
    python run_live.py --live                # use live account (default: paper)

Hyperparameter overrides (override whatever is in the params JSON):
    python run_live.py --status --use-validated   # use γ=5 EWMA SPO (validated 2022-2026)
    python run_live.py --rebalance --dry-run --gamma 5 --cov-halflife 21 --mu-method ewma
    python run_live.py --rebalance --dry-run --gamma 10 --max-wt 0.40 --no-mpo

Param sources (checked in order):
    1. --use-validated  → hardcoded γ=5, cov_hl=21, max_wt=0.40, ewma, SPO (full-backtest winner)
    2. --use-grid-best  → load cvxport_best_params.json (grid search result, default)
    3. --params PATH    → load custom JSON file
    4. Per-flag overrides (--gamma, --cov-halflife, etc.) applied on top of any source

Requires:
    .env with ALPACA_API_KEY, ALPACA_SECRET_KEY
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from min_var.config import EXTENDED_SYMBOLS, EXPANDED_SYMBOLS, OUTPUT_DIR
from min_var.rebalance_state import load_state, trading_days_since, should_rebalance


def _print_forecasts(ctrl, arith_rets: pd.DataFrame, tgt_wts: dict) -> None:
    """Print EWMA return forecasts and covariance used by the optimizer."""
    try:
        mu, sig = ctrl.compute_ewma_forecasts(arith_rets)
    except Exception as exc:
        print(f"\n  (Could not compute forecasts: {exc})")
        return

    vols = pd.Series(np.sqrt(np.diag(sig.values)), index=sig.columns)
    corr = sig.divide(vols, axis=0).divide(vols, axis=1).clip(-1, 1)

    active = [s for s in ctrl.symbols if tgt_wts.get(s, 0) > 0.005]

    shrinkage = getattr(ctrl, "mu_shrinkage", 0.0)
    shrink_note = f", shrinkage={shrinkage:.0%}" if shrinkage > 0 else ""
    print(f"\n── Optimizer inputs (EWMA, μ_hl={ctrl.mu_halflife}d, Σ_hl={ctrl.cov_halflife}d{shrink_note}) ─")

    if shrinkage > 0:
        # Compute effective (shrunk) μ so user sees what the optimizer actually uses
        mu_bar  = mu.mean()
        mu_eff  = mu_bar + (1 - shrinkage) * (mu - mu_bar)
        print(f"  {'Asset':<6}  {'Raw μ (ann)':>13}  {'Eff μ (ann)':>13}  {'Pred σ (ann)':>13}  {'Target wt':>10}")
        print("  " + "─" * 62)
        for sym in ctrl.symbols:
            tw = tgt_wts.get(sym, 0.0)
            if tw > 0.001 or abs(mu.get(sym, 0)) > 0.01:
                marker = " ◀" if tw > 0.01 else ""
                print(f"  {sym:<6}  {mu.get(sym, 0):>+13.1%}  "
                      f"{mu_eff.get(sym, 0):>+13.1%}  "
                      f"{vols.get(sym, 0):>13.1%}  {tw:>10.1%}{marker}")
    else:
        print(f"  {'Asset':<6}  {'Pred μ (ann)':>13}  {'Pred σ (ann)':>13}  {'Target wt':>10}")
        print("  " + "─" * 50)
        for sym in ctrl.symbols:
            tw = tgt_wts.get(sym, 0.0)
            if tw > 0.001 or abs(mu.get(sym, 0)) > 0.01:
                marker = " ◀" if tw > 0.01 else ""
                print(f"  {sym:<6}  {mu.get(sym, 0):>+13.1%}  "
                      f"{vols.get(sym, 0):>13.1%}  {tw:>10.1%}{marker}")

    if len(active) >= 2:
        print(f"\n── Predicted correlations (active positions) ───────────────────────────")
        print(f"  {'':6}", end="")
        for s in active:
            print(f"  {s:>6}", end="")
        print()
        for s1 in active:
            print(f"  {s1:<6}", end="")
            for s2 in active:
                c = corr.loc[s1, s2] if (s1 in corr.index and s2 in corr.columns) else 0.0
                print(f"  {c:>6.2f}", end="")
            print()

    # Diversification check
    if active:
        max_sym = max(active, key=lambda s: tgt_wts.get(s, 0))
        max_wt  = tgt_wts.get(max_sym, 0)
        if max_wt > ctrl.max_wt * 0.99:
            print(f"\n  ⚠  {max_sym} at max_wt cap ({max_wt:.1%}). "
                  f"Predicted μ={mu.get(max_sym, 0):+.1%}, σ={vols.get(max_sym, 0):.1%}")

# ── Validated config (γ=5, EWMA, SPO, monthly) — full 2022-2026 winner ────────
# max_wt=0.20 enforces broad diversification across the 31-symbol expanded universe.
# At γ=5 the optimizer still runs momentum-weighted risk management, but no single
# asset can exceed 20% — preventing the NVDA concentration seen with max_wt=0.40.
VALIDATED_PARAMS = {
    "gamma":            5.0,
    "cov_halflife":     21,
    "mu_halflife":      252,
    "mu_shrinkage":     0.80,
    "max_wt":           0.20,
    "mu_method":        "ewma",
    "use_mpo":          False,
    "rebal_freq":       "quarterly",
    "planning_horizon": 3,
    "rebal_freq_days":  63,
    "source":           "validated_2022_2026",
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Live/paper portfolio rebalancer",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# Mode flags
parser.add_argument("--status",    action="store_true",
                    help="Show positions, target weights, and rebalance status")
parser.add_argument("--rebalance", action="store_true",
                    help="Run rebalance pipeline (checks period unless --force)")
parser.add_argument("--dry-run",   action="store_true",
                    help="Compute orders but don't submit (use with --rebalance)")
parser.add_argument("--force",     action="store_true",
                    help="Bypass period check — always rebalance")
parser.add_argument("--live",      action="store_true",
                    help="Use live account (default: paper trading)")
parser.add_argument("--symbols",   nargs="*", default=None,
                    help="Override symbol universe")

# Param source flags
src_group = parser.add_mutually_exclusive_group()
src_group.add_argument("--use-validated", action="store_true",
                       help="Use γ=5 EWMA SPO (validated over full 2022-2026 period, "
                            "recommended for live trading)")
src_group.add_argument("--use-grid-best", action="store_true",
                       help="Use cvxport_best_params.json (grid-search winner, default)")
src_group.add_argument("--params",        default=None,
                       help="Path to a custom params JSON file")

# Hyperparameter override flags (applied on top of any param source)
parser.add_argument("--gamma",        type=float, default=None,
                    help="Risk-aversion coefficient (default: from params JSON)")
parser.add_argument("--cov-halflife", type=int,   default=None,
                    help="EWMA half-life for covariance in trading days (default: from params)")
parser.add_argument("--mu-method",    choices=["ewma", "momentum", "zero"], default=None,
                    help="Return forecast method (default: from params)")
parser.add_argument("--max-wt",       type=float, default=None,
                    help="Per-asset weight cap 0-1 (default: from params)")
parser.add_argument("--use-mpo",      action="store_true", default=None,
                    help="Force MultiPeriodOptimization ON")
parser.add_argument("--no-mpo",       action="store_true",
                    help="Force MultiPeriodOptimization OFF (use SPO)")
parser.add_argument("--mu-shrinkage", type=float, default=None,
                    help="James-Stein shrinkage 0-1: compress EWMA forecasts toward "
                         "cross-sectional mean (0=pure momentum, 0.5=half-shrunk, "
                         "1=all assets same expected return)")

args = parser.parse_args()

if not (args.status or args.rebalance):
    parser.print_help()
    sys.exit(0)

symbols = args.symbols or EXPANDED_SYMBOLS
paper   = not args.live


# ── Build effective params dict ───────────────────────────────────────────────
def _load_effective_params(args) -> dict:
    """Load base params then apply any CLI overrides."""
    import json

    if args.use_validated:
        base = dict(VALIDATED_PARAMS)
        print("  Param source: validated 2022-2026 config (γ=5, EWMA, SPO)")
    elif args.params:
        with open(args.params) as f:
            base = json.load(f)
        print(f"  Param source: {args.params}")
    else:
        # Default: cvxport_best_params.json
        from min_var.cvxport_grid_search import load_cvxport_best_params

        cvxport_path = OUTPUT_DIR / "cvxport_best_params.json"

        if cvxport_path.exists():
            base = load_cvxport_best_params(cvxport_path)
            print(f"  Param source: {cvxport_path.name}")
        else:
            print("  ⚠  No params file found — using validated config as fallback")
            base = dict(VALIDATED_PARAMS)

    # Apply per-flag overrides
    overrides = {}
    if args.gamma        is not None: overrides["gamma"]        = args.gamma
    if args.cov_halflife  is not None: overrides["cov_halflife"] = args.cov_halflife
    if args.mu_method    is not None: overrides["mu_method"]    = args.mu_method
    if args.max_wt       is not None: overrides["max_wt"]       = args.max_wt
    if args.mu_shrinkage is not None: overrides["mu_shrinkage"] = args.mu_shrinkage
    if args.use_mpo:                   overrides["use_mpo"]      = True
    if args.no_mpo:                    overrides["use_mpo"]      = False

    if overrides:
        print(f"  CLI overrides applied: {overrides}")
        base.update(overrides)

    return base


# ── Status-only mode ──────────────────────────────────────────────────────────
if args.status:
    state  = load_state()
    last   = state.get("last_rebal_date")
    n_days = trading_days_since(last)

    print(f"\nStatus report")
    print(f"  Last rebalance : {last or 'never'}")
    print(f"  Trading days since last rebal: {n_days}")

    params     = _load_effective_params(args)
    rebal_days = int(params.get("rebal_freq_days", 21))
    due, reason = should_rebalance(rebal_days)
    print(f"  Rebalance due  : {'YES — ' + reason if due else 'NO  — ' + reason}")

    mode = "MPO" if params.get("use_mpo") else "SPO"
    print(f"  Strategy: cvxportfolio {mode}, "
          f"γ={params.get('gamma', '?')}, "
          f"Σ_hl={params.get('cov_halflife', '?')}d, "
          f"μ={params.get('mu_method', '?')}, "
          f"rebal={params.get('rebal_freq', 'monthly')}, "
          f"max_wt={params.get('max_wt', 0.40):.0%}")

    # Show live positions + target weights
    try:
        from min_var.live_trader import AlpacaPortfolioController
        ctrl = AlpacaPortfolioController(symbols=symbols, paper=paper)
        # Inject overridden params
        ctrl._params        = params
        ctrl._use_cvxport   = True
        ctrl.gamma          = float(params.get("gamma", 5.0))
        ctrl.cov_halflife   = int(params.get("cov_halflife", 21))
        ctrl.mu_halflife    = int(params.get("mu_halflife", 252))
        ctrl.mu_method      = str(params.get("mu_method", "ewma"))
        ctrl.use_mpo        = bool(params.get("use_mpo", False))
        ctrl.max_wt         = float(params.get("max_wt", 0.40))
        ctrl.mu_shrinkage   = float(params.get("mu_shrinkage", 0.0))
        ctrl.rebal_freq_str  = str(params.get("rebal_freq", "monthly"))
        ctrl.rebal_freq_days = int(params.get("rebal_freq_days", 21))

        cur_wts, pv = ctrl.get_portfolio_state()
        arith_rets  = ctrl.fetch_latest_returns(arith=True)
        print(f"  Data: {len(arith_rets)} days  "
              f"({arith_rets.index[0].date()} → {arith_rets.index[-1].date()})")
        tgt_wts     = ctrl.compute_target_weights(arith_rets)

        print(f"\n  Portfolio value: ${pv:,.2f}")
        print(f"\n  {'Symbol':<8}  {'Current':>8}  {'Target':>8}  {'Drift':>8}")
        print("  " + "─" * 42)
        for sym in symbols:
            cw = cur_wts.get(sym, 0.0)
            tw = tgt_wts.get(sym, 0.0)
            print(f"  {sym:<8}  {cw:>8.1%}  {tw:>8.1%}  {tw - cw:>+7.1%}")

        _print_forecasts(ctrl, arith_rets, tgt_wts)
    except Exception as exc:
        import traceback
        print(f"\n  Could not fetch live data: {exc}")
        traceback.print_exc()

    sys.exit(0)


# ── Rebalance mode ────────────────────────────────────────────────────────────
if args.rebalance:
    from min_var.live_trader import AlpacaPortfolioController

    print(f"\nAccount mode   : {'PAPER' if paper else 'LIVE'}")
    print(f"Symbols        : {symbols}")
    if args.dry_run:
        print("Mode           : DRY RUN (no orders will be submitted)")
    if args.force:
        print("Mode           : FORCED (bypassing period check)")

    params = _load_effective_params(args)

    ctrl = AlpacaPortfolioController(symbols=symbols, paper=paper)
    # Inject params (overriding whatever was loaded from file)
    ctrl._params        = params
    ctrl._use_cvxport   = True
    ctrl.gamma          = float(params.get("gamma", 5.0))
    ctrl.cov_halflife   = int(params.get("cov_halflife", 21))
    ctrl.mu_halflife    = int(params.get("mu_halflife", 252))
    ctrl.mu_method      = str(params.get("mu_method", "ewma"))
    ctrl.use_mpo        = bool(params.get("use_mpo", False))
    ctrl.planning_horizon = int(params.get("planning_horizon", 3))
    ctrl.max_wt         = float(params.get("max_wt", 0.40))
    ctrl.mu_shrinkage   = float(params.get("mu_shrinkage", 0.0))
    ctrl.rebal_freq_str  = str(params.get("rebal_freq", "monthly"))
    ctrl.rebal_freq_days = int(params.get("rebal_freq_days", 21))

    # Pre-fetch data and compute forecasts before running the pipeline,
    # so we can display them regardless of whether orders are submitted.
    arith_rets = ctrl.fetch_latest_returns(arith=True)
    print(f"  Data: {len(arith_rets)} days  "
          f"({arith_rets.index[0].date()} → {arith_rets.index[-1].date()})")

    # Compute target weights directly (for forecast display; ctrl.run() will recompute)
    tgt_wts_preview = ctrl.compute_target_weights(arith_rets)
    _print_forecasts(ctrl, arith_rets, tgt_wts_preview)

    summary = ctrl.run(dry_run=args.dry_run, force=args.force)

    print(f"\n── Rebalance Summary ───────────────────────────────────────────────────")
    print(f"  Portfolio value: ${summary.attrs.get('portfolio_value', 0):,.2f}")
    print(f"\n  {'Symbol':<8}  {'Current':>8}  {'Target':>8}  {'Drift':>8}  "
          f"{'Notional':>10}  {'Action':<6}")
    print("  " + "─" * 60)
    for sym, row in summary.iterrows():
        print(f"  {sym:<8}  {row.current_wt:>8.1%}  {row.target_wt:>8.1%}  "
              f"{row.delta_wt:>+7.1%}  {row.notional:>10,.0f}  {row.action:<6}")

    if not args.dry_run and summary["executed"].any():
        print(f"\n  Orders submitted. State saved to min_var_output/rebal_state.json")
    elif args.dry_run:
        print(f"\n  Dry-run complete — no orders submitted.")
