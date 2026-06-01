"""
min_var — walk-forward min-variance (and mean-variance) portfolio backtests
          using lognormal GN fitting and Delta-Gamma-Vega covariance.

Public API:
    run_walk_forward  — unified backtest engine
    run_all_experiments — run the full 6-experiment matrix
    port_stats        — fixed performance statistics
    build_stats_table — assemble stats DataFrame (fixes missing _stats_df from notebook)
    build_sigma_mu    — DGV Σ + delta-lev μ at a snapshot date
    trace_frontier    — efficient frontier
    run_accuracy_test — predict μ/Σ vs realized, return test data
    print_accuracy_report — print full accuracy summary
    print_allocation_report — print allocation diagnostics
"""
from .config import (  # noqa: F401
    R_ANN, SYMS, UNIV_STOCKS, UNIV_A, UNIV_B,
    LOOKBACK, REBAL_FREQ, MAX_WT, MAX_OPT_WT, MV_GAMMA,
    PANEL_PATH, TEST_DTE, N_SAMPLES,
)
from .pricing import lognormal_price, lognormal_price_vec  # noqa: F401
from .fitting import fit_lognormal, fit_lognormal_gn       # noqa: F401
from .data_loader import (  # noqa: F401
    load_spy_prices, discover_chain_dates, load_research_panel,
    build_returns_matrix, build_rp_index,
)
from .covariance import build_full_cov, _gn_cov_dgv        # noqa: F401
from .backtest import run_walk_forward                      # noqa: F401
from .metrics import port_stats, build_stats_table, format_stats_table  # noqa: F401
from .optimizer import build_sigma_mu, solve_portfolio, trace_frontier  # noqa: F401
from .experiments import run_all_experiments, summarize     # noqa: F401
from .accuracy import run_accuracy_test, print_accuracy_report  # noqa: F401
from .allocation import (  # noqa: F401
    allocation_summary, option_attribution, weight_stability_table,
    compare_allocation_experiments, print_allocation_report,
)
from .data_equity import ensure_equity_data, load_equity_returns, build_equity_panel  # noqa: F401
from .cvxport_backtest import run_cvxportfolio_backtest                       # noqa: F401
from .cvxport_accuracy import (  # noqa: F401
    run_ewma_forecast_accuracy, print_accuracy_report, summarize_forecast_accuracy,
)
from .cvxport_grid_search import (  # noqa: F401
    run_cvxport_grid_search, save_cvxport_best_params, load_cvxport_best_params,
)
from .rebalance_state import (  # noqa: F401
    load_state, save_state, trading_days_since, should_rebalance,
)
