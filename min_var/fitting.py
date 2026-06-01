"""
Lognormal fitting: one-step QP (fit_lognormal) and iterative GN/LM (fit_lognormal_gn).
Migrated from log_norm.ipynb Cells 8, 10.
"""
import numpy as np
import cvxpy as cp

from .config import R_ANN
from .pricing import lognormal_price_vec


def _numerical_jacobian(df, mu, sigma, spot_0, T, r):
    """Numerical Jacobian of lognormal_price_vec w.r.t. (mu, sigma)."""
    h_mu    = max(1e-4, 0.01 * max(abs(mu), 0.01))
    h_sigma = max(1e-4, 0.01 * sigma)
    base = lognormal_price_vec(df, mu, sigma, spot_0, T, r)
    d_mu    = (lognormal_price_vec(df, mu + h_mu,      sigma,           spot_0, T, r) - base) / h_mu
    d_sigma = (lognormal_price_vec(df, mu,             sigma + h_sigma, spot_0, T, r) - base) / h_sigma
    return np.column_stack([d_mu, d_sigma]), base


def fit_lognormal(option_df, spot_0, T, r=R_ANN,
                  mu_guess=None, sigma_guess=None,
                  recent_log_rets=None):
    """
    Fit (mu_ann, sigma_ann) to option_df market mids via one linearized CVXPY QP step.

    Returns dict: mu_guess, sigma_guess, mu_fit, sigma_fit, param_cov,
                  train_rmse_guess, train_rmse_fit, holdout_rmse_guess, holdout_rmse_fit,
                  option_df, T, r, spot.
    """
    if mu_guess is None:
        mu_guess = (float(recent_log_rets.mean() * 252)
                    if recent_log_rets is not None and len(recent_log_rets) else 0.0)
    if sigma_guess is None:
        sigma_guess = max(
            float(recent_log_rets.std() * np.sqrt(252))
            if recent_log_rets is not None and len(recent_log_rets) >= 2 else 0.15,
            1e-4,
        )

    df = option_df.sort_values(["type", "strike"]).reset_index(drop=True).copy()
    holdout_mask = (np.arange(len(df)) % 5) == 0
    train_df = df.loc[~holdout_mask]

    bump_mu    = max(0.01, 0.25 * max(abs(mu_guess), 0.01))
    bump_sigma = max(0.01, 0.10 * sigma_guess)

    base = lognormal_price_vec(train_df, mu_guess, sigma_guess, spot_0, T, r)
    jac  = np.column_stack([
        (lognormal_price_vec(train_df, mu_guess + bump_mu,    sigma_guess,             spot_0, T, r) - base) / bump_mu,
        (lognormal_price_vec(train_df, mu_guess,             sigma_guess + bump_sigma, spot_0, T, r) - base) / bump_sigma,
    ])

    step = cp.Variable(2)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(base + jac @ step - train_df["market_mid"].to_numpy())),
        [sigma_guess + step[1] >= 0.02]
    )
    prob.solve()
    sv = np.zeros(2) if step.value is None else np.asarray(step.value).reshape(-1)

    mu_fit    = mu_guess    + float(sv[0])
    sigma_fit = max(0.02,    sigma_guess + float(sv[1]))

    df["guess_price"] = lognormal_price_vec(df, mu_guess, sigma_guess, spot_0, T, r)
    df["fit_price"]   = lognormal_price_vec(df, mu_fit,   sigma_fit,   spot_0, T, r)

    def rmse(mask, col):
        return float(np.sqrt(((df.loc[mask, col] - df.loc[mask, "market_mid"]) ** 2).mean()))

    # Local parameter covariance (for sampling uncertainty)
    base_fit = lognormal_price_vec(train_df, mu_fit, sigma_fit, spot_0, T, r)
    bm_f = max(0.01, 0.25 * max(abs(mu_fit), 0.01))
    bs_f = max(0.01, 0.10 * sigma_fit)
    jac_fit = np.column_stack([
        (lognormal_price_vec(train_df, mu_fit + bm_f, sigma_fit,       spot_0, T, r) - base_fit) / bm_f,
        (lognormal_price_vec(train_df, mu_fit,        sigma_fit + bs_f, spot_0, T, r) - base_fit) / bs_f,
    ])
    resid = train_df["market_mid"].to_numpy() - base_fit
    dof   = max(len(train_df) - 2, 1)
    mse   = float(np.sum(resid**2) / dof)
    param_cov = mse * np.linalg.pinv(jac_fit.T @ jac_fit)
    param_cov = np.nan_to_num(0.5 * (param_cov + param_cov.T), nan=0.0, posinf=0.0, neginf=0.0)

    return dict(
        mu_guess=mu_guess, sigma_guess=sigma_guess,
        mu_fit=mu_fit,     sigma_fit=sigma_fit,
        param_cov=param_cov,
        train_rmse_guess=rmse(~holdout_mask, "guess_price"),
        train_rmse_fit=rmse(~holdout_mask, "fit_price"),
        holdout_rmse_guess=rmse(holdout_mask, "guess_price"),
        holdout_rmse_fit=rmse(holdout_mask, "fit_price"),
        option_df=df, T=T, r=r, spot=spot_0,
    )


def fit_lognormal_gn(option_df, spot_0, T, r=R_ANN,
                     mu_guess=None, sigma_guess=None,
                     recent_log_rets=None,
                     max_iter=25, tol=1e-7,
                     lm_lambda_init=1e-2, lm_shrink=0.5, lm_grow=2.0,
                     verbose=True):
    """
    Gauss-Newton (Levenberg-Marquardt) iterative log-normal fit.

    Each iteration solves a CVXPY QP subproblem:
        min  ||r + J*delta||^2 + lambda * ||delta||^2
        s.t. sigma + delta_sigma >= 0.02

    Returns same dict as fit_lognormal plus 'gn_history'.
    """
    if mu_guess is None:
        mu_guess = (float(recent_log_rets.mean() * 252)
                    if recent_log_rets is not None and len(recent_log_rets) else 0.0)
    if sigma_guess is None:
        sigma_guess = max(
            float(recent_log_rets.std() * np.sqrt(252))
            if recent_log_rets is not None and len(recent_log_rets) >= 2 else 0.15,
            1e-4,
        )

    df = option_df.sort_values(["type", "strike"]).reset_index(drop=True).copy()
    holdout_mask = (np.arange(len(df)) % 5) == 0
    train_df     = df.loc[~holdout_mask].reset_index(drop=True)
    mids_train   = train_df["market_mid"].to_numpy()

    mu    = float(mu_guess)
    sigma = max(float(sigma_guess), 0.02)
    lam   = lm_lambda_init

    history = []

    for it in range(max_iter):
        J, prices_cur = _numerical_jacobian(train_df, mu, sigma, spot_0, T, r)
        residuals = mids_train - prices_cur
        sse_cur   = float(np.dot(residuals, residuals))

        history.append({"iter": it, "mu": mu, "sigma": sigma, "sse": sse_cur, "lambda": lam})

        delta = cp.Variable(2)
        lm_reg    = lam * cp.sum_squares(delta)
        objective = cp.Minimize(cp.sum_squares(residuals - J @ delta) + lm_reg)
        cp.Problem(objective, [sigma + delta[1] >= 0.02]).solve(warm_start=True)

        if delta.value is None:
            break

        dv = np.asarray(delta.value).reshape(-1)
        mu_new    = mu    + float(dv[0])
        sigma_new = max(0.02, sigma + float(dv[1]))

        _, prices_new = _numerical_jacobian(train_df, mu_new, sigma_new, spot_0, T, r)
        sse_new = float(np.sum((mids_train - prices_new) ** 2))

        if sse_new < sse_cur:
            mu, sigma = mu_new, sigma_new
            lam = max(lam * lm_shrink, 1e-8)
            if abs(sse_cur - sse_new) / max(sse_cur, 1e-12) < tol:
                break
        else:
            lam = min(lam * lm_grow, 1e3)

    # Final annotation
    df["guess_price"]    = lognormal_price_vec(df, mu_guess, sigma_guess, spot_0, T, r)
    df["fit_price_gn"]   = lognormal_price_vec(df, mu,       sigma,       spot_0, T, r)

    def rmse(mask, col):
        return float(np.sqrt(((df.loc[mask, col] - df.loc[mask, "market_mid"]) ** 2).mean()))

    J_final, prices_final = _numerical_jacobian(train_df, mu, sigma, spot_0, T, r)
    resid_final = mids_train - prices_final
    dof         = max(len(train_df) - 2, 1)
    mse_final   = float(np.dot(resid_final, resid_final) / dof)
    param_cov   = mse_final * np.linalg.pinv(J_final.T @ J_final)
    param_cov   = np.nan_to_num(0.5 * (param_cov + param_cov.T), nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"  GN converged in {len(history)} iters  "
              f"μ={mu:+.4f}  σ={sigma:.4f}  "
              f"train RMSE={rmse(~holdout_mask, 'fit_price_gn'):.4f}  "
              f"holdout RMSE={rmse(holdout_mask, 'fit_price_gn'):.4f}")

    return dict(
        mu_guess=mu_guess, sigma_guess=sigma_guess,
        mu_fit=mu,         sigma_fit=sigma,
        param_cov=param_cov,
        train_rmse_guess   =rmse(~holdout_mask, "guess_price"),
        train_rmse_fit     =rmse(~holdout_mask, "fit_price_gn"),
        holdout_rmse_guess =rmse( holdout_mask, "guess_price"),
        holdout_rmse_fit   =rmse( holdout_mask, "fit_price_gn"),
        option_df=df, T=T, r=r, spot=spot_0,
        gn_history=history,
    )
