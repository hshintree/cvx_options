"""
Accuracy evaluation: predicted μ and Σ vs realized outcomes.
Migrated and extended from log_norm.ipynb Cells 24-25, 33.

New metrics added beyond the notebook:
  - RMSE
  - Hit rate (direction accuracy)
  - Spearman rank correlation
  - Covariance: variance ratio, log variance ratio, correlation bias, Frobenius error
  - Elliptical calibration: fraction of realized returns inside predicted 68%/95% ellipsoids
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2
from scipy.optimize import brentq

from .config import (
    R_ANN, TEST_DTE, N_SAMPLES, MIN_DAYS_BACK, LOOKBACK_DAYS,
    N_SCENARIOS, N_PARAM_DRAWS,
)
from .data_loader import load_spy_prices, eligible_chain_dates
from .pricing import load_options_for_dte, lognormal_price
from .fitting import fit_lognormal

ASSET_NAMES = ["underlying", "call", "put"]


def run_accuracy_test(
    asset: str = "SPY",
    test_dte: int = TEST_DTE,
    n_samples: int = N_SAMPLES,
    min_days_back: int = MIN_DAYS_BACK,
    lookback_days: int = LOOKBACK_DAYS,
    n_scenarios: int = N_SCENARIOS,
    n_param_draws: int = N_PARAM_DRAWS,
    r_ann: float = R_ANN,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
    """
    For each of n_samples historical chain dates, fit a lognormal distribution,
    predict μ and Σ for (underlying, ATM call, ATM put), then compare to
    realized outcomes at expiry.

    Returns
    -------
    mu_df        : pd.DataFrame — one row per date with pred/real mu + vol columns
    cov_pred_list: list of 3×3 np.ndarray — predicted covariance per window
    cov_real_list: list of 3×3 np.ndarray — realized covariance per window
    """
    from data.forecasts import _bs_call_price, _bs_put_price  # noqa

    prices      = load_spy_prices(asset)
    last_close  = prices.index.max()
    chain_dates = eligible_chain_dates(asset, min_days_back)

    rng = np.random.default_rng(seed)
    if len(chain_dates) <= n_samples:
        sampled = chain_dates
    else:
        idx = rng.choice(len(chain_dates), size=n_samples, replace=False)
        sampled = [chain_dates[i] for i in sorted(idx)]

    test_rows      = []
    cov_pred_list  = []
    cov_real_list  = []

    for chain_date in sampled:
        try:
            ref_ts  = pd.Timestamp(chain_date)
            idx_ref = prices.index.get_indexer([ref_ts], method="ffill")[0]
            if idx_ref < 0:
                continue
            spot_ref = float(prices.iloc[idx_ref]["close"])

            opt_df, expiry_str, T_ref = load_options_for_dte(
                chain_date, test_dte, spot_ref, asset=asset, r=r_ann
            )
            recent_lr = prices.loc[prices.index <= ref_ts, "log_ret"].iloc[-lookback_days:].dropna()
            mu_g  = float(recent_lr.mean() * 252) if len(recent_lr) else 0.0
            sig_g = max(float(recent_lr.std() * np.sqrt(252)) if len(recent_lr) >= 2 else 0.15, 1e-4)

            res = fit_lognormal(opt_df, spot_ref, T_ref, r=r_ann,
                                mu_guess=mu_g, sigma_guess=sig_g)

            calls_sub = opt_df[opt_df["type"] == "call"]
            puts_sub  = opt_df[opt_df["type"] == "put"]
            if len(calls_sub) == 0 or len(puts_sub) == 0:
                continue

            K_call = float(calls_sub.loc[(calls_sub["strike"] - spot_ref).abs().idxmin(), "strike"])
            K_put  = float(puts_sub.loc[(puts_sub["strike"]   - spot_ref).abs().idxmin(), "strike"])
            C0     = lognormal_price(K_call, True,  res["mu_fit"], res["sigma_fit"], spot_ref, T_ref)
            P0     = lognormal_price(K_put,  False, res["mu_fit"], res["sigma_fit"], spot_ref, T_ref)
            C0_mkt = float(calls_sub[calls_sub["strike"] == K_call]["market_mid"].iloc[0])
            P0_mkt = float(puts_sub[puts_sub["strike"]  == K_put ]["market_mid"].iloc[0])
            if C0 < 1e-4 or P0 < 1e-4 or C0_mkt < 1e-4 or P0_mkt < 1e-4:
                continue

            # ── Predicted distribution (parameter-uncertainty MC) ─────────────
            evals, evecs = np.linalg.eigh(res["param_cov"])
            evals = np.clip(evals, 1e-10, None)
            sqrt_cov = evecs @ np.diag(np.sqrt(evals))
            theta_draws = np.array([res["mu_fit"], res["sigma_fit"]]) + \
                rng.standard_normal((n_param_draws, 2)) @ sqrt_cov.T
            mu_draws    = np.clip(theta_draws[:, 0], -2.0, 2.0)
            sigma_draws = np.clip(theta_draws[:, 1], 0.02, 1.0)

            pidx = rng.integers(0, n_param_draws, size=n_scenarios)
            z    = rng.standard_normal(n_scenarios)
            log_st = (np.log(spot_ref)
                      + (mu_draws[pidx] - 0.5 * sigma_draws[pidx]**2) * T_ref
                      + sigma_draws[pidx] * np.sqrt(T_ref) * z)
            S_T_s = np.exp(log_st)

            R_und_p  = (S_T_s - spot_ref) / spot_ref
            R_call_p = (np.maximum(S_T_s - K_call, 0.0) - C0) / C0
            R_put_p  = (np.maximum(K_put - S_T_s, 0.0) - P0) / P0
            R_pred   = np.column_stack([R_und_p, R_call_p, R_put_p])
            mu_pred_v = R_pred.mean(axis=0)
            Sig_pred  = np.cov(R_pred, rowvar=False)

            # ── Realized outcome ──────────────────────────────────────────────
            exp_ts       = pd.Timestamp(expiry_str)
            real_end_ts  = min(last_close, exp_ts)
            idx_end      = prices.index.get_indexer([real_end_ts], method="ffill")[0]
            if idx_end < 0:
                continue
            spot_end    = float(prices.iloc[idx_end]["close"])
            R_und_real  = (spot_end - spot_ref) / spot_ref
            R_call_real = (max(spot_end - K_call, 0.0) - C0_mkt) / C0_mkt
            R_put_real  = (max(K_put - spot_end,  0.0) - P0_mkt) / P0_mkt
            mu_real_v   = np.array([R_und_real, R_call_real, R_put_real])

            # ── Realized daily covariance (scaled to period) ──────────────────
            try:
                iv_obj = lambda sig: _bs_call_price(spot_ref, K_call, r_ann, T_ref, sig) - C0_mkt
                iv_atm = float(brentq(iv_obj, 0.01, 2.0))
            except Exception:
                iv_atm = res["sigma_fit"]

            spy_slice = prices.loc[ref_ts:real_end_ts, "close"]
            dates_sl  = spy_slice.index
            spots_sl  = spy_slice.values.astype(float)
            nd        = len(spots_sl)
            if nd < 3:
                continue

            T_rem = np.array([max((exp_ts - d).days / 365.0, 1e-6) for d in dates_sl])
            cv    = np.zeros(nd)
            pv    = np.zeros(nd)
            for i in range(nd):
                if dates_sl[i] < exp_ts and T_rem[i] > 1 / 365.0:
                    cv[i] = max(_bs_call_price(spots_sl[i], K_call, r_ann, T_rem[i], iv_atm), 0.01)
                    pv[i] = max(_bs_put_price( spots_sl[i], K_put,  r_ann, T_rem[i], iv_atm), 0.01)
                else:
                    cv[i] = max(spots_sl[i] - K_call, 0.01)
                    pv[i] = max(K_put - spots_sl[i],  0.01)

            r_und_d  = (spots_sl[1:] - spots_sl[:-1]) / spots_sl[:-1]
            r_call_d = (cv[1:] - cv[:-1]) / cv[:-1]
            r_put_d  = (pv[1:] - pv[:-1]) / pv[:-1]
            R_daily  = np.column_stack([r_und_d, r_call_d, r_put_d])
            Sig_real = R_daily.shape[0] * np.cov(R_daily, rowvar=False)

            test_rows.append(dict(
                chain_date=chain_date,
                expiry=expiry_str,
                actual_dte=round(T_ref * 365),
                spot_entry=spot_ref,
                spot_end=spot_end,
                mu_pred_und=mu_pred_v[0],  mu_real_und=mu_real_v[0],
                mu_pred_call=mu_pred_v[1], mu_real_call=mu_real_v[1],
                mu_pred_put=mu_pred_v[2],  mu_real_put=mu_real_v[2],
                vol_pred_und=float(np.sqrt(max(Sig_pred[0, 0], 0))),
                vol_pred_call=float(np.sqrt(max(Sig_pred[1, 1], 0))),
                vol_pred_put=float(np.sqrt(max(Sig_pred[2, 2], 0))),
                vol_real_und=float(np.sqrt(max(Sig_real[0, 0], 0))),
                vol_real_call=float(np.sqrt(max(Sig_real[1, 1], 0))),
                vol_real_put=float(np.sqrt(max(Sig_real[2, 2], 0))),
            ))
            cov_pred_list.append(Sig_pred)
            cov_real_list.append(Sig_real)

        except Exception:
            continue

    mu_df = pd.DataFrame(test_rows)
    print(f"Accuracy test complete: {len(mu_df)} windows with valid predictions.")
    return mu_df, cov_pred_list, cov_real_list


def summarize_mu_accuracy(mu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset prediction accuracy metrics.

    Metrics (beyond notebook's bias/MAE/corr):
      bias         = mean(realized − predicted)
      MAE          = mean(|realized − predicted|)
      RMSE         = sqrt(mean((realized − predicted)^2))
      hit_rate     = fraction where sign(realized) == sign(predicted)
      pearson_corr = Pearson correlation of predicted and realized
      spearman_rho = Spearman rank correlation
      vol_ratio    = median(vol_pred / vol_real)
    """
    if len(mu_df) == 0:
        return pd.DataFrame()

    rows = []
    for asset, p_col, r_col, pv_col, rv_col in [
        ("underlying", "mu_pred_und",  "mu_real_und",  "vol_pred_und",  "vol_real_und"),
        ("call",       "mu_pred_call", "mu_real_call", "vol_pred_call", "vol_real_call"),
        ("put",        "mu_pred_put",  "mu_real_put",  "vol_pred_put",  "vol_real_put"),
    ]:
        pred = mu_df[p_col].dropna()
        real = mu_df[r_col].dropna()
        common = pred.index.intersection(real.index)
        p, r = pred.loc[common].values, real.loc[common].values

        err = r - p
        hit = np.mean(np.sign(r) == np.sign(p)) if len(p) > 0 else np.nan
        pearson = float(np.corrcoef(p, r)[0, 1]) if len(p) > 1 else np.nan
        spearman = float(spearmanr(p, r).correlation) if len(p) > 1 else np.nan

        vp = mu_df[pv_col].replace(0, np.nan)
        vr = mu_df[rv_col].replace(0, np.nan)
        vol_ratio = float((vp / vr).median())

        rows.append({
            "asset": asset,
            "n": len(p),
            "bias": float(err.mean()),
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err**2).mean())),
            "hit_rate": float(hit),
            "pearson_corr": pearson,
            "spearman_rho": spearman,
            "vol_ratio_median": vol_ratio,
        })

    return pd.DataFrame(rows).set_index("asset")


def summarize_cov_accuracy(
    cov_pred_list: list[np.ndarray],
    cov_real_list: list[np.ndarray],
) -> pd.DataFrame:
    """
    Compute per-matrix-entry covariance accuracy metrics.

    Metrics:
      variance_ratio      = mean(pred_var / real_var)  [diagonal only]
      log_variance_ratio  = mean(log(pred_var / real_var))
      correlation_bias    = mean(pred_corr − real_corr)  [off-diagonal]
      frobenius_error     = mean(||Σ_pred - Σ_real||_F)
      elliptical_68       = fraction of realized returns inside predicted 68% ellipsoid
      elliptical_95       = fraction of realized returns inside predicted 95% ellipsoid
    """
    if not cov_pred_list or not cov_real_list:
        return pd.DataFrame()

    n_mats   = min(len(cov_pred_list), len(cov_real_list))
    n_assets = cov_pred_list[0].shape[0]

    var_ratios   = []
    log_var_ratios = []
    corr_biases  = []
    frob_errors  = []

    for Sp, Sr in zip(cov_pred_list[:n_mats], cov_real_list[:n_mats]):
        for i in range(n_assets):
            vp, vr = Sp[i, i], Sr[i, i]
            if vr > 1e-14:
                ratio = vp / vr
                var_ratios.append((i, ratio))
                if ratio > 0:
                    log_var_ratios.append((i, np.log(ratio)))

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                vip = max(Sp[i, i], 1e-14)
                vjp = max(Sp[j, j], 1e-14)
                vir = max(Sr[i, i], 1e-14)
                vjr = max(Sr[j, j], 1e-14)
                corr_p = Sp[i, j] / np.sqrt(vip * vjp)
                corr_r = Sr[i, j] / np.sqrt(vir * vjr)
                corr_biases.append(((i, j), corr_p - corr_r))

        frob_errors.append(np.linalg.norm(Sp - Sr, "fro"))

    rows = []
    labels = ASSET_NAMES[:n_assets]
    for i, lbl in enumerate(labels):
        ratios = [r for idx, r in var_ratios if idx == i]
        lratios = [r for idx, r in log_var_ratios if idx == i]
        rows.append({
            "asset": lbl,
            "mean_var_ratio": float(np.mean(ratios)) if ratios else np.nan,
            "log_var_ratio":  float(np.mean(lratios)) if lratios else np.nan,
        })

    cov_df = pd.DataFrame(rows).set_index("asset")
    cov_df.loc["(off-diag)", "corr_bias_mean"] = float(
        np.mean([b for _, b in corr_biases]) if corr_biases else np.nan
    )
    cov_df.loc["(overall)", "frobenius_mean"] = float(np.mean(frob_errors)) if frob_errors else np.nan

    return cov_df


def elliptical_calibration(
    cov_pred_list: list[np.ndarray],
    mu_df: pd.DataFrame,
    levels: tuple = (0.68, 0.95),
) -> dict:
    """
    Compute the fraction of realized return vectors that fall inside
    the predicted confidence ellipsoid at each level.

    Uses the chi-squared CDF to determine the ellipsoid threshold:
      Mahalanobis² = (r_real - mu_pred)' Σ_pred^{-1} (r_real - mu_pred)
      covered if Mahalanobis² ≤ chi2.ppf(level, df=3)
    """
    if len(mu_df) == 0 or not cov_pred_list:
        return {}

    n = min(len(mu_df), len(cov_pred_list))
    coverage = {lv: [] for lv in levels}

    for i in range(n):
        Sp = cov_pred_list[i]
        row = mu_df.iloc[i]
        mu_p  = np.array([row["mu_pred_und"], row["mu_pred_call"], row["mu_pred_put"]])
        r_real = np.array([row["mu_real_und"],  row["mu_real_call"],  row["mu_real_put"]])

        try:
            Sp_inv = np.linalg.pinv(Sp)
            diff   = r_real - mu_p
            maha2  = float(diff @ Sp_inv @ diff)
            df     = len(mu_p)
            for lv in levels:
                threshold = chi2.ppf(lv, df=df)
                coverage[lv].append(1.0 if maha2 <= threshold else 0.0)
        except Exception:
            continue

    return {f"coverage_{int(lv*100)}pct": float(np.mean(v)) if v else np.nan
            for lv, v in coverage.items()}


def print_accuracy_report(
    mu_df: pd.DataFrame,
    cov_pred_list: list,
    cov_real_list: list,
) -> None:
    """Print a full accuracy report to stdout."""
    print("\n── μ Prediction Accuracy ────────────────────────────────────────────────")
    mu_summary = summarize_mu_accuracy(mu_df)
    if not mu_summary.empty:
        print(mu_summary.to_string(float_format="{:.4f}".format))

    print("\n── Σ Accuracy ───────────────────────────────────────────────────────────")
    cov_summary = summarize_cov_accuracy(cov_pred_list, cov_real_list)
    if not cov_summary.empty:
        print(cov_summary.to_string(float_format="{:.4f}".format))

    print("\n── Elliptical Calibration (fraction realized inside predicted ellipsoid) ─")
    calib = elliptical_calibration(cov_pred_list, mu_df)
    for k, v in calib.items():
        print(f"  {k}: {v:.1%}" if not np.isnan(v) else f"  {k}: —")
