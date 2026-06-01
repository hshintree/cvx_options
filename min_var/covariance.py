"""
Delta-Gamma-Vega (DGV) daily covariance construction.
Migrated from log_norm.ipynb Cell 36.

The DGV decomposition approximates daily option variance via second-order Taylor:
    Var(R_opt) ≈ (Δ·S/C)²·σ²/252           ← delta systematic
               + (Γ·S²/2C)²·2σ⁴/252²       ← gamma convexity
               + (Vega/C)²·σ_IV_daily²     ← vega vol-of-vol
"""
import contextlib
import io as _io
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import R_ANN, LOOKBACK
from .fitting import fit_lognormal_gn, fit_lognormal
from .pricing import lognormal_price


def _silent(fn, *args, **kwargs):
    with contextlib.redirect_stdout(_io.StringIO()):
        return fn(*args, **kwargs)


def _mini_optdf(row: dict) -> pd.DataFrame | None:
    """Build a minimal option DataFrame from a single research-panel row."""
    rows = []
    for slot, is_call in [("call_1", True), ("call_2", True), ("put_1", False), ("put_2", False)]:
        K   = row.get(f"{slot}_strike")
        mid = row.get(f"{slot}_mid_t")
        if pd.notna(K) and pd.notna(mid) and float(mid) > 0.005:
            rows.append({
                "strike":     float(K),
                "type":       "call" if is_call else "put",
                "market_mid": float(mid),
            })
    return pd.DataFrame(rows) if len(rows) >= 2 else None


def _gn_cov_dgv(row: dict, r: float = R_ANN, n_param: int = 30,
                rng=None) -> dict | None:
    """
    Fit lognormal GN to one research-panel row and return the 3×3 DGV daily covariance
    for (underlying, call_1, put_1), plus fitting metadata.

    Returns dict with keys:
        mu_fit, sigma_fit, K_call, K_put, C0, P0, T, spot,
        Sigma_daily (3×3 ndarray), lev_c, lev_p
    Returns None if fit or option data is unavailable.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    opt_df = _mini_optdf(row)
    if opt_df is None:
        return None

    spot = float(row["spot_t"])
    T    = max(float(row.get("chosen_dte_t") or 21) / 365.0, 2 / 365)
    sig0 = max(float(row.get("call_1_iv_t")  or 0.20), 0.05)

    # Step 1: GN fit for forward-looking sigma
    try:
        res = _silent(fit_lognormal_gn, opt_df, spot_0=spot, T=T, r=r,
                      mu_guess=r, sigma_guess=sig0, max_iter=15, verbose=False)
    except Exception:
        try:
            res = fit_lognormal(opt_df, spot_0=spot, T=T, r=r,
                                mu_guess=r, sigma_guess=sig0)
        except Exception:
            return None

    # Sample sigma from Fisher distribution (uncertainty averaging)
    evals, evecs = np.linalg.eigh(res["param_cov"])
    evals        = np.clip(evals, 1e-12, None)
    sqrt_cov     = evecs @ np.diag(np.sqrt(evals))
    draws        = (np.array([res["mu_fit"], res["sigma_fit"]])
                    + rng.standard_normal((n_param, 2)) @ sqrt_cov.T)
    sigma        = float(np.clip(draws[:, 1], 0.02, 1.5).mean())
    var_und      = sigma**2 / 252   # daily underlying variance

    # Step 2: Greeks from research panel (preferred over BS approximation)
    delta_c = float(row.get("call_1_delta_t") or np.nan)
    gamma_c = float(row.get("call_1_gamma_t") or 0.0)
    vega_c  = float(row.get("call_1_vega_t")  or 0.0)
    C0_mkt  = float(row.get("call_1_mid_t")   or 0.0)
    delta_p = float(row.get("put_1_delta_t")  or np.nan)
    gamma_p = float(row.get("put_1_gamma_t")  or 0.0)
    vega_p  = float(row.get("put_1_vega_t")   or 0.0)
    P0_mkt  = float(row.get("put_1_mid_t")    or 0.0)

    # Fallback: BS Greeks when panel Greeks are missing
    calls = opt_df[opt_df["type"] == "call"]
    puts  = opt_df[opt_df["type"] == "put"]
    if (np.isnan(delta_c) or np.isnan(delta_p)
            or C0_mkt < spot * 1e-4 or P0_mkt < spot * 1e-4):
        K_c_fb = float(calls.loc[(calls["strike"] - spot).abs().idxmin(), "strike"])
        K_p_fb = float(puts.loc[(puts["strike"]   - spot).abs().idxmin(), "strike"])
        sqrtT  = max(sigma * np.sqrt(T), 1e-6)
        d1_c   = (np.log(spot / K_c_fb) + (r + 0.5 * sigma**2) * T) / sqrtT
        d1_p   = (np.log(spot / K_p_fb) + (r + 0.5 * sigma**2) * T) / sqrtT
        delta_c = float(norm.cdf(d1_c))
        delta_p = float(norm.cdf(d1_p) - 1.0)
        phi     = float(norm.pdf(d1_c))
        gamma_c = gamma_p = phi / (spot * sigma * np.sqrt(T))
        vega_c  = vega_p  = spot * phi * np.sqrt(T)
        C0_mkt  = max(lognormal_price(K_c_fb, True,  res["mu_fit"], sigma, spot, T), spot * 1e-4)
        P0_mkt  = max(lognormal_price(K_p_fb, False, res["mu_fit"], sigma, spot, T), spot * 1e-4)

    # ATM strikes
    K_c = float(calls.loc[(calls["strike"] - spot).abs().idxmin(), "strike"])
    K_p = float(puts.loc[(puts["strike"]   - spot).abs().idxmin(), "strike"])

    # Step 3: return-space leverages
    C_s    = max(C0_mkt, spot * 1e-4)
    P_s    = max(P0_mkt, spot * 1e-4)
    lev_c  = delta_c * spot / C_s            # Δ·S/C  (call: +15 to +100)
    lev_p  = delta_p * spot / P_s            # Δ·S/P  (put: -15 to -100)
    glev_c = 0.5 * gamma_c * spot**2 / C_s   # Γ·S²/(2C)
    glev_p = 0.5 * gamma_p * spot**2 / P_s
    vret_c = vega_c / max(C0_mkt, 1e-4)      # Vega / C
    vret_p = vega_p / max(P0_mkt, 1e-4)

    # Step 4: DGV covariance (daily)
    levs        = np.array([1.0, lev_c, lev_p])
    Sigma_daily = np.outer(levs, levs) * var_und          # delta component
    IV_VOL_DAILY = 0.01                                    # ~1 vol-pt daily IV move
    Sigma_daily[1, 1] += glev_c**2 * 2.0 * var_und**2    # gamma component (call)
    Sigma_daily[2, 2] += glev_p**2 * 2.0 * var_und**2    # gamma component (put)
    Sigma_daily[1, 1] += (vret_c * IV_VOL_DAILY)**2       # vega component (call)
    Sigma_daily[2, 2] += (vret_p * IV_VOL_DAILY)**2       # vega component (put)

    return dict(
        mu_fit=res["mu_fit"], sigma_fit=sigma,
        K_call=K_c, K_put=K_p, C0=C0_mkt, P0=P0_mkt, T=T, spot=spot,
        Sigma_daily=Sigma_daily, lev_c=lev_c, lev_p=lev_p,
    )


def build_full_cov(rebal_date, rp_idx: dict, und_rets_df: pd.DataFrame,
                   syms: list, asset_cols: list,
                   lookback: int = LOOKBACK) -> tuple[np.ndarray, dict]:
    """
    Assemble the full N×N cross-asset covariance at rebal_date using:
      - Per-symbol DGV daily covariance (from GN fit)
      - Rolling realized underlying cross-correlation (off-diagonal blocks)

    Returns (Sigma [n×n, daily], gn_results dict keyed by symbol).
    """
    rng = np.random.default_rng(int(pd.Timestamp(rebal_date).value % (2**31)))

    gn = {}
    for sym in syms:
        df_sym = rp_idx.get(sym)
        if df_sym is None:
            continue
        avail = df_sym.loc[:rebal_date]
        if avail.empty:
            continue
        result = _gn_cov_dgv(avail.iloc[-1].to_dict(), rng=rng)
        if result:
            gn[sym] = result

    if not gn:
        raise ValueError(f"No DGV fits succeeded for {rebal_date}")

    # Rolling underlying cross-correlation
    end_loc   = und_rets_df.index.get_indexer([rebal_date], method="ffill")[0]
    start_loc = max(0, end_loc - lookback)
    corr_df   = und_rets_df.iloc[start_loc:end_loc][syms].dropna().corr()
    Corr      = corr_df.reindex(index=syms, columns=syms).fillna(0).values
    np.fill_diagonal(Corr, 1.0)

    TYPE_IDX  = {"": 0, "_call1": 1, "_put1": 2}
    sym_order = {s: i for i, s in enumerate(syms)}
    n         = len(asset_cols)
    sym_of    = [c.split("_")[0] for c in asset_cols]
    suf_of    = [c[len(c.split("_")[0]):] for c in asset_cols]

    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si, sj = sym_of[i], sym_of[j]
            ti = TYPE_IDX.get(suf_of[i], -1)
            tj = TYPE_IDX.get(suf_of[j], -1)
            if ti < 0 or tj < 0 or si not in gn or sj not in gn:
                continue
            gi, gj = gn[si], gn[sj]
            if si == sj:
                Sigma[i, j] = gi["Sigma_daily"][ti, tj]
            else:
                var_i  = max(gi["Sigma_daily"][0, 0], 1e-12)
                var_j  = max(gj["Sigma_daily"][0, 0], 1e-12)
                sens_i = 1.0 if ti == 0 else gi["Sigma_daily"][ti, 0] / var_i
                sens_j = 1.0 if tj == 0 else gj["Sigma_daily"][tj, 0] / var_j
                rho    = Corr[sym_order[si], sym_order[sj]]
                Sigma[i, j] = sens_i * rho * sens_j * np.sqrt(var_i * var_j)

    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-7 * np.eye(n)
    return Sigma, gn
