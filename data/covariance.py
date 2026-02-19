"""
Covariance prediction via Iterated EWMA (IEWMA) and CM-IEWMA.

Reference:
  Johansson, Ogut, Pelger, Schmelzer, Boyd (2023),
  "A Simple Method for Predicting Covariance Matrices of Financial Returns"

Key idea: separate volatility and correlation estimation with different
half-lives — fast half-life for volatility (captures vol clustering),
slower half-life for correlation (more stable).

  1. Estimate per-asset volatilities with fast-halflife EWMA (H_vol)
  2. Standardize returns by estimated vol, winsorize at ±4.2
  3. Estimate correlation with slower-halflife EWMA on standardized returns
  4. Σ̂ = D̂_vol R̂ D̂_vol
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WINSORIZE_CLIP = 4.2


def _halflife_to_beta(halflife: float) -> float:
    """EWMA forgetting factor β such that β^H = 0.5."""
    return 2.0 ** (-1.0 / halflife)


class IEWMAPredictor:
    """Single Iterated-EWMA covariance predictor.

    Parameters
    ----------
    n_assets : number of assets
    h_vol    : half-life (in periods) for per-asset variance estimation
    h_cor    : half-life (in periods) for correlation estimation
    """

    def __init__(self, n_assets: int, h_vol: float, h_cor: float):
        self.n = n_assets
        self.h_vol = h_vol
        self.h_cor = h_cor
        self.beta_vol = _halflife_to_beta(h_vol)
        self.beta_cor = _halflife_to_beta(h_cor)

        self._var: Optional[np.ndarray] = None      # per-asset EWMA variance
        self._R_tilde: Optional[np.ndarray] = None   # EWMA of outer products of std returns
        self._t = 0

    def update(self, r: np.ndarray) -> None:
        """Feed one return observation of shape (n_assets,)."""
        r = np.asarray(r, dtype=float).ravel()
        self._t += 1

        r_sq = r ** 2
        if self._var is None:
            self._var = r_sq.copy()
        else:
            bv = self.beta_vol
            self._var = bv * self._var + (1.0 - bv) * r_sq

        sigma = np.sqrt(np.maximum(self._var, 1e-14))
        r_std = np.clip(r / sigma, -WINSORIZE_CLIP, WINSORIZE_CLIP)

        outer = np.outer(r_std, r_std)
        if self._R_tilde is None:
            self._R_tilde = outer.copy()
        else:
            bc = self.beta_cor
            self._R_tilde = bc * self._R_tilde + (1.0 - bc) * outer

    def predict(self) -> Optional[np.ndarray]:
        """Return predicted Σ̂ = D̂_vol R̂ D̂_vol, or None if < 2 observations."""
        if self._var is None or self._R_tilde is None or self._t < 2:
            return None

        d = np.sqrt(np.maximum(np.diag(self._R_tilde), 1e-14))
        D_inv = np.diag(1.0 / d)
        R_hat = D_inv @ self._R_tilde @ D_inv
        np.fill_diagonal(R_hat, 1.0)

        sigma = np.sqrt(np.maximum(self._var, 1e-14))
        D_vol = np.diag(sigma)
        Sigma = D_vol @ R_hat @ D_vol

        return (Sigma + Sigma.T) / 2.0

    def precision_cholesky(self) -> Optional[np.ndarray]:
        """Lower-triangular Cholesky of Σ̂^{-1} (for CM-IEWMA combination)."""
        Sigma = self.predict()
        if Sigma is None:
            return None
        Sigma_reg = Sigma + np.eye(self.n) * 1e-8
        try:
            L_cov = np.linalg.cholesky(Sigma_reg)
            L_prec = np.linalg.inv(L_cov).T
            return L_prec
        except np.linalg.LinAlgError:
            return None


class CMIEWMAPredictor:
    """Combined Multiple IEWMA predictor.

    Maintains K IEWMA predictors with different (H_vol, H_cor) pairs and
    combines them via inverse-variance weighting on trailing log-likelihood.
    Falls back to equal-weight averaging when optimization is not possible.

    Parameters
    ----------
    n_assets       : number of assets
    halflife_pairs : list of (h_vol, h_cor) tuples for each sub-predictor
    lookback       : trailing window for log-likelihood scoring
    """

    DEFAULT_HALFLIFE_PAIRS = [
        (2.0, 5.0),    # fast:   ~1 month vol, ~2.5 month cor   (biweekly periods)
        (4.0, 10.0),   # medium: ~2 month vol, ~5 month cor
        (8.0, 20.0),   # slow:   ~4 month vol, ~10 month cor
    ]

    def __init__(
        self,
        n_assets: int,
        halflife_pairs: Optional[list[tuple[float, float]]] = None,
        lookback: int = 10,
    ):
        self.n = n_assets
        pairs = halflife_pairs or self.DEFAULT_HALFLIFE_PAIRS
        self.K = len(pairs)
        self.lookback = lookback
        self.predictors = [
            IEWMAPredictor(n_assets, hv, hc) for hv, hc in pairs
        ]
        self._recent_returns: list[np.ndarray] = []

    def update(self, r: np.ndarray) -> None:
        r = np.asarray(r, dtype=float).ravel()
        for p in self.predictors:
            p.update(r)
        self._recent_returns.append(r.copy())
        if len(self._recent_returns) > self.lookback + 1:
            self._recent_returns.pop(0)

    def predict(self) -> Optional[np.ndarray]:
        """Weighted combination of K IEWMA covariance predictions."""
        preds = [p.predict() for p in self.predictors]
        valid = [(i, S) for i, S in enumerate(preds) if S is not None]

        if not valid:
            return None
        if len(valid) == 1:
            return valid[0][1]

        weights = self._score_weights(valid)
        Sigma = sum(w * S for w, (_, S) in zip(weights, valid))
        return (Sigma + Sigma.T) / 2.0

    def _score_weights(self, valid: list) -> np.ndarray:
        """Inverse-variance scoring on trailing log-likelihood.

        For each predictor k, compute trailing average log-likelihood:
          LL_k = mean_j [ -0.5 * (n*log(2π) + log|Σ_k| + r_j' Σ_k^{-1} r_j) ]
        Then softmax to get weights.
        """
        recent = self._recent_returns[-self.lookback:]
        K_valid = len(valid)

        if len(recent) < 2:
            return np.ones(K_valid) / K_valid

        scores = np.zeros(K_valid)
        for ki, (_, Sigma_k) in enumerate(valid):
            Sigma_reg = Sigma_k + np.eye(self.n) * 1e-8
            try:
                L = np.linalg.cholesky(Sigma_reg)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError:
                scores[ki] = -1e12
                continue

            try:
                Sigma_inv = np.linalg.inv(Sigma_reg)
            except np.linalg.LinAlgError:
                scores[ki] = -1e12
                continue

            ll_sum = 0.0
            for r_j in recent:
                mahal = r_j @ Sigma_inv @ r_j
                ll_sum += -0.5 * (log_det + mahal)
            scores[ki] = ll_sum / len(recent)

        # Softmax with temperature (prevents one predictor from dominating)
        scores -= scores.max()
        w = np.exp(scores / 0.5)
        w_sum = w.sum()
        if w_sum < 1e-15:
            return np.ones(K_valid) / K_valid
        return w / w_sum


def build_iewma_covariance_series(
    returns_df: "pd.DataFrame",
    halflife_pairs: Optional[list[tuple[float, float]]] = None,
) -> dict[int, np.ndarray]:
    """Run CM-IEWMA over a returns DataFrame and return {period_index: Σ̂}.

    The predictor is updated with returns[0..t-1] and predict() gives Σ̂_t,
    so Σ̂_t is out-of-sample (uses only past data).
    """
    n_assets = returns_df.shape[1]
    predictor = CMIEWMAPredictor(n_assets, halflife_pairs=halflife_pairs)

    sigma_series: dict[int, np.ndarray] = {}
    for t in range(len(returns_df)):
        sigma_t = predictor.predict()
        if sigma_t is not None:
            sigma_series[t] = sigma_t
        r_t = returns_df.iloc[t].values.astype(float)
        predictor.update(r_t)

    return sigma_series
