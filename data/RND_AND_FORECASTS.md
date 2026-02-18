# RND forecasts: what changed, how it works, how to read μ and Σ

## What the last change did

Previously we set **μ (expected return) identical for all four assets** to the period risk-free rate (`use_risk_neutral_mu=True` by default). That meant:

- The optimizer could not tell SPY, SPY_CALL, SPY_PUT, and cash apart by expected return.
- Weights were driven only by risk (Σ), constraints, and costs, not by “alpha” per sleeve.

We changed to:

- **Default `use_risk_neutral_mu=False`**: μ is now the **sample mean** of the Monte Carlo returns per sleeve (with cash forced to the risk-free period return and optional clipping).
- So **SPY, SPY_CALL, SPY_PUT, and USDOLLAR have different μ** (e.g. call higher convexity, put often lower or negative carry), and the optimizer can use expected return to choose between them.
- **Σ** is unchanged: still from the same RND samples (option vols rescaled to a multiple of SPY vol).

So the change was: stop broadcasting a single scalar μ; use **per-sleeve sample means** (and cash = r_period) so the optimizer gets distinct expected returns.

---

## How RND works here

1. **Breeden–Litzenberger**  
   From the option chain we get call prices C(K) by strike. The risk-neutral density is  
   `q(K) = exp(r*T) * ∂²C/∂K²`.  
   We approximate the second derivative with a central difference on the strike grid.

2. **Sampling S_T**  
   We build a discrete CDF from q(K), draw uniform u, and invert to get samples of terminal spot S_T.

3. **Per-sample returns**  
   For each S_T we compute:
   - **SPY**: (S_T / S_0) − 1  
   - **Call**: max(S_T − K, 0) / C_0 − 1 (payoff at expiry / price today)  
   - **Put**: max(K − S_T, 0) / P_0 − 1  
   - **Cash**: exp(r*T) − 1  

4. **μ and Σ**  
   μ = sample mean of those four return series; Σ = sample covariance. Cash μ is then forced to the risk-free period return; with `use_risk_neutral_mu=False`, SPY and option means are (optionally) clipped to plausible ranges.

Important limitation: we use **terminal payoffs only**. We do **not** reprice options at an intermediate horizon (e.g. t+1) with remaining time and IV. So over the single period to expiry, when the call pays the put doesn’t and vice versa → **call/put correlation ≈ −1**. For less symmetric, more realistic option μ and Σ you’d need either historical sleeve returns (price t+1 / price t − 1) or repricing at horizon.

---

## How to read μ and Σ

**μ (expected return, one number per asset)**  
- **USDOLLAR**: Period risk-free return (e.g. ~0.0001 for a short period).  
- **SPY**: Sample mean of (S_T/S_0 − 1); with RND it’s often close to r*T; we allow clipping so it stays in a plausible band (e.g. ±15%).  
- **SPY_CALL**: Sample mean of (payoff_call / C_0 − 1). With positive skew of S_T, this can be higher than SPY (convexity). Clipped to e.g. [−0.5, 2.0].  
- **SPY_PUT**: Sample mean of (payoff_put / P_0 − 1). Often lower than call (insurance premium, negative carry). Same clip range.

So after the change, **μ is not a single scalar**: each sleeve has its own expected return, and the optimizer can use them.

**Σ (covariance of returns)**  
- **Diagonals** = variances (per period).  
  - SPY ~0.002–0.003 → vol ~5–5.5%.  
  - SPY_CALL / SPY_PUT rescaled to ~2× SPY vol → var ~0.01, vol ~10%.  
  - USDOLLAR ≈ 0 (riskless).  
- **Off-diagonals**:  
  - Cov(SPY, CALL) > 0, Cov(SPY, PUT) < 0 (call moves with spot, put opposite).  
  - Cov(CALL, PUT) strongly negative (terminal payoff mirror → correlation ≈ −1).  

So Σ is coherent with “options are levered/hedging on SPY” and “call vs put are opposite at expiry”. The only oddity is the near −1 call/put correlation, which comes from using terminal payoffs only; repricing at t+1 would soften that.

---

## Summary

- **Change**: We stopped broadcasting one μ to all assets. μ is now **per-sleeve** (sample means, cash = r_period, optional clipping).  
- **RND**: Build q(K) from the chain → sample S_T → compute return per asset → μ = sample mean, Σ = sample cov; option vols rescaled.  
- **Interpretation**: μ different across sleeves so the optimizer can use alpha; Σ reflects option vol and strong call/put opposition at expiry. For more realistic call/put correlation and carry, use historical sleeve returns or horizon repricing.
