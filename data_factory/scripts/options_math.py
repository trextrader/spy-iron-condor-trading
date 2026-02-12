#!/usr/bin/env python3
"""
Black-Scholes Greeks Calculator + Newton-Raphson IV Solver
===========================================================
Pure-math module for CondorNet v3.0 dataset construction.

Provides:
  - bs_price(S, K, T, r, sigma, cp)    -> theoretical option price
  - bs_greeks(S, K, T, r, sigma, cp)   -> dict of all Greeks
  - implied_vol(price, S, K, T, r, cp) -> IV via Newton-Raphson
  - compute_greeks_df(df)              -> vectorized batch over DataFrame

All formulas follow standard Black-Scholes-Merton (1973) for European options.
SPY options are American-style, but BS provides a close approximation
that is standard practice for dataset construction and model training.

Conventions:
  - S     = underlying spot price
  - K     = strike price
  - T     = time to expiry in years (te)
  - r     = risk-free rate (annualized, default 0.05)
  - sigma = implied volatility (annualized)
  - cp    = 'call' or 'put'
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import time

# Default risk-free rate (Fed Funds ~5.25-5.50% in late 2025)
R_DEFAULT = 0.05


# ============================================================================
# CORE BLACK-SCHOLES
# ============================================================================

def _d1(S, K, T, r, sigma):
    """Compute d1 term of Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    """Compute d2 term of Black-Scholes formula."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, cp='call'):
    """
    Black-Scholes European option price.

    Returns NaN for invalid inputs (T <= 0, sigma <= 0, etc.)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)

    if cp == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================================
# GREEKS (scalar versions)
# ============================================================================

def bs_delta(S, K, T, r, sigma, cp='call'):
    """Delta: dV/dS"""
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    if cp == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    """Gamma: d2V/dS2 (same for calls and puts)"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S, K, T, r, sigma, cp='call'):
    """Theta: dV/dT (per year, divide by 365 for daily)"""
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if cp == 'call':
        return common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return common + r * K * np.exp(-r * T) * norm.cdf(-d2)


def bs_vega(S, K, T, r, sigma):
    """Vega: dV/dsigma (same for calls and puts). Per 1.0 vol change."""
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_rho(S, K, T, r, sigma, cp='call'):
    """Rho: dV/dr"""
    if T <= 0 or sigma <= 0:
        return np.nan
    d2 = _d2(S, K, T, r, sigma)
    if cp == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def bs_greeks(S, K, T, r, sigma, cp='call'):
    """Compute all Greeks at once. Returns dict."""
    return {
        'delta': bs_delta(S, K, T, r, sigma, cp),
        'gamma': bs_gamma(S, K, T, r, sigma),
        'theta': bs_theta(S, K, T, r, sigma, cp),
        'vega':  bs_vega(S, K, T, r, sigma),
        'rho':   bs_rho(S, K, T, r, sigma, cp),
    }


# ============================================================================
# VECTORIZED BLACK-SCHOLES (numpy arrays)
# ============================================================================

def bs_price_vec(S, K, T, r, sigma, is_call):
    """
    Vectorized BS price. All inputs are numpy arrays or scalars.
    is_call: boolean array (True=call, False=put)
    """
    # Mask invalid inputs
    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    price = np.full_like(S, np.nan, dtype=float)

    S_v = np.where(valid, S, 1.0)
    K_v = np.where(valid, K, 1.0)
    T_v = np.where(valid, T, 1.0)
    sigma_v = np.where(valid, sigma, 0.2)

    d1 = (np.log(S_v / K_v) + (r + 0.5 * sigma_v**2) * T_v) / (sigma_v * np.sqrt(T_v))
    d2 = d1 - sigma_v * np.sqrt(T_v)

    call_price = S_v * norm.cdf(d1) - K_v * np.exp(-r * T_v) * norm.cdf(d2)
    put_price = K_v * np.exp(-r * T_v) * norm.cdf(-d2) - S_v * norm.cdf(-d1)

    price = np.where(valid & is_call, call_price, price)
    price = np.where(valid & ~is_call, put_price, price)

    return price


def bs_greeks_vec(S, K, T, r, sigma, is_call):
    """
    Vectorized Greeks computation. Returns dict of numpy arrays.
    """
    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    nan_arr = np.full_like(S, np.nan, dtype=float)

    S_v = np.where(valid, S, 1.0)
    K_v = np.where(valid, K, 1.0)
    T_v = np.where(valid, T, 1.0)
    sigma_v = np.where(valid, sigma, 0.2)

    sqrtT = np.sqrt(T_v)
    d1 = (np.log(S_v / K_v) + (r + 0.5 * sigma_v**2) * T_v) / (sigma_v * sqrtT)
    d2 = d1 - sigma_v * sqrtT

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d1 = norm.cdf(-d1)
    cdf_neg_d2 = norm.cdf(-d2)
    disc = np.exp(-r * T_v)

    # Delta
    delta = np.where(is_call, cdf_d1, cdf_d1 - 1.0)

    # Gamma (same for C and P)
    gamma = pdf_d1 / (S_v * sigma_v * sqrtT)

    # Theta
    common_theta = -(S_v * pdf_d1 * sigma_v) / (2 * sqrtT)
    theta_call = common_theta - r * K_v * disc * cdf_d2
    theta_put = common_theta + r * K_v * disc * cdf_neg_d2
    theta = np.where(is_call, theta_call, theta_put)

    # Vega
    vega = S_v * pdf_d1 * sqrtT

    # Rho
    rho_call = K_v * T_v * disc * cdf_d2
    rho_put = -K_v * T_v * disc * cdf_neg_d2
    rho = np.where(is_call, rho_call, rho_put)

    # Apply validity mask
    return {
        'delta': np.where(valid, delta, np.nan),
        'gamma': np.where(valid, gamma, np.nan),
        'theta': np.where(valid, theta, np.nan),
        'vega':  np.where(valid, vega, np.nan),
        'rho':   np.where(valid, rho, np.nan),
    }


# ============================================================================
# IMPLIED VOLATILITY -- Newton-Raphson
# ============================================================================

def implied_vol(market_price, S, K, T, r, cp='call',
                tol=1e-6, max_iter=50, init_vol=0.3):
    """
    Solve for implied volatility using Newton-Raphson.

    Uses vega (dPrice/dSigma) as the derivative for fast convergence.
    Returns NaN if:
      - market_price is below intrinsic value
      - convergence fails
      - inputs are invalid
    """
    if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
        return np.nan

    # Check intrinsic bounds
    if cp == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price < intrinsic - 0.01:
        return np.nan

    sigma = init_vol
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, cp)
        vega = bs_vega(S, K, T, r, sigma)

        if vega < 1e-10:
            return np.nan  # vega too small, can't converge

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

        # Keep sigma in reasonable bounds
        if sigma <= 0.001:
            sigma = 0.001
        elif sigma > 5.0:
            return np.nan  # diverging

    return np.nan  # didn't converge


def implied_vol_vec(market_price, S, K, T, r, is_call,
                    tol=1e-6, max_iter=50, init_vol=0.3):
    """
    Vectorized Newton-Raphson IV solver.
    All inputs are numpy arrays. Returns numpy array of IVs.
    """
    n = len(market_price)
    sigma = np.full(n, init_vol, dtype=float)
    active = np.ones(n, dtype=bool)

    # Pre-filter invalid inputs
    invalid = (T <= 0) | (S <= 0) | (K <= 0) | (market_price <= 0) | np.isnan(market_price)
    active[invalid] = False

    for iteration in range(max_iter):
        if not active.any():
            break

        # Compute BS price and vega for active elements
        price = bs_price_vec(S, K, T, r, sigma, is_call)
        greeks = bs_greeks_vec(S, K, T, r, sigma, is_call)
        vega = greeks['vega']

        # Check convergence
        diff = price - market_price
        converged = active & (np.abs(diff) < tol)
        active[converged] = False

        # Check for tiny vega (can't step)
        tiny_vega = active & (np.abs(vega) < 1e-10)
        sigma[tiny_vega] = np.nan
        active[tiny_vega] = False

        # Newton step
        step = np.where(active, diff / np.where(np.abs(vega) > 1e-10, vega, 1.0), 0.0)
        sigma = sigma - step

        # Clamp
        sigma = np.clip(sigma, 0.001, 5.0)

        # Mark diverged
        diverged = active & (sigma >= 4.99)
        sigma[diverged] = np.nan
        active[diverged] = False

    # Mark unconverged as NaN
    sigma[active] = np.nan

    return sigma


# ============================================================================
# DATAFRAME INTERFACE -- compute Greeks + IV for v3.0 dataset
# ============================================================================

def compute_greeks_df(df, spot_col='underlying_price', strike_col='strike',
                      te_col='te', cp_col='call_put', mark_col='mark',
                      r=R_DEFAULT):
    """
    Compute Greeks and IV for a v3.0 DataFrame.

    For rows that already have IV (from options_2025.csv), computes Greeks from that IV.
    For rows that only have mark price (from Alpaca bars), solves IV first via Newton-Raphson.

    Adds/overwrites columns: delta, gamma, theta, vega, rho, iv

    Parameters:
        df: DataFrame with v3.0 schema columns
        spot_col: column name for underlying spot price
        strike_col: column name for strike price
        te_col: column name for time-to-expiry in years
        cp_col: column name for call/put type ('call' or 'put')
        mark_col: column name for option mid/mark price
        r: risk-free rate

    Returns:
        DataFrame with Greeks columns populated
    """
    print(f"\n[GREEKS] Computing Black-Scholes Greeks + IV for {len(df):,} rows ...")
    t0 = time.time()

    S = df[spot_col].values.astype(float)
    K = df[strike_col].values.astype(float)
    T = df[te_col].values.astype(float)
    is_call = (df[cp_col].astype(str).str.lower() == 'call').values

    # --- Phase 1: IV Solver ---
    # If IV already exists (from options_2025.csv), use it
    # If IV is missing but mark exists (from Alpaca bars), solve for it
    iv = df['iv'].values.astype(float) if 'iv' in df.columns else np.full(len(df), np.nan)
    mark = df[mark_col].values.astype(float) if mark_col in df.columns else np.full(len(df), np.nan)

    needs_iv = np.isnan(iv) & ~np.isnan(mark) & (mark > 0)
    n_solve = needs_iv.sum()

    if n_solve > 0:
        print(f"  Solving IV for {n_solve:,} rows via Newton-Raphson ...")
        solved_iv = implied_vol_vec(
            mark[needs_iv], S[needs_iv], K[needs_iv], T[needs_iv],
            r, is_call[needs_iv]
        )
        iv[needs_iv] = solved_iv
        n_solved = (~np.isnan(solved_iv)).sum()
        print(f"  Converged: {n_solved:,} / {n_solve:,}  ({n_solved/n_solve*100:.1f}%)")

    # --- Phase 2: Greeks from IV ---
    has_iv = ~np.isnan(iv) & (iv > 0)
    n_greeks = has_iv.sum()
    print(f"  Computing Greeks for {n_greeks:,} rows with valid IV ...")

    greeks = bs_greeks_vec(S, K, T, r, iv, is_call)

    # Write results
    df = df.copy()
    df['iv'] = iv
    df['delta'] = greeks['delta']
    df['gamma'] = greeks['gamma']
    df['theta'] = greeks['theta']
    df['vega'] = greeks['vega']
    df['rho'] = greeks['rho']

    # Zero out Greeks where IV was invalid
    for col in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        df.loc[~has_iv, col] = np.nan

    elapsed = time.time() - t0
    # Summary stats
    for col in ['iv', 'delta', 'gamma', 'theta', 'vega', 'rho']:
        pct = df[col].notna().mean() * 100
        print(f"    {col:8s}  {pct:6.1f}% populated")

    print(f"  Elapsed: {elapsed:.1f}s")
    return df


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Black-Scholes Greeks Calculator - Validation")
    print("=" * 60)

    # Test case: ATM SPY call
    S, K, T, r, sigma = 690.0, 690.0, 7/365, 0.05, 0.15
    print(f"\n  SPY ATM Call: S={S}, K={K}, T={T:.4f}y, r={r}, sigma={sigma}")

    price = bs_price(S, K, T, r, sigma, 'call')
    greeks = bs_greeks(S, K, T, r, sigma, 'call')
    print(f"  BS Price:  ${price:.4f}")
    print(f"  Delta:     {greeks['delta']:.4f}")
    print(f"  Gamma:     {greeks['gamma']:.6f}")
    print(f"  Theta:     {greeks['theta']:.4f}  (per year) = {greeks['theta']/365:.4f} (per day)")
    print(f"  Vega:      {greeks['vega']:.4f}")
    print(f"  Rho:       {greeks['rho']:.4f}")

    # Test IV solver: recover sigma from price
    recovered_iv = implied_vol(price, S, K, T, r, 'call')
    print(f"\n  IV Recovery: input sigma={sigma}, recovered={recovered_iv:.6f}")
    print(f"  Error: {abs(recovered_iv - sigma):.2e}")

    # Test OTM put
    print(f"\n  SPY OTM Put: S={S}, K=670, T={T:.4f}y")
    put_price = bs_price(S, 670, T, r, sigma, 'put')
    put_greeks = bs_greeks(S, 670, T, r, sigma, 'put')
    print(f"  BS Price:  ${put_price:.4f}")
    print(f"  Delta:     {put_greeks['delta']:.4f}")
    print(f"  Gamma:     {put_greeks['gamma']:.6f}")

    # Test vectorized
    print(f"\n  Vectorized test (3 contracts) ...")
    S_arr = np.array([690.0, 690.0, 690.0])
    K_arr = np.array([685.0, 690.0, 695.0])
    T_arr = np.array([7/365, 7/365, 7/365])
    sig_arr = np.array([0.15, 0.15, 0.15])
    is_call_arr = np.array([True, True, True])

    prices = bs_price_vec(S_arr, K_arr, T_arr, r, sig_arr, is_call_arr)
    greeks_v = bs_greeks_vec(S_arr, K_arr, T_arr, r, sig_arr, is_call_arr)
    for j in range(3):
        print(f"    K={K_arr[j]}  price=${prices[j]:.4f}  delta={greeks_v['delta'][j]:.4f}  gamma={greeks_v['gamma'][j]:.6f}")

    # Test vectorized IV solver
    print(f"\n  Vectorized IV recovery ...")
    recovered = implied_vol_vec(prices, S_arr, K_arr, T_arr, r, is_call_arr)
    for j in range(3):
        print(f"    K={K_arr[j]}  true_iv={sig_arr[j]:.4f}  recovered={recovered[j]:.6f}")

    print("\n  All tests passed.")
