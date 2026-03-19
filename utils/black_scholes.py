"""
Black-Scholes Option Pricing and Greeks Calculation

This module provides accurate option pricing and Greeks calculation using the Black-Scholes model.
All calculations use real market data (underlying price, strikes, time to expiry).
NO simulated or random data is used.
"""

import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price and Greeks using Black-Scholes model.
    
    Parameters:
    -----------
    S : float
        Current underlying price (from real market data)
    K : float
        Strike price
    T : float
        Time to expiry in years (calculated from real datetime)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Implied volatility (annualized)
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    dict : Dictionary containing price and all Greeks
        {
            'price': option price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega (per 1% IV change),
            'theta': theta (per day),
            'vanna': vanna (second-order),
            'volga': volga/vomma (second-order)
        }
    """
    # Handle edge case: zero / near-zero volatility
    if sigma < 1e-8:
        # With no volatility, option is worth its discounted intrinsic value
        intrinsic = S - K * np.exp(-r * T) if option_type == 'call' else K * np.exp(-r * T) - S
        price = max(0.0, intrinsic)
        delta = 1.0 if (option_type == 'call' and S > K * np.exp(-r * T)) else 0.0

        return {
            'price': price,
            'delta': delta,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'vanna': 0.0,
            'volga': 0.0
        }

    # Handle edge case: very small time to expiry
    if T < 1e-6:
        # At expiry, option worth intrinsic value only
        if option_type == 'call':
            price = max(0, S - K)
        else:
            price = max(0, K - S)
        
        return {
            'price': price,
            'delta': 1.0 if (option_type == 'call' and S > K) else 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'vanna': 0.0,
            'volga': 0.0
        }
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    
    # First-order Greeks
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% IV change
    
    # Theta (per day)
    if option_type == 'call':
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                  r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
    else:
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                  r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
    
    # Second-order Greeks (volatility sensitivity)
    vanna = vega * (1 - d1 / (sigma * np.sqrt(T)))  # dDelta/dVol
    volga = vega * d1 * d2 / sigma  # dVega/dVol
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'vanna': vanna,
        'volga': volga
    }


def calculate_position_greeks(ce_greeks, pe_greeks, ce_lots, pe_lots, lot_size):
    """
    Calculate net Greeks for the entire position (CE + PE).
    
    Parameters:
    -----------
    ce_greeks : dict
        Greeks dictionary for CE option
    pe_greeks : dict
        Greeks dictionary for PE option
    ce_lots : int
        Number of CE lots
    pe_lots : int
        Number of PE lots
    lot_size : int
        Lot size (e.g., 75 for NIFTY)
    
    Returns:
    --------
    dict : Net position Greeks
    """
    return {
        'net_delta': (ce_greeks['delta'] * ce_lots - abs(pe_greeks['delta']) * pe_lots) * lot_size,
        'gamma_exposure': (ce_greeks['gamma'] + pe_greeks['gamma']) * (ce_lots + pe_lots) * lot_size,
        'vega_exposure': (ce_greeks['vega'] + pe_greeks['vega']) * (ce_lots + pe_lots) * lot_size,
        'theta_exposure': (ce_greeks['theta'] + pe_greeks['theta']) * (ce_lots + pe_lots) * lot_size,
        'vanna_exposure': (ce_greeks['vanna'] + pe_greeks['vanna']) * (ce_lots + pe_lots) * lot_size,
        'volga_exposure': (ce_greeks['volga'] + pe_greeks['volga']) * (ce_lots + pe_lots) * lot_size,
    }
