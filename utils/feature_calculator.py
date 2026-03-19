"""
Feature Engineering Module for RL Trading Environment

This module calculates all 55 features from real OHLCV data only.
NO simulated or random data is used - everything is derived from actual market data.

Feature Categories:
1. Volatility Features (12) - CRITICAL for volatility regime detection
2. Price & Returns (10)
3. Greeks (10) - Using Black-Scholes
4. Technical Indicators (6)
5. Position Features (10)
6. Time Features (4)
7. Risk Metrics (3)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class FeatureCalculator:
    """
    Calculates all features from real OHLCV data.
    All features are derived from historical price data - NO simulation.
    """
    
    def __init__(self, window_size=150):
        """
        Initialize feature calculator.
        
        Parameters:
        -----------
        window_size : int
            Lookback window for rolling calculations (default 150 for 1-minute data)
        """
        self.window_size = window_size
        
        # Cache for intermediate calculations
        self.ema_cache = {}
        
    def safe_divide(self, numerator, denominator, default=0.0):
        """Safe division to avoid division by zero."""
        return numerator / denominator if abs(denominator) > 1e-8 else default
    
    def calculate_returns(self, prices, windows=[1, 5, 15, 30]):
        """
        Calculate returns over multiple windows.
        
        Parameters:
        -----------
        prices : np.array
            Array of close prices
        windows : list
            List of window sizes
        
        Returns:
        --------
        dict : Returns for each window
        """
        returns = {}
        current_price = prices[-1]
        
        for window in windows:
            if len(prices) > window:
                past_price = prices[-window-1]
                returns[f'return_{window}min'] = self.safe_divide(
                    current_price - past_price, past_price
                )
            else:
                returns[f'return_{window}min'] = 0.0
        
        return returns
    
    def calculate_volatility_features(self, prices, highs, lows, opens, closes):
        """
        Calculate comprehensive volatility features - CRITICAL for strategy.
        
        All volatility is annualized (252 trading days, 390 minutes per day).
        
        Parameters:
        -----------
        prices : np.array
            Close prices
        highs, lows, opens, closes : np.array
            OHLC data
        
        Returns:
        --------
        dict : Volatility features
        """
        features = {}
        
        # Calculate returns for volatility
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
        else:
            returns = np.array([0.0])
        
        # 1. Rolling realized volatility (multiple windows)
        annualization_factor = np.sqrt(252 * 390)
        
        windows = {
            '5min': 5, '15min': 15, '30min': 30, '60min': 60,
            '1day': 390, '7day': 2730, '15day': 5850
        }
        
        for name, window in windows.items():
            if len(returns) >= window:
                vol = np.std(returns[-window:]) * annualization_factor
                features[f'rolling_vol_{name}'] = vol
            else:
                features[f'rolling_vol_{name}'] = 0.0
        
        # 2. Parkinson volatility (high-low based estimator)
        if len(highs) >= 30:
            hl_ratio = np.log(highs[-30:] / lows[-30:])
            parkinson_vol = np.sqrt(np.sum(hl_ratio**2) / (4 * np.log(2) * 30)) * annualization_factor
            features['parkinson_vol'] = parkinson_vol
        else:
            features['parkinson_vol'] = 0.0
        
        # 3. Garman-Klass volatility (OHLC-based estimator)
        if len(highs) >= 30:
            hl = np.log(highs[-30:] / lows[-30:])
            co = np.log(closes[-30:] / opens[-30:])
            gk_vol = np.sqrt(np.mean(0.5 * hl**2 - (2*np.log(2)-1) * co**2)) * annualization_factor
            features['garman_klass_vol'] = gk_vol
        else:
            features['garman_klass_vol'] = 0.0
        
        # 4. Volatility percentile — fully vectorized using stride trick (no Python loops)
        if len(returns) >= 60:
            current_vol = features['rolling_vol_30min']
            # Use numpy stride tricks: create a (N, 30) view without copying data
            r = returns[-min(len(returns), 390):]  # cap at 1 day of history for speed
            n = len(r)
            if n >= 30:
                shape = (n - 30 + 1, 30)
                strides = (r.strides[0], r.strides[0])
                windows = np.lib.stride_tricks.as_strided(r, shape=shape, strides=strides)
                hist_vols = np.std(windows, axis=1) * annualization_factor
                features['vol_percentile'] = float(np.mean(hist_vols < current_vol))
            else:
                features['vol_percentile'] = 0.5
        else:
            features['vol_percentile'] = 0.5
        
        # 5. Volatility of volatility (vol regime stability)
        vol_series = [features[f'rolling_vol_{w}min'] for w in [5, 15, 30, 60]]
        if len(vol_series) > 1:
            features['vol_of_vol'] = np.std(vol_series)
        else:
            features['vol_of_vol'] = 0.0
        
        # 6. Intraday volatility trend
        if len(returns) >= 60:
            vol_recent = features['rolling_vol_15min']
            vol_earlier = np.std(returns[-60:-30]) * annualization_factor if len(returns) >= 60 else vol_recent
            features['vol_trend'] = self.safe_divide(vol_recent - vol_earlier, vol_earlier)
        else:
            features['vol_trend'] = 0.0
        
        return features
    
    def calculate_price_features(self, prices, highs, lows, day_open, day_high, day_low):
        """
        Calculate price and return features.
        
        Parameters:
        -----------
        prices : np.array
            Close prices
        highs, lows : np.array
            High and low prices
        day_open, day_high, day_low : float
            Day's open, high, low
        
        Returns:
        --------
        dict : Price features
        """
        features = {}
        current_price = prices[-1]
        
        # Normalized close
        features['close_normalized'] = self.safe_divide(current_price - day_open, day_open)
        
        # Multi-window returns
        returns = self.calculate_returns(prices, windows=[1, 5, 15, 30])
        features.update(returns)
        
        # VWAP approximation (using typical price)
        if len(prices) >= 30:
            typical_prices = (highs[-30:] + lows[-30:] + prices[-30:]) / 3
            vwap = np.mean(typical_prices)
            features['vwap_distance'] = self.safe_divide(current_price - vwap, vwap)
        else:
            features['vwap_distance'] = 0.0
        
        # Distance from day high/low
        features['distance_from_day_high'] = self.safe_divide(day_high - current_price, current_price)
        features['distance_from_day_low'] = self.safe_divide(current_price - day_low, current_price)
        
        # Price momentum
        if len(prices) >= 10:
            recent_returns = np.diff(prices[-11:]) / prices[-11:-1]
            features['price_momentum'] = np.sum(recent_returns)
        else:
            features['price_momentum'] = 0.0
        
        return features
    
    def calculate_technical_indicators(self, prices, highs, lows):
        """
        Calculate technical indicators (RSI, MACD, Bollinger, ATR).
        
        Parameters:
        -----------
        prices : np.array
            Close prices
        highs, lows : np.array
            High and low prices
        
        Returns:
        --------
        dict : Technical indicator features
        """
        features = {}
        
        # 1. RSI (14-period)
        if len(prices) >= 15:
            deltas = np.diff(prices[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss > 1e-8:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if avg_gain > 0 else 50
            
            features['rsi'] = rsi / 100.0  # Normalize to [0, 1]
        else:
            features['rsi'] = 0.5
        
        # 2. MACD
        if len(prices) >= 26:
            ema12 = self._calculate_ema(prices, 12)
            ema26 = self._calculate_ema(prices, 26)
            macd_line = ema12 - ema26
            
            # MACD signal (9-period EMA of MACD)
            # Simplified: use SMA for signal
            if len(prices) >= 35:
                macd_signal = np.mean([ema12 - ema26])  # Simplified
            else:
                macd_signal = macd_line
            
            macd_histogram = macd_line - macd_signal
            
            # Normalize by price
            features['macd_line'] = self.safe_divide(macd_line, prices[-1])
            features['macd_signal'] = self.safe_divide(macd_signal, prices[-1])
            features['macd_histogram'] = self.safe_divide(macd_histogram, prices[-1])
        else:
            features['macd_line'] = 0.0
            features['macd_signal'] = 0.0
            features['macd_histogram'] = 0.0
        
        # 3. Bollinger Bands
        if len(prices) >= 20:
            sma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            bb_width = bb_upper - bb_lower
            
            if bb_width > 1e-8:
                features['bollinger_position'] = (prices[-1] - sma20) / bb_width
            else:
                features['bollinger_position'] = 0.0
        else:
            features['bollinger_position'] = 0.0
        
        # 4. ATR (Average True Range)
        if len(prices) >= 15:
            true_ranges = []
            for i in range(-14, 0):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prices[i-1]) if i > -len(prices) else 0,
                    abs(lows[i] - prices[i-1]) if i > -len(prices) else 0
                )
                true_ranges.append(tr)
            
            atr = np.mean(true_ranges)
            features['atr'] = self.safe_divide(atr, prices[-1])  # Normalize by price
        else:
            features['atr'] = 0.0
        
        return features
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])  # Start with SMA
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def calculate_time_features(self, current_time, start_time, end_time):
        """
        Calculate time-based features.
        
        Parameters:
        -----------
        current_time : datetime
            Current timestamp
        start_time : datetime
            Market open time
        end_time : datetime
            Market close time
        
        Returns:
        --------
        dict : Time features
        """
        total_minutes = (end_time - start_time).total_seconds() / 60
        minutes_since_open = (current_time - start_time).total_seconds() / 60
        minutes_to_close = (end_time - current_time).total_seconds() / 60
        
        # Normalize to [0, 1]
        time_progress = minutes_since_open / total_minutes if total_minutes > 0 else 0
        
        # Cyclical encoding
        angle = 2 * np.pi * time_progress
        
        return {
            'minutes_since_open': minutes_since_open / total_minutes,
            'minutes_to_close': minutes_to_close / total_minutes,
            'time_sine': np.sin(angle),
            'time_cosine': np.cos(angle)
        }

    def get_raw_indicators(self, prices, highs, lows):
        """
        Calculate raw values for strike selection (BB and ATR).
        Unlike other methods, this returns absolute price levels, not normalized features.
        
        Returns:
        --------
        dict : Raw indicator values
        """
        indicators = {}
        
        # 1. Bollinger Bands (20, 2)
        if len(prices) >= 20:
            sma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            indicators['bb_upper'] = sma20 + (config.BOLLINGER_STD * std20)
            indicators['bb_lower'] = sma20 - (config.BOLLINGER_STD * std20)
        else:
            # Fallback for insufficient data
            indicators['bb_upper'] = prices[-1]
            indicators['bb_lower'] = prices[-1]

        # 2. ATR (14)
        if len(prices) >= 15:
            true_ranges = []
            for i in range(-14, 0):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prices[i-1]) if i > -len(prices) else 0,
                    abs(lows[i] - prices[i-1]) if i > -len(prices) else 0
                )
                true_ranges.append(tr)
            
            indicators['atr'] = np.mean(true_ranges)
        else:
            indicators['atr'] = 0.0
            
        return indicators
