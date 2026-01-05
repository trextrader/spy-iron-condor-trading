from enum import Enum
import pandas as pd
import numpy as np

class MarketRegime(Enum):
    LOW_VOL_RANGE = "Low Volatility Range"      # Ideal for standard IC
    HIGH_VOL_RANGE = "High Volatility Range"    # Good for wide IC
    BULL_TREND = "Bull Trend"                   # Directional risk
    BEAR_TREND = "Bear Trend"                   # Directional risk
    CRASH_MODE = "Crash Mode"                   # VIX Explosion - unsafe

def classify_regime(df: pd.DataFrame, current_idx: int, vix_level: float) -> MarketRegime:
    """
    Classify market regime based on technicals and VIX.
    
    Params:
        df: DataFrame with OHLCV and indicators (ADX, SMA)
        current_idx: Index to classify (usually -1 or 0 depending on backtrader)
        vix_level: Current VIX value
    """
    # 1. Volatility Filter (Crash Mode)
    if vix_level > 35.0:
        return MarketRegime.CRASH_MODE
        
    # Safeguard for logic if indicators aren't ready
    if len(df) <= current_idx:
        return MarketRegime.LOW_VOL_RANGE # Default
        
    # 2. Trend Filter (ADX + SMA)
    # Using 'close' vs 'sma_200' and 'adx'
    # Fallback values if columns missing
    close = df['close'].iloc[current_idx]
    sma = df['sma_200'].iloc[current_idx] if 'sma_200' in df.columns else close
    adx = df['adx'].iloc[current_idx] if 'adx' in df.columns else 0.0
    
    is_trending = adx > 25.0
    is_bullish = close > sma
    
    # 3. Regime Determination
    if is_trending:
        return MarketRegime.BULL_TREND if is_bullish else MarketRegime.BEAR_TREND
    else:
        # Ranging
        if vix_level > 20.0:
            return MarketRegime.HIGH_VOL_RANGE
        else:
            return MarketRegime.LOW_VOL_RANGE

def check_liquidity_gate(data, spread_estimate=1.0):
    """
    Checks if the recent price action suggests enough 
    liquidity to fill an Iron Condor.
    """
    try:
        # Check volume of the last 5 bars
        recent_volume = data.volume.get(size=5)
        avg_vol = sum(recent_volume) / len(recent_volume)
        
        # Rule: Minimum volume threshold (adjust based on symbol)
        if avg_vol < 100:
            return False
            
        # Check volatility (True Range)
        # If the range is too massive, spreads usually widen
        highs = data.high.get(size=5)
        lows = data.low.get(size=5)
        avg_range = sum([h - l for h, l in zip(highs, lows)]) / 5
        
        if avg_range > (data.close[0] * 0.02): # 2% swing in 5 mins is too wild
            return False

        return True
    except Exception:
        # If we don't have enough bars yet, skip trading
        return False