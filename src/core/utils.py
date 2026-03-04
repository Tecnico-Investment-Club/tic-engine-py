# src/core/utils.py
from datetime import timedelta
import re

def normalize_symbol(symbol: str) -> str:
    """
    Standardizes ticker symbols across exchanges to internal format.
    Example: BTCUSDT -> BTCUSD
    """
    # List of stablecoins to strip
    stablecoins = ["USDT", "BUSD", "TUSD", "USDC"]
    
    for stable in stablecoins:
        if symbol.endswith(stable):
            return symbol[:-len(stable)] + "USD"
    
    return symbol


def timeframe_to_table(timeframe: str) -> str:
    """
    Maps a logical timeframe identifier to the physical candles table name.
    Shared between the trading engine and the ETL pipeline.
    """
    if timeframe == "1h":
        return "candles_1h"
    if timeframe == "1d":
        return "candles_1d"
    raise ValueError(f"Unsupported timeframe '{timeframe}'. Expected '1h' or '1d'.")

def parse_time_interval(interval_str: str) -> timedelta:
    """Converts a string like '1d', '1w', '4h' into a timedelta."""
    if not interval_str:
        return timedelta(seconds=0)
        
    match = re.match(r"(\d+)([dhw])", interval_str.lower())
    if not match:
        raise ValueError(f"Invalid interval format: {interval_str}. Use '1h', '1d', '1w'.")
        
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'h': return timedelta(hours=value)
    if unit == 'd': return timedelta(days=value)
    if unit == 'w': return timedelta(weeks=value)