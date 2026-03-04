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