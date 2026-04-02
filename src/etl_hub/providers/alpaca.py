import logging
import requests
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Literal

from core.datatypes import Candle
from etl_hub.interfaces.IProvider import IProvider

logger = logging.getLogger("ETL.ALPACA")

class AlpacaProvider(IProvider):
    """
    Alpaca Data Source (Raw REST Implementation).
    Fetches public candles in batches, bypassing the slow SDK.
    """
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://data.alpaca.markets/v2/stocks/bars"
        self.eastern = ZoneInfo("America/New_York")
    
    def fetch_candles(self, symbols: List[str], timeframe: Literal["1h", "1d"], limit: int = 500) -> Dict[str, List[Candle]]:
        tf = "1Hour" if timeframe == "1h" else "1Day"
        buffer_minutes = 16 
        delayed_now = datetime.now(timezone.utc) - timedelta(minutes=buffer_minutes)
        
        if timeframe == "1d":
            days_back = int(limit * 1.5) + 10  
        else:
            days_back = int((limit / 6.5) * 1.5) + 10 
            
        start_date = delayed_now - timedelta(days=days_back)

        # Fix the ticker format for Alpaca
        clean_symbols = [sym.replace('-', '.') for sym in symbols]

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "accept": "application/json"
        }

        params = {
            "symbols": ",".join(clean_symbols),
            "timeframe": tf,
            "start": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "end": delayed_now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "limit": 10000, # Max allowed per Alpaca page
            "feed": "iex"   # Note: Change to "sip" if you have a paid data plan
        }

        results_dict = {sym: [] for sym in symbols}
        
        rth_start = datetime.strptime("09:30", "%H:%M").time()
        rth_end = datetime.strptime("16:00", "%H:%M").time()

        page_token = None
        page_count = 1

        try:
            while True:
                if page_token:
                    params["page_token"] = page_token

                logger.info(f"Fetching Alpaca API page {page_count} for {len(clean_symbols)} symbols...")

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.get(self.base_url, headers=headers, params=params, timeout=10)
                        break  # If successful, break out of the retry loop
                    except requests.exceptions.ConnectionError as e:
                        if attempt == max_retries - 1:
                            raise e  # If we failed 3 times, actually throw the error
                        logger.warning(f"Connection dropped. Retrying ({attempt + 1}/{max_retries})...")
                        time.sleep(2)
                
                # Handle Rate Limits explicitly
                if response.status_code == 429:
                    logger.warning("Alpaca rate limit hit. Sleeping for 10 seconds...")
                    time.sleep(10)
                    continue

                response.raise_for_status()
                data = response.json()

                bars_dict = data.get("bars", {})
                
                if not bars_dict and page_count == 1:
                    logger.warning("No stock data returned from Alpaca.")
                    break

                for clean_symbol, bar_list in bars_dict.items():
                    original_symbol = clean_symbol.replace('.', '-') if clean_symbol.replace('.', '-') in symbols else clean_symbol

                    for bar in bar_list:
                        # Raw Alpaca JSON looks like: {'t': '2026-04-01T09:30:00Z', 'o': 100, 'h': 105, 'l': 99, 'c': 104, 'v': 10000}
                        dt = datetime.fromisoformat(bar['t'].replace('Z', '+00:00'))
                        
                        if timeframe == "1h":
                            et_time = dt.astimezone(self.eastern).time()
                            if not (rth_start <= et_time < rth_end):
                                continue 
                        
                        results_dict[original_symbol].append(Candle(
                            symbol=original_symbol,
                            timestamp=dt,
                            open=float(bar['o']),
                            high=float(bar['h']),
                            low=float(bar['l']),
                            close=float(bar['c']),
                            volume=float(bar['v'])
                        ))

                page_token = data.get("next_page_token")
                if not page_token:
                    break
                
                page_count += 1

            # Slice the final arrays to guarantee we only return the requested limit
            for sym in results_dict:
                results_dict[sym] = results_dict[sym][-limit:]

            return results_dict
            
        except Exception as e:
            logger.error(f"Alpaca API Raw HTTP Error: {e}")
            return results_dict

    def get_provider_name(self) -> str:
        return "Alpaca"