- [ ] Dar fix do scaling horizontal das estratégias não estar a funcionar corretamente
- [ ] Fix this Bullshit 
     ````
     2026-03-08 15:33:01,560 - CORE.MESSAGING - INFO - Listening for updates on channel 'market_data_ready'...
    2026-03-08 15:33:14,114 - CORE.MESSAGING - INFO - Received Pub/Sub trigger on 'market_data_ready'!
    2026-03-08 15:33:14,115 - TRADING.PIPELINE - INFO - Pub/Sub Event Received: New candles available
    2026-03-08 15:33:14,115 - TRADING.PIPELINE - INFO - Cooldown passed. Starting Trading Cycle...
    2026-03-08 15:33:14,123 - TRADING.INGESTION - INFO - Successfully connected to the internal database.
    2026-03-08 15:33:14,135 - TRADING.INGESTION - INFO - Loaded 50 candles for 3 symbols from candles_1h.
    2026-03-08 15:33:14,711 - TRADING.INGESTION - INFO - Portfolio Synced: $97961.79 Total Equity | Buying Power: $4952.87
    2026-03-08 15:33:14,711 - TRADING.STRATEGY - INFO - PingPong: 3 assets under buy threshold. Weight per asset: 0.3333
    2026-03-08 15:33:14,711 - TRADING.TRANSFORMER - INFO - Transformer: Converted 3 weights into 3 order requests.
    2026-03-08 15:33:14,711 - TRADING.EXECUTION - INFO - Submitting MARKET SELL order for 0.2504 AAPL [TIF: DAY]...
    2026-03-08 15:33:15,121 - TRADING.EXECUTION - INFO - Order c09634d1-7906-4ce7-9e4f-283c7f145cad submitted successfully. Status: ACCEPTED
    2026-03-08 15:33:15,122 - TRADING.EXECUTION - INFO - Submitting MARKET SELL order for 0.0181 SPY [TIF: DAY]...
    2026-03-08 15:33:15,261 - TRADING.EXECUTION - INFO - Order 1bd6efe2-7d35-46c3-8535-7b840ff8572d submitted successfully. Status: ACCEPTED
    2026-03-08 15:33:15,261 - TRADING.EXECUTION - INFO - Submitting MARKET BUY order for 0.3661 TSLA [TIF: DAY]...
    2026-03-08 15:33:15,397 - TRADING.EXECUTION - INFO - Order f173d92a-4833-4791-ae9f-aac0655b4667 submitted successfully. Status: ACCEPTED
    2026-03-08 15:33:15,397 - TRADING.PIPELINE - INFO -  -> SELL 0.25037146 AAPL | Status: ACCEPTED
    2026-03-08 15:33:15,397 - TRADING.PIPELINE - INFO -  -> SELL 0.018094585 SPY | Status: ACCEPTED
    2026-03-08 15:33:15,397 - TRADING.PIPELINE - INFO -  -> BUY 0.366096654 TSLA | Status: ACCEPTED
    2026-03-08 16:01:04,591 - CORE.MESSAGING - INFO - Received Pub/Sub trigger on 'market_data_ready'!
    2026-03-08 16:01:04,591 - TRADING.PIPELINE - INFO - Pub/Sub Event Received: New candles available
    2026-03-08 16:01:04,591 - TRADING.PIPELINE - INFO - Skipping execution. Cooldown active. (Needs 1:00:00, elapsed 0:27:49.193948)
    ```` 