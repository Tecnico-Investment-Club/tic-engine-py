-- Table for 1-hour candles
CREATE TABLE IF NOT EXISTS candles_1h (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);

-- Table for 1-day candles
CREATE TABLE IF NOT EXISTS candles_1d (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);

-- OPTIMIZATIONS
-- DESC because we want recent data.
CREATE INDEX IF NOT EXISTS idx_candles_1h_timestamp ON candles_1h(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candles_1d_timestamp ON candles_1d(timestamp DESC);