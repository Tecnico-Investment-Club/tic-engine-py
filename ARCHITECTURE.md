# Architecture: TIC-Trading-Pod

The **TIC-Trading-Pod** is a modular, stateless trading engine designed to replace the previous engine, with a clear focus on modularity and ease of strategy implementation. 

The architecture is built on three core pillars:
1. **DB-Less State Management:** The engine maintains no local persistence. The broker is the absolute source of truth for portfolio balances and open positions, what is totally fine for low frequency trading.
2. **Containerized Isolation:** A single Docker container runs exactly one strategy so scaling is exclusively achieved in a horizontal fashion by deploying more containers, not by adding internal complexity. This ensures that if one strategy encounters an error or crash, it doesn't affect the others.
3. **Stateless Async Loop:** The pod wakes up at a set interval (e.g., 1 hour), executes a deterministic pipeline, and goes back to sleep. If the pod crashes, it can restart safely without data corruption.

---

## 1. System Pipeline Diagram

The engine follows a strict, unidirectional data flow every execution cycle:

```
┌─────────────────────────────────────────┐
│              main.py (Loop)             │
│    [Sleep 1h] ──--> [Awake & Execute]   │
└────────────────────┬────────────────────┘
                     │
 1. Fetch State      ▼     2. Fetch Data
┌─────────────────────────────────────────┐
│        src/broker/ (External I/O)       │
│  Queries Broker for Cash Balance &      │
│  Last Y OHLCV Candles                   │
└────────────────────┬────────────────────┘
                     │
 3. OHLCV Data       ▼     4. Weights (%)
┌─────────────────────────────────────────┐
│       src/strategy/ (Trading Logic)     │
│  Blackbox calculation outputting target │
│  allocation weights per asset           │
└────────────────────┬────────────────────┘
                     │
 5. Weights + Cash   ▼     6. Allocations
┌─────────────────────────────────────────┐
│       src/engine/transformer.py         │
│  Converts percentage weights into       │
│  absolute cash/broker units             │
└────────────────────┬────────────────────┘
                     │
 7. Allocations      ▼     8. Valid Orders
┌─────────────────────────────────────────┐
│          src/engine/risk.py             │
│  Validates allocations against max      │
│  exposure and hard caps                 │
└────────────────────┬────────────────────┘
                     │
 9. Valid Orders     ▼     10. API Request
┌─────────────────────────────────────────┐
│         src/broker/execution.py         │
│  Simulates execution via Broker Testnet │
└─────────────────────────────────────────┘

```

---

## 2. Core Modules

### A. **The Broker Layer:** (`src/broker/`)

Handles all external network I/O.

* **`market_data.py`**: Queries the broker REST API and standardizes raw data into a unified OHLCV internal format.
* **`portfolio.py`**: Queries the broker for live cash availability. This fully replaces a local database.
* **`execution.py`**: The order gateway. Submits orders to the broker's paper trading environment.

### B. The Strategy Layer (`src/strategy/`)

* **`IStrategy (Interface)`**: The abstract interface. Strategies are decoupled from execution logic. They ingest unified candle data and strictly return an array of JSON objects that has the following format: 
`{"symbol": "...", "weight": 0.XX}`.

### C. The Engine Layer (`src/engine/`)

The middleman between abstract strategy and the broker.

* **`transformer.py`**: Calculates absolute cash allocations based on the strategy's target weights and the exact portfolio balance fetched from the broker. It handles rounding and broker precision rules.
* **`risk.py`**: The final safety net. Validates proposed orders against maximum exposure limits, daily hard caps and basic sanity checks before they reach the execution gateway.

So the data flow is:

```
Sleeping --> Broker Ingestion --> Strategy Logic --> Transformer --> Risk Validation --> Execution
```

---

## 3. Design Decisions & Trade-offs

* **REST API over WebSockets:** To keep the MVP simple and reliable, we use REST APIs. Since the execution frequency is low (e.g., hourly), maintaining long-lived WebSocket connections introduces unnecessary complexity and failure points.
* **No Database:** By relying on the broker's Testnet for state, we completely eliminate local state drift. If a network call fails, the pod simply skips the hour and tries again later, fetching the freshest, most accurate portfolio balance.
* **Single Strategy per Container:** This design enforces strict isolation. If one strategy crashes, it doesn't affect others. Scaling is achieved by deploying more containers, not by adding internal complexity.
* **Synchronous Execution Flow:** The entire pipeline runs synchronously within the awake cycle. This ensures deterministic behavior and simplifies error handling. If any step fails, the pod can log the error and safely return to sleep without risking inconsistent state.