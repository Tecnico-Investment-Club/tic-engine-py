Architecture: TIC-Trading-Pod

The TIC-Trading-Pod is a modular trading engine designed to replace the previous engine, with a clear focus on modularity, horizontal scalability, and ease of strategy implementation.

The architecture is built on three core pillars:

Hub-and-Spoke Data Model: To avoid API rate limits when scaling horizontally, a centralized ETL "Hub" fetches market data from the broker once and stores it in an internal database. The trading "Spokes" (pods) query this internal DB.

Containerized Isolation: A single Docker container runs exactly one strategy. Scaling is exclusively achieved horizontally by deploying more containers. This ensures that if one strategy encounters an error or crash, it doesn't affect the others.

Stateless Async Loop: The trading pod maintains no local state. It wakes up, executes a deterministic pipeline by reading the freshest data, and goes back to sleep. If the pod crashes, it can restart safely without data corruption.

1. System Pipeline Diagram (Hub and Spoke)

The system operates in a Monorepo containing two independent services that communicate via an internal database.

=================== THE HUB (ETL) ===================
[Alpaca API] ---> src/etl_db/ingestor.py ---> [Internal Postgres DB]
(Fetches data hourly for all tracked assets)

=================== THE SPOKE (POD) =================
┌─────────────────────────────────────────┐
│         src/engine_py/main.py           │
│    [Sleep 1h] ──--> [Awake & Execute]   │
└────────────────────┬────────────────────┘
                     │
 1. Fetch State      ▼     2. Fetch Data
┌─────────────────────────────────────────┐
│      src/engine_py/pipeline/ingestion.py│
│  Queries Alpaca for Live Cash Balance   │
│  Queries Internal DB for OHLCV Candles  │
└────────────────────┬────────────────────┘
                     │
 3. OHLCV Data       ▼     4. Weights (%)
┌─────────────────────────────────────────┐
│     src/engine_py/pipeline/strategy/    │
│  Blackbox calculation outputting target │
│  allocation weights per asset           │
└────────────────────┬────────────────────┘
                     │
 5. Weights + Cash   ▼     6. Allocations
┌─────────────────────────────────────────┐
│      src/engine_py/pipeline/transformer │
│  Converts percentage weights into       │
│  absolute cash/broker units             │
└────────────────────┬────────────────────┘
                     │
 7. Allocations      ▼     8. Valid Orders
┌─────────────────────────────────────────┐
│          src/engine_py/pipeline/risk.py │
│  Validates allocations against max      │
│  exposure and hard caps                 │
└────────────────────┬────────────────────┘
                     │
 9. Valid Orders     ▼     10. API Request
┌─────────────────────────────────────────┐
│      src/engine_py/pipeline/execution.py│
│  Submits live orders to Alpaca Testnet  │
└─────────────────────────────────────────┘


2. Core Modules

A. The Core Layer (src/core/)

The shared DNA. Contains strictly typed Pydantic models (datatypes.py) and abstract base classes (contracts.py) used by both the Hub and the Spokes to ensure data consistency.

B. The Hub Layer (src/etl_db/)

ingestor.py: A centralized worker that queries the broker REST API for a master list of assets and saves them to the internal database, preventing API rate limits.

janitor.py: A cleanup utility that prunes database records older than X days to keep the system footprint extremely lean.

C. The Spoke Layer (src/engine_py/)

The trading pod. It remains completely stateless internally.

ingestion.py: Queries the internal Postgres DB for historical candles and the broker for live cash balances.

strategy/: The IStrategy blackbox that outputs target weights.

transformer.py & risk.py: Converts weights to safe, lot-sized broker orders.

execution.py: Submits the final orders to the broker.

3. Design Decisions & Trade-offs

Internal DB vs Direct Broker Queries: While checking portfolio state directly from the broker is necessary, querying historical candles directly from the broker per-pod leads to immediate API rate limiting when running multiple strategies. The internal Postgres DB acts as a shared cache, solving this bottleneck.

REST API over WebSockets: To keep the MVP simple and reliable, we use REST APIs. Since the execution frequency is low (e.g., hourly), maintaining long-lived WebSocket connections introduces unnecessary complexity and failure points.

Single Strategy per Container: This design enforces strict isolation. If one strategy crashes, it doesn't affect others. Scaling is achieved by deploying more containers, not by adding internal complexity.