# TIC-Trading-Pod: Development Roadmap (v1.0 MVP)

This document outlines the chronological build order for the v1.0 MVP. 

**Core Architectural Principles:**
1. **Lego Brick Modularity:** Modules communicate strictly via Data Transfer Objects (DTOs), never raw dictionaries.
2. **Pure Function Strategies:** The strategy logic is a black box that only processes historical data and outputs target weights; it knows nothing about the broker, cash balance, or timezones.
3. **DB-Less State:** The broker is the single source of truth for portfolio balances and execution states.

---

## Phase 1: Contracts & Configurations (The C-Headers)
*Establish the strict rules and inputs before building the engine pipeline.*
- [ ] **1. [CORE] Datatypes & Strategy Contracts:** Build the `Candle`, `TargetWeight`, and `BrokerOrder` DTOs (using Python `@dataclass`). Define the `IStrategy` abstract base class.
- [ ] **2. [CORE] Configuration & Env Loader:** Wire up `config.yaml` and `.env` parsing. Implement fail-fast validation to crash on startup if required keys/parameters are missing.

## Phase 2: Ingestion & The Dummy Strategy
*Get safe, verified data flowing into a temporary strategy stub.*
- [ ] **3. [PIPELINE] Broker Ingestion (Data & State):** Query broker API for the last `Y` candle data points and current cash balance. Implementing "Target Timestamp Polling", dropping the currently open candle to guarantee the strategy only receives fully closed, historical candles and repeat the query if the desired timestamp isn't available yet.
- [ ] **4. [PIPELINE] Build Dummy Strategy [TEMPORARY]:** Implement `example_strat.py` inheriting from `IStrategy` that will have two hardcoded rules:
    - If the last candle closed under price `X`, target 100% weight in the asset.
    - If the last candle closed above price `Y`, target 0% weight (cash position).

## Phase 3: The Math Engine & Safety Guardrails
*Translate percentages to reality and protect the portfolio.*
- [ ] **5. [PIPELINE] The Transformer:** Convert strategy weights into absolute cash allocations based on the pod’s portfolio balance.
- [ ] **6. [PIPELINE] Risk Validator:** Validates proposed orders against risk rules. Clamp or reject allocations that violate maximum exposure per asset or total daily turnover limits.

## Phase 4: Execution & The Loop
*Close the circuit and let it run autonomously.*
- [ ] **7. [PIPELINE] Order Gateway to Testnet:** Fire the risk-approved orders to the broker's API, simulate execution, and log the exact responses.
- [ ] **8. [ENGINE] Main Async Loop & Scheduler:** Tie the pipeline together in `main.py` using Dependency Injection by initializing every module with their own dependencies. Wrap the execution in a `try/except/finally` block. Adding a 5-15 second offset to the hourly sleep timer to account for exchange latency before waking up.

## Phase 5: Containerization & E2E Testing
*Isolate and run.*
- [ ] **9. [INFRA] Dockerize & E2E Run:** Write the `Dockerfile`, pass in the `.env` Testnet keys, and watch the container execute one full, successful paper-trading cycle against the Testnet.

---

## Beyond v1.0 (Future Outlook)
* **v1.1:** Implement the ability to trade CRYPTO and STOCKS in the engine.
* **v2.0:** Add webhook alerting (Discord/Telegram) and 
* **v3.0:** Optimize for horizontal scaling by spinning up multiple strategy pods in their own isolated Docker containers.