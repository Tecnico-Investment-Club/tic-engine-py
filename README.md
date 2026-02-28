# TIC-Trading-Pod

A modular, db-less trading engine designed for Técnico Investment Club trading strategies with the intent of replacing the previous one. Each Docker container encapsulates a single strategy with its own isolated portfolio, executing on a configurable asynchronous schedule (e.g., hourly, daily, trimestraly).

## Core Philosophy

* **Paper Trading Engine:** Until said otherwise the engine is strictly for paper trading.
* **DB-Less State:** Portfolio state is queried directly from the broker in each execution cycle.
* **One Strategy per Container:** Each Docker container runs exactly one strategy, ensuring strict isolation and easy horizontal scaling.
* **Modular Pipeline:** The dataflow pipeline is unidirectional and modular, that means that plugging a new module (e.g, MODULE X) should only require changes to the previous and next module (e.g., MODULE W and MODULE Y) without affecting the rest of the system.

## Prerequisites

* Docker
* Python 3.10+ (for local development/testing)
* A broker Testnet API key (e.g., Binance Testnet)

## Quick Start

1. **Clone the repository:**
   ```
   git clone git@github.com:Tecnico-Investment-Club/tic-engine-py.git
   cd tic-engine-py
   ```

2. **Set up environment variables:**
Copy the template and add your Testnet API credentials.
```
cp .env.example .env
```


3. **Configure the pod:**
Edit `config.yaml` to set your strategy sleep interval, starting portfolio balance, and risk parameters.

4. **Configure the internal strategy parameters:**
Go to `src/strategy/internal_config.yaml` and set the parameters for your strategy (e.g., lookback window, target assets, etc).

5. **Build and Run:**
Run the Docker container with your strategy.



## Project Structure
* `src/broker/`: External I/O for fetching market data, checking live portfolio balances, and submitting simulated orders.
* `src/strategy/`: The black-box trading logic that outputs target allocation weights.
* `src/engine/`: Core logic for transforming weights into cash allocations and validating against risk rules.
* `src/utils/`: Standardized logging and the asynchronous sleep/wake scheduling loop.