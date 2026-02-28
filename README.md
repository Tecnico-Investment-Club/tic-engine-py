# TIC-Trading-Pod

A modular, db-less trading engine designed for Técnico Investment Club trading strategies with the intent of replacing the previous one. Each Docker container encapsulates a single strategy with its own isolated portfolio, executing on a configurable asynchronous schedule (e.g., hourly, daily, quarterly).

## Core Philosophy

* **Paper Trading Engine:** Unless otherwise stated, the engine is strictly for paper trading.
* **DB-Less State:** Portfolio state is queried directly from the broker in each execution cycle.
* **One Strategy per Container:** Each Docker container runs exactly one strategy, ensuring strict isolation and easy horizontal scaling.
* **Modular Pipeline:** The dataflow pipeline is unidirectional and modular, meaning that plugging a new module should only require changes to the adjacent modules without affecting the rest of the system.

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
Copy the template and set up the .env file.
```
cp .env.example .env
```

3. **Configure the pod:**
   Edit `config.yaml` to set your strategy sleep interval, starting portfolio balance, risk parameters, and specific strategy variables under the `strategy_params` block.

4. **Build and Run:**
   Run the Docker container with your strategy.



## Project Structure
* `src/broker/`: External I/O for fetching market data, checking live portfolio balances, and submitting simulated orders.
* `src/strategy/`: The black-box trading logic that outputs target allocation weights.
* `src/engine/`: Core logic for transforming weights into cash allocations and validating against risk rules.
* `src/utils/`: Standardized logging and the asynchronous sleep/wake scheduling loop.

## Local Development Setup

While the TIC-Trading-Pod is ultimately designed to run isolated within a Docker container, you will need a local Python environment for development, writing strategies, and running tests. 

For now Python virtual environments (`venv`) are the choice to manage dependencies.

### 1. Create and Activate the Virtual Environment
Install and activate a virtual environment in the project root:
```
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Requirements

Once activated (you should see (.venv) at the beginning of your terminal prompt), to install the project dependencies run these commands:
```
pip install --upgrade pip
pip install -r requirements.txt
```