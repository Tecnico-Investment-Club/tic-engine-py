TIC-Trading-Pod

A modular, db-less trading engine designed for Técnico Investment Club trading strategies with the intent of replacing the previous one. Each Docker container encapsulates a single strategy with its own isolated portfolio, executing on a configurable asynchronous schedule (e.g., hourly, daily, quarterly).

Core Philosophy

Paper Trading Engine: Unless otherwise stated, the engine is strictly for paper trading.

DB-Less State (For the Pod): Portfolio state is queried directly from the broker in each execution cycle. Market data is queried from a centralized internal database to prevent rate limits.

One Strategy per Container: Each Docker container runs exactly one strategy, ensuring strict isolation and easy horizontal scaling.

Modular Pipeline: The dataflow pipeline is unidirectional and modular, meaning that plugging a new module should only require changes to the adjacent modules without affecting the rest of the system.

Prerequisites

Docker & Docker Compose

Python 3.10+ (for local development/testing)

A broker Testnet API key (e.g., Alpaca Paper Trading)

Quick Start

Clone the repository:

git clone git@github.com:Tecnico-Investment-Club/tic-engine-py.git
cd tic-engine-py


Set up environment variables:
Copy the template and set up the .env file with your API keys and internal DB passwords.
```
cp .env.example .env
```

Configure the services:

Edit src/etl_db/config.yaml to set your master list of tracked assets.

Edit src/engine_py/config.yaml to set your strategy sleep interval, starting balance, risk parameters, and strategy logic.

Build and Run:
Run the Docker Compose stack to spin up the Database, the ETL Hub, and your Trading Pod.

Project Structure (Monorepo)

src/core/: Shared data contracts (datatypes.py), interfaces (contracts.py), and standardized logging used by all services.

src/etl_db/: The Data Hub. Centralized service to fetch market data from the broker and store it in a lightweight internal Postgres DB.

src/engine_py/: The Strategy Pod (Spoke). The actual trading engine that reads historical data from the internal DB, executes the strategy math, and fires orders to the broker.

Local Development Setup

While the TIC-Trading-Pod is ultimately designed to run isolated within Docker containers, you will need a local Python environment for development, writing strategies, and running tests.

We strictly use Python virtual environments (venv) to manage dependencies.

1. Create and Activate the Virtual Environment

Install and activate a virtual environment in the project root:

On WSL/Linux/Mac:
```
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Requirements

Once activated (you should see (.venv) at the beginning of your terminal prompt), install the project dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```