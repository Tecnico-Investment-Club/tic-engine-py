# Implementing a New Trading Strategy: A Step-by-Step Guide

This guide outlines the standardized process for seamlessly integrating a new trading strategy into the trading engine. Following these steps ensures your strategy is automatically recognized by the Docker orchestration and operates in isolation from other trading pods.

## 1. Logic Implementation

All strategy code resides within the `src/trading_pod/strategy/` directory.

*   **Create a Directory:**
    *   Create a new folder under `strategy/` named after your strategy (e.g., `your_strat_name/`).

*   **Implement the Interface:**
    *   Your strategy class **must** implement the `IStrategy` interface.

*   **Internal Structure:**
    *   Maintain a clean and organized directory structure for your strategy files:

    ```
    strategy/
    └── momentum_alpha/
        ├── __init__.py
        ├── momentum_alpha.py      # Define your strategy class here
        └── your_file/folder_here  # Additional supporting files and modules
    ```

## 2. Register in Factory

The `StrategyFactory` acts as the crucial link between your configuration and the executable code.

*   **Open `factory.py`:**
    *   Locate and open the `src/trading_pod/strategy/factory.py` file.

*   **Import Your Strategy:**
    *   Import your new strategy class using its **absolute** path within the project.

*   **Update the Switch:**
    *   Add a new `elif` block within the `get_strategy` method:

    ```python
    if strategy_name == "XYZ":
        return XYZ(**params)
    ```

## 3. Configuration & Secrets

Each strategy pod necessitates its own unique configuration and API credentials.

*   **Create Configuration File:**
    *   Add a new YAML file to `src/trading_pod/configs/XYZ.yaml`.
    *   **Crucially:** Define the `strategy_name` in this file. It **must** match the name used in the `factory.py` switch (e.g., `"XYZ"`).
    *   Define the specific assets to trade and any strategy-specific parameters (e.g., SMA windows, thresholds).

*   **Create Secrets File:**
    *   Add a new `.env` file to the `secrets/XYZ.env` directory.
    *   Include API keys, secrets, and any environment-specific flags required by your strategy.

## 4. Docker Orchestration

To deploy your strategy as a self-contained, scalable pod, update the `docker-compose.yml` file.

*   **Leverage the Template:**
    *   Utilize the `&trading_template` anchor to maintain a clean and concise definition.
    *   By other words, copy psate one of the above and change the name and mounting env file

*   **Add the Service Block:**
    *   Add the service block precisely as formatted below, ensuring correct indentation:

    ```yaml
    tic_pod_momentum:
      <<: *trading_template
      container_name: tic_pod_momentum
      environment:
        - STRAT_NAME=XYZ  # Match the .yaml filename exactly
      env_file:
        - ./secrets/XYZ.env
    ```