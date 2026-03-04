"""
Pydantic-validated configuration for the Trading Engine pod.

Loads the pod-specific YAML config and validates every field at startup,
so typos / missing keys surface immediately instead of mid-trade.
"""

import logging
import os
import sys
from typing import Optional

from pydantic import BaseModel, Field

from src.core.config import load_yaml_config

logger = logging.getLogger("TradingEngine.Config")


# Nested config sections
class StrategyConfig(BaseModel):
    name: str = "ExampleStrategy"
    class_path: Optional[str] = None
    timeframe: str = "1h"
    trade_every: Optional[str] = None
    params: dict = Field(default_factory=dict)


class RiskConfig(BaseModel):
    max_notional_per_order: Optional[float] = None
    max_total_notional_per_symbol: Optional[float] = None


class ExecutionConfig(BaseModel):
    type: str = "alpaca"  # or "mock"
    paper: bool = True


# Top-level engine config
class EngineConfig(BaseModel):
    poll_interval_seconds: int = 60
    lookback: int = 50
    assets: list[str] = Field(default_factory=list)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    discord_webhook_url: Optional[str] = None


# Loader
def load_engine_config() -> EngineConfig:
    """
    Load and validate the engine YAML config.
    Uses the shared ``load_yaml_config`` from core so there is
    exactly one YAML-loading code path in the project.
    """
    path = os.getenv("CONFIG_PATH", "src/engine_py/config.yaml")
    logger.info(f"Loading configuration from: {path}")

    try:
        raw = load_yaml_config(path)
    except FileNotFoundError:
        logger.error(f"Config file not found at {path}.")
        sys.exit(1)

    return EngineConfig(**raw)

