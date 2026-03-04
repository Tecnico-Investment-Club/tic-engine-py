import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

class GlobalSettings(BaseSettings):
    """
    Loads and validates global secrets from environment variables.
    Will crash on startup if required keys are missing.
    
    Environment variables expected:
    ALPACA_KEY: Alpaca API key
    ALPACA_SECRET: Alpaca API secret
    DATABASE_URL: PostgreSQL connection string
    """
    env: str = "testnet"
    alpaca_key: str
    alpaca_secret: str
    database_url: str

    # Read from environment variables
    model_config = SettingsConfigDict(extra="ignore")

settings = GlobalSettings()

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Utility to load a YAML config file into a Python dictionary.
    Used by both engine_py and etl_db to load their specific settings.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(path, "r") as file:
        config = yaml.safe_load(file)
        
    return config if config else {}