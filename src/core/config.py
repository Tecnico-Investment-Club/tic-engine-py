import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

class GlobalSettings(BaseSettings):
    """
    Loads and validates global secrets from the .env file.
    Will crash on startup if required keys are missing.
    """
    env: str = "testnet"
    alpaca_api_key: str
    alpaca_api_secret: str
    database_url: str

    # Tells pydantic to look for a file named .env in the root folder
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instantiate it immediately so other modules can just import it and access settings as needed.
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