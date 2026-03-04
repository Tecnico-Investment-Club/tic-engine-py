import os
import yaml
from pydantic_settings import BaseSettings

class SystemSettings(BaseSettings):
    """
    Loads secrets and environment variables.
    In Docker, these are populated by your secrets/etl.env file.
    """
    database_url: str
    alpaca_api_key: str
    alpaca_api_secret: str
    discord_webhook_url: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# This is the object your main.py is trying to import
settings = SystemSettings()

def load_yaml_config(file_path: str) -> dict:
    """
    Loads structural configurations like asset lists and risk parameters.
    Allows easy tweaking without touching code.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, "r") as f:
        return yaml.safe_load(f)