"""Configuration utilities using dotenv."""
from functools import lru_cache
from pydantic import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    device: str = os.getenv("DEVICE", "cpu")

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
