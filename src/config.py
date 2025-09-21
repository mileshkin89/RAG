import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for resolving relative paths
BASE_DIR = Path(__file__).parent.parent

load_dotenv()


class AppConfig(BaseSettings):
    DATASET_NAME: str = "food_recipes.csv"

    DATASET_DIR: Path = BASE_DIR / 'dataset'
    DATASET_PATH: Path = DATASET_DIR / DATASET_NAME

    DB_DIR: Path = BASE_DIR / 'db'
    LOG_PATH: Path = BASE_DIR / "logs" / "app.log"

    COLLECTION_NAME: str = 'recipes'
    EMBEDDER_NAME: str = 'all-MiniLM-L6-v2'

    BATCH_SIZE: int = 2000

    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / "src/.env"),
        env_file_encoding="utf-8"
    )


# Global application config instance
config = AppConfig()

