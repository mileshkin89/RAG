
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for resolving relative paths
BASE_DIR = Path(__file__).parent.parent

load_dotenv()


class AppConfig(BaseSettings):

    DATASET_DIR: Path = BASE_DIR / 'dataset'
    DB_DIR: Path = BASE_DIR / 'db'
    LOG_PATH: Path = BASE_DIR / "logs" / "app.log"

    COLLECTION_NAME: str = 'recipes'
    EMBEDDER: str = 'all-MiniLM-L6-v2'

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / "src/.env"),
        env_file_encoding="utf-8"
    )



# Global application config instance
config = AppConfig()

