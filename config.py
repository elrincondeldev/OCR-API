from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Google Document AI
    GOOGLE_APPLICATION_CREDENTIALS: str
    GOOGLE_CLOUD_PROJECT: str
    DOCUMENT_AI_PROCESSOR_ID: str
    DOCUMENT_AI_LOCATION: str = "us"

    # Replicate
    REPLICATE_API_TOKEN: str
    REPLICATE_MODEL: str = "meta/meta-llama-3-70b-instruct"

    # Algolia
    ALGOLIA_APP_ID: str
    ALGOLIA_API_KEY: str
    ALGOLIA_INDEX_NAME: str = "restaurant_inventory"

    # MongoDB
    MONGODB_URI: str
    MONGODB_DB_NAME: str = "ocr_pipeline"
    MONGODB_COLLECTION: str = "delivery_notes"

    # Pipeline
    CONFIDENCE_THRESHOLD: float = 0.80

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
