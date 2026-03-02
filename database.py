import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from config import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


async def connect_db() -> None:
    global _client
    _client = AsyncIOMotorClient(settings.MONGODB_URI)
    # Ping to validate connection on startup
    await _client.admin.command("ping")
    logger.info("Connected to MongoDB Atlas.")


async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_collection() -> AsyncIOMotorCollection:
    if _client is None:
        raise RuntimeError("Database client is not initialised. Call connect_db() first.")
    return _client[settings.MONGODB_DB_NAME][settings.MONGODB_COLLECTION]
