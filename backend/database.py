from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import settings
import logging

logger = logging.getLogger(__name__)


class Database:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB"""
    try:
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}")
        db.client = AsyncIOMotorClient(settings.MONGODB_URL)
        db.db = db.client[settings.MONGODB_DB_NAME]
        
        # Test connection
        await db.client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection"""
    try:
        if db.client:
            db.client.close()
            logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")


async def create_indexes():
    """Create necessary database indexes"""
    try:
        # Stock data indexes
        await db.db.stock_data_daily.create_index([("symbol", 1), ("date", 1)], unique=True)
        await db.db.stock_data_weekly.create_index([("symbol", 1), ("date", 1)], unique=True)
        
        # Heikin Ashi indexes
        await db.db.heikin_ashi_daily.create_index([("symbol", 1), ("date", 1)], unique=True)
        await db.db.heikin_ashi_weekly.create_index([("symbol", 1), ("date", 1)], unique=True)
        
        # Metadata indexes
        await db.db.stock_metadata.create_index("symbol", unique=True)
        await db.db.symbols.create_index("symbol", unique=True)
        
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    return db.db
