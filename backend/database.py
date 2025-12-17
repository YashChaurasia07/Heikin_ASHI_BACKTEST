from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import settings
import logging

logger = logging.getLogger(__name__)


class Database:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB with optimized connection pooling"""
    try:
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}")
        # Optimized connection with pooling
        db.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            maxPoolSize=100,  # Increased pool size for concurrent requests
            minPoolSize=10,
            maxIdleTimeMS=45000,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=45000
        )
        db.db = db.client[settings.MONGODB_DB_NAME]
        
        # Test connection
        await db.client.admin.command('ping')
        logger.info("Successfully connected to MongoDB with optimized pooling")
        
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
    """Create comprehensive database indexes for optimal query performance"""
    try:
        # Stock data indexes - compound indexes for efficient queries
        await db.db.stock_data_daily.create_index([("symbol", 1), ("date", 1)], unique=True)
        await db.db.stock_data_daily.create_index([("symbol", 1), ("date", -1)])  # For latest date queries
        await db.db.stock_data_daily.create_index([("date", 1)])  # For date range queries
        
        await db.db.stock_data_weekly.create_index([("symbol", 1), ("date", 1)], unique=True)
        await db.db.stock_data_weekly.create_index([("symbol", 1), ("date", -1)])
        await db.db.stock_data_weekly.create_index([("date", 1)])
        
        # Heikin Ashi indexes
        await db.db.heikin_ashi_daily.create_index([("symbol", 1), ("date", 1)], unique=True)
        await db.db.heikin_ashi_daily.create_index([("symbol", 1), ("date", -1)])
        await db.db.heikin_ashi_daily.create_index([("date", 1)])
        
        await db.db.heikin_ashi_weekly.create_index([("symbol", 1), ("date", 1)], unique=True)
        await db.db.heikin_ashi_weekly.create_index([("symbol", 1), ("date", -1)])
        await db.db.heikin_ashi_weekly.create_index([("date", 1)])
        
        # Metadata indexes
        await db.db.stock_metadata.create_index("symbol", unique=True)
        await db.db.symbols.create_index("symbol", unique=True)
        await db.db.symbols.create_index("active")  # For filtering active symbols
        
        logger.info("Comprehensive database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    return db.db
