"""
Database optimization script - Add indexes for better query performance
Run this once to optimize MongoDB queries
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_indexes():
    """Create indexes on MongoDB collections for optimal performance"""
    
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    
    logger.info("Creating indexes for optimal query performance...")
    
    try:
        # Stock data indexes (daily and weekly)
        for interval in ['daily', 'weekly']:
            collection_name = f"stock_data_{interval}"
            logger.info(f"Creating indexes for {collection_name}...")
            
            # Compound index on symbol + date (most common query pattern)
            await db[collection_name].create_index(
                [("symbol", 1), ("date", 1)],
                unique=True,
                name=f"idx_{interval}_symbol_date"
            )
            
            # Index on date for date range queries
            await db[collection_name].create_index(
                [("date", -1)],
                name=f"idx_{interval}_date"
            )
            
            # Index on symbol for symbol-specific queries
            await db[collection_name].create_index(
                [("symbol", 1)],
                name=f"idx_{interval}_symbol"
            )
            
            logger.info(f"✓ Indexes created for {collection_name}")
        
        # Heikin Ashi indexes (daily and weekly)
        for interval in ['daily', 'weekly']:
            ha_collection_name = f"heikin_ashi_{interval}"
            logger.info(f"Creating indexes for {ha_collection_name}...")
            
            # Compound index on symbol + date
            await db[ha_collection_name].create_index(
                [("symbol", 1), ("date", 1)],
                unique=True,
                name=f"idx_ha_{interval}_symbol_date"
            )
            
            # Index on date
            await db[ha_collection_name].create_index(
                [("date", -1)],
                name=f"idx_ha_{interval}_date"
            )
            
            # Index on symbol
            await db[ha_collection_name].create_index(
                [("symbol", 1)],
                name=f"idx_ha_{interval}_symbol"
            )
            
            logger.info(f"✓ Indexes created for {ha_collection_name}")
        
        # Stock metadata indexes
        logger.info("Creating indexes for stock_metadata...")
        await db.stock_metadata.create_index(
            [("symbol", 1)],
            unique=True,
            name="idx_metadata_symbol"
        )
        
        await db.stock_metadata.create_index(
            [("last_updated_daily", -1)],
            name="idx_metadata_last_updated_daily"
        )
        
        await db.stock_metadata.create_index(
            [("last_updated_weekly", -1)],
            name="idx_metadata_last_updated_weekly"
        )
        
        logger.info("✓ Indexes created for stock_metadata")
        
        # Symbols collection index
        logger.info("Creating indexes for symbols...")
        await db.symbols.create_index(
            [("symbol", 1)],
            unique=True,
            name="idx_symbols_symbol"
        )
        
        await db.symbols.create_index(
            [("active", 1)],
            name="idx_symbols_active"
        )
        
        logger.info("✓ Indexes created for symbols")
        
        logger.info("="*50)
        logger.info("ALL INDEXES CREATED SUCCESSFULLY!")
        logger.info("="*50)
        
        # List all indexes
        for collection_name in ['stock_data_daily', 'stock_data_weekly', 
                                'heikin_ashi_daily', 'heikin_ashi_weekly',
                                'stock_metadata', 'symbols']:
            indexes = await db[collection_name].list_indexes().to_list(length=None)
            logger.info(f"\n{collection_name} indexes:")
            for idx in indexes:
                logger.info(f"  - {idx['name']}: {idx.get('key', {})}")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(create_indexes())
