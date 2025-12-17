"""
Setup and initialization script for the Heikin Ashi backend
"""
import asyncio
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from database import connect_to_mongo, close_mongo_connection, get_database
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_database():
    """Initialize database with indexes and collections"""
    try:
        await connect_to_mongo()
        logger.info("Database initialized successfully")
        
        db = get_database()
        
        # Get collection stats
        collections = await db.list_collection_names()
        logger.info(f"Collections: {collections}")
        
        # Check symbol count
        symbol_count = await db.symbols.count_documents({})
        logger.info(f"Symbols in database: {symbol_count}")
        
        await close_mongo_connection()
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)


async def import_symbols_from_excel(file_path: str):
    """Import symbols from Excel file"""
    try:
        await connect_to_mongo()
        db = get_database()
        
        # Read Excel
        df = pd.read_excel(file_path)
        logger.info(f"Excel columns: {df.columns.tolist()}")
        
        # Extract symbols
        if 'Row Labels' in df.columns:
            symbols = df['Row Labels'].dropna().tolist()
        elif 'symbol' in df.columns or 'Symbol' in df.columns:
            col = 'symbol' if 'symbol' in df.columns else 'Symbol'
            symbols = df[col].dropna().tolist()
        else:
            symbols = df[df.columns[0]].dropna().tolist()
        
        # Clean symbols
        symbols = [str(s).strip() for s in symbols if str(s).strip() and not str(s).lower() in ['nan', 'none', '']]
        
        logger.info(f"Found {len(symbols)} symbols in Excel")
        
        # Import to database
        added = 0
        for symbol in symbols:
            existing = await db.symbols.find_one({"symbol": symbol})
            if not existing:
                await db.symbols.insert_one({
                    "symbol": symbol,
                    "exchange": "NSE",
                    "active": True,
                    "added_date": pd.Timestamp.utcnow()
                })
                added += 1
        
        logger.info(f"Imported {added} new symbols")
        
        await close_mongo_connection()
        
    except Exception as e:
        logger.error(f"Error importing symbols: {e}")
        sys.exit(1)


async def sync_all_data(interval: str = "daily"):
    """Sync all symbol data"""
    try:
        from services.data_fetcher import DataFetcher
        from services.heikin_ashi import HeikinAshiCalculator
        from models import IntervalType
        
        await connect_to_mongo()
        db = get_database()
        
        interval_type = IntervalType.DAILY if interval == "daily" else IntervalType.WEEKLY
        
        symbols = await db.symbols.find({"active": True}).to_list(length=None)
        logger.info(f"Syncing data for {len(symbols)} symbols ({interval})")
        
        fetcher = DataFetcher()
        
        for i, symbol_doc in enumerate(symbols, 1):
            symbol = symbol_doc["symbol"]
            logger.info(f"[{i}/{len(symbols)}] Syncing {symbol}...")
            
            # Sync data
            success, message, records = await fetcher.sync_stock_to_db(
                symbol,
                interval_type,
                force_update=False
            )
            
            if success:
                # Calculate HA
                ha_success, ha_msg = await HeikinAshiCalculator.calculate_and_store(
                    symbol,
                    interval_type
                )
                logger.info(f"  ✓ {symbol}: {records} records, HA: {ha_success}")
            else:
                logger.warning(f"  ✗ {symbol}: {message}")
        
        logger.info("Data sync completed")
        
        await close_mongo_connection()
        
    except Exception as e:
        logger.error(f"Error syncing data: {e}")
        sys.exit(1)


def print_menu():
    """Print setup menu"""
    print("\n" + "="*60)
    print("  Heikin Ashi Backend - Setup & Management")
    print("="*60)
    print("1. Initialize Database")
    print("2. Import Symbols from Excel")
    print("3. Sync All Data (Daily)")
    print("4. Sync All Data (Weekly)")
    print("5. Check Database Status")
    print("0. Exit")
    print("="*60)


async def check_status():
    """Check database status"""
    try:
        await connect_to_mongo()
        db = get_database()
        
        # Symbol count
        symbol_count = await db.symbols.count_documents({"active": True})
        
        # Data counts
        daily_count = await db.stock_data_daily.count_documents({})
        weekly_count = await db.stock_data_weekly.count_documents({})
        ha_daily_count = await db.heikin_ashi_daily.count_documents({})
        ha_weekly_count = await db.heikin_ashi_weekly.count_documents({})
        
        print("\n" + "="*60)
        print("  Database Status")
        print("="*60)
        print(f"Active Symbols: {symbol_count}")
        print(f"\nDaily Data:")
        print(f"  Stock Candles: {daily_count}")
        print(f"  HA Candles: {ha_daily_count}")
        print(f"\nWeekly Data:")
        print(f"  Stock Candles: {weekly_count}")
        print(f"  HA Candles: {ha_weekly_count}")
        print("="*60)
        
        await close_mongo_connection()
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")


async def main():
    """Main setup function"""
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            await initialize_database()
        elif choice == "2":
            file_path = input("Enter Excel file path: ").strip()
            await import_symbols_from_excel(file_path)
        elif choice == "3":
            await sync_all_data("daily")
        elif choice == "4":
            await sync_all_data("weekly")
        elif choice == "5":
            await check_status()
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    asyncio.run(main())
