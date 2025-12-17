import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import settings
from database import get_database
from models import IntervalType, StockData, StockMetadata
from functools import lru_cache

logger = logging.getLogger(__name__)

# Thread pool for blocking TvDatafeed operations (reduced to 10 to avoid rate limiting)
_executor = ThreadPoolExecutor(max_workers=10)

# In-memory cache for metadata checks
_metadata_cache: Dict[str, Dict] = {}


class DataFetcher:
    def __init__(self):
        """Initialize TvDatafeed"""
        if settings.TV_USERNAME and settings.TV_PASSWORD:
            self.tv = TvDatafeed(username=settings.TV_USERNAME, password=settings.TV_PASSWORD)
            logger.info("TvDatafeed initialized with credentials")
        else:
            self.tv = TvDatafeed()
            logger.info("TvDatafeed initialized without credentials (limited access)")
    
    def get_tv_interval(self, interval: IntervalType) -> Interval:
        """Convert IntervalType to TvDatafeed Interval"""
        if interval == IntervalType.DAILY:
            return Interval.in_daily
        elif interval == IntervalType.WEEKLY:
            return Interval.in_weekly
        else:
            raise ValueError(f"Unsupported interval: {interval}")
    
    async def fetch_stock_data(
        self, 
        symbol: str, 
        interval: IntervalType = IntervalType.DAILY,
        n_bars: int = 5000
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from TradingView (async wrapper)
        
        Args:
            symbol: Stock symbol
            interval: Data interval (daily or weekly)
            n_bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"Fetching {interval.value} data for {symbol}...")
            
            tv_interval = self.get_tv_interval(interval)
            
            # Run blocking operation in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                _executor,
                self.tv.get_hist,
                symbol,
                'NSE',
                tv_interval,
                n_bars
            )
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.2)
            
            if data is None or data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Reset index to make datetime a column
            data = data.reset_index()
            data.rename(columns={'datetime': 'Date'}, inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Ensure lowercase column names
            data.columns = [col.lower() if col != 'Date' else 'Date' for col in data.columns]
            
            # Sort by date
            data = data.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Fetched {len(data)} bars for {symbol} ({interval.value})")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def sync_stock_to_db(
        self,
        symbol: str,
        interval: IntervalType = IntervalType.DAILY,
        force_update: bool = False,
        timeout: int = 60
    ) -> Tuple[bool, str, int]:
        """
        Sync stock data to MongoDB with timeout
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            force_update: Force update even if data is fresh
            timeout: Timeout in seconds (default 60)
        
        Returns:
            (success, message, records_added)
        """
        try:
            # Wrap the sync operation with timeout
            return await asyncio.wait_for(
                self._sync_stock_to_db_impl(symbol, interval, force_update),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            error_msg = f"Sync timeout for {symbol} after {timeout}s"
            logger.error(error_msg)
            return False, error_msg, 0
        except Exception as e:
            error_msg = f"Error syncing {symbol}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, 0
    
    async def _sync_stock_to_db_impl(
        self,
        symbol: str,
        interval: IntervalType = IntervalType.DAILY,
        force_update: bool = False
    ) -> Tuple[bool, str, int]:
        """
        Internal implementation of sync_stock_to_db with smart incremental updates
        
        Returns:
            (success, message, records_added)
        """
        try:
            db = get_database()
            today = datetime.utcnow().date()
            
            # Determine collection
            collection_name = f"stock_data_{interval.value}"
            collection = db[collection_name]
            
            # Check if already synced today (unless force update)
            if not force_update:
                metadata = await db.stock_metadata.find_one({"symbol": symbol})
                if metadata:
                    last_sync_field = f"last_sync_date_{interval.value}"
                    last_sync_date = metadata.get(last_sync_field)
                    
                    # If synced today, skip
                    if last_sync_date and last_sync_date.date() >= today:
                        return True, f"Already synced today ({last_sync_date.date()})", 0
                    
                    # Check how much data is missing
                    last_data_field = f"last_data_date_{interval.value}"
                    last_data_date = metadata.get(last_data_field)
                    
                    if last_data_date:
                        days_missing = (datetime.utcnow() - last_data_date).days
                        logger.info(f"{symbol}: Last data from {last_data_date.date()}, {days_missing} days gap")
                        
                        # For incremental update, fetch only recent data
                        if days_missing < 100:  # Incremental update for small gaps
                            data = await self.fetch_stock_data(symbol, interval, n_bars=days_missing + 50)
                        else:  # Full refresh for large gaps
                            data = await self.fetch_stock_data(symbol, interval)
                    else:
                        # No previous data, do full fetch
                        data = await self.fetch_stock_data(symbol, interval)
                else:
                    # First time sync
                    data = await self.fetch_stock_data(symbol, interval)
            else:
                # Force update - full fetch
                data = await self.fetch_stock_data(symbol, interval)
            
            if data is None or data.empty:
                return False, "No data fetched", 0
            
            # Prepare documents for bulk insert
            documents = []
            for _, row in data.iterrows():
                doc = {
                    "symbol": symbol,
                    "date": row['Date'].to_pydatetime(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']) if 'volume' in row else 0.0,
                    "interval": interval.value
                }
                documents.append(doc)
            
            # Bulk insert with upsert - OPTIMIZED for speed
            records_added = 0
            latest_date = None
            if len(documents) > 0:
                # Use bulk_write for much better performance
                from pymongo import UpdateOne
                
                operations = [
                    UpdateOne(
                        {"symbol": symbol, "date": doc["date"]},
                        {"$set": doc},
                        upsert=True
                    )
                    for doc in documents
                ]
                
                # Execute in batches for very large datasets
                batch_size = 1000
                for i in range(0, len(operations), batch_size):
                    batch = operations[i:i + batch_size]
                    result = await collection.bulk_write(batch, ordered=False)
                    records_added += result.upserted_count + result.modified_count
                
                # Get latest date from synced data
                latest_date = max(doc["date"] for doc in documents)
            
            # Update metadata with sync tracking
            await db.stock_metadata.update_one(
                {"symbol": symbol},
                {
                    "$set": {
                        f"last_updated_{interval.value}": datetime.utcnow(),
                        f"last_sync_date_{interval.value}": datetime.utcnow(),
                        f"last_data_date_{interval.value}": latest_date if latest_date else datetime.utcnow(),
                        f"total_candles_{interval.value}": len(documents)
                    }
                },
                upsert=True
            )
            
            logger.info(f"Synced {records_added} records for {symbol} ({interval.value}), latest: {latest_date.date() if latest_date else 'N/A'}")
            return True, f"Successfully synced {records_added} records", records_added
            
        except Exception as e:
            error_msg = f"Error syncing {symbol}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, 0
    
    async def is_data_up_to_date(
        self,
        symbol: str,
        interval: IntervalType = IntervalType.DAILY
    ) -> bool:
        """Check if stock data is up-to-date with caching"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval.value}"
            if cache_key in _metadata_cache:
                cached_data = _metadata_cache[cache_key]
                hours_since_cache = (datetime.utcnow() - cached_data['cache_time']).total_seconds() / 3600
                if hours_since_cache < 0.5:  # Cache for 30 minutes
                    return cached_data['is_fresh']
            
            db = get_database()
            metadata = await db.stock_metadata.find_one({"symbol": symbol})
            
            if not metadata:
                _metadata_cache[cache_key] = {'is_fresh': False, 'cache_time': datetime.utcnow()}
                return False
            
            last_update_field = f"last_updated_{interval.value}"
            last_update = metadata.get(last_update_field)
            
            if not last_update:
                _metadata_cache[cache_key] = {'is_fresh': False, 'cache_time': datetime.utcnow()}
                return False
            
            hours_since_update = (datetime.utcnow() - last_update).total_seconds() / 3600
            is_fresh = hours_since_update < settings.DATA_UPDATE_HOURS
            
            # Update cache
            _metadata_cache[cache_key] = {'is_fresh': is_fresh, 'cache_time': datetime.utcnow()}
            return is_fresh
            
        except Exception as e:
            logger.error(f"Error checking data freshness for {symbol}: {e}")
            return False
    
    async def get_latest_date_in_db(
        self,
        symbol: str,
        interval: IntervalType = IntervalType.DAILY
    ) -> Optional[datetime]:
        """Get the latest date available for a symbol"""
        try:
            db = get_database()
            collection_name = f"stock_data_{interval.value}"
            
            result = await db[collection_name].find_one(
                {"symbol": symbol},
                sort=[("date", -1)]
            )
            
            return result["date"] if result else None
            
        except Exception as e:
            logger.error(f"Error getting latest date for {symbol}: {e}")
            return None
