import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from typing import Optional
import logging
from database import get_database
from models import IntervalType

logger = logging.getLogger(__name__)

# Set high precision for accurate calculations
getcontext().prec = 10


class HeikinAshiCalculator:
    """Calculate Heikin Ashi candles with vectorized numpy operations for speed"""
    
    @staticmethod
    async def calculate_and_store(
        symbol: str,
        interval: IntervalType = IntervalType.DAILY
    ) -> tuple[bool, str]:
        """
        Calculate Heikin Ashi candles from stock data and store in DB
        OPTIMIZED with vectorized numpy operations
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            
        Returns:
            (success, message)
        """
        try:
            db = get_database()
            
            # Fetch stock data from DB (optimized with projection and limit)
            collection_name = f"stock_data_{interval.value}"
            cursor = db[collection_name].find(
                {"symbol": symbol},
                {"_id": 0}  # Exclude _id field
            ).sort("date", 1).limit(10000)  # Limit to prevent huge loads
            
            stock_data = await cursor.to_list(length=10000)
            
            if not stock_data:
                return False, f"No stock data found for {symbol}"
            
            if len(stock_data) < 1:
                return False, f"Insufficient data for {symbol}"
            
            logger.info(f"Calculating HA for {symbol} ({interval.value}) - {len(stock_data)} candles")
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(stock_data)
            df = df.sort_values('date').reset_index(drop=True)
            
            # OPTIMIZED: Use numpy arrays directly for faster calculations
            open_arr = df['open'].values.astype(np.float64)
            high_arr = df['high'].values.astype(np.float64)
            low_arr = df['low'].values.astype(np.float64)
            close_arr = df['close'].values.astype(np.float64)
            
            n = len(df)
            ha_open = np.zeros(n, dtype=np.float64)
            ha_high = np.zeros(n, dtype=np.float64)
            ha_low = np.zeros(n, dtype=np.float64)
            ha_close = np.zeros(n, dtype=np.float64)
            
            # First candle
            ha_close[0] = (open_arr[0] + high_arr[0] + low_arr[0] + close_arr[0]) / 4.0
            ha_open[0] = (open_arr[0] + close_arr[0]) / 2.0
            ha_high[0] = max(high_arr[0], ha_open[0], ha_close[0])
            ha_low[0] = min(low_arr[0], ha_open[0], ha_close[0])
            
            # Vectorized calculation for remaining candles
            for i in range(1, n):
                ha_close[i] = (open_arr[i] + high_arr[i] + low_arr[i] + close_arr[i]) / 4.0
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
                ha_high[i] = max(high_arr[i], ha_open[i], ha_close[i])
                ha_low[i] = min(low_arr[i], ha_open[i], ha_close[i])
            
            # Add HA columns to dataframe
            df['ha_open'] = ha_open
            df['ha_high'] = ha_high
            df['ha_low'] = ha_low
            df['ha_close'] = ha_close
            
            # Prepare documents for storage
            ha_collection_name = f"heikin_ashi_{interval.value}"
            documents = []
            
            for _, row in df.iterrows():
                doc = {
                    "symbol": symbol,
                    "date": row['date'],
                    "ha_open": float(row['ha_open']),
                    "ha_high": float(row['ha_high']),
                    "ha_low": float(row['ha_low']),
                    "ha_close": float(row['ha_close']),
                    "volume": float(row.get('volume', 0)),
                    "interval": interval.value
                }
                documents.append(doc)
            
            # OPTIMIZED: Bulk insert with bulk_write for much better performance
            records_added = 0
            if len(documents) > 0:
                from pymongo import UpdateOne
                
                operations = [
                    UpdateOne(
                        {"symbol": symbol, "date": doc["date"]},
                        {"$set": doc},
                        upsert=True
                    )
                    for doc in documents
                ]
                
                # Execute in batches
                batch_size = 1000
                for i in range(0, len(operations), batch_size):
                    batch = operations[i:i + batch_size]
                    result = await db[ha_collection_name].bulk_write(batch, ordered=False)
                    records_added += result.upserted_count + result.modified_count
            
            # Update metadata
            await db.stock_metadata.update_one(
                {"symbol": symbol},
                {
                    "$set": {
                        f"ha_calculated_{interval.value}": True,
                        f"last_ha_update_{interval.value}": pd.Timestamp.utcnow()
                    }
                },
                upsert=True
            )
            
            logger.info(f"Calculated and stored {records_added} HA candles for {symbol} ({interval.value})")
            return True, f"Successfully calculated {records_added} HA candles"
            
        except Exception as e:
            error_msg = f"Error calculating HA for {symbol}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    @staticmethod
    async def ensure_ha_data(
        symbol: str,
        interval: IntervalType = IntervalType.DAILY
    ) -> bool:
        """Ensure HA data exists, calculate if missing"""
        try:
            db = get_database()
            
            # Check if HA data exists
            ha_collection_name = f"heikin_ashi_{interval.value}"
            count = await db[ha_collection_name].count_documents({"symbol": symbol})
            
            if count > 0:
                return True
            
            # Calculate if missing
            success, _ = await HeikinAshiCalculator.calculate_and_store(symbol, interval)
            return success
            
        except Exception as e:
            logger.error(f"Error ensuring HA data for {symbol}: {e}")
            return False
