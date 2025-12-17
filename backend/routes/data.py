from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from models import DataSyncRequest, DataSyncStatus, IntervalType
from services.data_fetcher import DataFetcher
from services.heikin_ashi import HeikinAshiCalculator
from database import get_database
import logging
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


@router.get("/symbols")
async def get_symbols():
    """Get all available symbols"""
    try:
        db = get_database()
        symbols = await db.symbols.find({"active": True}).to_list(length=None)
        return {"symbols": [s["symbol"] for s in symbols]}
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync", response_model=List[DataSyncStatus])
async def sync_data(request: DataSyncRequest):
    """
    Sync stock data from TradingView to MongoDB
    HIGHLY OPTIMIZED with 500 concurrent stock processing
    
    - Checks if already synced today to avoid redundant API calls
    - Fetches only missing data incrementally
    - Syncs both DAILY and WEEKLY intervals
    - Calculates Heikin Ashi for both intervals
    - Updates metadata with sync tracking
    """
    try:
        db = get_database()
        fetcher = DataFetcher()
        
        # Get symbols to sync
        if request.symbols:
            symbols = request.symbols
        else:
            # Sync all active symbols
            symbol_docs = await db.symbols.find({"active": True}).to_list(length=None)
            symbols = [s["symbol"] for s in symbol_docs]
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols to sync")
        
        logger.info(f"Starting sync for {len(symbols)} symbols (daily & weekly sequentially)")
        
        # Process symbols with limited concurrency to avoid overwhelming tvdatafeed server
        async def sync_single_symbol_all_intervals(symbol: str):
            results = []
            
            # SEQUENTIAL sync for DAILY and WEEKLY to avoid rate limiting
            intervals_to_sync = [IntervalType.DAILY, IntervalType.WEEKLY]
            
            for interval in intervals_to_sync:
                try:
                    # Check if update needed (unless force update)
                    if not request.force_update:
                        # Check if synced today
                        metadata = await db.stock_metadata.find_one({"symbol": symbol})
                        if metadata:
                            last_sync_field = f"last_sync_date_{interval.value}"
                            last_sync_date = metadata.get(last_sync_field)
                            from datetime import datetime
                            today = datetime.utcnow().date()
                            
                            if last_sync_date and last_sync_date.date() >= today:
                                results.append(DataSyncStatus(
                                    symbol=symbol,
                                    status="skipped",
                                    message=f"{interval.value}: Already synced today",
                                    records_added=0
                                ))
                                continue
                    
                    # Sync stock data (SEQUENTIAL external API call)
                    success, message, records_added = await fetcher.sync_stock_to_db(
                        symbol,
                        interval,
                        request.force_update
                    )
                    
                    if not success:
                        results.append(DataSyncStatus(
                            symbol=symbol,
                            status="failed",
                            message=f"{interval.value}: {message}",
                            records_added=0
                        ))
                        continue
                    
                    # Get latest date
                    latest_date = await fetcher.get_latest_date_in_db(symbol, interval)
                    
                    results.append(DataSyncStatus(
                        symbol=symbol,
                        status="success",
                        message=f"{interval.value}: {records_added} records",
                        records_added=records_added,
                        last_date=latest_date
                    ))
                    
                except Exception as e:
                    logger.error(f"Error syncing {symbol} {interval.value}: {e}")
                    results.append(DataSyncStatus(
                        symbol=symbol,
                        status="failed",
                        message=f"{interval.value}: {str(e)}",
                        records_added=0
                    ))
            
            return results
        
        # Process symbols in smaller batches to avoid rate limiting (10 at a time)
        batch_size = 10
        all_results = []
        
        # Phase 1: Sync data from external API (with rate limiting)
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Syncing data batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} symbols)")
            
            # Process batch with limited concurrency
            batch_results = await asyncio.gather(*[sync_single_symbol_all_intervals(sym) for sym in batch])
            
            # Flatten results (each symbol returns 2 results - daily & weekly)
            for symbol_results in batch_results:
                all_results.extend(symbol_results)
            
            logger.info(f"Completed data sync batch {i//batch_size + 1}")
            
            # Small delay between batches to avoid overwhelming the server
            await asyncio.sleep(0.5)
        
        # Phase 2: Calculate Heikin Ashi in parallel (internal MongoDB operations)
        logger.info(f"Starting parallel Heikin Ashi calculations for all symbols")
        
        async def calculate_ha_for_symbol(symbol: str):
            ha_results = []
            for interval in [IntervalType.DAILY, IntervalType.WEEKLY]:
                try:
                    ha_success, ha_message = await HeikinAshiCalculator.calculate_and_store(
                        symbol,
                        interval
                    )
                    if not ha_success:
                        logger.warning(f"HA calculation failed for {symbol} {interval.value}: {ha_message}")
                except Exception as e:
                    logger.error(f"Error calculating HA for {symbol} {interval.value}: {e}")
            return ha_results
        
        # Calculate HA for all successfully synced symbols in parallel (no rate limiting needed)
        synced_symbols = list(set([r.symbol for r in all_results if r.status in ['success', 'skipped']]))
        if synced_symbols:
            logger.info(f"Calculating HA for {len(synced_symbols)} symbols in parallel...")
            await asyncio.gather(*[calculate_ha_for_symbol(sym) for sym in synced_symbols])
            logger.info("Heikin Ashi calculations completed")
        
        # Calculate summary
        success_count = sum(1 for r in all_results if r.status == 'success')
        skipped_count = sum(1 for r in all_results if r.status == 'skipped')
        failed_count = sum(1 for r in all_results if r.status == 'failed')
        
        logger.info(f"Data sync completed: Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}, Total: {len(all_results)}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error in data sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{symbol}")
async def get_data_status(
    symbol: str,
    interval: IntervalType = Query(IntervalType.DAILY)
):
    """Get data status for a symbol"""
    try:
        db = get_database()
        fetcher = DataFetcher()
        
        metadata = await db.stock_metadata.find_one({"symbol": symbol})
        
        if not metadata:
            return {
                "symbol": symbol,
                "exists": False,
                "up_to_date": False
            }
        
        is_fresh = await fetcher.is_data_up_to_date(symbol, interval)
        latest_date = await fetcher.get_latest_date_in_db(symbol, interval)
        
        return {
            "symbol": symbol,
            "exists": True,
            "up_to_date": is_fresh,
            "last_updated": metadata.get(f"last_updated_{interval.value}"),
            "latest_date": latest_date,
            "total_candles": metadata.get(f"total_candles_{interval.value}", 0),
            "ha_calculated": metadata.get(f"ha_calculated_{interval.value}", False)
        }
        
    except Exception as e:
        logger.error(f"Error getting data status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-all")
async def check_all_data_status(interval: IntervalType = Query(IntervalType.DAILY)):
    """Check data status for all symbols"""
    try:
        db = get_database()
        fetcher = DataFetcher()
        
        symbols = await db.symbols.find({"active": True}).to_list(length=None)
        
        results = []
        for symbol_doc in symbols:
            symbol = symbol_doc["symbol"]
            is_fresh = await fetcher.is_data_up_to_date(symbol, interval)
            
            results.append({
                "symbol": symbol,
                "up_to_date": is_fresh
            })
        
        total = len(results)
        up_to_date = sum(1 for r in results if r["up_to_date"])
        stale = total - up_to_date
        
        return {
            "total_symbols": total,
            "up_to_date": up_to_date,
            "stale": stale,
            "details": results
        }
        
    except Exception as e:
        logger.error(f"Error checking all data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}")
async def get_stock_ohlc_data(symbol: str, interval: IntervalType = Query(IntervalType.DAILY)):
    """Get OHLC data for a specific symbol"""
    try:
        db = get_database()
        collection = db.daily_candles if interval == IntervalType.DAILY else db.weekly_candles
        
        # Fetch data
        candles = await collection.find(
            {"symbol": symbol},
            {"_id": 0}
        ).sort("Date", 1).to_list(length=None)
        
        if not candles:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        return candles
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching OHLC data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/ha")
async def get_stock_ha_data(symbol: str, interval: IntervalType = Query(IntervalType.DAILY)):
    """Get Heikin Ashi data for a specific symbol"""
    try:
        db = get_database()
        collection = db.daily_ha if interval == IntervalType.DAILY else db.weekly_ha
        
        # Fetch data
        ha_candles = await collection.find(
            {"symbol": symbol},
            {"_id": 0}
        ).sort("Date", 1).to_list(length=None)
        
        if not ha_candles:
            raise HTTPException(status_code=404, detail=f"No HA data found for {symbol}")
        
        return ha_candles
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching HA data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
