from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import asyncio
from collections import defaultdict
import logging
import pandas as pd

from models import BacktestParams, IntervalType
from database import get_database
from services.backtest_engine import BacktestEngine
from services.data_fetcher import DataFetcher
from services.heikin_ashi import HeikinAshiCalculator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/run")
async def run_backtest(params: BacktestParams) -> Dict[str, Any]:
    """
    Run backtest on all available symbols with OPTIMIZED concurrent processing
    
    IMPORTANT: Does NOT sync data - uses existing data in database
    User should manually sync data before running backtest
    
    Backtest process:
    1. Load existing data from database (NO SYNC)
    2. Validate data availability
    3. Run backtest with concurrent processing
    """
    try:
        logger.info(f"="*50)
        logger.info(f"BACKTEST STARTED - NO DATA SYNC")
        logger.info(f"Parameters: {params.dict()}")
        logger.info(f"="*50)
        
        db = get_database()
        
        # Get all active symbols
        logger.info("Fetching active symbols from database...")
        symbols_docs = await db.symbols.find({"active": True}).to_list(length=None)
        symbols = [s["symbol"] for s in symbols_docs]
        
        if not symbols:
            logger.error("No active symbols found in database")
            raise HTTPException(status_code=400, detail="No active symbols found")
        
        logger.info(f"Found {len(symbols)} active symbols")
        
        # SKIP DATA SYNC - just validate existing data
        logger.info("Validating existing data (NO SYNC DURING BACKTEST)...")
        
        async def check_data_availability(symbol):
            try:
                # Check if data exists
                collection_name = f"stock_data_{params.interval.value}"
                ha_collection_name = f"heikin_ashi_{params.interval.value}"
                
                data_count = await db[collection_name].count_documents({"symbol": symbol})
                ha_count = await db[ha_collection_name].count_documents({"symbol": symbol})
                
                return symbol, data_count > 0, ha_count > 0, data_count
            except Exception as e:
                logger.error(f"Error checking data for {symbol}: {e}")
                return symbol, False, False, 0
        
        # Check data availability for all symbols
        availability_results = await asyncio.gather(*[check_data_availability(sym) for sym in symbols])
        
        # Filter symbols with available data
        symbols_with_data = []
        symbols_missing_data = []
        symbols_missing_ha = []
        
        for symbol, has_data, has_ha, count in availability_results:
            if has_data and has_ha:
                symbols_with_data.append(symbol)
            elif not has_data:
                symbols_missing_data.append(symbol)
            elif not has_ha:
                symbols_missing_ha.append(symbol)
        
        logger.info(f"Data availability: {len(symbols_with_data)} ready, {len(symbols_missing_data)} missing data, {len(symbols_missing_ha)} missing HA")
        
        if len(symbols_missing_data) > 0:
            logger.warning(f"Symbols missing data: {', '.join(symbols_missing_data[:10])}{'...' if len(symbols_missing_data) > 10 else ''}")
        
        if len(symbols_missing_ha) > 0:
            logger.warning(f"Symbols missing HA: {', '.join(symbols_missing_ha[:10])}{'...' if len(symbols_missing_ha) > 10 else ''}")
        
        if len(symbols_with_data) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No symbols have data available. Please sync data first using /data/sync endpoint."
            )
        
        # Use only symbols with complete data
        symbols = symbols_with_data
        logger.info(f"Proceeding with backtest on {len(symbols)} symbols with complete data")
        
        # OPTIMIZATION: Run backtest with concurrent processing
        logger.info("="*50)
        logger.info("STARTING CONCURRENT BACKTEST EXECUTION")
        logger.info(f"Processing {len(symbols)} symbols...")
        logger.info("="*50)
        
        all_trades = []
        daily_signals = defaultdict(list)
        stock_performance = {}
        stocks_results = {}
        
        # Process stocks concurrently in batches
        async def process_stock(symbol: str):
            try:
                # Load data
                normal_df, ha_df = await BacktestEngine.load_stock_data(
                    symbol,
                    params.interval,
                    params.start_date,
                    params.end_date
                )
                
                if normal_df is None or ha_df is None:
                    return None
                
                # Run backtest for this stock (CPU-bound, but fast with vectorization)
                result = BacktestEngine.vectorized_backtest(
                    normal_df,
                    ha_df,
                    params.initial_capital,
                    symbol
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None
        
        # Process in batches to control memory usage
        batch_size = 50
        processed_count = 0
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_results = await asyncio.gather(*[process_stock(sym) for sym in batch])
            
            for result in batch_results:
                if result is None:
                    continue
                
                symbol = result['symbol']
                trades = result['trades']
                signal_dates = result['signal_dates']
                
                processed_count += 1
                logger.info(f"[{processed_count}/{len(symbols)}] ✓ {symbol}: {len(trades)} trades, P&L: {result['pnl']:.2f}")
                
                # Track signals by date
                for sig_date in signal_dates:
                    daily_signals[sig_date].append(symbol)
                
                # Calculate stock performance
                win_trades = [t for t in trades if t['pnl'] > 0]
                loss_trades = [t for t in trades if t['pnl'] <= 0]
                
                stock_performance[symbol] = {
                    "pnl": result['pnl'],
                    "return_pct": (result['pnl'] / params.initial_capital * 100) if params.initial_capital > 0 else 0,
                    "num_trades": len(trades),
                    "win_trades": len(win_trades),
                    "loss_trades": len(loss_trades),
                    "win_rate": (len(win_trades) / len(trades) * 100) if len(trades) > 0 else 0
                }
                
                stocks_results[symbol] = {
                    "initial_capital": result['initial_capital'],
                    "final_capital": result['final_capital'],
                    "pnl": result['pnl'],
                    "num_trades": len(trades),
                    "trades": trades,
                    "ema_filter_stats": result['ema_filter_stats']
                }
                
                # Enrich trades
                for trade in trades:
                    entry_date = pd.to_datetime(trade['entry_date'])
                    exit_date = pd.to_datetime(trade['exit_date'])
                    holding_days = (exit_date - entry_date).days
                    return_pct = (
                        (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
                    ) if trade['entry_price'] > 0 else 0
                    
                    all_trades.append({
                        **trade,
                        "symbol": symbol,
                        "return_pct": return_pct,
                        "holding_days": holding_days
                    })
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
        
        # Clear cache after backtest
        BacktestEngine.clear_cache()
        
        # Calculate portfolio metrics
        logger.info("="*50)
        logger.info("CALCULATING PORTFOLIO METRICS")
        logger.info("="*50)
        
        total_initial = len(stocks_results) * params.initial_capital
        total_final = sum(result['final_capital'] for result in stocks_results.values())
        total_pnl = total_final - total_initial
        
        logger.info(f"Stocks Processed: {len(stocks_results)}/{len(symbols_with_data)}")
        logger.info(f"Total Trades: {len(all_trades)}")
        logger.info(f"Initial Capital: ₹{total_initial:,.2f}")
        logger.info(f"Final Capital: ₹{total_final:,.2f}")
        logger.info(f"Total P&L: ₹{total_pnl:,.2f} ({(total_pnl / total_initial * 100):.2f}%)")
        
        # Calculate statistics
        logger.info("Calculating statistics...")
        statistics = BacktestEngine.calculate_statistics(
            all_trades,
            daily_signals,
            stock_performance,
            total_initial,
            total_final,
            params.start_date,
            params.end_date
        )
        
        # Prepare response
        portfolio = {
            "total_initial_capital": total_initial,
            "total_final_capital": total_final,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / total_initial * 100) if total_initial > 0 else 0,
            "num_stocks_processed": len(stocks_results),
            "stocks": stocks_results,
            "params": {
                "start_date": params.start_date,
                "end_date": params.end_date,
                "initial_capital": params.initial_capital,
                "interval": params.interval.value,
                "strategy": "heikin_ashi_with_50_ema_filter"
            },
            "statistics": statistics,
            "data_sync_skipped": True,
            "symbols_with_data": len(symbols_with_data),
            "symbols_missing_data": len(symbols_missing_data),
            "symbols_missing_ha": len(symbols_missing_ha),
            "total_symbols": len(symbols_docs)
        }
        
        logger.info("="*50)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: Check logs above for timing")
        logger.info("="*50)
        
        return portfolio
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_backtest_status():
    """Get status of data readiness for backtesting"""
    try:
        db = get_database()
        
        # Get all active symbols
        symbols = await db.symbols.find({"active": True}).to_list(length=None)
        
        daily_ready = 0
        weekly_ready = 0
        
        fetcher = DataFetcher()
        
        for symbol_doc in symbols:
            symbol = symbol_doc["symbol"]
            
            # Check daily
            is_daily_fresh = await fetcher.is_data_up_to_date(symbol, IntervalType.DAILY)
            if is_daily_fresh:
                daily_ready += 1
            
            # Check weekly
            is_weekly_fresh = await fetcher.is_data_up_to_date(symbol, IntervalType.WEEKLY)
            if is_weekly_fresh:
                weekly_ready += 1
        
        total = len(symbols)
        
        return {
            "total_symbols": total,
            "daily": {
                "ready": daily_ready,
                "stale": total - daily_ready,
                "ready_pct": round((daily_ready / total * 100), 2) if total > 0 else 0
            },
            "weekly": {
                "ready": weekly_ready,
                "stale": total - weekly_ready,
                "ready_pct": round((weekly_ready / total * 100), 2) if total > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def run_backtest_get(
    start_date: str = Query(...),
    initial_capital: float = Query(...),
    end_date: Optional[str] = Query(None),
    interval: str = Query("daily")
) -> Dict[str, Any]:
    """
    GET endpoint wrapper for backtest (for frontend compatibility)
    Converts to POST endpoint internally
    """
    params = BacktestParams(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        interval=IntervalType(interval)
    )
    return await run_backtest(params)
