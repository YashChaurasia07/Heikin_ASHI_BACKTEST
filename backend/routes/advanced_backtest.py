"""
Advanced backtest strategy endpoints:
- HA + T-Score strategy
- Smart portfolio with ranking and compounding
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from collections import defaultdict
import logging
import pandas as pd
import numpy as np

from models import IntervalType, SmartBacktestParams
from database import get_database
from services.backtest_engine import BacktestEngine
from services.data_fetcher import DataFetcher
from services.heikin_ashi import HeikinAshiCalculator
from services.advanced_strategies import calculate_t_score, rank_stocks_for_selection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/advanced", tags=["advanced-backtest"])


@router.post("/ha-tscore")
async def run_ha_tscore_backtest(params: SmartBacktestParams) -> Dict[str, Any]:
    """
    Heikin Ashi + T-Score Strategy Backtest
    
    Combines:
    - Heikin Ashi pattern recognition for entry signals
    - T-Score momentum/strength filtering for stock prioritization
    
    Higher T-Score = Higher priority for entry
    """
    try:
        logger.info(f"="*50)
        logger.info(f"HA+T-SCORE BACKTEST STARTED")
        logger.info(f"Parameters: {params.dict()}")
        logger.info(f"="*50)
        
        db = get_database()
        
        # Get all active symbols (optimized)
        logger.info("Fetching active symbols from database...")
        symbols_docs = await db.symbols.find(
            {"active": True},
            {"_id": 0, "symbol": 1}
        ).to_list(length=500)  # Limit for performance
        symbols = [s["symbol"] for s in symbols_docs]
        
        if not symbols:
            logger.error("No active symbols found in database")
            raise HTTPException(status_code=400, detail="No active symbols found")
        
        logger.info(f"Found {len(symbols)} active symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        
        # SKIP DATA SYNC - just validate existing data
        logger.info("Validating existing data (NO SYNC DURING BACKTEST)...")
        
        # Check data availability
        symbols_with_data = []
        for symbol in symbols:
            collection_name = f"stock_data_{params.interval.value}"
            ha_collection_name = f"heikin_ashi_{params.interval.value}"
            
            data_count = await db[collection_name].count_documents({"symbol": symbol})
            ha_count = await db[ha_collection_name].count_documents({"symbol": symbol})
            
            if data_count > 0 and ha_count > 0:
                symbols_with_data.append(symbol)
        
        if not symbols_with_data:
            raise HTTPException(
                status_code=400,
                detail="No symbols have data available. Please sync data first using /data/sync endpoint."
            )
        
        symbols = symbols_with_data
        logger.info(f"Found {len(symbols)} symbols with available data")
        
        # Load all stock data
        logger.info("Loading stock data...")
        stock_data_cache = {}
        for idx, symbol in enumerate(symbols, 1):
            normal_df, ha_df = await BacktestEngine.load_stock_data(
                symbol,
                params.interval,
                params.start_date,
                params.end_date
            )
            if normal_df is not None and ha_df is not None and len(ha_df) >= 4:
                stock_data_cache[symbol] = (normal_df, ha_df)
                if idx % 10 == 0:
                    logger.info(f"Loaded {idx}/{len(symbols)} symbols...")
        
        logger.info(f"Loaded {len(stock_data_cache)} stocks for HA+T-Score backtest")
        
        # Get all unique dates
        all_dates = set()
        for normal_df, ha_df in stock_data_cache.values():
            all_dates.update(ha_df.index)
        all_dates = sorted(list(all_dates))
        
        if not all_dates:
            raise HTTPException(
                status_code=400,
                detail="No trading dates found in the data. The selected date range may not have any data."
            )
        
        logger.info(f"Backtest period: {len(all_dates)} days from {all_dates[0].date()} to {all_dates[-1].date()}")
        
        # Portfolio state
        base_capital_per_position = params.total_investment / params.num_stocks
        active_positions = {}
        total_portfolio_value = params.total_investment
        cash_balance = params.total_investment
        all_trades = []
        entry_signals_by_date = defaultdict(list)
        equity_curve = []
        
        # Track actual capital deployed per stock for accurate return calculation
        stock_capital_deployed = defaultdict(float)
        
        # Pre-calculate all HA signals WITH 50 EMA FILTER (matching main.py logic)
        logger.info("Pre-calculating HA signals for all stocks with 50 EMA filter...")
        stock_signals = {}
        total_signals_before_ema = 0
        total_signals_after_ema = 0
        
        for idx, (symbol, (normal_df, ha_df)) in enumerate(stock_data_cache.items(), 1):
            if idx % 10 == 0:
                logger.info(f"Calculated signals for {idx}/{len(stock_data_cache)} stocks...")
            signals = []
            
            ha_open = ha_df['ha_open'].values
            ha_low = ha_df['ha_low'].values
            ha_high = ha_df['ha_high'].values
            ha_close = ha_df['ha_close'].values
            ha_dates = ha_df.index.values
            
            normal_dates = normal_df.index.values
            normal_close = normal_df['close'].values
            ema_50 = normal_df['ema_50'].values if 'ema_50' in normal_df.columns else None
            normal_date_to_idx = {date: idx for idx, date in enumerate(normal_dates)}
            
            for i in range(3, len(ha_df)):
                date = ha_dates[i]
                
                cond_today = (ha_open[i] == ha_low[i] and ha_close[i] > ha_open[i])
                cond_yest = (ha_open[i-1] == ha_low[i-1] and ha_close[i-1] > ha_open[i-1])
                cond_yest2 = (ha_open[i-2] != ha_low[i-2] and ha_high[i-2] != ha_close[i-2] and ha_close[i-2] > ha_open[i-2])
                
                if cond_today and cond_yest and cond_yest2:
                    entry_idx = i + 1
                    if entry_idx < len(ha_df):
                        entry_date = ha_dates[entry_idx]
                        if entry_date in normal_date_to_idx:
                            total_signals_before_ema += 1
                            
                            # 50 EMA FILTER: Check if price is above 50 EMA (matching main.py)
                            entry_normal_idx = normal_date_to_idx[entry_date]
                            if ema_50 is not None:
                                entry_close_price = normal_close[entry_normal_idx]
                                entry_ema_50_value = ema_50[entry_normal_idx]
                                
                                # Skip signal if price is NOT above 50 EMA
                                if entry_close_price <= entry_ema_50_value:
                                    continue
                            
                            total_signals_after_ema += 1
                            signals.append({
                                'signal_date': date,
                                'entry_date': entry_date,
                                'sl': ha_low[i-2]
                            })
            
            if signals:
                stock_signals[symbol] = signals
        
        ema_rejection_rate = ((total_signals_before_ema - total_signals_after_ema) / total_signals_before_ema * 100) if total_signals_before_ema > 0 else 0
        logger.info(f"Found {len(stock_signals)} stocks with signals")
        logger.info(f"Total signals BEFORE 50 EMA filter: {total_signals_before_ema}")
        logger.info(f"Total signals AFTER 50 EMA filter: {total_signals_after_ema}")
        logger.info(f"Signals rejected by EMA filter: {total_signals_before_ema - total_signals_after_ema} ({ema_rejection_rate:.1f}%)")
        logger.info("="*50)
        logger.info("STARTING DAY-BY-DAY SIMULATION")
        logger.info("="*50)
        
        # Day-by-day simulation with T-Score ranking
        last_log_pct = 0
        for day_idx, current_date in enumerate(all_dates, 1):
            # Log progress every 10%
            progress_pct = int((day_idx / len(all_dates)) * 100)
            if progress_pct >= last_log_pct + 10:
                logger.info(f"Progress: {progress_pct}% ({day_idx}/{len(all_dates)} days)")
                last_log_pct = progress_pct
            # Update existing positions
            positions_to_exit = []
            
            for symbol, position in list(active_positions.items()):
                if symbol not in stock_data_cache:
                    continue
                
                normal_df, ha_df = stock_data_cache[symbol]
                
                if current_date not in normal_df.index or current_date not in ha_df.index:
                    continue
                
                normal_low = normal_df.loc[current_date, 'low']
                normal_close = normal_df.loc[current_date, 'close']
                ha_open_today = ha_df.loc[current_date, 'ha_open']
                ha_low_today = ha_df.loc[current_date, 'ha_low']
                
                # Update current value
                current_value = position['qty'] * normal_close
                position['current_value'] = current_value
                
                # Check SL hit
                if normal_low <= position['sl']:
                    exit_price = position['sl']
                    pnl = (exit_price - position['entry_price']) * position['qty']
                    # Return deployed capital plus P&L (matches main.py logic)
                    cash_balance += position['capital_deployed'] + pnl
                    
                    all_trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': str(current_date.date()),
                        'exit_price': exit_price,
                        'qty': position['qty'],
                        'pnl': pnl,
                        'exit_type': 'SL hit',
                        't_score': position.get('t_score'),
                        'capital_deployed': position['capital_deployed']
                    })
                    positions_to_exit.append(symbol)
                else:
                    # Trail SL
                    if ha_open_today != ha_low_today:
                        position['sl'] = max(position['sl'], ha_low_today)
            
            for symbol in positions_to_exit:
                del active_positions[symbol]
            
            # Check for new entries with T-Score ranking
            if len(active_positions) < params.num_stocks and cash_balance > 0:
                candidates = []
                
                for symbol, signals in stock_signals.items():
                    if symbol in active_positions:
                        continue
                    
                    for signal in signals:
                        if signal['entry_date'] == current_date:
                            normal_df, ha_df = stock_data_cache[symbol]
                            t_score = calculate_t_score(normal_df, current_date)
                            
                            priority_score = t_score if t_score is not None else 0
                            
                            candidates.append({
                                'symbol': symbol,
                                'signal': signal,
                                'priority': priority_score,
                                't_score': t_score
                            })
                            break
                
                # Sort by T-Score (higher first)
                candidates.sort(key=lambda x: x['priority'], reverse=True)
                
                open_slots = params.num_stocks - len(active_positions)
                entries_today = candidates[:open_slots]
                
                for candidate in entries_today:
                    symbol = candidate['symbol']
                    signal = candidate['signal']
                    t_score = candidate['t_score']
                    
                    normal_df, ha_df = stock_data_cache[symbol]
                    
                    if current_date not in normal_df.index:
                        continue
                    
                    entry_price = normal_df.loc[current_date, 'open']
                    
                    if params.enable_compounding:
                        # Calculate portfolio value for compounding
                        deployed_value = sum(p.get('current_value', p['capital_deployed']) for p in active_positions.values())
                        current_portfolio_value = cash_balance + deployed_value
                        capital_for_position = current_portfolio_value / params.num_stocks
                    else:
                        capital_for_position = base_capital_per_position
                    
                    capital_for_position = min(capital_for_position, cash_balance)
                    qty = int(capital_for_position / entry_price)
                    
                    if qty < 1:
                        continue
                    
                    actual_capital = entry_price * qty
                    
                    active_positions[symbol] = {
                        'entry_date': str(current_date.date()),
                        'entry_price': entry_price,
                        'qty': qty,
                        'sl': signal['sl'],
                        'capital_deployed': actual_capital,
                        'current_value': actual_capital,
                        't_score': t_score
                    }
                    
                    # Track capital deployed per stock
                    stock_capital_deployed[symbol] += actual_capital
                    
                    entry_signals_by_date[str(current_date.date())].append({
                        'symbol': symbol,
                        't_score': t_score
                    })
                    cash_balance -= actual_capital
            
            # Track equity - recalculate portfolio value each day
            deployed_value = sum(p.get('current_value', p['capital_deployed']) for p in active_positions.values())
            total_portfolio_value = cash_balance + deployed_value
            
            equity_curve.append({
                'date': str(current_date.date()),
                'equity': round(total_portfolio_value, 2),
                'cash': round(cash_balance, 2),
                'deployed': round(deployed_value, 2)
            })
        
        # Close remaining positions
        logger.info("Closing remaining positions...")
        if active_positions:
            logger.info(f"{len(active_positions)} positions to close")
            last_date = all_dates[-1]
            for symbol, position in active_positions.items():
                normal_df, ha_df = stock_data_cache[symbol]
                if last_date in normal_df.index:
                    exit_price = normal_df.loc[last_date, 'close']
                    pnl = (exit_price - position['entry_price']) * position['qty']
                    # Return deployed capital plus P&L (matches main.py logic)
                    cash_balance += position['capital_deployed'] + pnl
                    
                    all_trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': str(last_date.date()),
                        'exit_price': exit_price,
                        'qty': position['qty'],
                        'pnl': pnl,
                        'exit_type': 'end of period',
                        't_score': position.get('t_score'),
                        'capital_deployed': position['capital_deployed']
                    })
        
        # Calculate results
        logger.info("="*50)
        logger.info("CALCULATING RESULTS")
        logger.info("="*50)
        
        total_pnl = sum(trade['pnl'] for trade in all_trades)
        final_capital = params.total_investment + total_pnl
        
        logger.info(f"Total Trades: {len(all_trades)}")
        logger.info(f"Initial Capital: ₹{params.total_investment:,.2f}")
        logger.info(f"Final Capital: ₹{final_capital:,.2f}")
        logger.info(f"Total P&L: ₹{total_pnl:,.2f} ({(total_pnl / params.total_investment * 100):.2f}%)")
        
        # Enrich trades
        all_trades_enriched = []
        for trade in all_trades:
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            holding_days = (exit_date - entry_date).days
            return_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
            
            all_trades_enriched.append({
                **trade,
                'return_pct': return_pct,
                'holding_days': holding_days
            })
        
        # Calculate stock performance
        stocks_by_symbol = defaultdict(lambda: {'trades': [], 'pnl': 0})
        for trade in all_trades_enriched:
            symbol = trade['symbol']
            stocks_by_symbol[symbol]['trades'].append(trade)
            stocks_by_symbol[symbol]['pnl'] += trade['pnl']
        
        stock_performance = {}
        stocks_detailed = {}
        for symbol, data in stocks_by_symbol.items():
            win_trades = [t for t in data['trades'] if t['pnl'] > 0]
            
            # Use actual capital deployed for this stock (sum of all entry capitals)
            actual_capital_deployed = stock_capital_deployed.get(symbol, base_capital_per_position)
            
            # Calculate initial and final capital for this stock
            stock_initial_capital = actual_capital_deployed
            stock_final_capital = stock_initial_capital + data['pnl']
            
            # Calculate latest T-Score (for current date/end of backtest period)
            latest_tscore = None
            if symbol in stock_data_cache:
                normal_df, ha_df = stock_data_cache[symbol]
                # Use the last date in the data (most recent)
                latest_date = normal_df.index[-1]
                latest_tscore = calculate_t_score(normal_df, latest_date)
            
            stock_performance[symbol] = {
                'pnl': data['pnl'],
                'return_pct': (data['pnl'] / stock_initial_capital * 100) if stock_initial_capital > 0 else 0,
                'num_trades': len(data['trades']),
                'win_trades': len(win_trades),
                'loss_trades': len(data['trades']) - len(win_trades),
                'win_rate': (len(win_trades) / len(data['trades']) * 100) if data['trades'] else 0
            }
            
            # Add detailed stock data for frontend
            stocks_detailed[symbol] = {
                'pnl': data['pnl'],
                'return_pct': (data['pnl'] / stock_initial_capital * 100) if stock_initial_capital > 0 else 0,
                'num_trades': len(data['trades']),
                'win_trades': len(win_trades),
                'loss_trades': len(data['trades']) - len(win_trades),
                'win_rate': (len(win_trades) / len(data['trades']) * 100) if data['trades'] else 0,
                'initial_capital': stock_initial_capital,
                'final_capital': stock_final_capital,
                'trades': data['trades'],
                'latest_tscore': latest_tscore  # Latest T-Score at end of backtest period
            }
        
        # Calculate statistics using portfolio-level final_capital (not stock-level)
        statistics = BacktestEngine.calculate_statistics(
            all_trades_enriched,
            entry_signals_by_date,
            stock_performance,
            params.total_investment,
            final_capital,
            params.start_date,
            params.end_date
        )
        
        # T-Score stats
        trades_with_tscore = [t for t in all_trades_enriched if t.get('t_score') is not None]
        avg_tscore = np.mean([t['t_score'] for t in trades_with_tscore]) if trades_with_tscore else 0
        
        logger.info("="*50)
        logger.info("HA+T-SCORE BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        return {
            "total_initial_capital": params.total_investment,
            "total_final_capital": final_capital,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / params.total_investment * 100) if params.total_investment > 0 else 0,
            "num_stocks_processed": len(stocks_by_symbol),
            "params": {
                "strategy": "heikin_ashi_tscore_with_ema_filter",
                "start_date": params.start_date,
                "end_date": params.end_date,
                "interval": params.interval.value,
                "total_investment": params.total_investment,
                "max_concurrent_positions": params.num_stocks,
                "compounding": params.enable_compounding,
                "ema_filter": "50 EMA (price must be above)"
            },
            "statistics": statistics,
            "equity_curve": equity_curve,
            "stocks": stocks_detailed,
            "tscore_stats": {
                "avg_tscore_at_entry": round(avg_tscore, 2),
                "trades_with_tscore": len(trades_with_tscore),
                "total_trades": len(all_trades_enriched)
            },
            "ema_filter_stats": {
                "total_signals_before_ema": total_signals_before_ema,
                "total_signals_after_ema": total_signals_after_ema,
                "rejected_by_ema": total_signals_before_ema - total_signals_after_ema,
                "rejection_rate_pct": round(ema_rejection_rate, 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in HA+T-Score backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/smart-portfolio")
async def run_smart_portfolio_backtest(params: SmartBacktestParams) -> Dict[str, Any]:
    """
    Smart Portfolio Backtest with Ranking and Compounding
    
    Features:
    - Intelligent stock ranking based on historical performance
    - Dynamic capital allocation
    - True compounding
    - Rebalancing support
    """
    try:
        logger.info(f"Starting Smart Portfolio backtest: {params.dict()}")
        
        db = get_database()
        
        # Get all active symbols (optimized)
        symbols_docs = await db.symbols.find(
            {"active": True},
            {"_id": 0, "symbol": 1}
        ).to_list(length=500)  # Limit for performance
        symbols = [s["symbol"] for s in symbols_docs]
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No active symbols found")
        
        # SKIP DATA SYNC - just validate existing data
        logger.info("Validating existing data (NO SYNC DURING BACKTEST)...")
        
        symbols_with_data = []
        for symbol in symbols:
            collection_name = f"stock_data_{params.interval.value}"
            ha_collection_name = f"heikin_ashi_{params.interval.value}"
            
            data_count = await db[collection_name].count_documents({"symbol": symbol})
            ha_count = await db[ha_collection_name].count_documents({"symbol": symbol})
            
            if data_count > 0 and ha_count > 0:
                symbols_with_data.append(symbol)
        
        if not symbols_with_data:
            raise HTTPException(
                status_code=400,
                detail="No symbols have data available. Please sync data first using /data/sync endpoint."
            )
        
        symbols = symbols_with_data
        logger.info(f"Found {len(symbols)} symbols with available data")
        
        # Rank stocks
        logger.info("Ranking stocks based on historical performance...")
        ranked_stocks = await rank_stocks_for_selection(
            symbols,
            params.start_date,
            params.end_date,
            params.interval,
            params.ranking_period_months
        )
        
        stock_rankings = {s['symbol']: s['score'] for s in ranked_stocks}
        
        logger.info(f"Ranked {len(stock_rankings)} stocks")
        
        # Load stock data
        logger.info("Loading stock data...")
        stock_data_cache = {}
        for idx, symbol in enumerate(symbols, 1):
            normal_df, ha_df = await BacktestEngine.load_stock_data(
                symbol,
                params.interval,
                params.start_date,
                params.end_date
            )
            if normal_df is not None and ha_df is not None and len(ha_df) >= 4:
                stock_data_cache[symbol] = (normal_df, ha_df)
                if idx % 10 == 0:
                    logger.info(f"Loaded {idx}/{len(symbols)} symbols...")
        
        logger.info(f"Loaded {len(stock_data_cache)} stocks for Smart Portfolio backtest")
        
        # Get all unique dates
        all_dates = set()
        for normal_df, ha_df in stock_data_cache.values():
            all_dates.update(ha_df.index)
        all_dates = sorted(list(all_dates))
        
        if not all_dates:
            raise HTTPException(
                status_code=400,
                detail="No trading dates found in the data. The selected date range may not have any data."
            )
        
        logger.info(f"Backtest period: {len(all_dates)} days from {all_dates[0].date()} to {all_dates[-1].date()}")
        
        # Portfolio state
        base_capital_per_position = params.total_investment / params.num_stocks
        total_portfolio_value = params.total_investment
        active_positions = {}
        # total_portfolio_value = params.total_investment
        cash_balance = params.total_investment
        all_trades = []
        entry_signals_by_date = defaultdict(list)
        equity_curve = []
        
        # Pre-calculate all HA signals WITH 50 EMA FILTER (matching main.py logic)
        logger.info("Pre-calculating HA signals for all stocks with 50 EMA filter...")
        stock_signals = {}
        total_signals_before_ema = 0
        total_signals_after_ema = 0
        
        for idx, (symbol, (normal_df, ha_df)) in enumerate(stock_data_cache.items(), 1):
            if idx % 10 == 0:
                logger.info(f"Calculated signals for {idx}/{len(stock_data_cache)} stocks...")
            signals = []
            
            ha_open = ha_df['ha_open'].values
            ha_low = ha_df['ha_low'].values
            ha_high = ha_df['ha_high'].values
            ha_close = ha_df['ha_close'].values
            ha_dates = ha_df.index.values
            
            normal_dates = normal_df.index.values
            normal_close = normal_df['close'].values
            ema_50 = normal_df['ema_50'].values if 'ema_50' in normal_df.columns else None
            normal_date_to_idx = {date: idx for idx, date in enumerate(normal_dates)}
            
            for i in range(3, len(ha_df)):
                date = ha_dates[i]
                
                cond_today = (ha_open[i] == ha_low[i] and ha_close[i] > ha_open[i])
                cond_yest = (ha_open[i-1] == ha_low[i-1] and ha_close[i-1] > ha_open[i-1])
                cond_yest2 = (ha_open[i-2] != ha_low[i-2] and ha_high[i-2] != ha_close[i-2] and ha_close[i-2] > ha_open[i-2])
                
                if cond_today and cond_yest and cond_yest2:
                    entry_idx = i + 1
                    if entry_idx < len(ha_df):
                        entry_date = ha_dates[entry_idx]
                        if entry_date in normal_date_to_idx:
                            total_signals_before_ema += 1
                            
                            # 50 EMA FILTER: Check if price is above 50 EMA (matching main.py)
                            entry_normal_idx = normal_date_to_idx[entry_date]
                            if ema_50 is not None:
                                entry_close_price = normal_close[entry_normal_idx]
                                entry_ema_50_value = ema_50[entry_normal_idx]
                                
                                # Skip signal if price is NOT above 50 EMA
                                if entry_close_price <= entry_ema_50_value:
                                    continue
                            
                            total_signals_after_ema += 1
                            signals.append({
                                'signal_date': date,
                                'entry_date': entry_date,
                                'sl': ha_low[i-2]
                            })
            
            if signals:
                stock_signals[symbol] = signals
        
        ema_rejection_rate = ((total_signals_before_ema - total_signals_after_ema) / total_signals_before_ema * 100) if total_signals_before_ema > 0 else 0
        logger.info(f"Found {len(stock_signals)} stocks with signals")
        logger.info(f"Total signals BEFORE 50 EMA filter: {total_signals_before_ema}")
        logger.info(f"Total signals AFTER 50 EMA filter: {total_signals_after_ema}")
        logger.info(f"Signals rejected by EMA filter: {total_signals_before_ema - total_signals_after_ema} ({ema_rejection_rate:.1f}%)")
        
        # Run backtest with ranking-based prioritization
        logger.info("Running backtest simulation...")
        for date_idx, date in enumerate(all_dates):
            if date_idx % 100 == 0:
                logger.info(f"Processing day {date_idx}/{len(all_dates)}...")
            
            # Get signals for this date
            signals_today = []
            for symbol, signals in stock_signals.items():
                for signal in signals:
                    if signal['entry_date'] == date:
                        ranking_score = stock_rankings.get(symbol, 0)
                        signals_today.append({
                            'symbol': symbol,
                            'signal': signal,
                            'ranking_score': ranking_score
                        })
            
            # Prioritize by ranking score (higher = better)
            signals_today.sort(key=lambda x: x['ranking_score'], reverse=True)
            entry_signals_by_date[str(date.date())].extend([s['symbol'] for s in signals_today])
            
            # Process entries
            for signal_info in signals_today:
                if len(active_positions) >= params.num_stocks:
                    break
                
                symbol = signal_info['symbol']
                signal = signal_info['signal']
                
                if symbol in active_positions:
                    continue
                
                normal_df, ha_df = stock_data_cache[symbol]
                try:
                    entry_price = normal_df.loc[date, 'open']
                    
                    # Use compounding if enabled (matches main.py logic)
                    if params.enable_compounding:
                        # Calculate portfolio value for compounding
                        deployed_value = sum(p.get('current_value', p.get('capital_deployed', p['entry_price'] * p['qty'])) for p in active_positions.values())
                        current_portfolio_value = cash_balance + deployed_value
                        capital_for_position = current_portfolio_value / params.num_stocks
                    else:
                        capital_for_position = base_capital_per_position
                    
                    capital_for_position = min(capital_for_position, cash_balance)
                    qty = int(capital_for_position / entry_price)
                    
                    if qty > 0 and cash_balance >= (entry_price * qty):
                        actual_capital_deployed = entry_price * qty
                        active_positions[symbol] = {
                            'entry_date': str(date.date()),
                            'entry_price': entry_price,
                            'qty': qty,
                            'sl': signal['sl'],
                            'ranking_score': signal_info['ranking_score'],
                            'capital_deployed': actual_capital_deployed,
                            'current_value': actual_capital_deployed
                        }
                        cash_balance -= actual_capital_deployed
                        logger.info(f"  ENTRY: {symbol} @ ₹{entry_price:.2f} x {qty} (Rank Score: {signal_info['ranking_score']:.2f})")
                except:
                    continue
            
            # Check exits for active positions - FIXED to match main.py logic
            # main.py only uses SL hit and trailing SL, NOT HA exit signal
            positions_to_exit = []
            for symbol, position in list(active_positions.items()):
                normal_df, ha_df = stock_data_cache[symbol]
                
                try:
                    if date not in normal_df.index or date not in ha_df.index:
                        continue
                    
                    low_price = normal_df.loc[date, 'low']
                    close_price = normal_df.loc[date, 'close']
                    ha_open_today = ha_df.loc[date, 'ha_open']
                    ha_low_today = ha_df.loc[date, 'ha_low']
                    
                    # Update current value for tracking
                    position['current_value'] = position['qty'] * close_price
                    
                    # Check stop loss hit
                    if low_price <= position['sl']:
                        exit_price = position['sl']
                        pnl = (exit_price - position['entry_price']) * position['qty']
                        capital_deployed = position.get('capital_deployed', position['entry_price'] * position['qty'])
                        # Return deployed capital plus P&L (matches main.py)
                        cash_balance += capital_deployed + pnl
                        
                        all_trades.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'entry_price': position['entry_price'],
                            'exit_date': str(date.date()),
                            'exit_price': exit_price,
                            'qty': position['qty'],
                            'pnl': pnl,
                            'exit_type': 'SL hit',
                            'ranking_score': position.get('ranking_score'),
                            'capital_deployed': capital_deployed
                        })
                        
                        positions_to_exit.append(symbol)
                        logger.info(f"  EXIT: {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:,.2f} | Type: SL hit")
                    else:
                        # Trail SL - only if ha_open != ha_low (matches main.py logic)
                        if ha_open_today != ha_low_today:
                            position['sl'] = max(position['sl'], ha_low_today)
                except:
                    continue
            
            for symbol in positions_to_exit:
                del active_positions[symbol]
        
        # Close remaining positions
        if active_positions:
            last_date = all_dates[-1]
            logger.info(f"Closing {len(active_positions)} remaining positions at end of period...")
            
            for symbol, position in active_positions.items():
                normal_df, _ = stock_data_cache[symbol]
                if last_date in normal_df.index:
                    exit_price = normal_df.loc[last_date, 'close']
                    pnl = (exit_price - position['entry_price']) * position['qty']
                    capital_deployed = position.get('capital_deployed', position['entry_price'] * position['qty'])
                    # Return deployed capital plus P&L (matches main.py logic)
                    cash_balance += capital_deployed + pnl
                    
                    all_trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': str(last_date.date()),
                        'exit_price': exit_price,
                        'qty': position['qty'],
                        'pnl': pnl,
                        'exit_type': 'end of period',
                        'ranking_score': position.get('ranking_score'),
                        'capital_deployed': capital_deployed
                    })
        
        # Calculate results
        logger.info("="*50)
        logger.info("CALCULATING RESULTS")
        logger.info("="*50)
        
        total_pnl = sum(trade['pnl'] for trade in all_trades)
        final_capital = params.total_investment + total_pnl
        
        logger.info(f"Total Trades: {len(all_trades)}")
        logger.info(f"Initial Capital: ₹{params.total_investment:,.2f}")
        logger.info(f"Final Capital: ₹{final_capital:,.2f}")
        logger.info(f"Total P&L: ₹{total_pnl:,.2f} ({(total_pnl / params.total_investment * 100):.2f}%)")
        
        # Enrich trades
        all_trades_enriched = []
        for trade in all_trades:
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            holding_days = (exit_date - entry_date).days
            return_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
            
            all_trades_enriched.append({
                **trade,
                'return_pct': return_pct,
                'holding_days': holding_days
            })
        
        # Calculate stock performance
        stocks_by_symbol = defaultdict(lambda: {'trades': [], 'pnl': 0})
        for trade in all_trades_enriched:
            symbol = trade['symbol']
            stocks_by_symbol[symbol]['trades'].append(trade)
            stocks_by_symbol[symbol]['pnl'] += trade['pnl']
        
        stock_performance = {}
        stocks_detailed = {}
        for symbol, data in stocks_by_symbol.items():
            win_trades = [t for t in data['trades'] if t['pnl'] > 0]
            
            # Calculate total actual capital deployed for this stock (sum of all entry capitals)
            total_capital = sum(t['entry_price'] * t['qty'] for t in data['trades'])
            
            # Calculate initial and final capital for this stock (use stock_ prefix to avoid shadowing)
            stock_initial_capital = total_capital  # Use actual deployed capital
            stock_final_capital = stock_initial_capital + data['pnl']
            
            # Calculate latest T-Score (for current date/end of backtest period)
            latest_tscore = None
            if symbol in stock_data_cache:
                normal_df, ha_df = stock_data_cache[symbol]
                # Use the last date in the data (most recent)
                latest_date = normal_df.index[-1]
                latest_tscore = calculate_t_score(normal_df, latest_date)
            
            stock_performance[symbol] = {
                'pnl': data['pnl'],
                'return_pct': (data['pnl'] / stock_initial_capital * 100) if stock_initial_capital > 0 else 0,
                'num_trades': len(data['trades']),
                'win_trades': len(win_trades),
                'loss_trades': len(data['trades']) - len(win_trades),
                'win_rate': (len(win_trades) / len(data['trades']) * 100) if data['trades'] else 0
            }
            
            # Add detailed stock data for frontend
            stocks_detailed[symbol] = {
                'pnl': data['pnl'],
                'return_pct': (data['pnl'] / stock_initial_capital * 100) if stock_initial_capital > 0 else 0,
                'num_trades': len(data['trades']),
                'win_trades': len(win_trades),
                'loss_trades': len(data['trades']) - len(win_trades),
                'win_rate': (len(win_trades) / len(data['trades']) * 100) if data['trades'] else 0,
                'initial_capital': stock_initial_capital,
                'final_capital': stock_final_capital,
                'trades': data['trades'],
                'latest_tscore': latest_tscore  # Latest T-Score at end of backtest period
            }
        
        # Calculate statistics using portfolio-level final_capital (not stock-level)
        statistics = BacktestEngine.calculate_statistics(
            all_trades_enriched,
            entry_signals_by_date,
            stock_performance,
            params.total_investment,
            final_capital,
            params.start_date,
            params.end_date
        )
        
        logger.info("="*50)
        logger.info("SMART PORTFOLIO BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        return {
            "total_initial_capital": params.total_investment,
            "total_final_capital": final_capital,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / params.total_investment * 100) if params.total_investment > 0 else 0,
            "num_stocks_processed": len(stocks_by_symbol),
            "params": {
                "strategy": "heikin_ashi_smart_selection",
                "start_date": params.start_date,
                "end_date": params.end_date,
                "interval": params.interval.value,
                "total_investment": params.total_investment,
                "max_concurrent_positions": params.num_stocks,
                "compounding": params.enable_compounding
            },
            "statistics": statistics,
            "equity_curve": equity_curve,
            "stocks": stocks_detailed,
            "selection_info": {
                "total_investment": params.total_investment,
                "num_stocks_selected": params.num_stocks,
                "total_stocks_analyzed": len(ranked_stocks),
                "capital_per_stock": base_capital_per_position,
                "ranking_period_months": params.ranking_period_months,
                "selected_stocks": ranked_stocks[:params.num_stocks]
            },
            "ema_filter_stats": {
                "total_signals_before_ema": total_signals_before_ema,
                "total_signals_after_ema": total_signals_after_ema,
                "rejected_by_ema": total_signals_before_ema - total_signals_after_ema,
                "rejection_rate_pct": round(ema_rejection_rate, 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in smart portfolio backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ha-tscore")
async def run_ha_tscore_backtest_get(
    total_investment: float = Query(...),
    num_stocks: int = Query(...),
    start_date: str = Query(...),
    end_date: Optional[str] = Query(None),
    interval: str = Query("daily")
) -> Dict[str, Any]:
    """GET endpoint wrapper for HA+T-Score backtest"""
    params = SmartBacktestParams(
        total_investment=total_investment,
        num_stocks=num_stocks,
        start_date=start_date,
        end_date=end_date,
        interval=IntervalType(interval)
    )
    return await run_ha_tscore_backtest(params)


@router.get("/smart-portfolio")
async def run_smart_portfolio_backtest_get(
    total_investment: float = Query(...),
    num_stocks: int = Query(...),
    start_date: str = Query(...),
    end_date: Optional[str] = Query(None),
    use_cached: bool = Query(True),
    interval: str = Query("daily")
) -> Dict[str, Any]:
    """GET endpoint wrapper for smart portfolio backtest"""
    params = SmartBacktestParams(
        total_investment=total_investment,
        num_stocks=num_stocks,
        start_date=start_date,
        end_date=end_date,
        interval=IntervalType(interval)
    )
    return await run_smart_portfolio_backtest(params)


@router.get("/tscores")
async def get_all_tscores(
    interval: str = Query("daily", description="Data interval: daily or weekly"),
    as_of_date: Optional[str] = Query(None, description="Calculate T-Score as of this date (default: latest)")
) -> Dict[str, Any]:
    """
    Get T-Scores for all active stocks
    
    Returns:
    - List of all stocks with their T-Scores
    - Latest data date for each stock
    - Sortable and filterable data
    """
    try:
        logger.info(f"Fetching T-Scores for all stocks (interval: {interval})")
        
        db = get_database()
        interval_type = IntervalType(interval)
        
        # Get all active symbols
        symbols_docs = await db.symbols.find(
            {"active": True},
            {"_id": 0, "symbol": 1}
        ).to_list(length=500)
        symbols = [s["symbol"] for s in symbols_docs]
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No active symbols found")
        
        logger.info(f"Calculating T-Scores for {len(symbols)} symbols...")
        
        # Calculate T-Scores for all stocks
        tscores_data = []
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                # Load stock data (only normal data needed for T-Score)
                collection_name = f"stock_data_{interval_type.value}"
                
                # Get stock data
                query = {"symbol": symbol}
                if as_of_date:
                    query["date"] = {"$lte": pd.to_datetime(as_of_date)}
                
                stock_data = await db[collection_name].find(
                    query,
                    {"_id": 0, "symbol": 0, "interval": 0}
                ).sort("date", 1).to_list(length=5000)
                
                if not stock_data or len(stock_data) < 21:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(stock_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                # Get latest date
                latest_date = df.index[-1]
                
                # Calculate T-Score for latest date
                tscore = calculate_t_score(df, latest_date)
                
                if tscore is not None:
                    latest_close = df.loc[latest_date, 'close']
                    latest_volume = df.loc[latest_date, 'volume'] if 'volume' in df.columns else 0
                    
                    # Calculate additional metrics
                    sma_21 = df['close'].rolling(21).mean().iloc[-1] if len(df) >= 21 else None
                    sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
                    sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else None
                    
                    # Handle NaN values
                    if sma_21 is not None and pd.isna(sma_21):
                        sma_21 = None
                    if sma_50 is not None and pd.isna(sma_50):
                        sma_50 = None
                    if sma_200 is not None and pd.isna(sma_200):
                        sma_200 = None
                    
                    # Calculate 52-week high
                    lookback_days = min(252, len(df))
                    week_52_high = df['high'].iloc[-lookback_days:].max()
                    distance_from_52w_high = ((week_52_high - latest_close) / week_52_high) * 100
                    
                    tscores_data.append({
                        'symbol': symbol,
                        't_score': tscore,
                        'latest_date': latest_date.strftime('%Y-%m-%d'),
                        'latest_close': float(latest_close),
                        'latest_volume': int(latest_volume),
                        'sma_21': float(sma_21) if sma_21 is not None else None,
                        'sma_50': float(sma_50) if sma_50 is not None else None,
                        'sma_200': float(sma_200) if sma_200 is not None else None,
                        'week_52_high': float(week_52_high),
                        'distance_from_52w_high_pct': round(float(distance_from_52w_high), 2),
                        'above_sma_21': bool(latest_close > sma_21) if sma_21 is not None else None,
                        'above_sma_50': bool(latest_close > sma_50) if sma_50 is not None else None,
                        'above_sma_200': bool(latest_close > sma_200) if sma_200 is not None else None
                    })
                
                if idx % 50 == 0:
                    logger.info(f"Processed {idx}/{len(symbols)} symbols...")
                    
            except Exception as e:
                logger.error(f"Error calculating T-Score for {symbol}: {e}")
                continue
        
        # Sort by T-Score (highest first)
        tscores_data.sort(key=lambda x: x['t_score'], reverse=True)
        
        logger.info(f"Successfully calculated T-Scores for {len(tscores_data)} stocks")
        
        return {
            "total_stocks": len(tscores_data),
            "interval": interval,
            "as_of_date": as_of_date or "latest",
            "tscores": tscores_data,
            "summary": {
                "avg_tscore": round(sum(d['t_score'] for d in tscores_data) / len(tscores_data), 2) if tscores_data else 0,
                "max_tscore": max(d['t_score'] for d in tscores_data) if tscores_data else 0,
                "min_tscore": min(d['t_score'] for d in tscores_data) if tscores_data else 0,
                "stocks_above_80": sum(1 for d in tscores_data if d['t_score'] >= 80),
                "stocks_above_60": sum(1 for d in tscores_data if d['t_score'] >= 60),
                "stocks_below_40": sum(1 for d in tscores_data if d['t_score'] < 40)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting T-Scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))

