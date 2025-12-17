import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import asyncio
from database import get_database
from models import IntervalType
from functools import lru_cache

logger = logging.getLogger(__name__)

# In-memory cache for loaded dataframes
_dataframe_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}


class BacktestEngine:
    """Core backtesting engine with optimized calculations"""
    
    @staticmethod
    async def load_stock_data(
        symbol: str,
        interval: IntervalType,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load stock and Heikin Ashi data from MongoDB with caching
        OPTIMIZED with in-memory caching for repeated requests
        
        Returns:
            (normal_df, ha_df) or (None, None) if data not available
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval.value}_{start_date}_{end_date}"
            if use_cache and cache_key in _dataframe_cache:
                logger.debug(f"Cache hit for {symbol}")
                return _dataframe_cache[cache_key]
            
            db = get_database()
            
            # Parse dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) if end_date else datetime.utcnow()
            
            # Load normal data and HA data in parallel
            collection_name = f"stock_data_{interval.value}"
            ha_collection_name = f"heikin_ashi_{interval.value}"
            
            # Extended query for EMA calculation
            extended_start = start_dt - timedelta(days=100)
            extended_query = {
                "symbol": symbol,
                "date": {"$gte": extended_start, "$lte": end_dt}
            }
            
            # Fetch both datasets (MongoDB can handle parallel queries)
            normal_data_task = db[collection_name].find(extended_query).sort("date", 1).to_list(length=None)
            ha_query = {
                "symbol": symbol,
                "date": {"$gte": start_dt, "$lte": end_dt}
            }
            ha_data_task = db[ha_collection_name].find(ha_query).sort("date", 1).to_list(length=None)
            
            # Await both
            normal_data, ha_data = await asyncio.gather(normal_data_task, ha_data_task)
            
            if not normal_data or not ha_data:
                _dataframe_cache[cache_key] = (None, None)
                return None, None
            
            # Convert to DataFrames
            normal_df = pd.DataFrame(normal_data)
            ha_df = pd.DataFrame(ha_data)
            
            # Set date as index
            normal_df['date'] = pd.to_datetime(normal_df['date'])
            ha_df['date'] = pd.to_datetime(ha_df['date'])
            
            normal_df = normal_df.set_index('date').sort_index()
            ha_df = ha_df.set_index('date').sort_index()
            
            # Calculate 50 EMA efficiently using numpy
            normal_df['ema_50'] = normal_df['close'].ewm(span=50, adjust=False).mean()
            
            # Filter to requested date range
            normal_df = normal_df[normal_df.index >= start_dt]
            
            # Validate data
            if len(normal_df) < 4 or len(ha_df) < 4:
                _dataframe_cache[cache_key] = (None, None)
                return None, None
            
            # Cache the result
            _dataframe_cache[cache_key] = (normal_df, ha_df)
            
            return normal_df, ha_df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            _dataframe_cache[cache_key] = (None, None)
            return None, None
    
    @staticmethod
    def clear_cache():
        """Clear the dataframe cache"""
        global _dataframe_cache
        _dataframe_cache.clear()
        logger.info("Dataframe cache cleared")
    
    @staticmethod
    def vectorized_backtest(
        normal_df: pd.DataFrame,
        ha_df: pd.DataFrame,
        initial_capital: float,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Vectorized backtesting logic for a single stock with 50 EMA filter
        
        Entry Signal:
        - Last 3 days of HA meet specific conditions
        - Price must be above 50 EMA on entry day
        
        Exit:
        - Stop loss hit
        - Trailing stop loss
        """
        
        # Convert to numpy arrays for speed
        ha_open = ha_df['ha_open'].values
        ha_low = ha_df['ha_low'].values
        ha_high = ha_df['ha_high'].values
        ha_close = ha_df['ha_close'].values
        ha_dates = ha_df.index.values
        
        normal_low = normal_df['low'].values
        normal_open = normal_df['open'].values
        normal_close = normal_df['close'].values
        normal_dates = normal_df.index.values
        
        # Get 50 EMA values
        ema_50 = normal_df['ema_50'].values if 'ema_50' in normal_df.columns else None
        
        # Create date to index mapping
        normal_date_to_idx = {date: idx for idx, date in enumerate(normal_dates)}
        
        trades = []
        cash_balance = initial_capital
        in_position = False
        entry_price = 0.0
        sl = 0.0
        qty = 0.0
        entry_date = None
        signal_dates = []
        rejected_signals = []
        
        # Main loop
        for i in range(3, len(ha_df)):
            date = ha_dates[i]
            
            # Check if date exists in normal_df
            if date not in normal_date_to_idx:
                continue
            
            normal_idx = normal_date_to_idx[date]
            
            if in_position:
                # Check for SL hit
                if normal_low[normal_idx] <= sl:
                    exit_price = sl
                    pnl = (exit_price - entry_price) * qty
                    # Return deployed capital plus P&L (matches main.py logic)
                    cash_balance += (entry_price * qty) + pnl
                    
                    trades.append({
                        "entry_date": entry_date,
                        "entry_price": float(entry_price),
                        "exit_date": str(pd.Timestamp(date).date()),
                        "exit_price": float(exit_price),
                        "qty": float(qty),
                        "pnl": float(pnl),
                        "exit_type": "SL hit"
                    })
                    in_position = False
                else:
                    # Trail SL
                    if ha_open[i] != ha_low[i]:
                        sl = max(sl, ha_low[i])
                continue
            
            # Check for entry signal
            cond_today = (ha_open[i] == ha_low[i] and ha_close[i] > ha_open[i])
            cond_yest = (ha_open[i-1] == ha_low[i-1] and ha_close[i-1] > ha_open[i-1])
            cond_yest2 = (ha_open[i-2] != ha_low[i-2] and ha_high[i-2] != ha_close[i-2] and ha_close[i-2] > ha_open[i-2])
            
            if cond_today and cond_yest and cond_yest2:
                # Entry on next day
                entry_idx = i + 1
                if entry_idx >= len(ha_df):
                    break
                
                entry_date_obj = ha_dates[entry_idx]
                if entry_date_obj not in normal_date_to_idx:
                    continue
                
                entry_normal_idx = normal_date_to_idx[entry_date_obj]
                
                # 50 EMA Filter
                if ema_50 is not None:
                    entry_close = normal_close[entry_normal_idx]
                    entry_ema_50 = ema_50[entry_normal_idx]
                    
                    if entry_close <= entry_ema_50:
                        rejected_signals.append({
                            'signal_date': str(pd.Timestamp(date).date()),
                            'entry_date': str(pd.Timestamp(entry_date_obj).date()),
                            'close': float(entry_close),
                            'ema_50': float(entry_ema_50),
                            'reason': 'Price below 50 EMA'
                        })
                        continue
                
                entry_date = str(pd.Timestamp(entry_date_obj).date())
                signal_dates.append(str(pd.Timestamp(date).date()))
                
                entry_price = normal_open[entry_normal_idx]
                sl = ha_low[i-2]
                qty = int(cash_balance / entry_price)
                
                if qty < 1:
                    continue
                
                # Deduct deployed capital
                entry_value = entry_price * qty
                cash_balance -= entry_value
                in_position = True
        
        # Close any open position at end
        if in_position:
            last_date = ha_dates[-1]
            if last_date in normal_date_to_idx:
                last_normal_idx = normal_date_to_idx[last_date]
                exit_price = normal_close[last_normal_idx]
                pnl = (exit_price - entry_price) * qty
                # Return deployed capital plus P&L (matches main.py logic)
                cash_balance += (entry_price * qty) + pnl
                
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": float(entry_price),
                    "exit_date": str(pd.Timestamp(last_date).date()),
                    "exit_price": float(exit_price),
                    "qty": float(qty),
                    "pnl": float(pnl),
                    "exit_type": "end of period"
                })
        
        final_capital = cash_balance
        stock_pnl = final_capital - initial_capital
        
        return {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "pnl": stock_pnl,
            "trades": trades,
            "signal_dates": signal_dates,
            "rejected_signals": rejected_signals,
            "ema_filter_stats": {
                "total_signals": len(signal_dates) + len(rejected_signals),
                "accepted_signals": len(signal_dates),
                "rejected_by_ema": len(rejected_signals),
                "ema_filter_rate": round(
                    len(rejected_signals) / (len(signal_dates) + len(rejected_signals)) * 100, 2
                ) if (len(signal_dates) + len(rejected_signals)) > 0 else 0
            }
        }
    
    @staticmethod
    def calculate_statistics(
        all_trades: List[Dict],
        daily_signals: Dict,
        stock_performance: Dict,
        total_initial: float,
        total_final: float,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive backtest statistics - FULL FEATURE SET"""
        
        stats = {}
        
        if not all_trades:
            return {
                "message": "No trades executed in the backtest period",
                "total_trades": 0
            }
        
        # Calculate CAGR
        if start_date and total_initial > 0:
            try:
                start_dt = pd.to_datetime(start_date)
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                else:
                    latest_exit = max(pd.to_datetime(t['exit_date']) for t in all_trades)
                    end_dt = latest_exit
                
                years = (end_dt - start_dt).days / 365.25
                
                if years > 0:
                    cagr = (((total_final / total_initial) ** (1 / years)) - 1) * 100
                    stats["cagr_pct"] = round(float(cagr), 2)
                else:
                    stats["cagr_pct"] = 0
            except Exception as e:
                logger.error(f"Error calculating CAGR: {e}")
                stats["cagr_pct"] = 0
        else:
            stats["cagr_pct"] = 0
        
        # Convert to numpy arrays
        pnls = np.array([t['pnl'] for t in all_trades])
        returns = np.array([t['return_pct'] for t in all_trades])
        holding_days = np.array([t['holding_days'] for t in all_trades])
        
        # 1. Daily Signal Statistics
        signal_counts = [len(stocks) for stocks in daily_signals.values()]
        if signal_counts:
            stats["avg_stocks_signaled_per_day"] = round(float(np.mean(signal_counts)), 2)
            stats["max_stocks_on_single_day"] = int(max(signal_counts))
            stats["min_stocks_on_single_day"] = int(min(signal_counts))
            stats["median_stocks_per_day"] = round(float(np.median(signal_counts)), 2)
            stats["total_signal_days"] = len(daily_signals)
            
            # Find day with most signals
            max_signal_day = max(daily_signals.items(), key=lambda x: len(x[1]))
            stats["day_with_most_signals"] = {
                "date": max_signal_day[0],
                "count": len(max_signal_day[1]),
                "stocks": max_signal_day[1]
            }
        
        # 2. Win/Loss Statistics
        winning_mask = pnls > 0
        winning_pnls = pnls[winning_mask]
        losing_pnls = pnls[~winning_mask]
        
        stats["total_trades"] = len(all_trades)
        stats["winning_trades"] = int(winning_mask.sum())
        stats["losing_trades"] = int((~winning_mask).sum())
        stats["win_rate_pct"] = round(float(winning_mask.mean() * 100), 2)
        
        # 3. Profit/Loss Analysis
        total_profit = float(winning_pnls.sum())
        total_loss = float(abs(losing_pnls.sum()))
        
        stats["total_profit"] = round(total_profit, 2)
        stats["total_loss"] = round(total_loss, 2)
        stats["net_pnl"] = round(total_profit - total_loss, 2)
        stats["profit_factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else float('inf')
        
        # 4. Average Performance
        stats["avg_profit_per_winning_trade"] = round(float(winning_pnls.mean()), 2) if len(winning_pnls) > 0 else 0
        stats["avg_loss_per_losing_trade"] = round(float(abs(losing_pnls.mean())), 2) if len(losing_pnls) > 0 else 0
        stats["avg_pnl_per_trade"] = round(float(pnls.mean()), 2)
        stats["avg_return_pct_per_trade"] = round(float(returns.mean()), 2)
        
        # 5. Holding Period
        stats["avg_holding_days"] = round(float(holding_days.mean()), 2)
        stats["max_holding_days"] = int(holding_days.max())
        stats["min_holding_days"] = int(holding_days.min())
        stats["median_holding_days"] = round(float(np.median(holding_days)), 2)
        
        # 6. Best/Worst Trades
        best_idx = pnls.argmax()
        worst_idx = pnls.argmin()
        
        stats["best_trade"] = {
            "symbol": all_trades[best_idx]['symbol'],
            "entry_date": all_trades[best_idx]['entry_date'],
            "exit_date": all_trades[best_idx]['exit_date'],
            "pnl": round(all_trades[best_idx]['pnl'], 2),
            "return_pct": round(all_trades[best_idx]['return_pct'], 2)
        }
        
        stats["worst_trade"] = {
            "symbol": all_trades[worst_idx]['symbol'],
            "entry_date": all_trades[worst_idx]['entry_date'],
            "exit_date": all_trades[worst_idx]['exit_date'],
            "pnl": round(all_trades[worst_idx]['pnl'], 2),
            "return_pct": round(all_trades[worst_idx]['return_pct'], 2)
        }
        
        # 7. Stock Performance Analysis
        if stock_performance:
            stock_perf_list = list(stock_performance.items())
            best_stock = max(stock_perf_list, key=lambda x: x[1]['return_pct'])
            worst_stock = min(stock_perf_list, key=lambda x: x[1]['return_pct'])
            
            stats["best_performing_stock"] = {
                "symbol": best_stock[0],
                "return_pct": round(best_stock[1]['return_pct'], 2),
                "pnl": round(best_stock[1]['pnl'], 2),
                "num_trades": best_stock[1]['num_trades'],
                "win_rate": round(best_stock[1]['win_rate'], 2)
            }
            
            stats["worst_performing_stock"] = {
                "symbol": worst_stock[0],
                "return_pct": round(worst_stock[1]['return_pct'], 2),
                "pnl": round(worst_stock[1]['pnl'], 2),
                "num_trades": worst_stock[1]['num_trades'],
                "win_rate": round(worst_stock[1]['win_rate'], 2)
            }
            
            # Stocks with highest win rate (min 3 trades)
            high_trade_stocks = {k: v for k, v in stock_performance.items() if v['num_trades'] >= 3}
            if high_trade_stocks:
                best_wr_stock = max(high_trade_stocks.items(), key=lambda x: x[1]['win_rate'])
                stats["highest_win_rate_stock"] = {
                    "symbol": best_wr_stock[0],
                    "win_rate": round(best_wr_stock[1]['win_rate'], 2),
                    "num_trades": best_wr_stock[1]['num_trades'],
                    "return_pct": round(best_wr_stock[1]['return_pct'], 2)
                }
        
        # 8. Monthly Performance Analysis
        monthly_pnl = defaultdict(float)
        monthly_trades = defaultdict(int)
        for trade in all_trades:
            month = trade['exit_date'][:7]  # YYYY-MM
            monthly_pnl[month] += trade['pnl']
            monthly_trades[month] += 1
        
        if monthly_pnl:
            best_month = max(monthly_pnl.items(), key=lambda x: x[1])
            worst_month = min(monthly_pnl.items(), key=lambda x: x[1])
            
            stats["best_month"] = {
                "month": best_month[0],
                "pnl": round(best_month[1], 2),
                "num_trades": monthly_trades[best_month[0]]
            }
            
            stats["worst_month"] = {
                "month": worst_month[0],
                "pnl": round(worst_month[1], 2),
                "num_trades": monthly_trades[worst_month[0]]
            }
            
            monthly_values = list(monthly_pnl.values())
            stats["avg_monthly_pnl"] = round(float(np.mean(monthly_values)), 2)
            stats["total_profitable_months"] = sum(1 for pnl in monthly_values if pnl > 0)
            stats["total_losing_months"] = sum(1 for pnl in monthly_values if pnl <= 0)
        
        # 9. Risk Metrics
        stats["std_deviation_returns"] = round(float(returns.std()), 2)
        stats["max_return_pct"] = round(float(returns.max()), 2)
        stats["min_return_pct"] = round(float(returns.min()), 2)
        
        if len(returns) > 1 and returns.std() > 0:
            avg_return = returns.mean()
            sharpe = (avg_return / returns.std()) * np.sqrt(252 / len(all_trades))
            stats["sharpe_ratio_approx"] = round(float(sharpe), 2)
        else:
            stats["sharpe_ratio_approx"] = 0
        
        # 10. Exit Type Breakdown
        exit_types = defaultdict(int)
        exit_type_pnl = defaultdict(float)
        for trade in all_trades:
            exit_type = trade.get('exit_type', 'unknown')
            exit_types[exit_type] += 1
            exit_type_pnl[exit_type] += trade['pnl']
        
        stats["exit_type_breakdown"] = {
            exit_type: {
                "count": count,
                "pct": round(count / len(all_trades) * 100, 2),
                "total_pnl": round(exit_type_pnl[exit_type], 2),
                "avg_pnl": round(exit_type_pnl[exit_type] / count, 2)
            }
            for exit_type, count in exit_types.items()
        }
        
        # 11. Consecutive Wins/Losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        # Sort trades by exit date for streak analysis
        sorted_trades = sorted(all_trades, key=lambda x: x['exit_date'])
        for trade in sorted_trades:
            if trade['pnl'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        stats["max_consecutive_wins"] = max_consecutive_wins
        stats["max_consecutive_losses"] = max_consecutive_losses
        
        # 12. Return on Capital
        stats["return_on_capital_pct"] = round(
            (total_final - total_initial) / total_initial * 100, 2
        ) if total_initial > 0 else 0
        
        # 13. Capital Efficiency
        capital_deployed = np.array([t['entry_price'] * t['qty'] for t in all_trades])
        stats["avg_capital_deployed_per_trade"] = round(float(capital_deployed.mean()), 2)
        
        # 14. Max Drawdown
        cumulative_pnl = np.cumsum([t['pnl'] for t in sorted_trades])
        
        if len(cumulative_pnl) > 0:
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = float(drawdown.max())
            
            stats["max_drawdown"] = round(max_drawdown, 2)
            stats["max_drawdown_pct"] = round(
                (max_drawdown / total_initial * 100), 2
            ) if total_initial > 0 else 0
        
        return stats
