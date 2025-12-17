"""
Advanced backtesting strategies:
- T-Score calculation
- HA + T-Score strategy
- Smart portfolio with ranking
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import logging
from database import get_database
from models import IntervalType
from services.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)


def calculate_t_score(normal_df: pd.DataFrame, current_date: pd.Timestamp) -> Optional[float]:
    """
    Calculate T-Score for a stock at a given date.
    
    T-Score Components:
    - SMA21: 1 point if close > 21-day SMA
    - SMA50: 1 point if close > 50-day SMA
    - SMA200: 1 point if close > 200-day SMA
    - closerange7d: 2 points if close > max of last 7 days
    - HR_52 (52-week high range): 0-3 points based on proximity to 52-week high
    
    Returns T-Score percentage (0-100%)
    """
    try:
        if current_date not in normal_df.index:
            return None
        
        # Get current index position
        current_idx = normal_df.index.get_loc(current_date)
        
        # Need at least 21 days of data for minimum calculation
        if current_idx < 21:
            return None
        
        # Get price data
        close_values = normal_df['close'].values
        high_values = normal_df['high'].values
        current_close = close_values[current_idx]
        
        # Initialize scores
        SMA21, SMA50, SMA200 = 0, 0, 0
        closerange7d = 0
        HR_52 = 0
        max_score = 6  # Base: SMA21(1) + closerange7d(2) + HR_52(3)
        
        # Calculate SMA21 (21-day Simple Moving Average)
        if current_idx >= 20:
            DMA_21 = np.mean(close_values[current_idx - 20:current_idx + 1])
            SMA21 = 1 if current_close > DMA_21 else 0
        
        # Calculate SMA50 (50-day Simple Moving Average)
        if current_idx >= 49:
            DMA_50 = np.mean(close_values[current_idx - 49:current_idx + 1])
            SMA50 = 1 if current_close > DMA_50 else 0
            max_score += 1
        
        # Calculate SMA200 (200-day Simple Moving Average)
        if current_idx >= 199:
            DMA_200 = np.mean(close_values[current_idx - 199:current_idx + 1])
            SMA200 = 1 if current_close > DMA_200 else 0
            max_score += 1
        
        # Calculate 7-day close range (2 points if current close > max of last 7 days)
        if current_idx >= 6:
            max_7d_close = np.max(close_values[current_idx - 6:current_idx])
            closerange7d = 2 if current_close > max_7d_close else 0
        
        # Calculate 52-week high range (HR_52)
        # Look back 252 trading days (approximately 1 year)
        lookback_days = min(252, current_idx)
        if lookback_days > 0:
            week_52_high = np.max(high_values[current_idx - lookback_days:current_idx + 1])
            wh_percentage_52 = ((week_52_high - current_close) / week_52_high) * 100
            
            # Score based on distance from 52-week high
            if wh_percentage_52 >= 40:
                HR_52 = 0
            elif 30 <= wh_percentage_52 < 40:
                HR_52 = 0.5
            elif 20 <= wh_percentage_52 < 30:
                HR_52 = 1
            elif 10 <= wh_percentage_52 < 20:
                HR_52 = 2
            else:  # < 10%
                HR_52 = 3
        
        # Calculate total T-Score
        tscore_total = SMA21 + SMA50 + SMA200 + closerange7d + HR_52
        t_score_percentage = (tscore_total / max_score) * 100
        
        return round(t_score_percentage, 2)
        
    except Exception as e:
        logger.error(f"Error calculating T-Score: {e}")
        return None


async def rank_stocks_for_selection(
    symbols: List[str],
    start_date: str,
    end_date: Optional[str],
    interval: IntervalType,
    ranking_period_months: int = 6
) -> List[Dict[str, Any]]:
    """
    Intelligent stock ranking algorithm for portfolio selection.
    
    Ranks stocks based on:
    - Win rate
    - Average return per trade
    - Sharpe ratio
    - Profit factor
    - Recovery rate from losses
    """
    
    backtest_start = pd.to_datetime(start_date)
    ranking_end = backtest_start
    ranking_start = (ranking_end - pd.DateOffset(months=ranking_period_months)).strftime('%Y-%m-%d')
    
    ranking_capital = 100000
    
    stock_metrics = []
    
    logger.info(f"Ranking {len(symbols)} stocks from {ranking_start} to {ranking_end.strftime('%Y-%m-%d')}")
    
    for symbol in symbols:
        try:
            # Load data for ranking period
            normal_df, ha_df = await BacktestEngine.load_stock_data(
                symbol,
                interval,
                ranking_start,
                ranking_end.strftime('%Y-%m-%d')
            )
            
            if normal_df is None or ha_df is None:
                continue
            
            # Run backtest for ranking period
            result = BacktestEngine.vectorized_backtest(
                normal_df,
                ha_df,
                ranking_capital,
                symbol
            )
            
            if not result['trades']:
                continue
            
            trades = result['trades']
            pnl = result['pnl']
            
            win_trades = [t for t in trades if t['pnl'] > 0]
            loss_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
            return_pct = (pnl / ranking_capital * 100) if ranking_capital > 0 else 0
            avg_return_per_trade = return_pct / len(trades) if len(trades) > 0 else 0
            
            returns = [(t['exit_price'] - t['entry_price']) / t['entry_price'] * 100 for t in trades]
            std_dev = np.std(returns) if len(returns) > 1 else 0
            sharpe_like = (avg_return_per_trade / std_dev) if std_dev > 0 else 0
            
            total_profit = sum(t['pnl'] for t in win_trades)
            total_loss = abs(sum(t['pnl'] for t in loss_trades))
            profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
            
            days_in_period = (ranking_end - pd.to_datetime(ranking_start)).days
            trades_per_month = len(trades) / (days_in_period / 30) if days_in_period > 0 else 0
            
            # Calculate recovery rate
            consecutive_losses = 0
            max_consecutive_losses = 0
            recovery_count = 0
            
            sorted_trades = sorted(trades, key=lambda x: x['exit_date'])
            for trade in sorted_trades:
                if trade['pnl'] <= 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    if consecutive_losses > 0:
                        recovery_count += 1
                    consecutive_losses = 0
            
            recovery_rate = recovery_count / max(max_consecutive_losses, 1) if max_consecutive_losses > 0 else 1
            
            # Calculate composite score
            score = (
                win_rate * 30 +
                min(avg_return_per_trade, 10) * 2.5 +
                min(sharpe_like, 3) * 8.33 +
                min(trades_per_month, 2) * 5 +
                recovery_rate * 10
            )
            
            stock_metrics.append({
                'symbol': symbol,
                'score': round(score, 2),
                'win_rate': round(win_rate * 100, 2),
                'return_pct': round(return_pct, 2),
                'avg_return_per_trade': round(avg_return_per_trade, 2),
                'sharpe_like': round(sharpe_like, 2),
                'profit_factor': round(profit_factor, 2),
                'num_trades': len(trades),
                'trades_per_month': round(trades_per_month, 2),
                'recovery_rate': round(recovery_rate * 100, 2),
                'std_dev': round(std_dev, 2),
                'max_consecutive_losses': max_consecutive_losses
            })
            
        except Exception as e:
            logger.error(f"Error ranking {symbol}: {e}")
            continue
    
    stock_metrics.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Ranking complete. Top 10: {[s['symbol'] for s in stock_metrics[:10]]}")
    
    return stock_metrics
