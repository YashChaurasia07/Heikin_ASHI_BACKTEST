from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import uvicorn
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing

app = FastAPI(title="Heikin Ashi Backtest Server", description="API for backtesting the specified strategy on historical data")

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Your React dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        # Add production origins, e.g., "https://yourdomain.com"
        # For all origins (development only): ["*"]
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Global constants
DEFAULT_INITIAL_CAPITAL_PER_STOCK = 50000
DEFAULT_START_DATE = '2022-06-01'
DATA_FOLDER = 'data'
HA_FOLDER = 'heikin_ashi'

# Cache for loaded dataframes
_dataframe_cache = {}

def load_and_filter_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and filter data with caching"""
    cache_key = (symbol, start_date, end_date)
    
    if cache_key in _dataframe_cache:
        return _dataframe_cache[cache_key]
    
    normal_path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
    ha_path = os.path.join(HA_FOLDER, f"{symbol}.csv")
    
    if not os.path.exists(ha_path) or not os.path.exists(normal_path):
        _dataframe_cache[cache_key] = (None, None)
        return None, None
    
    try:
        # Load normal data
        normal_df = pd.read_csv(normal_path)
        normal_df['Date'] = pd.to_datetime(normal_df['Date'])
        
        # Load HA data
        ha_df = pd.read_csv(ha_path)
        ha_df['Date'] = pd.to_datetime(ha_df['Date'])
        
        # Filter by date range (do it before sorting to reduce data)
        normal_df = normal_df[(normal_df['Date'] >= start_date)]
        ha_df = ha_df[(ha_df['Date'] >= start_date)]
        
        if end_date:
            normal_df = normal_df[normal_df['Date'] <= end_date]
            ha_df = ha_df[ha_df['Date'] <= end_date]
        
        # Sort and set index
        normal_df = normal_df.sort_values('Date').set_index('Date')
        ha_df = ha_df.sort_values('Date').set_index('Date')
        
        if len(ha_df) < 4 or len(normal_df) < 4:
            _dataframe_cache[cache_key] = (None, None)
            return None, None
        
        _dataframe_cache[cache_key] = (normal_df, ha_df)
        return normal_df, ha_df
        
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        _dataframe_cache[cache_key] = (None, None)
        return None, None


def vectorized_backtest(normal_df: pd.DataFrame, ha_df: pd.DataFrame, initial_capital: float, symbol: str) -> Dict[str, Any]:
    """Vectorized backtesting logic for a single stock"""
    
    # Convert to numpy arrays for faster access
    ha_open = ha_df['ha_open'].values
    ha_low = ha_df['ha_low'].values
    ha_high = ha_df['ha_high'].values
    ha_close = ha_df['ha_close'].values
    ha_dates = ha_df.index.values
    
    normal_low = normal_df['low'].values
    normal_open = normal_df['open'].values
    normal_close = normal_df['close'].values
    normal_dates = normal_df.index.values
    
    # Create date to index mapping for normal_df
    normal_date_to_idx = {date: idx for idx, date in enumerate(normal_dates)}
    
    trades = []
    current_cap = initial_capital
    in_position = False
    entry_price = 0.0
    sl = 0.0
    qty = 0.0
    entry_date = None
    signal_dates = []
    
    # Vectorize condition checks where possible
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
                current_cap += pnl
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
                # Check for SL revision (trailing up if condition met)
                if ha_open[i] != ha_low[i]:
                    sl = max(sl, ha_low[i])
            continue
        
        # Check for entry signal: last 3 days (i-2, i-1, i) meet conditions
        cond_today = (ha_open[i] == ha_low[i] and ha_close[i] > ha_open[i])
        cond_yest = (ha_open[i-1] == ha_low[i-1] and ha_close[i-1] > ha_open[i-1])
        cond_yest2 = (ha_open[i-2] != ha_low[i-2] and ha_high[i-2] != ha_close[i-2] and ha_close[i-2] > ha_open[i-2])
        
        if cond_today and cond_yest and cond_yest2:
            # Entry on next day (i+1)
            entry_idx = i + 1
            if entry_idx >= len(ha_df):
                break
            
            entry_date_obj = ha_dates[entry_idx]
            if entry_date_obj not in normal_date_to_idx:
                continue
            
            entry_normal_idx = normal_date_to_idx[entry_date_obj]
            entry_date = str(pd.Timestamp(entry_date_obj).date())
            signal_dates.append(str(pd.Timestamp(date).date()))
            
            entry_price = normal_open[entry_normal_idx]
            sl = ha_low[i-2]
            # Buy whole shares only (integer quantity)
            qty = int(current_cap / entry_price)
            
            # Skip if we can't afford even 1 share
            if qty < 1:
                continue
            
            in_position = True
    
    # Close any open position at end
    if in_position:
        last_date = ha_dates[-1]
        if last_date in normal_date_to_idx:
            last_normal_idx = normal_date_to_idx[last_date]
            exit_price = normal_close[last_normal_idx]
            pnl = (exit_price - entry_price) * qty
            current_cap += pnl
            trades.append({
                "entry_date": entry_date,
                "entry_price": float(entry_price),
                "exit_date": str(pd.Timestamp(last_date).date()),
                "exit_price": float(exit_price),
                "qty": float(qty),
                "pnl": float(pnl),
                "exit_type": "end of period"
            })
    
    stock_pnl = current_cap - initial_capital
    
    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_capital": current_cap,
        "pnl": stock_pnl,
        "trades": trades,
        "signal_dates": signal_dates
    }


def process_single_stock(args: Tuple[str, str, Optional[str], float]) -> Optional[Dict[str, Any]]:
    """Process a single stock - used for parallel processing"""
    symbol, start_date, end_date, initial_capital = args
    
    normal_df, ha_df = load_and_filter_data(symbol, start_date, end_date)
    
    if normal_df is None or ha_df is None:
        return None
    
    try:
        result = vectorized_backtest(normal_df, ha_df, initial_capital, symbol)
        return result
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None


def calculate_statistics(all_trades: List[Dict], daily_signals: Dict, stock_performance: Dict, 
                         total_initial: float, total_final: float) -> Dict[str, Any]:
    """Calculate comprehensive backtest statistics - optimized version"""
    
    stats = {}
    
    if not all_trades:
        return {
            "message": "No trades executed in the backtest period",
            "total_trades": 0
        }
    
    # Convert to numpy arrays for faster calculations
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
        
        # Find the day with most signals
        max_signal_day = max(daily_signals.items(), key=lambda x: len(x[1]))
        stats["day_with_most_signals"] = {
            "date": max_signal_day[0],
            "count": len(max_signal_day[1]),
            "stocks": max_signal_day[1]
        }
    
    # 2. Trade Win/Loss Statistics (vectorized)
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
    
    # 4. Average Trade Performance
    stats["avg_profit_per_winning_trade"] = round(float(winning_pnls.mean()), 2) if len(winning_pnls) > 0 else 0
    stats["avg_loss_per_losing_trade"] = round(float(abs(losing_pnls.mean())), 2) if len(losing_pnls) > 0 else 0
    stats["avg_pnl_per_trade"] = round(float(pnls.mean()), 2)
    stats["avg_return_pct_per_trade"] = round(float(returns.mean()), 2)
    
    # 5. Holding Period Analysis
    stats["avg_holding_days"] = round(float(holding_days.mean()), 2)
    stats["max_holding_days"] = int(holding_days.max())
    stats["min_holding_days"] = int(holding_days.min())
    stats["median_holding_days"] = round(float(np.median(holding_days)), 2)
    
    # 6. Best and Worst Trades
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
    
    # 9. Risk Metrics (vectorized)
    stats["std_deviation_returns"] = round(float(returns.std()), 2)
    stats["max_return_pct"] = round(float(returns.max()), 2)
    stats["min_return_pct"] = round(float(returns.min()), 2)
    
    # Sharpe Ratio approximation (assuming 252 trading days and 0% risk-free rate)
    if len(returns) > 1 and returns.std() > 0:
        avg_return = returns.mean()
        sharpe = (avg_return / returns.std()) * np.sqrt(252 / len(all_trades))
        stats["sharpe_ratio_approx"] = round(float(sharpe), 2)
    else:
        stats["sharpe_ratio_approx"] = 0
    
    # 10. Exit Type Analysis
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
    
    # 12. Capital Efficiency
    stats["return_on_capital_pct"] = round((total_final - total_initial) / total_initial * 100, 2) if total_initial > 0 else 0
    
    capital_deployed = np.array([t['entry_price'] * t['qty'] for t in all_trades])
    stats["avg_capital_deployed_per_trade"] = round(float(capital_deployed.mean()), 2)
    
    # 13. Recovery Analysis (largest drawdown recovery)
    cumulative_pnl = np.cumsum([t['pnl'] for t in sorted_trades])
    
    if len(cumulative_pnl) > 0:
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        max_drawdown = float(drawdown.max())
        
        stats["max_drawdown"] = round(max_drawdown, 2)
        stats["max_drawdown_pct"] = round((max_drawdown / total_initial * 100), 2) if total_initial > 0 else 0
    
    return stats

@app.get("/")
def root():
    return {"message": "Heikin Ashi Strategy Backtest Server is running. Use /backtest for results, /symbols for list of stocks."}

@app.get("/symbols")
def get_symbols() -> Dict[str, List[str]]:
    if not os.path.exists(DATA_FOLDER):
        return {"symbols": []}
    symbols = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    return {"symbols": symbols}

@app.get("/data/{symbol}")
def get_data(symbol: str) -> List[Dict[str, Any]]:
    path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": f"Data for {symbol} not found."})
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    # Assume columns: Date, open, high, low, close
    return df[['Date', 'open', 'high', 'low', 'close']].to_dict('records')

@app.get("/backtest")
def run_backtest(start_date: str = Query(DEFAULT_START_DATE, description="Start date YYYY-MM-DD"),
                 end_date: str = Query(None, description="End date YYYY-MM-DD (optional, defaults to latest data)"),
                 initial_capital: float = Query(DEFAULT_INITIAL_CAPITAL_PER_STOCK, description="Initial capital per stock"),
                 use_parallel: bool = Query(True, description="Use parallel processing for faster execution")) -> Dict[str, Any]:
    
    if not os.path.exists(DATA_FOLDER) or not os.path.exists(HA_FOLDER):
        return JSONResponse(status_code=404, content={"error": "Data or Heikin Ashi folders not found."})
    
    # Clear cache for fresh data
    global _dataframe_cache
    _dataframe_cache.clear()
    
    symbols = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    
    # Statistics tracking
    all_trades = []
    daily_signals = defaultdict(list)
    stock_performance = {}
    stocks_results = {}
    
    if use_parallel:
        # Parallel processing with ProcessPoolExecutor
        max_workers = min(multiprocessing.cpu_count(), len(symbols))
        args_list = [(symbol, start_date, end_date, initial_capital) for symbol in symbols]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(process_single_stock, args): args[0] for args in args_list}
            
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result is None:
                    continue
                
                symbol = result['symbol']
                
                # Process result
                trades = result['trades']
                signal_dates = result['signal_dates']
                
                # Track signals
                for sig_date in signal_dates:
                    daily_signals[sig_date].append(symbol)
                
                # Stock performance
                win_trades = [t for t in trades if t['pnl'] > 0]
                loss_trades = [t for t in trades if t['pnl'] <= 0]
                
                stock_performance[symbol] = {
                    "pnl": result['pnl'],
                    "return_pct": (result['pnl'] / initial_capital * 100) if initial_capital > 0 else 0,
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
                    "trades": trades
                }
                
                # Add trades to all_trades for global statistics
                for trade in trades:
                    entry_date = pd.to_datetime(trade['entry_date'])
                    exit_date = pd.to_datetime(trade['exit_date'])
                    holding_days = (exit_date - entry_date).days
                    return_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
                    
                    all_trades.append({
                        **trade,
                        "symbol": symbol,
                        "return_pct": return_pct,
                        "holding_days": holding_days
                    })
    else:
        # Sequential processing (fallback)
        for symbol in symbols:
            result = process_single_stock((symbol, start_date, end_date, initial_capital))
            
            if result is None:
                continue
            
            # Process result (same as above)
            trades = result['trades']
            signal_dates = result['signal_dates']
            
            for sig_date in signal_dates:
                daily_signals[sig_date].append(symbol)
            
            win_trades = [t for t in trades if t['pnl'] > 0]
            loss_trades = [t for t in trades if t['pnl'] <= 0]
            
            stock_performance[symbol] = {
                "pnl": result['pnl'],
                "return_pct": (result['pnl'] / initial_capital * 100) if initial_capital > 0 else 0,
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
                "trades": trades
            }
            
            for trade in trades:
                entry_date = pd.to_datetime(trade['entry_date'])
                exit_date = pd.to_datetime(trade['exit_date'])
                holding_days = (exit_date - entry_date).days
                return_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
                
                all_trades.append({
                    **trade,
                    "symbol": symbol,
                    "return_pct": return_pct,
                    "holding_days": holding_days
                })
    
    # Calculate totals
    total_initial = len(stocks_results) * initial_capital
    total_final = sum(result['final_capital'] for result in stocks_results.values())
    
    # Calculate comprehensive statistics
    statistics = calculate_statistics(all_trades, daily_signals, stock_performance, total_initial, total_final)
    
    total_pnl = total_final - total_initial
    portfolio = {
        "total_initial_capital": total_initial,
        "total_final_capital": total_final,
        "total_pnl": total_pnl,
        "total_return_pct": (total_pnl / total_initial * 100) if total_initial > 0 else 0,
        "num_stocks_processed": len(stocks_results),
        "stocks": stocks_results,
        "params": {
            "start_date": start_date, 
            "end_date": end_date, 
            "initial_capital": initial_capital,
            "strategy": "heikin_ashi",
            "parallel_processing": use_parallel
        },
        "statistics": statistics
    }
    
    return portfolio


@app.get("/ha_data/{symbol}")
def get_ha_data(symbol: str) -> List[Dict[str, Any]]:
    path = os.path.join(HA_FOLDER, f"{symbol}.csv")
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": f"Heikin Ashi data for {symbol} not found."})
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    # Assume columns: Date, ha_open, ha_high, ha_low, ha_close
    return df[['Date', 'ha_open', 'ha_high', 'ha_low', 'ha_close']].to_dict('records')


def rank_stocks_for_selection(symbols: List[str], start_date: str, end_date: Optional[str], 
                               ranking_period_months: int = 6) -> List[Dict[str, Any]]:
    """
    Intelligent stock ranking algorithm for portfolio selection.
    
    Strategy:
    1. Run a preliminary backtest on all stocks for a ranking period
    2. Calculate comprehensive metrics for each stock
    3. Score stocks based on multiple factors:
       - Win rate (consistency)
       - Average return per trade (profitability)
       - Risk-adjusted returns (Sharpe-like ratio)
       - Trade frequency (sufficient opportunities)
       - Recovery ability (bounce back from losses)
       - Profit factor (reward/risk ratio)
    4. Return ranked list of stocks
    
    Args:
        symbols: List of stock symbols to analyze
        start_date: Start date for analysis
        end_date: End date for analysis
        ranking_period_months: Months of historical data to use for ranking (default 6)
    
    Returns:
        List of dicts with symbol, score, and metrics
    """
    
    # Calculate ranking period: Use 1 year BEFORE the user's requested start_date
    # This ensures we rank stocks based on their performance BEFORE the backtest period
    backtest_start = pd.to_datetime(start_date)
    ranking_end = backtest_start  # End ranking period at the backtest start date
    ranking_start = (ranking_end - pd.DateOffset(months=ranking_period_months)).strftime('%Y-%m-%d')
    
    # Use a standard capital for ranking to ensure fair comparison
    ranking_capital = 100000
    
    stock_metrics = []
    
    print(f"Analyzing {len(symbols)} stocks from {ranking_start} to {ranking_end.strftime('%Y-%m-%d')}...")
    print(f"(Ranking period: {ranking_period_months} months BEFORE backtest start date)")
    
    for symbol in symbols:
        # Use ranking_end as the end date for ranking period (which is the backtest start_date)
        result = process_single_stock((symbol, ranking_start, ranking_end.strftime('%Y-%m-%d'), ranking_capital))
        
        if result is None or len(result['trades']) == 0:
            continue
        
        trades = result['trades']
        pnl = result['pnl']
        
        # Calculate metrics
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
        
        # Return metrics
        return_pct = (pnl / ranking_capital * 100) if ranking_capital > 0 else 0
        avg_return_per_trade = return_pct / len(trades) if len(trades) > 0 else 0
        
        # Risk metrics
        returns = [(t['exit_price'] - t['entry_price']) / t['entry_price'] * 100 for t in trades]
        std_dev = np.std(returns) if len(returns) > 1 else 0
        sharpe_like = (avg_return_per_trade / std_dev) if std_dev > 0 else 0
        
        # Profit factor
        total_profit = sum(t['pnl'] for t in win_trades)
        total_loss = abs(sum(t['pnl'] for t in loss_trades))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        # Trade frequency (opportunities)
        days_in_period = (ranking_end - pd.to_datetime(ranking_start)).days
        trades_per_month = len(trades) / (days_in_period / 30) if days_in_period > 0 else 0
        
        # Recovery metric (ability to recover from losses)
        consecutive_losses = 0
        max_consecutive_losses = 0
        recovery_count = 0
        
        sorted_trades = sorted(trades, key=lambda x: x['exit_date'])
        for i, trade in enumerate(sorted_trades):
            if trade['pnl'] <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                if consecutive_losses > 0:
                    recovery_count += 1
                consecutive_losses = 0
        
        recovery_rate = recovery_count / max(max_consecutive_losses, 1) if max_consecutive_losses > 0 else 1
        
        # Calculate composite score (weighted)
        # Weights can be adjusted based on preference
        score = (
            win_rate * 30 +  # 30% weight on consistency
            min(avg_return_per_trade, 10) * 2.5 +  # 25% weight on profitability (capped at 10%)
            min(sharpe_like, 3) * 8.33 +  # 25% weight on risk-adjusted returns (capped at 3)
            min(trades_per_month, 2) * 5 +  # 10% weight on trade frequency (capped at 2/month)
            recovery_rate * 10  # 10% weight on recovery ability
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
    
    # Sort by score (descending)
    stock_metrics.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Ranking complete. Top 10 stocks:")
    for i, stock in enumerate(stock_metrics[:10], 1):
        print(f"{i}. {stock['symbol']} - Score: {stock['score']}, Win Rate: {stock['win_rate']}%, Return: {stock['return_pct']}%")
    
    return stock_metrics


def dynamic_portfolio_backtest(
    all_symbols: List[str],
    total_capital: float,
    max_positions: int,
    start_date: str,
    end_date: Optional[str],
    ranking_period_months: int = 6,
    use_ranking: bool = True
) -> Dict[str, Any]:
    """
    Dynamic portfolio management backtest.
    
    Instead of selecting fixed stocks upfront, this maintains a rolling portfolio:
    - Maximum N positions at any time
    - All stocks compete for entry when signals appear
    - When a position exits, capital becomes available for new entries
    - Prioritizes best-performing stocks for entry using composite scoring
    - Compounds profits by reinvesting available capital
    
    Args:
        all_symbols: All available stock symbols
        total_capital: Total portfolio capital
        max_positions: Maximum concurrent positions
        start_date: Backtest start date
        end_date: Backtest end date
        ranking_period_months: Months of historical data for stock ranking
        use_ranking: Whether to use intelligent ranking (True) or simple signal count (False)
    
    Returns:
        Comprehensive backtest results
    """
    
    # Note: capital_per_position will be calculated dynamically based on available capital
    
    # Step 1: Rank stocks using intelligent scoring algorithm
    stock_rankings = {}  # {symbol: score}
    if use_ranking:
        print(f"\n{'='*60}")
        print(f"PHASE 1: INTELLIGENT STOCK RANKING")
        print(f"{'='*60}")
        ranked_stocks = rank_stocks_for_selection(
            symbols=all_symbols,
            start_date=start_date,
            end_date=end_date,
            ranking_period_months=ranking_period_months
        )
        
               # Create lookup dict for scores
        for stock in ranked_stocks:
            stock_rankings[stock['symbol']] = stock['score']
        
        print(f"Ranked {len(stock_rankings)} stocks with valid trading history")
        print(f"{'='*60}\n")
    else:
        # Fallback: all stocks get equal score
        for symbol in all_symbols:
            stock_rankings[symbol] = 1.0
    
    # Load all stock data
    print(f"\n{'='*60}")
    print(f"PHASE 2: LOADING STOCK DATA")
    print(f"{'='*60}")
    print(f"Loading data for {len(all_symbols)} stocks...")
    stock_data_cache = {}
    for symbol in all_symbols:
        normal_df, ha_df = load_and_filter_data(symbol, start_date, end_date)
        if normal_df is not None and ha_df is not None and len(ha_df) >= 4:
            stock_data_cache[symbol] = (normal_df, ha_df)
    
    print(f"Loaded {len(stock_data_cache)} valid stocks")
    
    # Get all unique dates across all stocks
    all_dates = set()
    for normal_df, ha_df in stock_data_cache.values():
        all_dates.update(ha_df.index)
    
    all_dates = sorted(list(all_dates))
    print(f"Backtest period: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")
    print(f"{'='*60}\n")
    
    # Portfolio state with proper capital tracking
    active_positions = {}  # {symbol: position_info}
    available_capital = total_capital  # Tracks free capital available for new positions
    total_portfolio_value = total_capital  # Total portfolio value (deployed + available)
    all_trades = []
    entry_signals_by_date = defaultdict(list)  # Track signals for statistics
    capital_utilization_history = []  # Track capital utilization over time
    
    # Pre-calculate all signals for all stocks
    print(f"{'='*60}")
    print(f"PHASE 3: CALCULATING ENTRY SIGNALS")
    print(f"{'='*60}")
    print("Calculating entry signals for all stocks...")
    stock_signals = {}  # {symbol: [(date, entry_details), ...]}
    
    for symbol, (normal_df, ha_df) in stock_data_cache.items():
        signals = []
        
        ha_open = ha_df['ha_open'].values
        ha_low = ha_df['ha_low'].values
        ha_high = ha_df['ha_high'].values
        ha_close = ha_df['ha_close'].values
        ha_dates = ha_df.index.values
        
        normal_dates = normal_df.index.values
        normal_date_to_idx = {date: idx for idx, date in enumerate(normal_dates)}
        
        # Check for entry signals
        for i in range(3, len(ha_df)):
            date = ha_dates[i]
            
            # Entry signal conditions
            cond_today = (ha_open[i] == ha_low[i] and ha_close[i] > ha_open[i])
            cond_yest = (ha_open[i-1] == ha_low[i-1] and ha_close[i-1] > ha_open[i-1])
            cond_yest2 = (ha_open[i-2] != ha_low[i-2] and ha_high[i-2] != ha_close[i-2] and ha_close[i-2] > ha_open[i-2])
            
            if cond_today and cond_yest and cond_yest2:
                # Entry would be on next day
                entry_idx = i + 1
                if entry_idx < len(ha_df):
                    entry_date = ha_dates[entry_idx]
                    if entry_date in normal_date_to_idx:
                        signals.append({
                            'signal_date': date,
                            'entry_date': entry_date,
                            'entry_idx': entry_idx,
                            'sl': ha_low[i-2]
                        })
        
        if signals:
            stock_signals[symbol] = signals
    
    print(f"Found signals for {len(stock_signals)} stocks")
    print(f"{'='*60}\n")
    
    # Simulate day-by-day
    print(f"{'='*60}")
    print(f"PHASE 4: DYNAMIC PORTFOLIO SIMULATION")
    print(f"{'='*60}")
    print("Running day-by-day simulation with intelligent stock selection...")
    
    for date_idx, current_date in enumerate(all_dates):
        # Step 1: Update existing positions (check SL and trail)
        positions_to_exit = []
        
        for symbol, position in list(active_positions.items()):
            if symbol not in stock_data_cache:
                continue
            
            normal_df, ha_df = stock_data_cache[symbol]
            
            if current_date not in normal_df.index or current_date not in ha_df.index:
                continue
            
            normal_low = normal_df.loc[current_date, 'low']
            ha_open_today = ha_df.loc[current_date, 'ha_open']
            ha_low_today = ha_df.loc[current_date, 'ha_low']
            
            # Check SL hit
            if normal_low <= position['sl']:
                exit_price = position['sl']
                pnl = (exit_price - position['entry_price']) * position['qty']
                
                # Return capital with P&L to available pool (COMPOUNDING)
                capital_returned = position['capital_deployed'] + pnl
                available_capital += capital_returned
                total_portfolio_value += pnl  # Update total portfolio value
                
                all_trades.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': str(current_date.date()),
                    'exit_price': exit_price,
                    'qty': position['qty'],
                    'pnl': pnl,
                    'exit_type': 'SL hit',
                    'capital_deployed': position['capital_deployed']
                })
                
                positions_to_exit.append(symbol)
            else:
                # Trail SL
                if ha_open_today != ha_low_today:
                    position['sl'] = max(position['sl'], ha_low_today)
        
        # Remove exited positions
        for symbol in positions_to_exit:
            del active_positions[symbol]
        
        # Step 2: Check for new entry signals (if we have capacity and capital)
        if len(active_positions) < max_positions and available_capital > 0:
            # Find all stocks with entry signal today
            candidates = []
            
            for symbol, signals in stock_signals.items():
                if symbol in active_positions:
                    continue  # Already in position
                
                # Check if this stock has entry signal for today
                for signal in signals:
                    if signal['entry_date'] == current_date:
                        # Use intelligent ranking score if available, otherwise signal count
                        if symbol in stock_rankings:
                            priority_score = stock_rankings[symbol]
                        else:
                            # Fallback to signal count method
                            priority_score = len([s for s in signals if s['entry_date'] < current_date])
                        
                        candidates.append({
                            'symbol': symbol,
                            'signal': signal,
                            'priority': priority_score
                        })
                        break
            
            # Sort by priority (higher score = better stock based on composite metrics)
            candidates.sort(key=lambda x: x['priority'], reverse=True)
            
            open_slots = max_positions - len(active_positions)
            entries_today = candidates[:open_slots]
            
            for candidate in entries_today:
                symbol = candidate['symbol']
                signal = candidate['signal']
                
                normal_df, ha_df = stock_data_cache[symbol]
                
                if current_date not in normal_df.index:
                    continue
                
                entry_price = normal_df.loc[current_date, 'open']
                
                # DYNAMIC CAPITAL ALLOCATION: Divide available capital by remaining slots
                remaining_slots = max_positions - len(active_positions)
                capital_for_this_position = available_capital / remaining_slots
                
                # Calculate quantity as WHOLE NUMBER (integer shares only)
                # Buy maximum whole shares within allocated capital
                qty = int(capital_for_this_position / entry_price)
                
                # Skip if we can't afford even 1 share
                if qty < 1:
                    continue
                
                # Calculate actual capital deployed (exact amount based on whole shares)
                actual_capital_deployed = entry_price * qty
                
                active_positions[symbol] = {
                    'entry_date': str(current_date.date()),
                    'entry_price': entry_price,
                    'qty': qty,
                    'sl': signal['sl'],
                    'capital_deployed': actual_capital_deployed
                }
                
                entry_signals_by_date[str(current_date.date())].append(symbol)
                available_capital -= actual_capital_deployed
        
        # Track capital utilization
        deployed_capital = sum(pos['capital_deployed'] for pos in active_positions.values())
        utilization_pct = ((deployed_capital + available_capital) / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        capital_utilization_history.append({
            'date': str(current_date.date()),
            'deployed': deployed_capital,
            'available': available_capital,
            'total_value': total_portfolio_value,
            'utilization_pct': utilization_pct,
            'num_positions': len(active_positions)
        })
    
    # Step 3: Close all remaining positions at end
    if len(active_positions) > 0:
        last_date = all_dates[-1]
        
        for symbol, position in active_positions.items():
            normal_df, ha_df = stock_data_cache[symbol]
            
            if last_date in normal_df.index:
                exit_price = normal_df.loc[last_date, 'close']
                pnl = (exit_price - position['entry_price']) * position['qty']
                
                # Return capital with P&L to available pool
                capital_returned = position['capital_deployed'] + pnl
                available_capital += capital_returned
                total_portfolio_value += pnl
                
                all_trades.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': str(last_date.date()),
                    'exit_price': exit_price,
                    'qty': position['qty'],
                    'pnl': pnl,
                    'exit_type': 'end of period',
                    'capital_deployed': position['capital_deployed']
                })
    
    # Calculate final results
    total_pnl = sum(trade['pnl'] for trade in all_trades)
    final_capital = total_portfolio_value  # Use the tracked portfolio value
    
    # Calculate capital utilization metrics
    avg_utilization = np.mean([h['utilization_pct'] for h in capital_utilization_history]) if capital_utilization_history else 100
    max_deployed = max([h['deployed'] for h in capital_utilization_history]) if capital_utilization_history else total_capital
    min_available = min([h['available'] for h in capital_utilization_history]) if capital_utilization_history else 0
    
    # Group trades by stock for results
    stocks_results = defaultdict(lambda: {
        'initial_capital': 0,
        'final_capital': 0,
        'pnl': 0,
        'num_trades': 0,
        'trades': [],
        'total_capital_deployed': 0
    })
    
    stock_performance = {}
    
    for trade in all_trades:
        symbol = trade['symbol']
        stocks_results[symbol]['trades'].append(trade)
        stocks_results[symbol]['pnl'] += trade['pnl']
        stocks_results[symbol]['num_trades'] += 1
        stocks_results[symbol]['total_capital_deployed'] += trade.get('capital_deployed', 0)
    
    # Calculate stock performance metrics
    for symbol, result in stocks_results.items():
        # Use actual capital deployed for this stock
        result['initial_capital'] = result['total_capital_deployed']
        result['final_capital'] = result['initial_capital'] + result['pnl']
        
        win_trades = [t for t in result['trades'] if t['pnl'] > 0]
        loss_trades = [t for t in result['trades'] if t['pnl'] <= 0]
        
        stock_performance[symbol] = {
            'pnl': result['pnl'],
            'return_pct': (result['pnl'] / result['initial_capital'] * 100) if result['initial_capital'] > 0 else 0,
            'num_trades': result['num_trades'],
            'win_trades': len(win_trades),
            'loss_trades': len(loss_trades),
            'win_rate': (len(win_trades) / result['num_trades'] * 100) if result['num_trades'] > 0 else 0
        }
    
    # Add metadata to trades
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
    
    return {
        'total_initial_capital': total_capital,
        'total_final_capital': final_capital,
        'total_pnl': total_pnl,
        'stocks_results': dict(stocks_results),
        'all_trades': all_trades_enriched,
        'entry_signals_by_date': dict(entry_signals_by_date),
        'stock_performance': stock_performance,
        'num_unique_stocks_traded': len(stocks_results),
        'capital_utilization': {
            'average_utilization_pct': round(avg_utilization, 2),
            'max_deployed_capital': round(max_deployed, 2),
            'min_available_capital': round(min_available, 2),
            'final_available_capital': round(available_capital, 2),
            'utilization_history': capital_utilization_history
        }
    }


@app.get("/smart_backtest")
def run_smart_backtest(
    total_investment: float = Query(..., description="Total investment amount"),
    num_stocks: int = Query(..., description="Maximum number of stocks to hold simultaneously"),
    start_date: str = Query(DEFAULT_START_DATE, description="Start date for backtest YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD (optional)"),
    ranking_period_months: int = Query(6, description="Months of historical data for stock ranking"),
    use_cached: bool = Query(True, description="Use cached rankings if available")
) -> Dict[str, Any]:
    """
    Dynamic portfolio management backtest endpoint.
    
    This endpoint implements a ROLLING PORTFOLIO strategy:
    1. Sets maximum N positions that can be held at any time
    2. All stocks compete for entry when signals appear
    3. When a position exits, capital becomes available for new entries
    4. Continuously evaluates and enters new opportunities throughout backtest
    
    Example: 
    - Total Investment: ₹100,000, Max Positions: 10
    - Each position gets ₹10,000
    - When Stock A exits, ₹10,000 is freed for Stock B to enter
    - Portfolio dynamically rotates through all available stocks
    
    Args:
        total_investment: Total capital for the portfolio
        num_stocks: Maximum concurrent positions (e.g., 10)
        start_date: Start date for the backtest period
        end_date: End date for the backtest period
        ranking_period_months: Months of historical data for intelligent stock ranking
        use_cached: Use cached rankings if available (future enhancement)
    
    Returns:
        Backtest results showing all trades across all stocks entered during period
    """
    
    if not os.path.exists(DATA_FOLDER) or not os.path.exists(HA_FOLDER):
        return JSONResponse(status_code=404, content={"error": "Data or Heikin Ashi folders not found."})
    
    if num_stocks <= 0 or total_investment <= 0:
        return JSONResponse(status_code=400, content={"error": "Invalid parameters. num_stocks and total_investment must be positive."})
    
    # Get all available symbols
    all_symbols = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    
    print(f"\n{'='*60}")
    print(f"SMART SELECTION BACKTEST STARTED")
    print(f"{'='*60}")
    print(f"Total Investment: ₹{total_investment:,.0f}")
    print(f"Max Concurrent Positions: {num_stocks}")
    print(f"Capital per Position: ₹{total_investment/num_stocks:,.0f}")
    print(f"Available Stocks: {len(all_symbols)}")
    print(f"Backtest Period: {start_date} to {end_date or 'latest'}")
    print(f"Ranking Period: {ranking_period_months} months")
    print(f"{'='*60}\n")
    
    # Clear cache
    global _dataframe_cache
    _dataframe_cache.clear()
    
    # Run dynamic portfolio backtest with intelligent ranking
    result = dynamic_portfolio_backtest(
        all_symbols=all_symbols,
        total_capital=total_investment,
        max_positions=num_stocks,
        start_date=start_date,
        end_date=end_date,
        ranking_period_months=ranking_period_months,
        use_ranking=True
    )
    
    # Calculate comprehensive statistics
    statistics = calculate_statistics(
        result['all_trades'],
        result['entry_signals_by_date'],
        result['stock_performance'],
        result['total_initial_capital'],
        result['total_final_capital']
    )
    
    total_return_pct = ((result['total_final_capital'] - result['total_initial_capital']) / 
                        result['total_initial_capital'] * 100) if result['total_initial_capital'] > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"SMART SELECTION BACKTEST COMPLETED")
    print(f"{'='*60}")
    print(f"Total Initial Capital: ₹{result['total_initial_capital']:,.0f}")
    print(f"Total Final Capital: ₹{result['total_final_capital']:,.0f}")
    print(f"Total P&L: ₹{result['total_pnl']:,.0f} ({total_return_pct:.2f}%)")
    print(f"Unique Stocks Traded: {result['num_unique_stocks_traded']}")
    print(f"Total Trades: {len(result['all_trades'])}")
    print(f"Avg Capital Utilization: {result['capital_utilization']['average_utilization_pct']:.2f}%")
    print(f"{'='*60}\n")
    
    # Build response
    portfolio = {
        "total_initial_capital": result['total_initial_capital'],
        "total_final_capital": result['total_final_capital'],
        "total_pnl": result['total_pnl'],
        "total_return_pct": total_return_pct,
        "num_stocks_processed": result['num_unique_stocks_traded'],
        "stocks": result['stocks_results'],
        "params": {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": result['total_initial_capital'] / num_stocks,
            "strategy": "heikin_ashi_smart_selection",
            "total_investment": total_investment,
            "max_concurrent_positions": num_stocks,
            "ranking_period_months": ranking_period_months
        },
        "statistics": statistics,
        "capital_utilization": result['capital_utilization'],
        "selection_info": {
            "total_investment": total_investment,
            "num_stocks_requested": num_stocks,
            "num_stocks_selected": result['num_unique_stocks_traded'],
            "capital_per_stock": result['total_initial_capital'] / num_stocks,
            "ranking_period_months": ranking_period_months,
            "total_stocks_analyzed": len(all_symbols),
            "selected_stocks": [
                {
                    "symbol": symbol,
                    "ranking_score": perf['return_pct'],  # Use return as score for display
                    "historical_win_rate": perf['win_rate'],
                    "historical_return_pct": perf['return_pct'],
                    "historical_trades": perf['num_trades']
                }
                for symbol, perf in sorted(result['stock_performance'].items(), 
                                          key=lambda x: x[1]['return_pct'], 
                                          reverse=True)
            ]
        }
    }
    
    return portfolio


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)