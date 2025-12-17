import { BacktestResult } from '../services/api';
import { PieChart, TrendingUp, TrendingDown, Target, Award } from 'lucide-react';
import { formatINR } from '../utils/formatters';

interface PortfolioSummaryProps {
  backtestResult: BacktestResult;
}

export default function PortfolioSummary({ backtestResult }: PortfolioSummaryProps) {
  // Check if stocks data is available
  if (!backtestResult.stocks || Object.keys(backtestResult.stocks).length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-2xl font-bold flex items-center gap-2 mb-4">
          <PieChart className="text-purple-400" size={28} />
          Portfolio Summary
        </h2>
        <div className="text-center py-12">
          <PieChart className="mx-auto text-gray-600 mb-4" size={48} />
          <p className="text-gray-400 text-lg mb-2">No stock data available</p>
          <p className="text-gray-500 text-sm">Run a backtest to see portfolio analysis</p>
        </div>
      </div>
    );
  }

  // Calculate portfolio metrics
  const totalTrades = Object.values(backtestResult.stocks).reduce(
    (sum, stock) => sum + stock.trades.length, 
    0
  );

  const winningTrades = Object.values(backtestResult.stocks).reduce(
    (sum, stock) => sum + stock.trades.filter(t => t.pnl > 0).length, 
    0
  );

  const losingTrades = totalTrades - winningTrades;
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

  // Find best and worst performing stocks
  const stockPerformances = Object.entries(backtestResult.stocks)
    .map(([symbol, data]) => ({
      symbol,
      pnl: data.pnl,
      returnPct: ((data.final_capital - data.initial_capital) / data.initial_capital) * 100,
      numTrades: data.trades.length,
    }))
    .filter(s => s.numTrades > 0);

  const bestStock = stockPerformances.sort((a, b) => b.pnl - a.pnl)[0];
  const worstStock = stockPerformances.sort((a, b) => a.pnl - b.pnl)[0];

  // Capital allocation
  const totalCapitalDeployed = backtestResult.total_initial_capital ?? 0;
  const currentCapital = backtestResult.total_final_capital ?? 0;
  const capitalGrowth = totalCapitalDeployed > 0 ? ((currentCapital - totalCapitalDeployed) / totalCapitalDeployed) * 100 : 0;

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <h2 className="text-2xl font-bold flex items-center gap-2 mb-6">
        <PieChart className="text-purple-400" size={28} />
        Portfolio Summary
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Capital Overview */}
        <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 rounded-lg p-5 border border-blue-700/50">
          <h3 className="text-sm font-semibold text-blue-300 mb-4 flex items-center gap-2">
            <Target size={18} />
            Capital Overview
          </h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-gray-400">Initial Capital</div>
              <div className="text-lg font-bold text-white">
                {formatINR(totalCapitalDeployed, 2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Current Capital</div>
              <div className="text-lg font-bold text-white">
                {formatINR(currentCapital, 2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Capital Growth</div>
              <div className={`text-xl font-bold ${
                capitalGrowth >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {capitalGrowth >= 0 ? '+' : ''}{capitalGrowth.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>

        {/* Trade Statistics */}
        <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 rounded-lg p-5 border border-green-700/50">
          <h3 className="text-sm font-semibold text-green-300 mb-4 flex items-center gap-2">
            <TrendingUp size={18} />
            Trade Statistics
          </h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-gray-400">Total Trades</div>
              <div className="text-lg font-bold text-white">{totalTrades}</div>
            </div>
            <div className="flex justify-between gap-4">
              <div>
                <div className="text-xs text-gray-400">Winning</div>
                <div className="text-lg font-bold text-green-400">{winningTrades}</div>
              </div>
              <div>
                <div className="text-xs text-gray-400">Losing</div>
                <div className="text-lg font-bold text-red-400">{losingTrades}</div>
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Win Rate</div>
              <div className="text-xl font-bold text-white">{winRate.toFixed(2)}%</div>
            </div>
          </div>
        </div>

        {/* P&L Breakdown */}
        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 rounded-lg p-5 border border-purple-700/50">
          <h3 className="text-sm font-semibold text-purple-300 mb-4 flex items-center gap-2">
            <Award size={18} />
            P&L Breakdown
          </h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-gray-400">Total P&L</div>
              <div className={`text-lg font-bold ${
                backtestResult.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {backtestResult.total_pnl >= 0 ? '+' : ''}
                {formatINR(backtestResult.total_pnl, 2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Return %</div>
              <div className={`text-xl font-bold ${
                backtestResult.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {backtestResult.total_return_pct >= 0 ? '+' : ''}
                {backtestResult.total_return_pct.toFixed(2)}%
              </div>
            </div>
            {backtestResult.statistics?.cagr_pct !== undefined && (
              <div>
                <div className="text-xs text-gray-400">CAGR</div>
                <div className={`text-xl font-bold ${
                  backtestResult.statistics.cagr_pct >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {backtestResult.statistics.cagr_pct >= 0 ? '+' : ''}
                  {backtestResult.statistics.cagr_pct.toFixed(2)}%
                </div>
              </div>
            )}
            <div>
              <div className="text-xs text-gray-400">Stocks Processed</div>
              <div className="text-lg font-bold text-white">
                {backtestResult.num_stocks_processed}
              </div>
            </div>
          </div>
        </div>

        {/* Best Performing Stock */}
        {bestStock && (
          <div className="bg-gradient-to-br from-emerald-900/30 to-emerald-800/20 rounded-lg p-5 border border-emerald-700/50">
            <h3 className="text-sm font-semibold text-emerald-300 mb-4 flex items-center gap-2">
              <TrendingUp size={18} />
              Best Performer
            </h3>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">{bestStock.symbol}</div>
              <div>
                <div className="text-xs text-gray-400">P&L</div>
                <div className="text-lg font-bold text-green-400">
                  +{formatINR(bestStock.pnl, 2)}
                </div>
              </div>
              <div className="flex justify-between gap-4">
                <div>
                  <div className="text-xs text-gray-400">Return</div>
                  <div className="text-sm font-bold text-green-400">
                    +{bestStock.returnPct.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Trades</div>
                  <div className="text-sm font-bold text-white">{bestStock.numTrades}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Worst Performing Stock */}
        {worstStock && (
          <div className="bg-gradient-to-br from-red-900/30 to-red-800/20 rounded-lg p-5 border border-red-700/50">
            <h3 className="text-sm font-semibold text-red-300 mb-4 flex items-center gap-2">
              <TrendingDown size={18} />
              Worst Performer
            </h3>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">{worstStock.symbol}</div>
              <div>
                <div className="text-xs text-gray-400">P&L</div>
                <div className="text-lg font-bold text-red-400">
                  {formatINR(worstStock.pnl, 2)}
                </div>
              </div>
              <div className="flex justify-between gap-4">
                <div>
                  <div className="text-xs text-gray-400">Return</div>
                  <div className="text-sm font-bold text-red-400">
                    {worstStock.returnPct.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Trades</div>
                  <div className="text-sm font-bold text-white">{worstStock.numTrades}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Period Info */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/30 rounded-lg p-5 border border-gray-600/50">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Backtest Period</h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-gray-400">Start Date</div>
              <div className="text-sm font-medium text-white">
                {backtestResult.params.start_date}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">End Date</div>
              <div className="text-sm font-medium text-white">
                {backtestResult.params.end_date || 'Latest Available'}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Strategy</div>
              <div className="text-sm font-medium text-blue-400">
                {backtestResult.params.strategy}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
