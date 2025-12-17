import { Trade, BacktestResult } from '../services/api';
import { Activity, TrendingUp, AlertCircle, DollarSign } from 'lucide-react';
import { formatINR } from '../utils/formatters';

interface CurrentPosition {
  symbol: string;
  trade: Trade;
  daysHeld: number;
  currentPnL: number;
  percentReturn: number;
}

interface CurrentPositionsProps {
  backtestResult: BacktestResult;
}

export default function CurrentPositions({ backtestResult }: CurrentPositionsProps) {
  // Check if stocks data is available
  if (!backtestResult.stocks || Object.keys(backtestResult.stocks).length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-2xl font-bold flex items-center gap-2 mb-4">
          <Activity className="text-blue-400" size={28} />
          Current Open Positions
        </h2>
        <div className="text-center py-12">
          <Activity className="mx-auto text-gray-600 mb-4" size={48} />
          <p className="text-gray-400 text-lg mb-2">No stock data available</p>
          <p className="text-gray-500 text-sm">Run a backtest to see open positions</p>
        </div>
      </div>
    );
  }

  // Find all positions that are still open (last trade with exit_date at end of data)
  const getCurrentPositions = (): CurrentPosition[] => {
    const positions: CurrentPosition[] = [];
    
    Object.entries(backtestResult.stocks).forEach(([symbol, stockData]) => {
      if (stockData.trades.length === 0) return;
      
      const lastTrade = stockData.trades[stockData.trades.length - 1];
      
      // Check if this is still an open position (exit at end of period)
      if (lastTrade.exit_type === 'end of period') {
        const entryDate = new Date(lastTrade.entry_date);
        const exitDate = new Date(lastTrade.exit_date);
        const daysHeld = Math.floor((exitDate.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24));
        
        positions.push({
          symbol,
          trade: lastTrade,
          daysHeld,
          currentPnL: lastTrade.pnl,
          percentReturn: (lastTrade.pnl / (lastTrade.entry_price * lastTrade.qty)) * 100,
        });
      }
    });
    
    return positions.sort((a, b) => b.currentPnL - a.currentPnL);
  };

  const positions = getCurrentPositions();
  const totalPositionValue = positions.reduce((sum, p) => sum + (p.trade.exit_price * p.trade.qty), 0);
  const totalPnL = positions.reduce((sum, p) => sum + p.currentPnL, 0);
  const avgReturn = positions.length > 0 ? positions.reduce((sum, p) => sum + p.percentReturn, 0) / positions.length : 0;

  if (positions.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-2xl font-bold flex items-center gap-2 mb-4">
          <Activity className="text-blue-400" size={28} />
          Current Open Positions
        </h2>
        <div className="text-center py-12">
          <AlertCircle className="mx-auto text-gray-600 mb-4" size={48} />
          <p className="text-gray-400">No open positions at the end of the backtest period</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <h2 className="text-2xl font-bold flex items-center gap-2 mb-6">
        <Activity className="text-blue-400" size={28} />
        Current Open Positions
      </h2>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-lg p-4 border border-blue-700/50">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="text-blue-400" size={20} />
            <span className="text-sm text-gray-300">Open Positions</span>
          </div>
          <div className="text-2xl font-bold text-white">{positions.length}</div>
        </div>

        <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-lg p-4 border border-purple-700/50">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="text-purple-400" size={20} />
            <span className="text-sm text-gray-300">Total Value</span>
          </div>
          <div className="text-2xl font-bold text-white">{formatINR(totalPositionValue, 0)}</div>
        </div>

        <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-lg p-4 border border-green-700/50">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="text-green-400" size={20} />
            <span className="text-sm text-gray-300">Total P&L</span>
          </div>
          <div className={`text-2xl font-bold ${
            totalPnL >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {totalPnL >= 0 ? '+' : ''}{formatINR(totalPnL, 2)}
          </div>
        </div>

        <div className="bg-gradient-to-br from-orange-900/50 to-orange-800/30 rounded-lg p-4 border border-orange-700/50">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="text-orange-400" size={20} />
            <span className="text-sm text-gray-300">Avg Return</span>
          </div>
          <div className={`text-2xl font-bold ${
            avgReturn >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {avgReturn >= 0 ? '+' : ''}{avgReturn.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
        <table className="w-full">
          <thead className="bg-gray-800 sticky top-0 z-10">
            <tr>
              <th className="text-left py-3 px-4 text-sm font-semibold text-gray-300">Symbol</th>
              <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Entry Date</th>
              <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Entry Price</th>
              <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Current Price</th>
              <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Qty</th>
              <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Days Held</th>
              <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">P&L</th>
              <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Return %</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position, idx) => (
              <tr
                key={position.symbol}
                className={`border-b border-gray-800 hover:bg-gray-800/50 transition-colors ${
                  idx % 2 === 0 ? 'bg-gray-900/30' : ''
                }`}
              >
                <td className="py-3 px-4">
                  <div className="font-semibold text-blue-400">{position.symbol}</div>
                </td>
                <td className="py-3 px-4 text-right text-sm text-gray-300">
                  {new Date(position.trade.entry_date).toLocaleDateString('en-IN', { 
                    month: 'short', 
                    day: 'numeric',
                    year: 'numeric'
                  })}
                </td>
                <td className="py-3 px-4 text-right font-medium text-white">
                  ₹{position.trade.entry_price.toFixed(2)}
                </td>
                <td className="py-3 px-4 text-right font-medium text-white">
                  ₹{position.trade.exit_price.toFixed(2)}
                </td>
                <td className="py-3 px-4 text-right text-sm text-gray-300">
                  {Math.floor(position.trade.qty)}
                </td>
                <td className="py-3 px-4 text-right">
                  <span className="bg-blue-900/30 px-2 py-1 rounded text-xs text-blue-400">
                    {position.daysHeld} days
                  </span>
                </td>
                <td className="py-3 px-4 text-right">
                  <div className={`font-bold ${
                    position.currentPnL >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {position.currentPnL >= 0 ? '+' : ''}{formatINR(position.currentPnL, 2)}
                  </div>
                </td>
                <td className="py-3 px-4 text-right">
                  <div className={`font-bold ${
                    position.percentReturn >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {position.percentReturn >= 0 ? '+' : ''}{position.percentReturn.toFixed(2)}%
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
