import { Trade } from '../services/api';
import { Calendar, TrendingUp, TrendingDown, Percent } from 'lucide-react';
import { formatINR } from '../utils/formatters';

interface TradeListProps {
  trades: Trade[];
  symbol: string;
}

const TradeList = ({ trades, symbol }: TradeListProps) => {
  if (trades.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-8 text-center border border-gray-800">
        <p className="text-gray-400">No trades executed for {symbol}</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <h3 className="text-xl font-bold text-white mb-6">Trade History - {symbol}</h3>
      <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
        {trades.map((trade, index) => {
          const isProfit = trade.pnl >= 0;
          const returnPct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100;

          return (
            <div
              key={index}
              className="bg-gradient-to-br from-gray-800 to-gray-850 rounded-lg p-5 border border-gray-700 hover:border-gray-600 transition-colors"
            >
              <div className="flex justify-between items-start mb-4">
                <span className="text-sm font-semibold text-gray-400">Trade #{index + 1}</span>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    isProfit ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                  }`}
                >
                  {isProfit ? '+' : ''}{formatINR(trade.pnl, 2)}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center gap-2 text-green-500 mb-1">
                      <TrendingUp size={16} />
                      <span className="text-xs font-semibold">ENTRY</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-400 text-sm">
                      <Calendar size={14} />
                      <span>{trade.entry_date}</span>
                    </div>
                    <div className="text-white font-semibold mt-1">
                      {formatINR(trade.entry_price, 2)}
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex items-center gap-2 text-red-500 mb-1">
                      <TrendingDown size={16} />
                      <span className="text-xs font-semibold">EXIT</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-400 text-sm">
                      <Calendar size={14} />
                      <span>{trade.exit_date}</span>
                    </div>
                    <div className="text-white font-semibold mt-1">
                      {formatINR(trade.exit_price, 2)}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-700 flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Percent size={14} className="text-gray-400" />
                  <span
                    className={`font-semibold ${
                      isProfit ? 'text-green-500' : 'text-red-500'
                    }`}
                  >
                    {isProfit ? '+' : ''}{returnPct.toFixed(2)}%
                  </span>
                </div>
                <div className="text-xs text-gray-500 bg-gray-800 px-3 py-1 rounded">
                  {trade.exit_type}
                </div>
                <div className="text-xs text-gray-400">
                  Qty: {Math.floor(trade.qty)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TradeList;
