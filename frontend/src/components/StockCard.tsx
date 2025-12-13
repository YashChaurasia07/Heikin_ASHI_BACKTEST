import { StockResult } from '../services/api';
import { TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react';
import { formatINR } from '../utils/formatters';

interface StockCardProps {
  symbol: string;
  result: StockResult;
  onClick: () => void;
}

const StockCard = ({ symbol, result, onClick }: StockCardProps) => {
  const returnPct = ((result.final_capital - result.initial_capital) / result.initial_capital) * 100;
  const isProfit = result.pnl >= 0;

  return (
    <div
      onClick={onClick}
      className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-5 border border-gray-700 hover:border-primary-500 transition-all cursor-pointer hover:scale-105 transform duration-200 shadow-lg hover:shadow-primary-500/20"
    >
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-xl font-bold text-white">{symbol}</h3>
        {isProfit ? (
          <TrendingUp className="text-green-500" size={24} />
        ) : (
          <TrendingDown className="text-red-500" size={24} />
        )}
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">P&L</span>
          <span className={`text-lg font-bold ${isProfit ? 'text-green-500' : 'text-red-500'}`}>
            {isProfit ? '+' : ''}{formatINR(result.pnl, 2)}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">Return</span>
          <span className={`text-lg font-semibold ${isProfit ? 'text-green-500' : 'text-red-500'}`}>
            {isProfit ? '+' : ''}{returnPct.toFixed(2)}%
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm flex items-center gap-1">
            <Activity size={14} />
            Trades
          </span>
          <span className="text-white font-medium">{result.num_trades}</span>
        </div>

        <div className="flex justify-between items-center pt-2 border-t border-gray-700">
          <span className="text-gray-400 text-sm flex items-center gap-1">
            <DollarSign size={14} />
            Final Capital
          </span>
          <span className="text-white font-medium">{formatINR(result.final_capital, 2)}</span>
        </div>
      </div>
    </div>
  );
};

export default StockCard;
