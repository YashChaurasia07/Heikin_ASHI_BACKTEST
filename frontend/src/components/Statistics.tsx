import { Statistics as StatsType } from '../services/api';
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Clock, 
  Award, 
  AlertTriangle,
  Calendar,
  BarChart3,
  DollarSign,
  Activity,
  Zap,
  Shield
} from 'lucide-react';
import { formatINR } from '../utils/formatters';

interface StatisticsProps {
  statistics: StatsType;
}

function Statistics({ statistics }: StatisticsProps) {
  if (!statistics || statistics.total_trades === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 text-center text-gray-400">
        No statistics available
      </div>
    );
  }

  const StatCard = ({ 
    icon: Icon, 
    title, 
    value, 
    subtitle, 
    color = 'blue' 
  }: { 
    icon: any; 
    title: string; 
    value: string | number; 
    subtitle?: string; 
    color?: string;
  }) => (
    <div className={`bg-gradient-to-br from-${color}-900/30 to-${color}-800/20 rounded-xl p-4 border border-${color}-700/30`}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`text-${color}-400`} size={20} />
        <span className="text-xs text-gray-400">{title}</span>
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      {subtitle && <div className="text-xs text-gray-500">{subtitle}</div>}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Daily Signal Statistics */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Calendar className="text-blue-400" size={24} />
          Daily Signal Statistics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
          <StatCard
            icon={Activity}
            title="Avg Stocks Per Day"
            value={statistics.avg_stocks_signaled_per_day}
            color="blue"
          />
          <StatCard
            icon={TrendingUp}
            title="Max Stocks (Single Day)"
            value={statistics.max_stocks_on_single_day}
            color="green"
          />
          <StatCard
            icon={TrendingDown}
            title="Min Stocks (Single Day)"
            value={statistics.min_stocks_on_single_day}
            color="orange"
          />
          <StatCard
            icon={Target}
            title="Median Stocks Per Day"
            value={statistics.median_stocks_per_day}
            color="purple"
          />
          <StatCard
            icon={Calendar}
            title="Total Signal Days"
            value={statistics.total_signal_days}
            color="indigo"
          />
        </div>
        {statistics.day_with_most_signals && (
          <div className="mt-4 p-4 bg-gray-800/50 rounded-lg">
            <div className="text-sm text-gray-400 mb-1">Day with Most Signals</div>
            <div className="font-semibold text-green-400">
              {statistics.day_with_most_signals.date} - {statistics.day_with_most_signals.count} stocks
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {statistics.day_with_most_signals.stocks.slice(0, 10).join(', ')}
              {statistics.day_with_most_signals.stocks.length > 10 && ` +${statistics.day_with_most_signals.stocks.length - 10} more`}
            </div>
          </div>
        )}
      </div>

      {/* Win/Loss & Profitability */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-green-400" size={24} />
          Win/Loss & Profitability Analysis
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          <StatCard
            icon={BarChart3}
            title="Total Trades"
            value={statistics.total_trades}
            color="blue"
          />
          <StatCard
            icon={TrendingUp}
            title="Winning Trades"
            value={statistics.winning_trades}
            subtitle={`${statistics.win_rate_pct}% Win Rate`}
            color="green"
          />
          <StatCard
            icon={TrendingDown}
            title="Losing Trades"
            value={statistics.losing_trades}
            color="red"
          />
          <StatCard
            icon={Award}
            title="Profit Factor"
            value={statistics.profit_factor === Infinity ? '∞' : statistics.profit_factor.toFixed(2)}
            color="yellow"
          />
          <StatCard
            icon={DollarSign}
            title="Total Profit"
            value={formatINR(statistics.total_profit)}
            color="green"
          />
          <StatCard
            icon={AlertTriangle}
            title="Total Loss"
            value={formatINR(statistics.total_loss)}
            color="red"
          />
          <StatCard
            icon={Activity}
            title="Net P&L"
            value={formatINR(statistics.net_pnl)}
            color={statistics.net_pnl >= 0 ? 'green' : 'red'}
          />
          <StatCard
            icon={TrendingUp}
            title="Avg P&L Per Trade"
            value={formatINR(statistics.avg_pnl_per_trade)}
            color={statistics.avg_pnl_per_trade >= 0 ? 'green' : 'red'}
          />
        </div>
      </div>

      {/* Holding Period Analysis */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Clock className="text-purple-400" size={24} />
          Holding Period Analysis
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            icon={Clock}
            title="Avg Holding Days"
            value={statistics.avg_holding_days}
            color="purple"
          />
          <StatCard
            icon={TrendingUp}
            title="Max Holding Days"
            value={statistics.max_holding_days}
            color="blue"
          />
          <StatCard
            icon={TrendingDown}
            title="Min Holding Days"
            value={statistics.min_holding_days}
            color="orange"
          />
          <StatCard
            icon={Target}
            title="Median Holding Days"
            value={statistics.median_holding_days}
            color="indigo"
          />
        </div>
      </div>

      {/* Best & Worst Performers */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Award className="text-yellow-400" size={24} />
          Best & Worst Performers
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Best Trade */}
          {statistics.best_trade && (
            <div className="p-4 bg-green-900/20 border border-green-700/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">🏆 Best Trade</div>
              <div className="font-bold text-lg text-green-400">{statistics.best_trade.symbol}</div>
              <div className="text-sm text-gray-300 mt-1">
                P&L: {formatINR(statistics.best_trade.pnl)} ({statistics.best_trade.return_pct.toFixed(2)}%)
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {statistics.best_trade.entry_date} → {statistics.best_trade.exit_date}
              </div>
            </div>
          )}

          {/* Worst Trade */}
          {statistics.worst_trade && (
            <div className="p-4 bg-red-900/20 border border-red-700/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">📉 Worst Trade</div>
              <div className="font-bold text-lg text-red-400">{statistics.worst_trade.symbol}</div>
              <div className="text-sm text-gray-300 mt-1">
                P&L: {formatINR(statistics.worst_trade.pnl)} ({statistics.worst_trade.return_pct.toFixed(2)}%)
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {statistics.worst_trade.entry_date} → {statistics.worst_trade.exit_date}
              </div>
            </div>
          )}

          {/* Best Performing Stock */}
          {statistics.best_performing_stock && (
            <div className="p-4 bg-blue-900/20 border border-blue-700/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">⭐ Best Stock</div>
              <div className="font-bold text-lg text-blue-400">{statistics.best_performing_stock.symbol}</div>
              <div className="text-sm text-gray-300 mt-1">
                Return: {statistics.best_performing_stock.return_pct.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {statistics.best_performing_stock.num_trades} trades | Win Rate: {statistics.best_performing_stock.win_rate.toFixed(1)}%
              </div>
            </div>
          )}

          {/* Worst Performing Stock */}
          {statistics.worst_performing_stock && (
            <div className="p-4 bg-orange-900/20 border border-orange-700/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">⚠️ Worst Stock</div>
              <div className="font-bold text-lg text-orange-400">{statistics.worst_performing_stock.symbol}</div>
              <div className="text-sm text-gray-300 mt-1">
                Return: {statistics.worst_performing_stock.return_pct.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {statistics.worst_performing_stock.num_trades} trades | Win Rate: {statistics.worst_performing_stock.win_rate.toFixed(1)}%
              </div>
            </div>
          )}

          {/* Highest Win Rate Stock */}
          {statistics.highest_win_rate_stock && (
            <div className="p-4 bg-purple-900/20 border border-purple-700/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">🎯 Best Win Rate</div>
              <div className="font-bold text-lg text-purple-400">{statistics.highest_win_rate_stock.symbol}</div>
              <div className="text-sm text-gray-300 mt-1">
                Win Rate: {statistics.highest_win_rate_stock.win_rate.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {statistics.highest_win_rate_stock.num_trades} trades | Return: {statistics.highest_win_rate_stock.return_pct.toFixed(2)}%
              </div>
            </div>
          )}

          {/* Best Month */}
          {statistics.best_month && (
            <div className="p-4 bg-green-900/20 border border-green-700/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">📅 Best Month</div>
              <div className="font-bold text-lg text-green-400">{statistics.best_month.month}</div>
              <div className="text-sm text-gray-300 mt-1">
                P&L: {formatINR(statistics.best_month.pnl)}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {statistics.best_month.num_trades} trades
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Shield className="text-red-400" size={24} />
          Risk Metrics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <StatCard
            icon={AlertTriangle}
            title="Std Deviation"
            value={`${statistics.std_deviation_returns}%`}
            color="orange"
          />
          <StatCard
            icon={TrendingUp}
            title="Max Return"
            value={`${statistics.max_return_pct.toFixed(2)}%`}
            color="green"
          />
          <StatCard
            icon={TrendingDown}
            title="Min Return"
            value={`${statistics.min_return_pct.toFixed(2)}%`}
            color="red"
          />
          <StatCard
            icon={Zap}
            title="Sharpe Ratio"
            value={statistics.sharpe_ratio_approx.toFixed(2)}
            color="purple"
          />
          <StatCard
            icon={AlertTriangle}
            title="Max Drawdown"
            value={formatINR(statistics.max_drawdown)}
            subtitle={`${statistics.max_drawdown_pct.toFixed(2)}%`}
            color="red"
          />
        </div>
      </div>

      {/* Exit Type Breakdown */}
      {statistics.exit_type_breakdown && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Activity className="text-indigo-400" size={24} />
            Exit Type Breakdown
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(statistics.exit_type_breakdown).map(([exitType, data]) => (
              <div key={exitType} className="p-4 bg-gray-800/50 rounded-lg">
                <div className="font-semibold text-white mb-2">{exitType}</div>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Count:</span>
                    <span className="text-white">{data.count} ({data.pct.toFixed(1)}%)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total P&L:</span>
                    <span className={data.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {formatINR(data.total_pnl)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Avg P&L:</span>
                    <span className={data.avg_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {formatINR(data.avg_pnl)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Streaks & Additional Metrics */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-yellow-400" size={24} />
          Streaks & Additional Metrics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            icon={TrendingUp}
            title="Max Consecutive Wins"
            value={statistics.max_consecutive_wins}
            color="green"
          />
          <StatCard
            icon={TrendingDown}
            title="Max Consecutive Losses"
            value={statistics.max_consecutive_losses}
            color="red"
          />
          <StatCard
            icon={DollarSign}
            title="Return on Capital"
            value={`${statistics.return_on_capital_pct.toFixed(2)}%`}
            color={statistics.return_on_capital_pct >= 0 ? 'green' : 'red'}
          />
          <StatCard
            icon={Activity}
            title="Avg Return Per Trade"
            value={`${statistics.avg_return_pct_per_trade.toFixed(2)}%`}
            color={statistics.avg_return_pct_per_trade >= 0 ? 'green' : 'red'}
          />
        </div>
      </div>

      {/* Monthly Performance */}
      {statistics.best_month && statistics.worst_month && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Calendar className="text-cyan-400" size={24} />
            Monthly Performance Summary
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <StatCard
              icon={DollarSign}
              title="Avg Monthly P&L"
              value={formatINR(statistics.avg_monthly_pnl)}
              color="blue"
            />
            <StatCard
              icon={TrendingUp}
              title="Profitable Months"
              value={statistics.total_profitable_months}
              color="green"
            />
            <StatCard
              icon={TrendingDown}
              title="Losing Months"
              value={statistics.total_losing_months}
              color="red"
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default Statistics;
