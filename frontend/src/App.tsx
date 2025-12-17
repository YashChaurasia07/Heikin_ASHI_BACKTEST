import { useState } from 'react';
import api, { BacktestResult, OHLCData, HAData } from './services/api';
import TradingChart from './components/TradingChart';
import StockCard from './components/StockCard';
import TradeList from './components/TradeList';
import CapitalGrowthChart from './components/CapitalGrowthChart';
import Statistics from './components/Statistics';
import PortfolioTimeline from './components/PortfolioTimeline';
import CurrentPositions from './components/CurrentPositions';
import PortfolioSummary from './components/PortfolioSummary';
import SymbolManager from './components/SymbolManager';
import DataSyncStatus from './components/DataSyncStatus';
import Instructions from './components/Instructions';
import TScoreDashboard from './components/TScoreDashboard';
import { BarChart3, TrendingUp, DollarSign, Activity, Search, Play, Loader2, PieChart, Sparkles, Calendar, ListChecks, Briefcase, Settings, BookOpen } from 'lucide-react';
import { formatINR } from './utils/formatters';

function App() {
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [stockData, setStockData] = useState<OHLCData[]>([]);
  const [haData, setHAData] = useState<HAData[]>([]);
  const [loading, setLoading] = useState(false);
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState(50000);
  const [searchQuery, setSearchQuery] = useState('');
  const [showStatistics, setShowStatistics] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'stocks' | 'timeline' | 'positions' | 'tscore' | 'settings' | 'instructions'>('overview');
  
  // Strategy selection: 'all' | 'smart' | 'tscore'
  const [strategy, setStrategy] = useState<'all' | 'smart' | 'tscore'>('all');
  const [interval, setInterval] = useState<'daily' | 'weekly'>('daily');
  const [totalInvestment, setTotalInvestment] = useState(500000);
  const [numStocksToSelect, setNumStocksToSelect] = useState(10);

  const runBacktest = async () => {
    setLoading(true);
    setSelectedSymbol(null);
    try {
      let result;
      
      if (strategy === 'smart') {
        // Smart backtest - backend selects best stocks with priority filtering
        result = await api.runSmartBacktest(
          totalInvestment,
          numStocksToSelect,
          startDate,
          endDate,
          interval
        );
      } else if (strategy === 'tscore') {
        // Heikin Ashi + T-Score strategy
        result = await api.runHATScoreBacktest(
          totalInvestment,
          numStocksToSelect,
          startDate,
          endDate,
          interval
        );
      } else {
        // Regular backtest - all stocks
        result = await api.runBacktest(
          startDate, 
          endDate, 
          initialCapital,
          interval
        );
      }
      
      setBacktestResult(result);
    } catch (error) {
      console.error('Error running backtest:', error);
      alert('Error running backtest. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleStockSelect = async (symbol: string) => {
    setSelectedSymbol(symbol);
    try {
      const [normal, ha] = await Promise.all([
        api.getStockData(symbol),
        api.getHAData(symbol),
      ]);
      setStockData(normal);
      setHAData(ha);
    } catch (error) {
      console.error('Error loading stock data:', error);
    }
  };

  const filteredStocks = backtestResult?.stocks
    ? Object.entries(backtestResult.stocks).filter(([symbol]) =>
        symbol.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : [];

  const sortedStocks = filteredStocks.sort((a, b) => b[1].pnl - a[1].pnl);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <BarChart3 className="text-primary-500" size={32} />
              <div>
                <h1 className="text-2xl font-bold">Trading Strategy Backtest</h1>
                <p className="text-sm text-gray-400">Multi-Strategy Dashboard</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4 flex-wrap">
              {/* Interval Selection */}
              <div className="flex flex-col items-start">
                <label className="text-xs text-gray-400 mb-1">Timeframe</label>
                <div className="flex items-center gap-2 bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5">
                  <button
                    onClick={() => setInterval('daily')}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      interval === 'daily' 
                        ? 'bg-blue-600 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    Daily
                  </button>
                  <button
                    onClick={() => setInterval('weekly')}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      interval === 'weekly' 
                        ? 'bg-blue-600 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    Weekly
                  </button>
                </div>
              </div>
              
              {/* Strategy Selection */}
              <div className="flex flex-col items-start">
                <label className="text-xs text-gray-400 mb-1">Strategy</label>
                <div className="flex items-center gap-2 bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5">
                  <button
                    onClick={() => setStrategy('all')}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      strategy === 'all' 
                        ? 'bg-primary-600 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    All Stocks
                  </button>
                  <button
                    onClick={() => setStrategy('smart')}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors flex items-center gap-1 ${
                      strategy === 'smart' 
                        ? 'bg-green-600 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <Sparkles size={14} />
                    Smart
                  </button>
                  <button
                    onClick={() => setStrategy('tscore')}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors flex items-center gap-1 ${
                      strategy === 'tscore' 
                        ? 'bg-purple-600 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <TrendingUp size={14} />
                    HA+TScore
                  </button>
                </div>
              </div>

              {(strategy === 'smart' || strategy === 'tscore') ? (
                /* Smart Mode & T-Score Inputs */
                <>
                  <div className="flex flex-col items-start">
                    <label className="text-xs text-gray-400 mb-1">Total Investment</label>
                    <input
                      type="number"
                      value={totalInvestment}
                      onChange={(e) => setTotalInvestment(Number(e.target.value))}
                      className={`bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-36 focus:outline-none ${
                        strategy === 'tscore' ? 'focus:border-purple-500' : 'focus:border-green-500'
                      }`}
                      placeholder="₹500000"
                    />
                  </div>
                  <div className="flex flex-col items-start">
                    <label className="text-xs text-gray-400 mb-1">Max Positions</label>
                    <input
                      type="number"
                      value={numStocksToSelect}
                      onChange={(e) => setNumStocksToSelect(Number(e.target.value))}
                      min={1}
                      max={50}
                      className={`bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-28 focus:outline-none ${
                        strategy === 'tscore' ? 'focus:border-purple-500' : 'focus:border-green-500'
                      }`}
                    />
                  </div>
                </>
              ) : (
                /* Regular Mode Inputs */
                <>
                  <div className="flex flex-col items-start">
                    <label className="text-xs text-gray-400 mb-1">Capital per Stock</label>
                    <input
                      type="number"
                      value={initialCapital}
                      onChange={(e) => setInitialCapital(Number(e.target.value))}
                      className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-32 focus:outline-none focus:border-primary-500"
                    />
                  </div>
                </>
              )}
              
              <div className="flex flex-col items-end">
                <label className="text-xs text-gray-400 mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary-500"
                />
              </div>
              <div className="flex flex-col items-end">
                <label className="text-xs text-gray-400 mb-1">End Date (Optional)</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary-500"
                  placeholder="Latest data"
                />
              </div>
              <button
                onClick={runBacktest}
                disabled={loading}
                className={`${
                  strategy === 'smart' 
                    ? 'bg-green-600 hover:bg-green-700' 
                    : strategy === 'tscore'
                    ? 'bg-purple-600 hover:bg-purple-700'
                    : 'bg-primary-600 hover:bg-primary-700'
                } disabled:bg-gray-700 px-6 py-2 rounded-lg font-semibold flex items-center gap-2 transition-colors mt-5`}
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Running...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    {strategy === 'smart' ? 'Smart Backtest' : strategy === 'tscore' ? 'HA+TScore Backtest' : 'Run Backtest'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* T-Score Strategy Info Banner */}
        {backtestResult && backtestResult.params?.strategy === 'heikin_ashi_tscore' && backtestResult.tscore_stats && (
          <div className="mb-6 bg-gradient-to-r from-purple-900/50 to-indigo-900/30 rounded-xl p-6 border border-purple-700/50">
            <div className="flex items-center gap-3 mb-4">
              <TrendingUp className="text-purple-400" size={24} />
              <h3 className="text-xl font-bold text-purple-400">Heikin Ashi + T-Score Strategy</h3>
            </div>
            <div className="mb-4 text-sm text-gray-300">
              <p>This strategy combines <span className="font-bold text-purple-300">Heikin Ashi pattern recognition</span> for entry signals with <span className="font-bold text-purple-300">T-Score momentum filtering</span> for intelligent stock prioritization.</p>
              <p className="mt-2">When multiple stocks signal on the same day, stocks with higher T-Score get priority (stronger momentum, better technical position).</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div>
                <p className="text-sm text-gray-400">Total Investment</p>
                <p className="text-lg font-bold text-white">₹{(backtestResult.total_initial_capital ?? 0).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Max Positions</p>
                <p className="text-lg font-bold text-white">{backtestResult.params?.max_concurrent_positions ?? numStocksToSelect}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Avg T-Score at Entry</p>
                <p className="text-lg font-bold text-purple-400">{backtestResult.tscore_stats?.avg_tscore_at_entry ?? 0}%</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Stocks Traded</p>
                <p className="text-lg font-bold text-white">{backtestResult.num_stocks_processed ?? 0}</p>
              </div>
            </div>
            <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-800/50">
              <p className="text-sm font-bold text-purple-300 mb-2">T-Score Components:</p>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• <span className="text-purple-400">SMA21, SMA50, SMA200</span>: Price vs moving averages (3 points)</li>
                <li>• <span className="text-purple-400">7-Day Momentum</span>: Close vs 7-day max (2 points)</li>
                <li>• <span className="text-purple-400">52-Week High Range</span>: Proximity to 52-week high (0-3 points)</li>
                <li className="mt-2 font-bold">• Higher T-Score = Stronger stock = Higher entry priority</li>
              </ul>
            </div>
          </div>
        )}

        {/* Smart Selection Info Banner */}
        {backtestResult && backtestResult.selection_info && (
          <div className="mb-6 bg-gradient-to-r from-green-900/50 to-emerald-900/30 rounded-xl p-6 border border-green-700/50">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="text-green-400" size={24} />
              <h3 className="text-xl font-bold text-green-400">Smart Stock Selection</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div>
                <p className="text-sm text-gray-400">Total Investment</p>
                <p className="text-lg font-bold text-white">₹{(backtestResult.selection_info?.total_investment ?? 0).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Stocks Selected</p>
                <p className="text-lg font-bold text-white">{backtestResult.selection_info?.num_stocks_selected ?? 0} of {backtestResult.selection_info?.total_stocks_analyzed ?? 0} analyzed</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Capital per Stock</p>
                <p className="text-lg font-bold text-white">₹{Math.round(backtestResult.selection_info?.capital_per_stock ?? 0).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Ranking Period</p>
                <p className="text-lg font-bold text-white">{backtestResult.selection_info?.ranking_period_months ?? 0} months</p>
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">Selected Stocks (ranked by composite score):</p>
              <div className="flex flex-wrap gap-2">
                {backtestResult.selection_info.selected_stocks?.map((stock) => (
                  <div key={stock.symbol} className="bg-gray-900/50 rounded-lg px-3 py-2 text-sm border border-gray-700 hover:border-green-600 transition-colors">
                    <span className="font-bold text-green-400">{stock.symbol}</span>
                    <span className="text-gray-400 ml-2">WR: {(stock.historical_win_rate ?? 0).toFixed(1)}%</span>
                    <span className="text-gray-500 ml-2">Score: {stock.ranking_score ?? 0}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Portfolio Summary */}
        {backtestResult && (
          <>
            {/* Strategy Info Banner */}
            <div className="mb-6 bg-gradient-to-r from-gray-900 to-gray-800 rounded-xl p-4 border border-gray-700">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-400">Active Strategy:</span>
                  <span className="text-lg font-bold text-primary-400">Heikin Ashi</span>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <span>Period: {backtestResult.params?.start_date ?? ''} to {backtestResult.params?.end_date || 'Latest'}</span>
                  {!backtestResult.selection_info && backtestResult.params?.initial_capital && (
                    <span>Capital: ₹{backtestResult.params.initial_capital.toLocaleString()}</span>
                  )}
                </div>
              </div>
            </div>
            
            <div className="mb-8 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-xl p-6 border border-blue-700/50">
              <div className="flex items-center gap-3 mb-2">
                <DollarSign className="text-blue-400" size={24} />
                <span className="text-sm text-gray-300">Total P&L</span>
              </div>
              <div
                className={`text-3xl font-bold ${
                  (backtestResult.total_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {(backtestResult.total_pnl ?? 0) >= 0 ? '+' : ''}
                {formatINR(backtestResult.total_pnl ?? 0, 2)}
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-6 border border-purple-700/50">
              <div className="flex items-center gap-3 mb-2">
                <TrendingUp className="text-purple-400" size={24} />
                <span className="text-sm text-gray-300">Total Return</span>
              </div>
              <div
                className={`text-3xl font-bold ${
                  (backtestResult.total_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {(backtestResult.total_return_pct ?? 0) >= 0 ? '+' : ''}
                {(backtestResult.total_return_pct ?? 0).toFixed(2)}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-6 border border-green-700/50">
              <div className="flex items-center gap-3 mb-2">
                <DollarSign className="text-green-400" size={24} />
                <span className="text-sm text-gray-300">Final Capital</span>
              </div>
              <div className="text-3xl font-bold text-white">
                {formatINR(backtestResult.total_final_capital ?? 0, 2)}
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-900/50 to-orange-800/30 rounded-xl p-6 border border-orange-700/50">
              <div className="flex items-center gap-3 mb-2">
                <Activity className="text-orange-400" size={24} />
                <span className="text-sm text-gray-300">Stocks Processed</span>
              </div>
              <div className="text-3xl font-bold text-white">
                {backtestResult.num_stocks_processed ?? 0}
              </div>
            </div>
          </div>
          </>
        )}

        {/* Overall Portfolio Capital Growth Chart */}
        {backtestResult && backtestResult.stocks && (() => {
          const allTrades = Object.values(backtestResult.stocks).flatMap(stock => stock.trades);
          return (
            <div className="mb-8">
              <CapitalGrowthChart
                initialCapital={backtestResult.total_initial_capital ?? 0}
                finalCapital={backtestResult.total_final_capital ?? 0}
                trades={allTrades}
                title="Overall Portfolio Capital Growth"
              />
            </div>
          );
        })()}

        {/* Info Banner to Guide Users */}
        {backtestResult && (
          <div className="mb-4 bg-gradient-to-r from-blue-900/40 to-purple-900/40 rounded-lg p-4 border border-blue-700/50">
            <div className="flex items-center gap-3">
              <BarChart3 className="text-blue-400 flex-shrink-0" size={24} />
              <p className="text-sm text-gray-200">
                <span className="font-semibold text-blue-300">Explore detailed analytics</span> using the tabs below: 
                Portfolio Overview, Day-wise Timeline, Open Positions, and Individual Stock Performance
              </p>
            </div>
          </div>
        )}

        {/* Tabs Navigation */}
        {backtestResult && (
          <div className="mb-8">
            <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden shadow-lg">
              {/* Tab Header with better visibility */}
              <div className="bg-gradient-to-r from-gray-800 to-gray-900 border-b-2 border-blue-600">
                <div className="flex overflow-x-auto">
                  <button
                    onClick={() => setActiveTab('overview')}
                    className={`flex items-center gap-2 px-6 py-4 font-semibold transition-all whitespace-nowrap relative ${
                      activeTab === 'overview'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <Briefcase size={20} />
                    Portfolio Overview
                    {activeTab === 'overview' && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-400"></div>
                    )}
                  </button>
                  <button
                    onClick={() => setActiveTab('timeline')}
                    className={`flex items-center gap-2 px-6 py-4 font-semibold transition-all whitespace-nowrap relative ${
                      activeTab === 'timeline'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <Calendar size={20} />
                    Day-wise Timeline
                    {activeTab === 'timeline' && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-400"></div>
                    )}
                  </button>
                  <button
                    onClick={() => setActiveTab('positions')}
                    className={`flex items-center gap-2 px-6 py-4 font-semibold transition-all whitespace-nowrap relative ${
                      activeTab === 'positions'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <ListChecks size={20} />
                    Open Positions
                    {activeTab === 'positions' && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-400"></div>
                    )}
                  </button>
                  <button
                    onClick={() => setActiveTab('stocks')}
                    className={`flex items-center gap-2 px-6 py-4 font-semibold transition-all whitespace-nowrap relative ${
                      activeTab === 'stocks'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <PieChart size={20} />
                    Stock Analysis
                    {activeTab === 'stocks' && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-400"></div>
                    )}
                  </button>
                  <button
                    onClick={() => setActiveTab('tscore')}
                    className={`flex items-center gap-2 px-6 py-4 font-semibold transition-all whitespace-nowrap relative ${
                      activeTab === 'tscore'
                        ? 'bg-purple-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <Activity size={20} />
                    T-Score Dashboard
                    {activeTab === 'tscore' && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-purple-400"></div>
                    )}
                  </button>
                  <button
                    onClick={() => setActiveTab('settings')}
                    className={`flex items-center gap-2 px-6 py-4 font-semibold transition-all whitespace-nowrap relative ${
                      activeTab === 'settings'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <Settings size={20} />
                    Data Settings
                    {activeTab === 'settings' && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-400"></div>
                    )}
                  </button>
                  <button
                    onClick={() => setActiveTab('instructions')}
                    className={`flex items-center gap-2 px-6 py-4 font-semibold transition-all whitespace-nowrap relative ${
                      activeTab === 'instructions'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <BookOpen size={20} />
                    Instructions
                    {activeTab === 'instructions' && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-400"></div>
                    )}
                  </button>
                </div>
              </div>

              {/* Tab Content */}
              <div className="p-6 fade-in">
                {activeTab === 'overview' && (
                  <PortfolioSummary backtestResult={backtestResult} />
                )}

                {activeTab === 'timeline' && backtestResult.stocks && (() => {
                  const allTrades = Object.values(backtestResult.stocks).flatMap(stock => stock.trades);
                  return (
                    <PortfolioTimeline 
                      trades={allTrades}
                      initialCapital={backtestResult.total_initial_capital ?? 0}
                    />
                  );
                })()}

                {activeTab === 'positions' && (
                  <CurrentPositions backtestResult={backtestResult} />
                )}

                {activeTab === 'stocks' && (
                  <div>
                    {/* Search Bar */}
                    <div className="mb-6">
                      <div className="relative">
                        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                        <input
                          type="text"
                          placeholder="Search stocks..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="w-full bg-gray-800 border border-gray-700 rounded-xl pl-12 pr-4 py-3 focus:outline-none focus:border-primary-500 transition-colors"
                        />
                      </div>
                    </div>

                    {/* Stock Count Info */}
                    {sortedStocks.length > 0 && (
                      <div className="mb-4 flex items-center justify-between">
                        <p className="text-sm text-gray-400">
                          Showing <span className="font-semibold text-white">{sortedStocks.length}</span> stock{sortedStocks.length !== 1 ? 's' : ''}
                          {searchQuery && <span> matching "{searchQuery}"</span>}
                        </p>
                      </div>
                    )}

                    {/* Stock Grid */}
                    {sortedStocks.length > 0 ? (
                      <div className="max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                          {sortedStocks.map(([symbol, result]) => (
                            <StockCard
                              key={symbol}
                              symbol={symbol}
                              result={result}
                              onClick={() => handleStockSelect(symbol)}
                            />
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12">
                        <BarChart3 className="mx-auto text-gray-600 mb-4" size={48} />
                        <p className="text-gray-400 text-lg mb-2">
                          {searchQuery ? 'No stocks found matching your search' : 'No stocks available'}
                        </p>
                        {searchQuery && (
                          <p className="text-gray-500 text-sm">Try adjusting your search query</p>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'tscore' && (
                  <TScoreDashboard />
                )}

                {activeTab === 'settings' && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <SymbolManager />
                      <DataSyncStatus />
                    </div>
                  </div>
                )}

                {activeTab === 'instructions' && (
                  <Instructions />
                )}
              </div>
            </div>
          </div>
        )}

        {/* Statistics Section */}
        {backtestResult && backtestResult.statistics && showStatistics && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold flex items-center gap-2">
                <PieChart className="text-primary-500" size={28} />
                Strategy Performance Statistics
              </h2>
              <button
                onClick={() => setShowStatistics(!showStatistics)}
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                {showStatistics ? 'Hide' : 'Show'} Statistics
              </button>
            </div>
            <Statistics statistics={backtestResult.statistics} />
          </div>
        )}

        {/* Selected Stock Details */}
        {selectedSymbol && backtestResult && (
          <div className="space-y-6">
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h2 className="text-2xl font-bold mb-6">{selectedSymbol} Analysis</h2>
              
              {/* Capital Growth Chart for Selected Stock */}
              <div className="mb-6">
                <CapitalGrowthChart
                  initialCapital={backtestResult.stocks[selectedSymbol]?.initial_capital || 0}
                  finalCapital={backtestResult.stocks[selectedSymbol]?.final_capital || 0}
                  trades={backtestResult.stocks[selectedSymbol]?.trades || []}
                  title={`${selectedSymbol} Capital Growth`}
                />
              </div>
              
              {/* Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                {stockData.length > 0 && (
                  <TradingChart
                    data={stockData.map((d) => ({
                      time: d.Date,
                      open: d.open,
                      high: d.high,
                      low: d.low,
                      close: d.close,
                    }))}
                    trades={backtestResult.stocks[selectedSymbol]?.trades || []}
                    title="Normal Candlesticks"
                  />
                )}
                {haData.length > 0 && (
                  <TradingChart
                    data={haData.map((d) => ({
                      time: d.Date,
                      open: d.ha_open,
                      high: d.ha_high,
                      low: d.ha_low,
                      close: d.ha_close,
                    }))}
                    trades={backtestResult.stocks[selectedSymbol]?.trades || []}
                    title="Heikin Ashi"
                  />
                )}
              </div>
            </div>

            {/* Trade List */}
            <TradeList
              trades={backtestResult.stocks[selectedSymbol]?.trades || []}
              symbol={selectedSymbol}
            />
          </div>
        )}

        {/* Welcome Message */}
        {!backtestResult && !loading && (
          <div className="text-center py-20">
            <BarChart3 className="mx-auto text-gray-600 mb-6" size={80} />
            <h2 className="text-3xl font-bold mb-4">Welcome to Heikin Ashi Strategy Dashboard</h2>
            <p className="text-gray-400 text-lg mb-8">
              Choose your mode and configure parameters to analyze performance
            </p>
            
            {/* Settings Management Section */}
            <div className="mb-12">
              <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden max-w-6xl mx-auto">
                <div className="bg-gray-800 p-4 border-b border-gray-700">
                  <h3 className="text-xl font-bold flex items-center gap-2 justify-center">
                    <Settings size={24} />
                    Data & Symbol Management
                  </h3>
                </div>
                <div className="p-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <SymbolManager />
                    <DataSyncStatus />
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto mb-8">
              {/* All Stocks Mode */}
              <div className="bg-gray-900/50 rounded-xl p-6 border border-gray-800">
                <h3 className="text-xl font-semibold mb-4 text-blue-400">All Stocks Mode</h3>
                <ul className="text-left text-gray-300 space-y-2 text-sm">
                  <li>• Test Heikin Ashi strategy on all available stocks</li>
                  <li>• Set equal capital per stock</li>
                  <li>• Automatic stop-loss trailing</li>
                  <li>• Comprehensive analysis of all trades</li>
                </ul>
              </div>
              
              {/* Smart Select Mode */}
              <div className="bg-gray-900/50 rounded-xl p-6 border border-green-800">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="text-green-400" size={20} />
                  <h3 className="text-xl font-semibold text-green-400">Smart Select Mode</h3>
                </div>
                <ul className="text-left text-gray-300 space-y-2 text-sm">
                  <li>• Set total investment amount</li>
                  <li>• Choose number of stocks to trade</li>
                  <li>• AI selects best performing stocks</li>
                  <li>• Based on historical win rates</li>
                  <li>• Equal capital distribution</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
