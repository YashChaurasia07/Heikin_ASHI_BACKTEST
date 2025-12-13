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
import { BarChart3, TrendingUp, DollarSign, Activity, Search, Play, Loader2, PieChart, Sparkles, Calendar, ListChecks, Briefcase } from 'lucide-react';
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
  const [activeTab, setActiveTab] = useState<'overview' | 'stocks' | 'timeline' | 'positions'>('overview');
  
  // Smart backtest mode
  const [useSmartMode, setUseSmartMode] = useState(false);
  const [totalInvestment, setTotalInvestment] = useState(500000);
  const [numStocksToSelect, setNumStocksToSelect] = useState(10);

  const runBacktest = async () => {
    setLoading(true);
    setSelectedSymbol(null);
    try {
      let result;
      
      if (useSmartMode) {
        // Smart backtest - backend selects best stocks with priority filtering
        result = await api.runSmartBacktest(
          totalInvestment,
          numStocksToSelect,
          startDate,
          endDate,
          true // use cached
        );
      } else {
        // Regular backtest - all stocks
        result = await api.runBacktest(
          startDate, 
          endDate, 
          initialCapital
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

  const filteredStocks = backtestResult
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
              {/* Smart Mode Toggle */}
              <div className="flex flex-col items-start">
                <label className="text-xs text-gray-400 mb-1">Mode</label>
                <div className="flex items-center gap-2 bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5">
                  <button
                    onClick={() => setUseSmartMode(false)}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      !useSmartMode 
                        ? 'bg-primary-600 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    All Stocks
                  </button>
                  <button
                    onClick={() => setUseSmartMode(true)}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors flex items-center gap-1 ${
                      useSmartMode 
                        ? 'bg-green-600 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <Sparkles size={14} />
                    Smart Select
                  </button>
                </div>
              </div>

              {useSmartMode ? (
                /* Smart Mode Inputs */
                <>
                  <div className="flex flex-col items-start">
                    <label className="text-xs text-gray-400 mb-1">Total Investment</label>
                    <input
                      type="number"
                      value={totalInvestment}
                      onChange={(e) => setTotalInvestment(Number(e.target.value))}
                      className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-36 focus:outline-none focus:border-green-500"
                      placeholder="₹500000"
                    />
                  </div>
                  <div className="flex flex-col items-start">
                    <label className="text-xs text-gray-400 mb-1">Number of Stocks</label>
                    <input
                      type="number"
                      value={numStocksToSelect}
                      onChange={(e) => setNumStocksToSelect(Number(e.target.value))}
                      min={1}
                      max={50}
                      className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-28 focus:outline-none focus:border-green-500"
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
                  useSmartMode ? 'bg-green-600 hover:bg-green-700' : 'bg-primary-600 hover:bg-primary-700'
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
                    {useSmartMode ? 'Smart Backtest' : 'Run Backtest'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
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
                <p className="text-lg font-bold text-white">₹{backtestResult.selection_info.total_investment.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Stocks Selected</p>
                <p className="text-lg font-bold text-white">{backtestResult.selection_info.num_stocks_selected} of {backtestResult.selection_info.total_stocks_analyzed} analyzed</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Capital per Stock</p>
                <p className="text-lg font-bold text-white">₹{Math.round(backtestResult.selection_info.capital_per_stock).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Ranking Period</p>
                <p className="text-lg font-bold text-white">{backtestResult.selection_info.ranking_period_months} months</p>
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">Selected Stocks (ranked by composite score):</p>
              <div className="flex flex-wrap gap-2">
                {backtestResult.selection_info.selected_stocks.map((stock) => (
                  <div key={stock.symbol} className="bg-gray-900/50 rounded-lg px-3 py-2 text-sm border border-gray-700 hover:border-green-600 transition-colors">
                    <span className="font-bold text-green-400">{stock.symbol}</span>
                    <span className="text-gray-400 ml-2">WR: {stock.historical_win_rate.toFixed(1)}%</span>
                    <span className="text-gray-500 ml-2">Score: {stock.ranking_score}</span>
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
                  <span>Period: {backtestResult.params.start_date} to {backtestResult.params.end_date || 'Latest'}</span>
                  {!backtestResult.selection_info && (
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
                  backtestResult.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {backtestResult.total_pnl >= 0 ? '+' : ''}
                {formatINR(backtestResult.total_pnl, 2)}
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-6 border border-purple-700/50">
              <div className="flex items-center gap-3 mb-2">
                <TrendingUp className="text-purple-400" size={24} />
                <span className="text-sm text-gray-300">Total Return</span>
              </div>
              <div
                className={`text-3xl font-bold ${
                  backtestResult.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {backtestResult.total_return_pct >= 0 ? '+' : ''}
                {backtestResult.total_return_pct.toFixed(2)}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-6 border border-green-700/50">
              <div className="flex items-center gap-3 mb-2">
                <DollarSign className="text-green-400" size={24} />
                <span className="text-sm text-gray-300">Final Capital</span>
              </div>
              <div className="text-3xl font-bold text-white">
                {formatINR(backtestResult.total_final_capital, 2)}
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-900/50 to-orange-800/30 rounded-xl p-6 border border-orange-700/50">
              <div className="flex items-center gap-3 mb-2">
                <Activity className="text-orange-400" size={24} />
                <span className="text-sm text-gray-300">Stocks Processed</span>
              </div>
              <div className="text-3xl font-bold text-white">
                {backtestResult.num_stocks_processed}
              </div>
            </div>
          </div>
          </>
        )}

        {/* Overall Portfolio Capital Growth Chart */}
        {backtestResult && (() => {
          const allTrades = Object.values(backtestResult.stocks).flatMap(stock => stock.trades);
          return (
            <div className="mb-8">
              <CapitalGrowthChart
                initialCapital={backtestResult.total_initial_capital}
                finalCapital={backtestResult.total_final_capital}
                trades={allTrades}
                title="Overall Portfolio Capital Growth"
              />
            </div>
          );
        })()}

        {/* Tabs Navigation */}
        {backtestResult && (
          <div className="mb-8">
            <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
              <div className="flex border-b border-gray-800 overflow-x-auto">
                <button
                  onClick={() => setActiveTab('overview')}
                  className={`flex items-center gap-2 px-6 py-4 font-semibold transition-colors whitespace-nowrap ${
                    activeTab === 'overview'
                      ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }`}
                >
                  <Briefcase size={20} />
                  Portfolio Overview
                </button>
                <button
                  onClick={() => setActiveTab('timeline')}
                  className={`flex items-center gap-2 px-6 py-4 font-semibold transition-colors whitespace-nowrap ${
                    activeTab === 'timeline'
                      ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }`}
                >
                  <Calendar size={20} />
                  Day-wise Timeline
                </button>
                <button
                  onClick={() => setActiveTab('positions')}
                  className={`flex items-center gap-2 px-6 py-4 font-semibold transition-colors whitespace-nowrap ${
                    activeTab === 'positions'
                      ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }`}
                >
                  <ListChecks size={20} />
                  Open Positions
                </button>
                <button
                  onClick={() => setActiveTab('stocks')}
                  className={`flex items-center gap-2 px-6 py-4 font-semibold transition-colors whitespace-nowrap ${
                    activeTab === 'stocks'
                      ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }`}
                >
                  <BarChart3 size={20} />
                  Individual Stocks
                </button>
              </div>

              {/* Tab Content */}
              <div className="p-6 fade-in">
                {activeTab === 'overview' && (
                  <PortfolioSummary backtestResult={backtestResult} />
                )}

                {activeTab === 'timeline' && (() => {
                  const allTrades = Object.values(backtestResult.stocks).flatMap(stock => stock.trades);
                  return (
                    <PortfolioTimeline 
                      trades={allTrades}
                      initialCapital={backtestResult.total_initial_capital}
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

                    {/* Stock Grid */}
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
                  </div>
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
