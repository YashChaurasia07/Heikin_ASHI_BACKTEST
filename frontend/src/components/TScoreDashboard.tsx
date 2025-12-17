import { useState, useEffect } from 'react';
import { Activity, Search, Filter, TrendingUp, TrendingDown, Calendar, Download, Loader2 } from 'lucide-react';
import api from '../services/api';

interface TScoreData {
  symbol: string;
  t_score: number;
  latest_date: string;
  latest_close: number;
  latest_volume: number;
  sma_21: number | null;
  sma_50: number | null;
  sma_200: number | null;
  week_52_high: number;
  distance_from_52w_high_pct: number;
  above_sma_21: boolean | null;
  above_sma_50: boolean | null;
  above_sma_200: boolean | null;
}

interface TScoreResponse {
  total_stocks: number;
  interval: string;
  as_of_date: string;
  tscores: TScoreData[];
  summary: {
    avg_tscore: number;
    max_tscore: number;
    min_tscore: number;
    stocks_above_80: number;
    stocks_above_60: number;
    stocks_below_40: number;
  };
}

export default function TScoreDashboard() {
  const [tscoreData, setTScoreData] = useState<TScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterScore, setFilterScore] = useState<'all' | '80+' | '60+' | '40-'>('all');
  const [filterSMA, setFilterSMA] = useState<'all' | 'above_50' | 'above_200' | 'below_50'>('all');
  const [sortBy, setSortBy] = useState<'tscore' | 'symbol' | 'close' | 'distance'>('tscore');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [interval, setInterval] = useState<'daily' | 'weekly'>('daily');

  const fetchTScores = async () => {
    setLoading(true);
    try {
      const data = await api.getTScores(interval);
      setTScoreData(data);
    } catch (error: any) {
      console.error('Error fetching T-Scores:', error);
      const errorMsg = error?.response?.data?.detail || error?.message || 'Unknown error';
      alert(`Error loading T-Scores: ${errorMsg}\n\nPlease make sure:\n1. Backend server is running\n2. Data is synced (use Settings > Data Sync)\n3. MongoDB is connected`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTScores();
  }, [interval]);

  const filterAndSortStocks = () => {
    if (!tscoreData) return [];

    let filtered = tscoreData.tscores.filter(stock => {
      // Search filter
      const matchesSearch = stock.symbol.toLowerCase().includes(searchQuery.toLowerCase());
      
      // T-Score filter
      let matchesScore = true;
      if (filterScore === '80+') matchesScore = stock.t_score >= 80;
      else if (filterScore === '60+') matchesScore = stock.t_score >= 60;
      else if (filterScore === '40-') matchesScore = stock.t_score < 40;
      
      // SMA filter
      let matchesSMA = true;
      if (filterSMA === 'above_50') matchesSMA = stock.above_sma_50 === true;
      else if (filterSMA === 'above_200') matchesSMA = stock.above_sma_200 === true;
      else if (filterSMA === 'below_50') matchesSMA = stock.above_sma_50 === false;
      
      return matchesSearch && matchesScore && matchesSMA;
    });

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'tscore':
          comparison = a.t_score - b.t_score;
          break;
        case 'symbol':
          comparison = a.symbol.localeCompare(b.symbol);
          break;
        case 'close':
          comparison = a.latest_close - b.latest_close;
          break;
        case 'distance':
          comparison = a.distance_from_52w_high_pct - b.distance_from_52w_high_pct;
          break;
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  };

  const downloadCSV = () => {
    if (!tscoreData) return;

    const stocks = filterAndSortStocks();
    
    const headers = [
      'Symbol',
      'T-Score',
      'Latest Date',
      'Close Price',
      'Distance from 52W High %',
      'SMA 21',
      'SMA 50',
      'SMA 200',
      'Above SMA 50',
      'Above SMA 200'
    ];

    const rows = stocks.map(stock => [
      stock.symbol,
      stock.t_score.toFixed(2),
      stock.latest_date,
      stock.latest_close.toFixed(2),
      stock.distance_from_52w_high_pct.toFixed(2),
      stock.sma_21?.toFixed(2) || 'N/A',
      stock.sma_50?.toFixed(2) || 'N/A',
      stock.sma_200?.toFixed(2) || 'N/A',
      stock.above_sma_50 ? 'Yes' : 'No',
      stock.above_sma_200 ? 'Yes' : 'No'
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `t-scores_${interval}_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredStocks = filterAndSortStocks();

  const getTScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-blue-400';
    if (score >= 40) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getTScoreBg = (score: number) => {
    if (score >= 80) return 'bg-green-900/30';
    if (score >= 60) return 'bg-blue-900/30';
    if (score >= 40) return 'bg-yellow-900/30';
    return 'bg-red-900/30';
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Activity className="text-purple-400" size={28} />
            T-Score Dashboard
          </h2>
          <p className="text-sm text-gray-400 mt-1">
            Technical Score analysis for all stocks
            {tscoreData && ` • Data as of ${tscoreData.as_of_date}`}
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Interval Toggle */}
          <div className="flex gap-2 bg-gray-800 p-1 rounded-lg">
            <button
              onClick={() => setInterval('daily')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                interval === 'daily'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Daily
            </button>
            <button
              onClick={() => setInterval('weekly')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                interval === 'weekly'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Weekly
            </button>
          </div>

          {/* Refresh Button */}
          <button
            onClick={fetchTScores}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors font-medium text-sm disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Loading...
              </>
            ) : (
              <>
                <Calendar size={16} />
                Refresh
              </>
            )}
          </button>

          {/* Download CSV */}
          <button
            onClick={downloadCSV}
            disabled={!tscoreData}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium text-sm disabled:opacity-50"
          >
            <Download size={16} />
            CSV
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      {tscoreData && (
        <div className="grid grid-cols-6 gap-4 mb-6">
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="text-xs text-gray-400 mb-1">Total Stocks</div>
            <div className="text-2xl font-bold text-white">{tscoreData.total_stocks}</div>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="text-xs text-gray-400 mb-1">Avg T-Score</div>
            <div className={`text-2xl font-bold ${getTScoreColor(tscoreData.summary.avg_tscore)}`}>
              {tscoreData.summary.avg_tscore.toFixed(1)}
            </div>
          </div>
          <div className="bg-green-900/30 rounded-lg p-4 border border-green-700">
            <div className="text-xs text-green-400 mb-1">Score ≥ 80</div>
            <div className="text-2xl font-bold text-green-400">{tscoreData.summary.stocks_above_80}</div>
          </div>
          <div className="bg-blue-900/30 rounded-lg p-4 border border-blue-700">
            <div className="text-xs text-blue-400 mb-1">Score ≥ 60</div>
            <div className="text-2xl font-bold text-blue-400">{tscoreData.summary.stocks_above_60}</div>
          </div>
          <div className="bg-red-900/30 rounded-lg p-4 border border-red-700">
            <div className="text-xs text-red-400 mb-1">Score &lt; 40</div>
            <div className="text-2xl font-bold text-red-400">{tscoreData.summary.stocks_below_40}</div>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="text-xs text-gray-400 mb-1">Filtered</div>
            <div className="text-2xl font-bold text-purple-400">{filteredStocks.length}</div>
          </div>
        </div>
      )}

      {/* Filters and Search */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        {/* Search */}
        <div className="flex-1 min-w-[250px]">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
            <input
              type="text"
              placeholder="Search stocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
            />
          </div>
        </div>

        {/* T-Score Filter */}
        <select
          value={filterScore}
          onChange={(e) => setFilterScore(e.target.value as any)}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All T-Scores</option>
          <option value="80+">T-Score ≥ 80</option>
          <option value="60+">T-Score ≥ 60</option>
          <option value="40-">T-Score &lt; 40</option>
        </select>

        {/* SMA Filter */}
        <select
          value={filterSMA}
          onChange={(e) => setFilterSMA(e.target.value as any)}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All SMA</option>
          <option value="above_50">Above SMA 50</option>
          <option value="above_200">Above SMA 200</option>
          <option value="below_50">Below SMA 50</option>
        </select>

        {/* Sort By */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="tscore">Sort by T-Score</option>
          <option value="symbol">Sort by Symbol</option>
          <option value="close">Sort by Price</option>
          <option value="distance">Sort by 52W Distance</option>
        </select>

        {/* Sort Order */}
        <button
          onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white hover:bg-gray-700 transition-colors flex items-center gap-2"
        >
          {sortOrder === 'desc' ? (
            <>
              <TrendingDown size={16} />
              Desc
            </>
          ) : (
            <>
              <TrendingUp size={16} />
              Asc
            </>
          )}
        </button>
      </div>

      {/* Stock List */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="animate-spin text-purple-400" size={48} />
        </div>
      ) : filteredStocks.length === 0 ? (
        <div className="text-center py-20">
          <Filter className="mx-auto text-gray-600 mb-4" size={48} />
          <p className="text-gray-400 text-lg mb-2">No stocks match the filters</p>
          <p className="text-gray-500 text-sm">Try adjusting your search or filters</p>
        </div>
      ) : (
        <div className="max-h-[600px] overflow-y-auto custom-scrollbar">
          <table className="w-full">
            <thead className="sticky top-0 bg-gray-800 z-10">
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-300">Symbol</th>
                <th className="text-center py-3 px-4 text-sm font-semibold text-gray-300">T-Score</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Price</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">52W Distance</th>
                <th className="text-center py-3 px-4 text-sm font-semibold text-gray-300">SMA 50</th>
                <th className="text-center py-3 px-4 text-sm font-semibold text-gray-300">SMA 200</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-gray-300">Latest Date</th>
              </tr>
            </thead>
            <tbody>
              {filteredStocks.map((stock, index) => (
                <tr
                  key={stock.symbol}
                  className={`border-b border-gray-800 hover:bg-gray-800/50 transition-colors ${
                    index % 2 === 0 ? 'bg-gray-900/50' : ''
                  }`}
                >
                  <td className="py-3 px-4">
                    <div className="font-semibold text-white">{stock.symbol}</div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center justify-center">
                      <div className={`px-3 py-1 rounded-full ${getTScoreBg(stock.t_score)}`}>
                        <span className={`font-bold ${getTScoreColor(stock.t_score)}`}>
                          {stock.t_score.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right text-white">
                    ₹{stock.latest_close.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span className={stock.distance_from_52w_high_pct < 10 ? 'text-green-400' : 'text-gray-400'}>
                      {stock.distance_from_52w_high_pct.toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    {stock.above_sma_50 === true ? (
                      <span className="text-green-400">✓</span>
                    ) : stock.above_sma_50 === false ? (
                      <span className="text-red-400">✗</span>
                    ) : (
                      <span className="text-gray-500">-</span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {stock.above_sma_200 === true ? (
                      <span className="text-green-400">✓</span>
                    ) : stock.above_sma_200 === false ? (
                      <span className="text-red-400">✗</span>
                    ) : (
                      <span className="text-gray-500">-</span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-right text-gray-400 text-sm">
                    {stock.latest_date}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
