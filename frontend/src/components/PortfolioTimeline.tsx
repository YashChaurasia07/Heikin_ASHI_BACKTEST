import { useState } from 'react';
import { Trade } from '../services/api';
import { Calendar, TrendingUp, TrendingDown, Activity, ChevronDown, ChevronUp, Download } from 'lucide-react';
import { formatINR } from '../utils/formatters';

interface DayPortfolio {
  date: string;
  entries: Trade[];
  exits: Trade[];
  activePositions: number;
  dayPnL: number;
  cumulativePnL: number;
  capitalDeployed: number;
}

interface PortfolioTimelineProps {
  trades: Trade[];
  initialCapital: number;
}

export default function PortfolioTimeline({ trades }: PortfolioTimelineProps) {
  const [expandedDate, setExpandedDate] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'all' | 'entries' | 'exits'>('all');

  // Check if trades are available
  if (!trades || trades.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-2xl font-bold flex items-center gap-2 mb-4">
          <Calendar className="text-blue-400" size={28} />
          Day-wise Portfolio Timeline
        </h2>
        <div className="text-center py-12">
          <Calendar className="mx-auto text-gray-600 mb-4" size={48} />
          <p className="text-gray-400 text-lg mb-2">No trades available</p>
          <p className="text-gray-500 text-sm">Run a backtest to see day-wise activity</p>
        </div>
      </div>
    );
  }

  // Group trades by date
  const groupTradesByDate = (): DayPortfolio[] => {
    const dateMap = new Map<string, { entries: Trade[], exits: Trade[] }>();
    
    trades.forEach(trade => {
      // Add to entry date
      if (!dateMap.has(trade.entry_date)) {
        dateMap.set(trade.entry_date, { entries: [], exits: [] });
      }
      dateMap.get(trade.entry_date)!.entries.push(trade);
      
      // Add to exit date
      if (!dateMap.has(trade.exit_date)) {
        dateMap.set(trade.exit_date, { entries: [], exits: [] });
      }
      dateMap.get(trade.exit_date)!.exits.push(trade);
    });

    // Sort dates and calculate metrics
    const sortedDates = Array.from(dateMap.keys()).sort();
    let cumulativePnL = 0;
    let activePositionsCount = 0;

    return sortedDates.map(date => {
      const dayData = dateMap.get(date)!;
      const dayPnL = dayData.exits.reduce((sum, t) => sum + t.pnl, 0);
      cumulativePnL += dayPnL;
      
      activePositionsCount += dayData.entries.length - dayData.exits.length;
      
      const capitalDeployed = dayData.entries.reduce((sum, t) => sum + (t.entry_price * t.qty), 0);

      return {
        date,
        entries: dayData.entries,
        exits: dayData.exits,
        activePositions: Math.max(0, activePositionsCount),
        dayPnL,
        cumulativePnL,
        capitalDeployed,
      };
    });
  };

  const portfolioTimeline = groupTradesByDate();

  const filteredTimeline = portfolioTimeline.filter(day => {
    if (viewMode === 'entries') return day.entries.length > 0;
    if (viewMode === 'exits') return day.exits.length > 0;
    return true;
  });

  // CSV Download Function
  const downloadCSV = () => {
    // Prepare CSV headers
    const headers = [
      'Date',
      'Day',
      'Entries Count',
      'Exits Count',
      'Active Positions',
      'Day P&L (₹)',
      'Cumulative P&L (₹)',
      'Capital Deployed (₹)'
    ];

    // Prepare CSV rows
    const rows = portfolioTimeline.map((day, index) => [
      day.date,
      (index + 1).toString(),
      day.entries.length.toString(),
      day.exits.length.toString(),
      day.activePositions.toString(),
      day.dayPnL.toFixed(2),
      day.cumulativePnL.toFixed(2),
      day.capitalDeployed.toFixed(2)
    ]);

    // Combine headers and rows
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `portfolio_timeline_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Calendar className="text-blue-400" size={28} />
          Portfolio Timeline
        </h2>
        
        <div className="flex items-center gap-3">
          {/* CSV Download Button */}
          <button
            onClick={downloadCSV}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium text-sm"
            title="Download Timeline as CSV"
          >
            <Download size={16} />
            Download CSV
          </button>

          {/* View Mode Toggle */}
          <div className="flex gap-2 bg-gray-800 p-1 rounded-lg">
            <button
              onClick={() => setViewMode('all')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'all' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              All Days
            </button>
            <button
              onClick={() => setViewMode('entries')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'entries' 
                  ? 'bg-green-600 text-white' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Entries
            </button>
            <button
              onClick={() => setViewMode('exits')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'exits' 
                  ? 'bg-red-600 text-white' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Exits
            </button>
          </div>
        </div>
      </div>

      {/* Scrollable Timeline */}
      <div className="max-h-[600px] overflow-y-auto pr-2 space-y-3 custom-scrollbar">
        {filteredTimeline.map((day, index) => (
          <div
            key={day.date}
            className="bg-gray-800/50 rounded-lg border border-gray-700 hover:border-blue-500 transition-all"
          >
            {/* Day Header */}
            <div
              onClick={() => setExpandedDate(expandedDate === day.date ? null : day.date)}
              className="p-4 cursor-pointer"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex flex-col">
                    <span className="text-sm font-semibold text-blue-400">
                      {new Date(day.date).toLocaleDateString('en-IN', { 
                        weekday: 'short', 
                        year: 'numeric', 
                        month: 'short', 
                        day: 'numeric' 
                      })}
                    </span>
                    <span className="text-xs text-gray-500">Day {index + 1}</span>
                  </div>
                  
                  <div className="flex gap-4">
                    {day.entries.length > 0 && (
                      <div className="flex items-center gap-1 bg-green-900/30 px-3 py-1 rounded-full">
                        <TrendingUp size={14} className="text-green-400" />
                        <span className="text-xs font-medium text-green-400">
                          {day.entries.length} {day.entries.length === 1 ? 'Entry' : 'Entries'}
                        </span>
                      </div>
                    )}
                    
                    {day.exits.length > 0 && (
                      <div className="flex items-center gap-1 bg-red-900/30 px-3 py-1 rounded-full">
                        <TrendingDown size={14} className="text-red-400" />
                        <span className="text-xs font-medium text-red-400">
                          {day.exits.length} {day.exits.length === 1 ? 'Exit' : 'Exits'}
                        </span>
                      </div>
                    )}
                    
                    {day.activePositions > 0 && (
                      <div className="flex items-center gap-1 bg-blue-900/30 px-3 py-1 rounded-full">
                        <Activity size={14} className="text-blue-400" />
                        <span className="text-xs font-medium text-blue-400">
                          {day.activePositions} Active
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="text-xs text-gray-400 mb-1">Day P&L</div>
                    <div className={`text-lg font-bold ${
                      day.dayPnL >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {day.dayPnL >= 0 ? '+' : ''}{formatINR(day.dayPnL, 2)}
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-xs text-gray-400 mb-1">Cumulative P&L</div>
                    <div className={`text-lg font-bold ${
                      day.cumulativePnL >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {day.cumulativePnL >= 0 ? '+' : ''}{formatINR(day.cumulativePnL, 2)}
                    </div>
                  </div>

                  {expandedDate === day.date ? (
                    <ChevronUp className="text-gray-400" size={20} />
                  ) : (
                    <ChevronDown className="text-gray-400" size={20} />
                  )}
                </div>
              </div>
            </div>

            {/* Expanded Details */}
            {expandedDate === day.date && (
              <div className="border-t border-gray-700 p-4 bg-gray-900/50">
                {/* Entries */}
                {day.entries.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-semibold text-green-400 mb-2 flex items-center gap-2">
                      <TrendingUp size={16} />
                      New Positions Entered ({day.entries.length})
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {day.entries.map((trade, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-green-500 transition-colors"
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <div className="text-base font-bold text-green-400 mb-1">
                                {trade.symbol}
                              </div>
                              <div className="text-sm text-gray-300">
                                Entry: ₹{trade.entry_price.toFixed(2)}
                              </div>
                              <div className="text-xs text-gray-400">
                                Qty: {Math.floor(trade.qty)} × ₹{trade.entry_price.toFixed(2)} = {formatINR(trade.entry_price * Math.floor(trade.qty), 2)}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Exits */}
                {day.exits.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold text-red-400 mb-2 flex items-center gap-2">
                      <TrendingDown size={16} />
                      Positions Closed ({day.exits.length})
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {day.exits.map((trade, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-red-500 transition-colors"
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <div className="text-base font-bold text-red-400 mb-1">
                                {trade.symbol}
                              </div>
                              <div className="text-sm text-gray-300">
                                Exit: ₹{trade.exit_price.toFixed(2)}
                              </div>
                              <div className="text-xs text-gray-400">
                                Entry: ₹{trade.entry_price.toFixed(2)} on {trade.entry_date}
                              </div>
                              <div className="text-xs text-gray-500">
                                {trade.exit_type}
                              </div>
                            </div>
                            <div className="text-right ml-2">
                              <div className={`text-base font-bold ${
                                trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                              }`}>
                                {trade.pnl >= 0 ? '+' : ''}{formatINR(trade.pnl, 2)}
                              </div>
                              <div className={`text-xs ${
                                trade.pnl >= 0 ? 'text-green-500' : 'text-red-500'
                              }`}>
                                {((trade.pnl / (trade.entry_price * trade.qty)) * 100).toFixed(2)}%
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="mt-6 pt-6 border-t border-gray-700 grid grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-sm text-gray-400 mb-1">Total Trading Days</div>
          <div className="text-xl font-bold text-white">{portfolioTimeline.length}</div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-400 mb-1">Total Entries</div>
          <div className="text-xl font-bold text-green-400">
            {portfolioTimeline.reduce((sum, d) => sum + d.entries.length, 0)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-400 mb-1">Total Exits</div>
          <div className="text-xl font-bold text-red-400">
            {portfolioTimeline.reduce((sum, d) => sum + d.exits.length, 0)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-400 mb-1">Final P&L</div>
          <div className={`text-xl font-bold ${
            portfolioTimeline[portfolioTimeline.length - 1]?.cumulativePnL >= 0 
              ? 'text-green-400' 
              : 'text-red-400'
          }`}>
            {formatINR(portfolioTimeline[portfolioTimeline.length - 1]?.cumulativePnL || 0, 2)}
          </div>
        </div>
      </div>
    </div>
  );
}
