import { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, LineData, Time } from 'lightweight-charts';
import { formatINR } from '../utils/formatters';
import { Trade } from '../services/api';

interface CapitalGrowthChartProps {
  initialCapital: number;
  finalCapital: number;
  trades: Trade[];
  title: string;
}

const CapitalGrowthChart = ({ initialCapital, finalCapital, trades, title }: CapitalGrowthChartProps) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const investedSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const capitalSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || trades.length === 0) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#0a0a0a' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#374151',
      },
    });

    chartRef.current = chart;

    // Create invested capital line (constant)
    const investedSeries = chart.addLineSeries({
      color: '#9ca3af',
      lineWidth: 2,
      title: 'Invested Capital',
      priceLineVisible: false,
      lastValueVisible: true,
    });
    investedSeriesRef.current = investedSeries;

    // Create capital growth line
    const capitalSeries = chart.addLineSeries({
      color: finalCapital >= initialCapital ? '#10b981' : '#ef4444',
      lineWidth: 3,
      title: 'Current Capital',
      priceLineVisible: true,
      lastValueVisible: true,
    });
    capitalSeriesRef.current = capitalSeries;

    // Calculate capital growth over time
    const capitalData: LineData[] = [];
    const investedData: LineData[] = [];
    
    let currentCapital = initialCapital;
    
    // Process each trade chronologically - sort by exit_date
    const sortedTrades = [...trades].sort((a, b) => 
      new Date(a.exit_date).getTime() - new Date(b.exit_date).getTime()
    );
    
    // Add initial point at the earliest entry date
    if (sortedTrades.length > 0) {
      const firstDate = sortedTrades.reduce((earliest, trade) => {
        const entryTime = new Date(trade.entry_date).getTime();
        const earliestTime = new Date(earliest).getTime();
        return entryTime < earliestTime ? trade.entry_date : earliest;
      }, sortedTrades[0].entry_date);
      
      capitalData.push({ time: firstDate as Time, value: currentCapital });
      investedData.push({ time: firstDate as Time, value: initialCapital });
    }

    // Group trades by exit_date to handle multiple trades on same date
    const tradesByDate = new Map<string, number>();
    sortedTrades.forEach((trade) => {
      const existingPnl = tradesByDate.get(trade.exit_date) || 0;
      tradesByDate.set(trade.exit_date, existingPnl + trade.pnl);
    });

    // Convert to sorted array of unique dates
    const sortedDates = Array.from(tradesByDate.entries())
      .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime());

    // Filter out any dates that are the same as the initial date
    const initialDateStr = sortedTrades.length > 0 ? capitalData[0].time as string : '';
    const filteredDates = sortedDates.filter(([date]) => date !== initialDateStr);

    filteredDates.forEach(([date, totalPnl]) => {
      // Add capital after all trades on this date complete
      currentCapital += totalPnl;
      capitalData.push({ 
        time: date as Time, 
        value: currentCapital 
      });
      investedData.push({ 
        time: date as Time, 
        value: initialCapital 
      });
    });

    // Set the data
    investedSeries.setData(investedData);
    capitalSeries.setData(capitalData);

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [initialCapital, finalCapital, trades]);

  if (trades.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4">{title}</h3>
        <div className="flex items-center justify-center h-[400px] text-gray-500">
          No trades to display
        </div>
      </div>
    );
  }

  const totalReturn = finalCapital - initialCapital;
  const returnPct = ((finalCapital - initialCapital) / initialCapital) * 100;

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-gray-400">Invested: </span>
            <span className="font-semibold">{formatINR(initialCapital, 2)}</span>
          </div>
          <div>
            <span className="text-gray-400">Final: </span>
            <span className={`font-semibold ${finalCapital >= initialCapital ? 'text-green-400' : 'text-red-400'}`}>
              {formatINR(finalCapital, 2)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Return: </span>
            <span className={`font-semibold ${totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {totalReturn >= 0 ? '+' : ''}{formatINR(totalReturn, 2)} ({returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>
      <div ref={chartContainerRef} />
    </div>
  );
};

export default CapitalGrowthChart;
