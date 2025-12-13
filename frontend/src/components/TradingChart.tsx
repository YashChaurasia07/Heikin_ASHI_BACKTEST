import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { Trade } from '../services/api';
import { formatINR } from '../utils/formatters';

interface ChartData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface TradingChartProps {
  data: ChartData[];
  trades: Trade[];
  title: string;
}

interface OHLCValues {
  open: number;
  high: number;
  low: number;
  close: number;
}

const TradingChart = ({ data, trades, title }: TradingChartProps) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [ohlcValues, setOhlcValues] = useState<OHLCValues | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

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
      height: 500,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: 1,
      },
      localization: {
        priceFormatter: (price: number) => {
          const absPrice = Math.abs(price);
          if (absPrice >= 10000000) {
            return `${(price / 10000000).toFixed(2)}Cr`;
          } else if (absPrice >= 100000) {
            return `${(price / 100000).toFixed(2)}L`;
          } else if (absPrice >= 1000) {
            return `${(price / 1000).toFixed(2)}K`;
          } else {
            return price.toFixed(2);
          }
        },
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;

    // Subscribe to crosshair move to show OHLC values
    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData.get(candlestickSeries)) {
        // Show last candle values when not hovering
        if (data.length > 0) {
          const lastCandle = data[data.length - 1];
          setOhlcValues({
            open: lastCandle.open,
            high: lastCandle.high,
            low: lastCandle.low,
            close: lastCandle.close,
          });
        }
        return;
      }

      const candleData = param.seriesData.get(candlestickSeries) as CandlestickData;
      if (candleData) {
        setOhlcValues({
          open: candleData.open,
          high: candleData.high,
          low: candleData.low,
          close: candleData.close,
        });
      }
    });

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!candlestickSeriesRef.current || !data.length) return;

    // Convert data to lightweight-charts format
    const chartData: CandlestickData[] = data.map((item) => ({
      time: item.time as Time,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));

    candlestickSeriesRef.current.setData(chartData);

    // Add buy/sell markers
    if (trades.length > 0 && chartRef.current) {
      const markers = trades.flatMap((trade) => {
        const buyMarker = {
          time: trade.entry_date as Time,
          position: 'belowBar' as const,
          color: '#10b981',
          shape: 'arrowUp' as const,
          text: `Buy @ ${trade.entry_price.toFixed(2)}`,
        };

        const sellMarker = {
          time: trade.exit_date as Time,
          position: 'aboveBar' as const,
          color: trade.pnl >= 0 ? '#10b981' : '#ef4444',
          shape: 'arrowDown' as const,
          text: `Sell @ ${trade.exit_price.toFixed(2)} | ${trade.exit_type}`,
        };

        return [buyMarker, sellMarker];
      });

      candlestickSeriesRef.current.setMarkers(markers);
    }

    chartRef.current?.timeScale().fitContent();

    // Set initial OHLC values to last candle
    if (data.length > 0) {
      const lastCandle = data[data.length - 1];
      setOhlcValues({
        open: lastCandle.open,
        high: lastCandle.high,
        low: lastCandle.low,
        close: lastCandle.close,
      });
    }
  }, [data, trades]);

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        
        {/* OHLC Display Widget */}
        {ohlcValues && (
          <div className="flex gap-4 text-sm">
            <div className="flex flex-col">
              <span className="text-gray-500 text-xs">O</span>
              <span className="text-white font-medium">{formatINR(ohlcValues.open, 2)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-gray-500 text-xs">H</span>
              <span className="text-green-500 font-medium">{formatINR(ohlcValues.high, 2)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-gray-500 text-xs">L</span>
              <span className="text-red-500 font-medium">{formatINR(ohlcValues.low, 2)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-gray-500 text-xs">C</span>
              <span className={`font-medium ${ohlcValues.close >= ohlcValues.open ? 'text-green-500' : 'text-red-500'}`}>
                {formatINR(ohlcValues.close, 2)}
              </span>
            </div>
          </div>
        )}
      </div>
      
      <div
        ref={chartContainerRef}
        className="rounded-lg overflow-hidden border border-gray-800"
      />
    </div>
  );
};

export default TradingChart;
