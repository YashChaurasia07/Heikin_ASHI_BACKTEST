import { BookOpen, TrendingUp, Calculator, BarChart, Target, Shield, Award, Activity, Zap, Brain, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { useState } from 'react';

interface SectionProps {
  title: string;
  icon: any;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function Section({ title, icon: Icon, children, defaultOpen = false }: SectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-5 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Icon className="text-blue-400" size={24} />
          <h2 className="text-xl font-bold text-white">{title}</h2>
        </div>
        {isOpen ? (
          <ChevronUp className="text-gray-400" size={20} />
        ) : (
          <ChevronDown className="text-gray-400" size={20} />
        )}
      </button>
      {isOpen && (
        <div className="p-6 pt-0 space-y-4 text-gray-300">
          {children}
        </div>
      )}
    </div>
  );
}

interface TermProps {
  term: string;
  definition: string;
  formula?: string;
  example?: string;
}

function Term({ term, definition, formula, example }: TermProps) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
      <h4 className="text-lg font-semibold text-blue-300 mb-2">{term}</h4>
      <p className="text-gray-300 mb-2">{definition}</p>
      {formula && (
        <div className="bg-gray-900 rounded p-3 my-2 font-mono text-sm text-green-300">
          <span className="text-gray-500">Formula: </span>{formula}
        </div>
      )}
      {example && (
        <div className="mt-2 text-sm text-gray-400 italic">
          <span className="text-blue-400">Example: </span>{example}
        </div>
      )}
    </div>
  );
}

export default function Instructions() {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/40 to-purple-900/40 rounded-xl p-8 border border-blue-700/30">
        <div className="flex items-center gap-3 mb-4">
          <BookOpen className="text-blue-400" size={40} />
          <h1 className="text-3xl font-bold text-white">Trading System Guide</h1>
        </div>
        <p className="text-gray-300 text-lg">
          Comprehensive documentation for understanding all metrics, calculations, and strategies used in this Heikin Ashi backtesting system.
        </p>
      </div>

      {/* Overview */}
      <Section title="System Overview" icon={Info} defaultOpen={true}>
        <p>
          This is an advanced backtesting system that combines <strong>Heikin Ashi candlestick patterns</strong> with multiple 
          strategic approaches to identify and backtest trading opportunities in the Indian stock market. The system analyzes 
          historical data to simulate trading strategies and provides detailed performance metrics.
        </p>
        <div className="bg-blue-900/30 rounded-lg p-4 border border-blue-700/30 mt-4">
          <h4 className="font-semibold text-blue-300 mb-2">Key Features:</h4>
          <ul className="list-disc list-inside space-y-1 text-gray-300">
            <li>Three distinct trading strategies (All Stocks, Smart Portfolio, HA + T-Score)</li>
            <li>Support for daily and weekly timeframes</li>
            <li>Advanced position sizing and capital management</li>
            <li>Comprehensive performance metrics and statistics</li>
            <li>Visual analysis with charts and timelines</li>
          </ul>
        </div>
      </Section>

      {/* Heikin Ashi Explanation */}
      <Section title="Heikin Ashi Candlesticks" icon={BarChart}>
        <p>
          Heikin Ashi is a modified candlestick charting technique that filters out market noise and helps identify trends more clearly.
        </p>
        
        <Term
          term="Heikin Ashi Calculation"
          definition="Unlike regular candlesticks that use actual OHLC (Open, High, Low, Close) prices, Heikin Ashi uses averaged values."
          formula="HA_Close = (Open + High + Low + Close) / 4 | HA_Open = (Previous HA_Open + Previous HA_Close) / 2 | HA_High = Max(High, HA_Open, HA_Close) | HA_Low = Min(Low, HA_Open, HA_Close)"
        />

        <Term
          term="Bullish Heikin Ashi Signal"
          definition="A buy signal is generated when Heikin Ashi candles show a reversal from bearish to bullish trend."
          formula="Signal: HA_Close &gt; HA_Open AND Previous HA_Close &lt;= Previous HA_Open"
          example="If today's HA candle is green (close &gt; open) and yesterday's was red (close &lt; open), a buy signal is triggered."
        />

        <Term
          term="Bearish Heikin Ashi Signal"
          definition="A sell signal is generated when Heikin Ashi candles show a reversal from bullish to bearish trend."
          formula="Signal: HA_Close &lt; HA_Open AND Previous HA_Close &gt;= Previous HA_Open"
          example="If today's HA candle is red and yesterday's was green, it indicates a potential trend reversal to downside."
        />

        <div className="bg-green-900/30 rounded-lg p-4 border border-green-700/30">
          <h4 className="font-semibold text-green-300 mb-2">Why Heikin Ashi?</h4>
          <ul className="list-disc list-inside space-y-1 text-gray-300">
            <li><strong>Smoother trends:</strong> Reduces false signals compared to regular candlesticks</li>
            <li><strong>Clear trend identification:</strong> Consecutive colored candles indicate strong trends</li>
            <li><strong>Better entry/exit timing:</strong> Helps identify trend reversals more effectively</li>
          </ul>
        </div>
      </Section>

      {/* Trading Strategies */}
      <Section title="Trading Strategies" icon={Brain}>
        <div className="space-y-6">
          {/* Strategy 1: All Stocks */}
          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="text-xl font-bold text-blue-300 mb-3">1. All Stocks Strategy</h3>
            <p className="mb-3">
              This strategy applies Heikin Ashi signals to all active stocks in the database independently. 
              Each stock gets an equal allocation of capital.
            </p>
            <Term
              term="How It Works"
              definition="For each stock, the system monitors Heikin Ashi candlestick patterns and enters positions when a bullish signal appears. Positions are exited when a bearish signal appears or when maximum holding period is reached."
              formula="Capital Per Stock = Initial Capital | Entry: Buy when HA bullish signal | Exit: Sell when HA bearish signal OR holding period &gt; 30 days"
              example="With ₹50,000 capital and 10 active stocks, each stock trades independently with ₹50,000."
            />
            <div className="bg-gray-800/50 rounded p-3 mt-3">
              <p className="text-sm"><strong>Best for:</strong> Broad market exposure, diversification across all available stocks</p>
            </div>
          </div>

          {/* Strategy 2: Smart Portfolio */}
          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="text-xl font-bold text-purple-300 mb-3">2. Smart Portfolio Strategy</h3>
            <p className="mb-3">
              An intelligent strategy that pre-selects the best performing stocks based on historical metrics, 
              then distributes capital among them with daily compounding.
            </p>
            
            <Term
              term="Stock Selection Process"
              definition="Before the backtest period, the system runs a 6-month ranking period to evaluate all stocks and select the top performers."
              formula="Composite Score = (Win Rate × 30) + (Avg Return × 2.5) + (Sharpe × 8.33) + (Trade Frequency × 5) + (Recovery Rate × 10)"
            />

            <Term
              term="Ranking Metrics"
              definition="Each stock is evaluated on multiple performance dimensions:"
            />
            <div className="ml-4 space-y-2 text-sm">
              <p><strong>• Win Rate (30% weight):</strong> Percentage of profitable trades</p>
              <p><strong>• Average Return (25% weight):</strong> Mean return per trade</p>
              <p><strong>• Sharpe-like Ratio (33% weight):</strong> Risk-adjusted return measure</p>
              <p><strong>• Trade Frequency (5% weight):</strong> Number of trades per month</p>
              <p><strong>• Recovery Rate (10% weight):</strong> Ability to bounce back after consecutive losses</p>
            </div>

            <Term
              term="Capital Distribution"
              definition="After stock selection, capital is distributed dynamically each day based on available signals."
              formula="Daily Capital Per Stock = Available Capital / Number of Stocks Signaling Today"
              example="With ₹5,00,000 total and 5 stocks signaling on a day, each gets ₹1,00,000. If only 2 signal the next day, each gets ₹2,50,000."
            />

            <Term
              term="Compounding Effect"
              definition="Profits and losses are reinvested daily. The available capital for new trades increases with profits and decreases with losses."
              example="Start with ₹5L. After first week profit of ₹50K, available capital becomes ₹5.5L for next trades."
            />

            <div className="bg-gray-800/50 rounded p-3 mt-3">
              <p className="text-sm"><strong>Best for:</strong> Concentrated exposure to historically proven performers with capital growth optimization</p>
            </div>
          </div>

          {/* Strategy 3: HA + T-Score */}
          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="text-xl font-bold text-green-300 mb-3">3. Heikin Ashi + T-Score Strategy</h3>
            <p className="mb-3">
              Combines Heikin Ashi signals with T-Score momentum filtering. Only enters positions in stocks 
              with strong technical momentum, prioritizing high T-Score stocks.
            </p>
            
            <Term
              term="T-Score System"
              definition="A momentum scoring system that rates stocks from 0-100% based on their technical strength across multiple indicators."
            />

            <div className="bg-gray-900 rounded-lg p-4 mt-3 space-y-3">
              <h4 className="font-semibold text-green-300">T-Score Components (Maximum Score: 6-8 points)</h4>
              
              <div className="space-y-3 text-sm">
                <div className="border-l-2 border-blue-500 pl-3">
                  <p className="font-semibold text-blue-300">SMA21 (1 point)</p>
                  <p>Close price above 21-day Simple Moving Average indicates short-term strength</p>
                  <p className="text-gray-400 text-xs mt-1">Formula: Current Close &gt; Average(Close, 21 days)</p>
                </div>

                <div className="border-l-2 border-purple-500 pl-3">
                  <p className="font-semibold text-purple-300">SMA50 (1 point)</p>
                  <p>Close price above 50-day Simple Moving Average indicates medium-term strength</p>
                  <p className="text-gray-400 text-xs mt-1">Formula: Current Close &gt; Average(Close, 50 days)</p>
                </div>

                <div className="border-l-2 border-indigo-500 pl-3">
                  <p className="font-semibold text-indigo-300">SMA200 (1 point)</p>
                  <p>Close price above 200-day Simple Moving Average indicates long-term uptrend</p>
                  <p className="text-gray-400 text-xs mt-1">Formula: Current Close &gt; Average(Close, 200 days)</p>
                </div>

                <div className="border-l-2 border-yellow-500 pl-3">
                  <p className="font-semibold text-yellow-300">7-Day Close Range (2 points)</p>
                  <p>Current close is higher than the maximum close of the last 7 days (breakout strength)</p>
                  <p className="text-gray-400 text-xs mt-1">Formula: Current Close &gt; Max(Close, last 7 days)</p>
                </div>

                <div className="border-l-2 border-green-500 pl-3">
                  <p className="font-semibold text-green-300">52-Week High Range (0-3 points)</p>
                  <p>Proximity to 52-week high - closer = stronger momentum</p>
                  <div className="text-gray-400 text-xs mt-1 space-y-1">
                    <p>• 3 points: Within 10% of 52-week high (very strong)</p>
                    <p>• 2 points: 10-20% below 52-week high (strong)</p>
                    <p>• 1 point: 20-30% below 52-week high (moderate)</p>
                    <p>• 0.5 points: 30-40% below 52-week high (weak)</p>
                    <p>• 0 points: More than 40% below 52-week high (very weak)</p>
                  </div>
                </div>
              </div>

              <div className="bg-green-900/30 rounded p-3 border border-green-700/30 mt-4">
                <p className="font-semibold text-green-300 mb-2">Final T-Score Calculation:</p>
                <p className="font-mono text-sm">T-Score % = (Total Points / Maximum Possible Points) × 100</p>
                <p className="text-gray-400 text-xs mt-2">
                  Example: If a stock scores 5 out of 8 possible points, T-Score = (5/8) × 100 = 62.5%
                </p>
              </div>
            </div>

            <Term
              term="Strategy Logic"
              definition="On each trading day, when multiple stocks show Heikin Ashi buy signals, the system prioritizes stocks with higher T-Scores for entry."
              formula="Entry Priority = Stocks sorted by T-Score (highest first) | Entry Condition = HA Buy Signal AND T-Score &gt; 0"
              example="If 15 stocks signal on a day but you can only buy 10, the system selects the 10 with highest T-Scores."
            />

            <div className="bg-gray-800/50 rounded p-3 mt-3">
              <p className="text-sm"><strong>Best for:</strong> Quality-focused trading with momentum confirmation, reduces false signals</p>
            </div>
          </div>
        </div>
      </Section>

      {/* Performance Metrics */}
      <Section title="Performance Metrics Explained" icon={Calculator}>
        <div className="space-y-4">
          <Term
            term="Total P&L (Profit & Loss)"
            definition="The total profit or loss across all trades in the backtest period."
            formula="Total P&L = Sum of all (Exit Price - Entry Price) × Quantity"
            example="If you made ₹10,000 on 5 trades and lost ₹3,000 on 2 trades, Total P&L = ₹7,000"
          />

          <Term
            term="Win Rate"
            definition="Percentage of trades that resulted in profit."
            formula="Win Rate = (Number of Winning Trades / Total Trades) × 100"
            example="If 60 out of 100 trades were profitable, Win Rate = 60%"
          />

          <Term
            term="Total Return %"
            definition="Overall percentage gain or loss on the initial capital."
            formula="Total Return % = ((Final Capital - Initial Capital) / Initial Capital) × 100"
            example="Starting with ₹1,00,000 and ending with ₹1,25,000 gives 25% return"
          />

          <Term
            term="Average Return per Trade"
            definition="Mean profit/loss percentage across all trades."
            formula="Avg Return = Sum of all (Return %) / Number of Trades"
            example="If total returns across 10 trades sum to 50%, average return = 5% per trade"
          />

          <Term
            term="Maximum Drawdown"
            definition="The largest peak-to-trough decline in portfolio value during the backtest period. Measures worst-case loss scenario."
            formula="Max Drawdown = ((Trough Value - Peak Value) / Peak Value) × 100"
            example="If portfolio peaked at ₹1,50,000 and then fell to ₹1,20,000, drawdown = -20%"
          />

          <Term
            term="Sharpe Ratio"
            definition="Risk-adjusted return metric. Higher Sharpe means better risk-adjusted performance."
            formula="Sharpe Ratio = (Average Return - Risk-free Rate) / Standard Deviation of Returns"
            example="Sharpe &gt; 1 is good, &gt; 2 is very good, &gt; 3 is excellent"
          />

          <Term
            term="Profit Factor"
            definition="Ratio of gross profits to gross losses. Indicates how much profit is made per unit of loss."
            formula="Profit Factor = Total Profit from Winning Trades / Total Loss from Losing Trades"
            example="If total profits = ₹50,000 and total losses = ₹25,000, Profit Factor = 2.0 (good)"
          />

          <Term
            term="Average Holding Period"
            definition="Average number of days a position is held before being closed."
            formula="Avg Holding Period = Sum of all (Exit Date - Entry Date) / Number of Trades"
            example="If trades are held for 10, 15, and 20 days, average = 15 days"
          />

          <Term
            term="Stocks Signaled Per Day"
            definition="Average number of stocks showing buy signals on any given trading day."
            formula="Avg Stocks Per Day = Total Signals / Number of Trading Days with Signals"
            example="If 500 signals occurred over 50 trading days, average = 10 stocks per day"
          />

          <Term
            term="Recovery Rate"
            definition="Ability of the strategy to recover from consecutive losing trades."
            formula="Recovery Rate = (Number of Recoveries / Max Consecutive Losses) × 100"
            example="If the strategy recovered 8 times after 10 losing streaks, recovery rate = 80%"
          />
        </div>
      </Section>

      {/* Position Sizing & Risk Management */}
      <Section title="Position Sizing & Risk Management" icon={Shield}>
        <Term
          term="Position Size Calculation"
          definition="The system calculates how much to invest in each trade based on available capital and number of signals."
          formula="Position Size = Available Capital / Number of Concurrent Signals"
          example="With ₹5,00,000 and 10 stocks signaling, each position = ₹50,000"
        />

        <Term
          term="Quantity Calculation"
          definition="Number of shares to buy based on position size and current price."
          formula="Quantity = Floor(Position Size / Entry Price)"
          example="With ₹50,000 position size and ₹450 stock price, buy 111 shares (50000/450)"
        />

        <Term
          term="Maximum Holding Period"
          definition="Maximum number of days a position can be held before forced exit (default: 30 days)."
          formula="If (Current Date - Entry Date) &gt; 30 days, Force Exit"
          example="If you entered on Jan 1 and no sell signal by Jan 31, position is automatically closed on Feb 1"
        />

        <Term
          term="Capital Management"
          definition="In Smart and T-Score strategies, capital compounds. Winners increase buying power, losers reduce it."
          example="Start: ₹5L → Week 1 profit ₹50K → Week 2 capital = ₹5.5L → Week 2 loss ₹30K → Week 3 capital = ₹5.2L"
        />
      </Section>

      {/* Daily Signal Statistics */}
      <Section title="Daily Signal Statistics" icon={Activity}>
        <Term
          term="Total Signal Days"
          definition="Number of trading days on which at least one stock showed a buy signal."
          example="Out of 252 trading days in a year, signals might appear on 180 days"
        />

        <Term
          term="Average Stocks Per Day"
          definition="Mean number of stocks showing buy signals on signal days."
          formula="Avg = Total Signals / Total Signal Days"
          example="1000 total signals over 200 signal days = 5 stocks per day average"
        />

        <Term
          term="Max Stocks on Single Day"
          definition="Highest number of stocks that signaled on any single day."
          example="Maximum of 25 stocks showed buy signals on March 15, 2024"
        />

        <Term
          term="Median Stocks Per Day"
          definition="Middle value of daily signal counts. Less affected by outliers than average."
          example="If daily counts are [2,3,5,5,6,8,20], median = 5"
        />
      </Section>

      {/* Chart Interpretations */}
      <Section title="Understanding Charts" icon={TrendingUp}>
        <Term
          term="Capital Growth Chart"
          definition="Shows how portfolio value changes over time. Helps visualize drawdowns and growth periods."
        />

        <Term
          term="Trading Chart (Candlesticks)"
          definition="Displays both regular and Heikin Ashi candlesticks with entry/exit markers. Green markers = Buy, Red markers = Sell."
        />

        <Term
          term="Portfolio Timeline"
          definition="Calendar view showing trading activity by day. Darker colors indicate more trades on that day."
        />

        <Term
          term="Current Positions"
          definition="Shows all open (unsold) positions at the end of backtest period with unrealized P&L."
        />
      </Section>

      {/* Data Intervals */}
      <Section title="Data Intervals" icon={Zap}>
        <Term
          term="Daily Interval"
          definition="Each candlestick represents one trading day. More granular, captures short-term movements."
          example="Jan 15 candlestick shows that day's open, high, low, close prices"
        />

        <Term
          term="Weekly Interval"
          definition="Each candlestick represents one week. Smoother trends, filters daily noise."
          example="Week of Jan 15-19 candlestick shows Monday's open, week's high/low, Friday's close"
        />

        <div className="bg-blue-900/30 rounded-lg p-4 border border-blue-700/30 mt-4">
          <h4 className="font-semibold text-blue-300 mb-2">Choosing an Interval:</h4>
          <ul className="list-disc list-inside space-y-1 text-gray-300">
            <li><strong>Daily:</strong> More trades, faster signals, higher transaction costs, suitable for active trading</li>
            <li><strong>Weekly:</strong> Fewer trades, stronger trends, lower costs, suitable for swing trading</li>
          </ul>
        </div>
      </Section>

      {/* Tips & Best Practices */}
      <Section title="Tips & Best Practices" icon={Award}>
        <div className="space-y-4">
          <div className="bg-green-900/30 rounded-lg p-4 border border-green-700/30">
            <h4 className="font-semibold text-green-300 mb-2 flex items-center gap-2">
              <Target size={18} />
              Strategy Selection
            </h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-300">
              <li><strong>All Stocks:</strong> Use for broad market analysis and diversification testing</li>
              <li><strong>Smart Portfolio:</strong> Use when you want to focus on proven winners with compounding</li>
              <li><strong>HA + T-Score:</strong> Use when you want quality over quantity with momentum confirmation</li>
            </ul>
          </div>

          <div className="bg-blue-900/30 rounded-lg p-4 border border-blue-700/30">
            <h4 className="font-semibold text-blue-300 mb-2 flex items-center gap-2">
              <Shield size={18} />
              Realistic Expectations
            </h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-300">
              <li>Backtest results show historical performance; future results may differ</li>
              <li>Consider transaction costs (brokerage, taxes) in real trading</li>
              <li>Slippage and liquidity issues aren't reflected in backtests</li>
              <li>Higher returns usually come with higher risk and drawdowns</li>
            </ul>
          </div>

          <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-700/30">
            <h4 className="font-semibold text-purple-300 mb-2 flex items-center gap-2">
              <Activity size={18} />
              Optimization Tips
            </h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-300">
              <li>Test different date ranges to validate strategy robustness</li>
              <li>Compare daily vs weekly intervals to find best timeframe</li>
              <li>Adjust number of stocks in Smart/T-Score strategies based on capital</li>
              <li>Look for consistent performance across different market conditions</li>
            </ul>
          </div>

          <div className="bg-yellow-900/30 rounded-lg p-4 border border-yellow-700/30">
            <h4 className="font-semibold text-yellow-300 mb-2 flex items-center gap-2">
              <Info size={18} />
              Important Notes
            </h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-300">
              <li>Past performance does not guarantee future results</li>
              <li>Always backtest on out-of-sample data before live trading</li>
              <li>Monitor max drawdown - ensure you can handle the worst-case losses</li>
              <li>Consider your risk tolerance and capital availability</li>
              <li>This is for educational purposes - consult a financial advisor for investment decisions</li>
            </ul>
          </div>
        </div>
      </Section>

      {/* Footer */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 text-center">
        <p className="text-gray-400">
          <strong className="text-white">Disclaimer:</strong> This backtesting system is for educational and research purposes only. 
          It does not constitute financial advice. Trading in stocks involves risk of loss. 
          Always conduct your own research and consider consulting with a qualified financial advisor.
        </p>
      </div>
    </div>
  );
}
