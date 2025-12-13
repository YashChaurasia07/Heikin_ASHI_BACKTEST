import axios from 'axios';

const API_BASE_URL = '/api';

export interface Trade {
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  qty: number;
  pnl: number;
  exit_type: string;
}

export interface StockResult {
  initial_capital: number;
  final_capital: number;
  pnl: number;
  num_trades: number;
  trades: Trade[];
}

export interface Statistics {
  avg_stocks_signaled_per_day: number;
  max_stocks_on_single_day: number;
  min_stocks_on_single_day: number;
  median_stocks_per_day: number;
  total_signal_days: number;
  day_with_most_signals?: {
    date: string;
    count: number;
    stocks: string[];
  };
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate_pct: number;
  total_profit: number;
  total_loss: number;
  net_pnl: number;
  profit_factor: number;
  avg_profit_per_winning_trade: number;
  avg_loss_per_losing_trade: number;
  avg_pnl_per_trade: number;
  avg_return_pct_per_trade: number;
  avg_holding_days: number;
  max_holding_days: number;
  min_holding_days: number;
  median_holding_days: number;
  best_trade?: {
    symbol: string;
    entry_date: string;
    exit_date: string;
    pnl: number;
    return_pct: number;
  };
  worst_trade?: {
    symbol: string;
    entry_date: string;
    exit_date: string;
    pnl: number;
    return_pct: number;
  };
  best_performing_stock?: {
    symbol: string;
    return_pct: number;
    pnl: number;
    num_trades: number;
    win_rate: number;
  };
  worst_performing_stock?: {
    symbol: string;
    return_pct: number;
    pnl: number;
    num_trades: number;
    win_rate: number;
  };
  highest_win_rate_stock?: {
    symbol: string;
    win_rate: number;
    num_trades: number;
    return_pct: number;
  };
  best_month?: {
    month: string;
    pnl: number;
    num_trades: number;
  };
  worst_month?: {
    month: string;
    pnl: number;
    num_trades: number;
  };
  avg_monthly_pnl: number;
  total_profitable_months: number;
  total_losing_months: number;
  std_deviation_returns: number;
  max_return_pct: number;
  min_return_pct: number;
  sharpe_ratio_approx: number;
  exit_type_breakdown?: Record<string, {
    count: number;
    pct: number;
    total_pnl: number;
    avg_pnl: number;
  }>;
  max_consecutive_wins: number;
  max_consecutive_losses: number;
  return_on_capital_pct: number;
  avg_capital_deployed_per_trade: number;
  max_drawdown: number;
  max_drawdown_pct: number;
}

export interface SelectedStock {
  symbol: string;
  ranking_score: number;
  historical_win_rate: number;
  historical_return_pct: number;
  historical_trades: number;
}

export interface SelectionInfo {
  total_investment: number;
  num_stocks_requested: number;
  num_stocks_selected: number;
  capital_per_stock: number;
  ranking_period_months: number;
  total_stocks_analyzed: number;
  selected_stocks: SelectedStock[];
}

export interface BacktestResult {
  total_initial_capital: number;
  total_final_capital: number;
  total_pnl: number;
  total_return_pct: number;
  num_stocks_processed: number;
  stocks: Record<string, StockResult>;
  params: {
    start_date: string;
    end_date: string | null;
    initial_capital: number;
    strategy: string;
  };
  statistics?: Statistics;
  selection_info?: SelectionInfo;
}

export interface OHLCData {
  Date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface HAData {
  Date: string;
  ha_open: number;
  ha_high: number;
  ha_low: number;
  ha_close: number;
}

const api = {
  async getSymbols(): Promise<string[]> {
    const response = await axios.get<{ symbols: string[] }>(`${API_BASE_URL}/symbols`);
    return response.data.symbols;
  },

  async runBacktest(startDate: string, endDate: string, initialCapital: number): Promise<BacktestResult> {
    let url = `${API_BASE_URL}/backtest?start_date=${startDate}&initial_capital=${initialCapital}`;
    if (endDate) {
      url += `&end_date=${endDate}`;
    }
    const response = await axios.get<BacktestResult>(url);
    return response.data;
  },

  async getStockData(symbol: string): Promise<OHLCData[]> {
    const response = await axios.get<OHLCData[]>(`${API_BASE_URL}/data/${symbol}`);
    return response.data;
  },

  async getHAData(symbol: string): Promise<HAData[]> {
    const response = await axios.get<HAData[]>(`${API_BASE_URL}/ha_data/${symbol}`);
    return response.data;
  },

  async runSmartBacktest(
    totalInvestment: number,
    numStocks: number,
    startDate: string,
    endDate: string,
    useCached: boolean = true
  ): Promise<BacktestResult> {
    let url = `${API_BASE_URL}/smart_backtest?total_investment=${totalInvestment}&num_stocks=${numStocks}&start_date=${startDate}&use_cached=${useCached}`;
    if (endDate) {
      url += `&end_date=${endDate}`;
    }
    const response = await axios.get<BacktestResult>(url);
    return response.data;
  },
};

export default api;
