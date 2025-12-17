import axios from 'axios';

// Use environment variable for API URL
const API_BASE_URL = import.meta.env.VITE_API_URL;

// Create axios instance with explicit configuration
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: false,
  // headers: {
  //   'Content-Type': 'application/json',
  // },
});

export interface Trade {
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  qty: number;
  pnl: number;
  exit_type: string;
  t_score?: number;  // T-Score at entry (for HA+T-Score strategy)
  return_pct?: number;
  holding_days?: number;
}

export interface StockResult {
  initial_capital: number;
  final_capital: number;
  pnl: number;
  num_trades: number;
  trades: Trade[];
  latest_tscore?: number;  // Latest T-Score at end of backtest period (for T-Score strategy)
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
  cagr_pct: number;
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
    max_concurrent_positions?: number;
  };
  statistics?: Statistics;
  selection_info?: SelectionInfo;
  tscore_stats?: {
    avg_tscore_at_entry: number;
    trades_with_tscore: number;
    total_trades: number;
  };
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
    const response = await axiosInstance.get<{ symbols: string[] }>('/data/symbols');
    return response.data.symbols;
  },

  async getAllSymbols(): Promise<any[]> {
    const response = await axiosInstance.get('/symbols/');
    return response.data;
  },

  async addSymbol(symbol: string, exchange: string = "NSE"): Promise<any> {
    const response = await axiosInstance.post(`/symbols/add?symbol=${symbol}&exchange=${exchange}`);
    return response.data;
  },

  async bulkAddSymbols(symbols: string[], exchange: string = "NSE"): Promise<any> {
    const response = await axiosInstance.post(`/symbols/bulk-add?exchange=${exchange}`, symbols);
    return response.data;
  },

  async deleteSymbol(symbol: string): Promise<any> {
    const response = await axiosInstance.delete(`/symbols/${symbol}`);
    return response.data;
  },

  async uploadSymbolsExcel(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axiosInstance.post('/symbols/upload-excel', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async syncData(symbols?: string[], interval: string = "daily", forceUpdate: boolean = false): Promise<any> {
    const response = await axiosInstance.post('/data/sync', {
      symbols,
      interval,
      force_update: forceUpdate
    });
    return response.data;
  },

  async getDataStatus(symbol: string, interval: string = "daily"): Promise<any> {
    const response = await axiosInstance.get(`/data/status/${symbol}?interval=${interval}`);
    return response.data;
  },

  async checkAllDataStatus(interval: string = "daily"): Promise<any> {
    const response = await axiosInstance.get(`/data/check-all?interval=${interval}`);
    return response.data;
  },

  async getBacktestStatus(): Promise<any> {
    const response = await axiosInstance.get('/backtest/status');
    return response.data;
  },

  async runBacktest(startDate: string, endDate: string, initialCapital: number, interval: string = 'daily'): Promise<BacktestResult> {
    let url = `/backtest?start_date=${startDate}&initial_capital=${initialCapital}&interval=${interval}`;
    if (endDate) {
      url += `&end_date=${endDate}`;
    }
    const response = await axiosInstance.get<BacktestResult>(url);
    return response.data;
  },

  async getStockData(symbol: string): Promise<OHLCData[]> {
    const response = await axiosInstance.get<OHLCData[]>(`/data/${symbol}`);
    return response.data;
  },

  async getHAData(symbol: string): Promise<HAData[]> {
    const response = await axiosInstance.get<HAData[]>(`/data/${symbol}/ha`);
    return response.data;
  },

  async runSmartBacktest(
    totalInvestment: number,
    numStocks: number,
    startDate: string,
    endDate: string,
    interval: string = 'daily'
  ): Promise<BacktestResult> {
    const response = await axiosInstance.post<BacktestResult>('/advanced/smart-portfolio', {
      total_investment: totalInvestment,
      num_stocks: numStocks,
      start_date: startDate,
      end_date: endDate || null,
      interval: interval,
      ranking_period_months: 6,
      enable_compounding: true,
      rebalancing_frequency: 'daily'
    });
    return response.data;
  },

  async runHATScoreBacktest(
    totalInvestment: number,
    numStocks: number,
    startDate: string,
    endDate: string,
    interval: string = 'daily'
  ): Promise<BacktestResult> {
    const response = await axiosInstance.post<BacktestResult>('/advanced/ha-tscore', {
      total_investment: totalInvestment,
      num_stocks: numStocks,
      start_date: startDate,
      end_date: endDate || null,
      interval: interval,
      ranking_period_months: 6,
      enable_compounding: true,
      rebalancing_frequency: 'daily'
    });
    return response.data;
  },

  async getTScores(interval: string = 'daily', asOfDate?: string): Promise<any> {
    let url = `/advanced/tscores?interval=${interval}`;
    if (asOfDate) {
      url += `&as_of_date=${asOfDate}`;
    }
    const response = await axiosInstance.get(url);
    return response.data;
  },
};

export default api;
