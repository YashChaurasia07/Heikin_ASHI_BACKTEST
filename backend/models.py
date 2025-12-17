from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class IntervalType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


class StockData(BaseModel):
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: IntervalType


class HeikinAshiData(BaseModel):
    symbol: str
    date: datetime
    ha_open: float
    ha_high: float
    ha_low: float
    ha_close: float
    volume: Optional[float] = None
    interval: IntervalType


class StockMetadata(BaseModel):
    symbol: str
    last_updated_daily: Optional[datetime] = None
    last_updated_weekly: Optional[datetime] = None
    total_candles_daily: int = 0
    total_candles_weekly: int = 0
    ha_calculated_daily: bool = False
    ha_calculated_weekly: bool = False
    last_ha_update_daily: Optional[datetime] = None
    last_ha_update_weekly: Optional[datetime] = None
    last_sync_date_daily: Optional[datetime] = None  # Last date we checked for sync (today's date)
    last_sync_date_weekly: Optional[datetime] = None  # Last date we checked for sync (today's date)
    last_data_date_daily: Optional[datetime] = None  # Latest data point available in DB
    last_data_date_weekly: Optional[datetime] = None  # Latest data point available in DB


class Symbol(BaseModel):
    symbol: str
    exchange: str = "NSE"
    active: bool = True
    added_date: datetime = Field(default_factory=datetime.utcnow)


class Trade(BaseModel):
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    qty: float
    pnl: float
    exit_type: str
    capital_deployed: Optional[float] = None
    t_score: Optional[float] = None
    return_pct: Optional[float] = None
    holding_days: Optional[int] = None


class BacktestParams(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    initial_capital: float = 50000
    interval: IntervalType = IntervalType.DAILY
    use_parallel: bool = True


class SmartBacktestParams(BaseModel):
    total_investment: float
    num_stocks: int
    start_date: str
    end_date: Optional[str] = None
    interval: IntervalType = IntervalType.DAILY
    ranking_period_months: int = 6
    enable_compounding: bool = True
    rebalancing_frequency: str = "daily"


class DataSyncRequest(BaseModel):
    symbols: Optional[List[str]] = None  # If None, sync all
    interval: IntervalType = IntervalType.DAILY
    force_update: bool = False


class DataSyncStatus(BaseModel):
    symbol: str
    status: str  # "success", "failed", "skipped"
    message: str
    records_added: int = 0
    last_date: Optional[datetime] = None
