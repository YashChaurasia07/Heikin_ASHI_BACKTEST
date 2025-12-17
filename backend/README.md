# Heikin Ashi Backtest Server - Backend

Professional FastAPI backend with MongoDB for Heikin Ashi backtesting.

## Features

- ✅ MongoDB integration for efficient data storage
- ✅ Daily and Weekly interval support
- ✅ Automatic data synchronization from TradingView
- ✅ Precise Heikin Ashi calculations using Decimal arithmetic
- ✅ Data freshness checking
- ✅ Bulk symbol management
- ✅ Excel file upload support
- ✅ RESTful API design
- ✅ Comprehensive error handling

## Setup

### 1. Install MongoDB

Download and install MongoDB from [mongodb.com](https://www.mongodb.com/try/download/community)

Start MongoDB service:
```bash
# Windows
net start MongoDB

# Linux/Mac
sudo systemctl start mongod
```

### 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=heikin_ashi_db
PORT=8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 4. Run Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

### Data Management

#### Sync Data
```http
POST /api/data/sync
Content-Type: application/json

{
  "symbols": ["RELIANCE", "TCS"],  // Optional, null for all
  "interval": "daily",              // "daily" or "weekly"
  "force_update": false
}
```

#### Check Data Status
```http
GET /api/data/status/RELIANCE?interval=daily
```

#### Check All Symbols
```http
GET /api/data/check-all?interval=daily
```

### Symbol Management

#### Get All Symbols
```http
GET /api/symbols/
```

#### Add Symbol
```http
POST /api/symbols/add?symbol=RELIANCE&exchange=NSE
```

#### Bulk Add Symbols
```http
POST /api/symbols/bulk-add
Content-Type: application/json

{
  "symbols": ["RELIANCE", "TCS", "INFY"],
  "exchange": "NSE"
}
```

#### Upload Excel File
```http
POST /api/symbols/upload-excel
Content-Type: multipart/form-data

file: symbol.xlsx
```

Excel file should have symbols in:
- Column named "Row Labels", "symbol", or "Symbol"
- Or first column

#### Delete Symbol
```http
DELETE /api/symbols/RELIANCE
```

## Database Schema

### Collections

#### symbols
```json
{
  "symbol": "RELIANCE",
  "exchange": "NSE",
  "active": true,
  "added_date": "2024-01-01T00:00:00"
}
```

#### stock_data_daily / stock_data_weekly
```json
{
  "symbol": "RELIANCE",
  "date": "2024-01-01T00:00:00",
  "open": 2500.0,
  "high": 2550.0,
  "low": 2480.0,
  "close": 2530.0,
  "volume": 1000000,
  "interval": "daily"
}
```

#### heikin_ashi_daily / heikin_ashi_weekly
```json
{
  "symbol": "RELIANCE",
  "date": "2024-01-01T00:00:00",
  "ha_open": 2500.0,
  "ha_high": 2550.0,
  "ha_low": 2480.0,
  "ha_close": 2530.0,
  "volume": 1000000,
  "interval": "daily"
}
```

#### stock_metadata
```json
{
  "symbol": "RELIANCE",
  "last_updated_daily": "2024-01-01T00:00:00",
  "last_updated_weekly": "2024-01-01T00:00:00",
  "total_candles_daily": 500,
  "total_candles_weekly": 100,
  "ha_calculated_daily": true,
  "ha_calculated_weekly": true,
  "last_ha_update_daily": "2024-01-01T00:00:00",
  "last_ha_update_weekly": "2024-01-01T00:00:00"
}
```

## Architecture

```
backend/
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── database.py            # MongoDB connection
├── models.py              # Pydantic models
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
├── routes/
│   ├── __init__.py
│   ├── data.py           # Data sync endpoints
│   ├── symbols.py        # Symbol management
│   └── backtest.py       # Backtesting endpoints (to be added)
└── services/
    ├── __init__.py
    ├── data_fetcher.py   # TradingView data fetching
    ├── heikin_ashi.py    # HA calculations
    └── backtest.py       # Backtesting engine (to be added)
```

## Data Flow

1. **Symbol Upload**: Upload symbols via API or Excel
2. **Data Sync**: Fetch historical data from TradingView
3. **HA Calculation**: Calculate Heikin Ashi candles with precision
4. **Backtest**: Run backtests on stored data
5. **Results**: Return comprehensive statistics

## Key Features

### Precise Calculations
- Uses Python `Decimal` for accurate HA calculations
- Avoids floating-point errors
- Maintains calculation chain integrity

### Smart Data Management
- Automatic freshness checking
- Configurable update intervals
- Bulk operations support
- Efficient MongoDB queries

### Error Handling
- Comprehensive logging
- Detailed error messages
- Graceful failure handling
- Status tracking

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

## Deployment

### Production Configuration

1. Set production environment variables
2. Use proper MongoDB credentials
3. Enable authentication
4. Configure CORS properly
5. Use production ASGI server

### Example with Gunicorn
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB is running
- Check connection string in .env
- Verify network/firewall settings

### Data Sync Failures
- Check TradingView credentials
- Verify symbol names
- Check API rate limits

### Missing Data
- Run data sync endpoint
- Check data status endpoint
- Verify symbol is active

## Support

For issues or questions, check the logs:
```bash
tail -f logs/app.log
```

## License

MIT License
