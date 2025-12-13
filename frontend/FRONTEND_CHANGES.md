# Frontend Smart Backtest Implementation

## Changes Made

### 1. **API Service (services/api.ts)**
- Added `SelectionInfo` and `SelectedStock` interfaces
- Updated `BacktestResult` to include optional `selection_info`
- Added `runSmartBacktest()` function

### 2. **Main App (App.tsx)**
- Added smart mode toggle button
- Added state variables:
  - `useSmartMode`: Toggle between regular and smart mode
  - `totalInvestment`: Total amount to invest
  - `numStocksToSelect`: Number of stocks to select
- Updated `runBacktest()` to handle both modes
- Added Smart Selection Info Banner showing:
  - Total investment
  - Number of stocks selected
  - Capital per stock
  - Analysis period
  - List of selected stocks with win rates

### 3. **UI Features**

#### Mode Toggle
- **All Stocks Mode**: Traditional backtest on all stocks
- **Smart Select Mode**: AI-powered stock selection (green theme with sparkle icon)

#### Smart Mode Inputs
- Total Investment (e.g., ₹500,000)
- Number of Stocks (e.g., 10)

#### Regular Mode Inputs (unchanged)
- Strategy selection
- EMA scores (if applicable)
- Capital per stock

#### Smart Selection Banner
Shows when smart mode results are displayed:
- Investment breakdown
- Selected stocks with historical win rates
- Analysis period used for ranking

## Usage

### Smart Mode
1. Click "Smart Select" button
2. Enter total investment amount (e.g., 500000)
3. Enter number of stocks to select (e.g., 10)
4. Set start and end dates
5. Click "Smart Backtest"

### Result
- Backend analyzes historical performance
- Selects top N stocks by win rate
- Divides capital equally
- Runs backtest with realistic slippage
- Displays comprehensive results

## Benefits
- ✅ Easy to use - just two inputs
- ✅ Data-driven stock selection
- ✅ Visual indication of selected stocks
- ✅ Shows why stocks were chosen
- ✅ Equal risk distribution
- ✅ Green theme for smart mode
