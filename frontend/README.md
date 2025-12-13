# Heikin Ashi Strategy - Frontend

Modern, high-performance frontend dashboard for backtesting Heikin Ashi trading strategy built with Vite, React, TypeScript, and TradingView Lightweight Charts.

## Features

- ⚡ **Lightning Fast**: Built with Vite for optimal dev experience and performance
- 📊 **Advanced Charts**: TradingView Lightweight Charts for smooth, professional candlestick visualization
- 🎯 **Buy/Sell Markers**: Visual indicators for all trade entries and exits on charts
- 🎨 **Modern UI**: Sleek dark theme with Tailwind CSS and smooth animations
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🔄 **Real-time Updates**: Live portfolio statistics and performance metrics
- 🔍 **Search & Filter**: Quickly find and analyze specific stocks
- 📈 **Dual Chart View**: Compare normal candlesticks with Heikin Ashi charts side-by-side

## Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite 5
- **Styling**: Tailwind CSS
- **Charts**: TradingView Lightweight Charts
- **HTTP Client**: Axios
- **Icons**: Lucide React

## Prerequisites

- Node.js 18+ and npm
- Backend server running on `http://localhost:8000`

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will open at `http://localhost:3000`

## Usage

1. **Configure Parameters**:
   - Set the start date for backtesting
   - Specify initial capital per stock

2. **Run Backtest**:
   - Click "Run Backtest" button
   - Wait for the backend to process all stocks

3. **View Results**:
   - Portfolio summary with total P&L and return percentage
   - Grid of all stocks sorted by profitability
   - Click any stock card to view detailed analysis

4. **Analyze Stocks**:
   - View normal and Heikin Ashi candlestick charts
   - See buy/sell markers on charts
   - Review detailed trade history

## API Integration

The frontend connects to the FastAPI backend through a proxy configuration:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- API calls are proxied from `/api/*` to backend endpoints

## Build for Production

```bash
npm run build
```

The production-ready files will be in the `dist` directory.

## Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── TradingChart.tsx      # Lightweight Charts wrapper
│   │   ├── StockCard.tsx         # Stock performance card
│   │   └── TradeList.tsx         # Trade history display
│   ├── services/
│   │   └── api.ts                # API client with TypeScript types
│   ├── App.tsx                    # Main application component
│   ├── main.tsx                   # Application entry point
│   └── index.css                  # Global styles with Tailwind
├── public/                        # Static assets
├── index.html                     # HTML template
├── vite.config.ts                # Vite configuration
├── tailwind.config.js            # Tailwind CSS configuration
└── package.json                   # Dependencies and scripts
```

## Performance Optimizations

- **Vite**: Ultra-fast HMR and optimized builds
- **Lightweight Charts**: GPU-accelerated canvas rendering
- **Code Splitting**: Automatic chunk splitting for optimal loading
- **Lazy Loading**: Components loaded on demand
- **Memoization**: React hooks for preventing unnecessary re-renders

## Troubleshooting

### Backend Connection Issues
- Ensure FastAPI server is running on port 8000
- Check CORS configuration in backend allows `localhost:3000`

### Chart Not Rendering
- Verify data format matches TradingView Lightweight Charts expectations
- Check browser console for errors

### Build Errors
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Clear Vite cache: `rm -rf node_modules/.vite`

## License

MIT License - See backend repository for details
