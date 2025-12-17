# import pandas as pd
# from tvDatafeed import TvDatafeed, Interval
# from datetime import datetime

# # Initialize TvDatafeed (use no-login mode; for more symbols/access, provide username/password)
# # tv = TvDatafeed(username='YourTradingViewUsername', password='YourTradingViewPassword')
# tv = TvDatafeed()  # No login - may have limitations

# # Read symbols from Excel file and debug structure
# symbols_df = pd.read_excel('symbol.xlsx')
# print("Columns in the DataFrame:", symbols_df.columns.tolist())
# print("First few rows:\n", symbols_df.head())

# # Try to extract symbols - adjust based on actual structure
# # Option 1: If 'Row Labels' is a column
# if 'Row Labels' in symbols_df.columns:
#     symbols = symbols_df['Row Labels'].dropna().tolist()
# # Option 2: If symbols are in the first column (e.g., unnamed or different name)
# else:
#     # Assume first column contains the symbols
#     first_col_name = symbols_df.columns[0]
#     symbols = symbols_df[first_col_name].dropna().tolist()
#     print(f"Using column '{first_col_name}' for symbols")

# # Remove any non-string or empty entries
# symbols = [str(s).strip() for s in symbols if str(s).strip() and not str(s).isdigit()]

# print(f"Extracted symbols: {symbols[:5]}... (total: {len(symbols)})")

# # Define start date
# start_date = '2022-01-01'

# # Fetch historical daily data for each symbol and save to individual CSVs
# for symbol in symbols:
#     try:
#         print(f"Fetching data for {symbol}...")
#         # Fetch last 5000 daily bars (covers ~20 years, sufficient for 2022-now)
#         data = tv.get_hist(
#             symbol=symbol,
#             exchange='NSE',  # Assuming NSE based on symbols
#             interval=Interval.in_daily,
#             n_bars=5000
#         )
        
#         if data is not None and not data.empty:
#             # Slice from start_date to now (data index is datetime)
#             data = data.loc[start_date:]
            
#             # Save to CSV with standard OHLCV columns: datetime, open, high, low, close, volume
#             # Ensure datetime index is a column
#             data.reset_index(inplace=True)
#             data.rename(columns={'datetime': 'Date'}, inplace=True)  # Rename index to 'Date'
#             data.to_csv(f'{symbol}.csv', index=False)
#             print(f"Saved {len(data)} bars for {symbol} to {symbol}.csv")
#         else:
#             print(f"No data available for {symbol}")
#     except Exception as e:
#         print(f"Error fetching {symbol}: {e}")

# print("Data fetching complete. Check individual CSV files.")

import pandas as pd
import os
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime

# Initialize TvDatafeed (use no-login mode; for more symbols/access, provide username/password)
# tv = TvDatafeed(username='YourTradingViewUsername', password='YourTradingViewPassword')
tv = TvDatafeed()  # No login - may have limitations

# Read symbols from Excel file and debug structure
symbols_df = pd.read_excel('symbol.xlsx')
print("Columns in the DataFrame:", symbols_df.columns.tolist())
print("First few rows:\n", symbols_df.head())

# Try to extract symbols - adjust based on actual structure
# Option 1: If 'Row Labels' is a column
if 'Row Labels' in symbols_df.columns:
    symbols = symbols_df['Row Labels'].dropna().tolist()
# Option 2: If symbols are in the first column (e.g., unnamed or different name)
else:
    # Assume first column contains the symbols
    first_col_name = symbols_df.columns[0]
    symbols = symbols_df[first_col_name].dropna().tolist()
    print(f"Using column '{first_col_name}' for symbols")

# Remove any non-string or empty entries
symbols = [str(s).strip() for s in symbols if str(s).strip() and not str(s).isdigit()]

print(f"Extracted symbols: {symbols[:5]}... (total: {len(symbols)})")

# Define start date
start_date = '2022-01-01'

# Dictionary to track failed symbols and reasons
failed_symbols = {}

# Ensure data folder exists
os.makedirs('data', exist_ok=True)

# Fetch historical daily data for each symbol and save to individual CSVs in 'data' folder
for symbol in symbols:
    csv_path = f'data/{symbol}.csv'
    if os.path.exists(csv_path):
        print(f"File {csv_path} already exists, skipping {symbol}.")
        continue
    
    try:
        print(f"Fetching data for {symbol}...")
        # Fetch last 5000 daily bars (covers ~20 years, sufficient for 2022-now)
        data = tv.get_hist(
            symbol=symbol,
            exchange='NSE',  # Assuming NSE based on symbols
            interval=Interval.in_weekly,
            n_bars=5000
        )
        
        if data is not None and not data.empty:
            # Store full historical data (no date filtering)
            # The backtest API will filter by date range when needed
            
            if not data.empty:
                # Save to CSV with standard OHLCV columns: datetime, open, high, low, close, volume
                # Ensure datetime index is a column
                data.reset_index(inplace=True)
                data.rename(columns={'datetime': 'Date'}, inplace=True)  # Rename index to 'Date'
                data.to_csv(csv_path, index=False)
                print(f"Saved {len(data)} bars for {symbol} to {csv_path}")
            else:
                reason = f"No data available after slicing from {start_date}"
                failed_symbols[symbol] = reason
                print(f"{reason} for {symbol}")
        else:
            reason = "No data fetched from TradingView"
            failed_symbols[symbol] = reason
            print(f"{reason} for {symbol}")
    except Exception as e:
        reason = f"Exception: {str(e)}"
        failed_symbols[symbol] = reason
        print(f"Error fetching {symbol}: {e}")

# Print summary of failed symbols
if failed_symbols:
    print("\nFailed symbols and reasons:")
    for symbol, reason in failed_symbols.items():
        print(f"- {symbol}: {reason}")
else:
    print("\nAll symbols processed successfully (skipped existing files).")

print("Data fetching complete. Check 'data' folder for CSV files.")