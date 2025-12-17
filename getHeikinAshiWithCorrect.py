import pandas as pd
import os
from decimal import Decimal, getcontext
from tvDatafeed import TvDatafeed, Interval

# Set decimal precision (e.g., 10 decimal places; adjust as needed for precision)
getcontext().prec = 10

# Ensure folders exist
os.makedirs('heikin_ashi', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Get all CSV files in data folder (or potential symbols)
data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
print(f"Found {len(data_files)} CSV files in 'data' folder.")

# Dictionary to track failed conversions
failed_conversions = {}

# Initialize TvDatafeed for fetching full history
tv = TvDatafeed()  # No login - may have limitations

start_date_str = '1945-01-01'
start_date = pd.to_datetime(start_date_str)

for filename in data_files:
    symbol = filename[:-4]
    csv_path = os.path.join('data', filename)
    ha_path = os.path.join('heikin_ashi', filename)
    
    if os.path.exists(ha_path):
        print(f"HA file {ha_path} already exists, skipping {filename}.")
        continue
    
    try:
        print(f"Processing {filename}...")
        
        # Fetch full historical data for accurate HA calculation
        print(f"Fetching full history for {symbol}...")
        full_data = tv.get_hist(
            symbol=symbol,
            exchange='NSE',
            interval=Interval.in_weekly,
            n_bars=5000
        )
        
        if full_data is None or full_data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Slice for saving (from 2022-01-01)
        sliced_data = full_data.loc[start_date:] if start_date in full_data.index else full_data[full_data.index >= start_date]
        
        if sliced_data.empty:
            raise ValueError(f"No data from {start_date_str} onwards for {symbol}")
        
        # Save sliced regular data if it doesn't exist
        if not os.path.exists(csv_path):
            sliced_df = sliced_data.reset_index().copy()
            sliced_df.rename(columns={'datetime': 'Date'}, inplace=True)
            # Ensure columns are lowercase if needed
            sliced_df.columns = [col.lower() if col != 'Date' else 'Date' for col in sliced_df.columns]
            sliced_df.to_csv(csv_path, index=False)
            print(f"Saved sliced regular data to {csv_path} ({len(sliced_df)} rows)")
        
        # Now compute HA on the FULL data (unsliced) for proper dependency chain
        full_df = full_data.reset_index().copy()
        full_df.rename(columns={'datetime': 'Date'}, inplace=True)
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        full_df = full_df.sort_values('Date').reset_index(drop=True)
        
        print(f"Computing HA on full history ({len(full_df)} rows) for {symbol}...")
        
        if len(full_df) < 1:
            raise ValueError("Empty full DataFrame")
        
        # Ensure required columns exist (lowercase)
        required_cols = ['Date', 'open', 'high', 'low', 'close']
        full_df.columns = [col.lower() if col != 'Date' else 'Date' for col in full_df.columns]
        if not all(col in full_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in full_df.columns]
            raise ValueError(f"Missing required columns in full data: {missing}")
        
        # Convert OHLC to Decimal for precise calculations
        full_df['open_dec'] = full_df['open'].apply(Decimal)
        full_df['high_dec'] = full_df['high'].apply(Decimal)
        full_df['low_dec'] = full_df['low'].apply(Decimal)
        full_df['close_dec'] = full_df['close'].apply(Decimal)
        
        # Initialize lists for HA values (as Decimal)
        ha_open = []
        ha_high = []
        ha_low = []
        ha_close = []
        
        # First bar (standard initialization)
        first_open = full_df['open_dec'].iloc[0]
        first_high = full_df['high_dec'].iloc[0]
        first_low = full_df['low_dec'].iloc[0]
        first_close = full_df['close_dec'].iloc[0]
        
        ha_close_val = (first_open + first_high + first_low + first_close) / Decimal('4')
        ha_open_val = (first_open + first_close) / Decimal('2')
        
        ha_close.append(ha_close_val)
        ha_open.append(ha_open_val)
        ha_high.append(max(first_high, ha_open_val, ha_close_val))
        ha_low.append(min(first_low, ha_open_val, ha_close_val))
        
        # Subsequent bars (dependent on previous HA)
        prev_ha_open = ha_open_val
        prev_ha_close = ha_close_val
        for i in range(1, len(full_df)):
            curr_open = full_df['open_dec'].iloc[i]
            curr_high = full_df['high_dec'].iloc[i]
            curr_low = full_df['low_dec'].iloc[i]
            curr_close = full_df['close_dec'].iloc[i]
            
            ha_close_val = (curr_open + curr_high + curr_low + curr_close) / Decimal('4')
            ha_open_val = (prev_ha_open + prev_ha_close) / Decimal('2')
            
            ha_close.append(ha_close_val)
            ha_open.append(ha_open_val)
            ha_high.append(max(curr_high, ha_open_val, ha_close_val))
            ha_low.append(min(curr_low, ha_open_val, ha_close_val))
            
            prev_ha_open = ha_open_val
            prev_ha_close = ha_close_val
        
        # Add HA columns to full_df
        full_df['ha_open'] = [float(x) for x in ha_open]
        full_df['ha_high'] = [float(x) for x in ha_high]
        full_df['ha_low'] = [float(x) for x in ha_low]
        full_df['ha_close'] = [float(x) for x in ha_close]
        
        # Keep only relevant columns
        ha_cols = ['Date', 'ha_open', 'ha_high', 'ha_low', 'ha_close']
        if 'volume' in full_df.columns:
            ha_cols.append('volume')
        
        ha_full_df = full_df[ha_cols].copy()
        
        # Slice HA to start_date
        ha_df = ha_full_df[ha_full_df['Date'] >= start_date].copy()
        
        if ha_df.empty:
            raise ValueError(f"No HA data from {start_date_str} onwards for {symbol}")
        
        # Save sliced HA data
        ha_df.to_csv(ha_path, index=False)
        print(f"Converted and saved {len(ha_df)} accurate HA candles (from full history) to {ha_path}")
        
    except Exception as e:
        reason = f"Error: {str(e)}"
        failed_conversions[filename] = reason
        print(f"Failed to process {filename}: {e}")

# Print summary of failed conversions
if failed_conversions:
    print("\nFailed conversions:")
    for filename, reason in failed_conversions.items():
        print(f"- {filename}: {reason}")
else:
    print("\nAll conversions successful (skipped existing HA files).")

print("Heikin Ashi conversion complete. Check 'heikin_ashi' folder for CSV files.")