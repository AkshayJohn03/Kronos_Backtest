import os
import requests
import json
import pandas as pd
import ssl
import urllib3
from datetime import datetime
import pytz

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

def fetch_nifty_ist(symbol="^NSEI", interval="1m", range_param="7d"):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_param}"
    
    print(f"Fetching {interval} data from Yahoo API: {url}")
    resp = requests.get(url, headers=headers, verify=False, timeout=15)
    data = resp.json()
    
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quote = result['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'timestamps_utc': pd.to_datetime(timestamps, unit='s', utc=True),
        'open': quote['open'],
        'high': quote['high'],
        'low': quote['low'],
        'close': quote['close'],
        'volume': quote.get('volume', [0]*len(timestamps))
    })
    
    # Drop NaNs
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    
    # Convert UTC timestamps to Indian Standard Time (Asia/Kolkata)
    ist_tz = pytz.timezone('Asia/Kolkata')
    df['timestamps'] = df['timestamps_utc'].dt.tz_convert(ist_tz).dt.tz_localize(None)
    
    # Filter strictly for NSE Market Hours (09:15 AM IST to 03:30 PM IST)
    df['time_only'] = df['timestamps'].dt.time
    market_start = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_end = datetime.strptime("15:30:00", "%H:%M:%S").time()
    
    df = df[(df['time_only'] >= market_start) & (df['time_only'] <= market_end)].copy()
    
    # Filter post June 2025
    df = df[df['timestamps'] >= '2025-06-01'].copy()
    
    df['amount'] = df['close'] * df['volume'].fillna(0)
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].sort_values('timestamps').reset_index(drop=True)
    
    return df

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)

print("=" * 80)
print("FETCHING REAL NIFTY 50 SPOT DATA IN INDIAN STANDARD TIME (IST)...")
print("Market Session Constraint: 09:15 AM IST to 03:30 PM IST | Dates >= 2025-06-01")
print("=" * 80)

# 1. 1-Minute Data (Last 7 days intraday IST)
df_1m = fetch_nifty_ist(interval="1m", range_param="7d")
csv_1m = os.path.join(data_dir, "nifty_1m_ist.csv")
df_1m.to_csv(csv_1m, index=False)
print(f"\n[1-MINUTE IST DATA] Rows: {len(df_1m)} | Range: {df_1m['timestamps'].min()} IST to {df_1m['timestamps'].max()} IST")
print(df_1m.head(3))
print("...")
print(df_1m.tail(3))

# 2. 5-Minute Data (Last 60 days intraday IST)
df_5m = fetch_nifty_ist(interval="5m", range_param="60d")
csv_5m = os.path.join(data_dir, "nifty_5m_ist.csv")
df_5m.to_csv(csv_5m, index=False)
print(f"\n[5-MINUTE IST DATA] Rows: {len(df_5m)} | Range: {df_5m['timestamps'].min()} IST to {df_5m['timestamps'].max()} IST")
print(df_5m.head(3))
print("...")
print(df_5m.tail(3))

# 3. 15-Minute Data (Last 60 days intraday IST)
df_15m = fetch_nifty_ist(interval="15m", range_param="60d")
csv_15m = os.path.join(data_dir, "nifty_15m_ist.csv")
df_15m.to_csv(csv_15m, index=False)
print(f"\n[15-MINUTE IST DATA] Rows: {len(df_15m)} | Range: {df_15m['timestamps'].min()} IST to {df_15m['timestamps'].max()} IST")
print(df_15m.head(3))
print("...")
print(df_15m.tail(3))

print("\nSUCCESS: ALL NIFTY 50 DATASETS CONVERTED TO IST (09:15 - 15:30 IST) AND SAVED!")
