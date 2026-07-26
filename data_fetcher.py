import os
import requests
import json
import pandas as pd
import ssl
import urllib3
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

def fetch_yahoo_direct(symbol="^NSEI", interval="1m", range_param="7d", period1=None, period2=None):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if period1 and period2:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&period1={period1}&period2={period2}"
    else:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_param}"
    
    print(f"Fetching from Yahoo direct URL: {url}")
    resp = requests.get(url, headers=headers, verify=False, timeout=15)
    data = resp.json()
    
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quote = result['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'timestamps': pd.to_datetime(timestamps, unit='s'),
        'open': quote['open'],
        'high': quote['high'],
        'low': quote['low'],
        'close': quote['close'],
        'volume': quote.get('volume', [0]*len(timestamps))
    })
    
    # Drop NaNs
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    df['amount'] = df['close'] * df['volume'].fillna(0)
    return df

data_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\data"
os.makedirs(data_dir, exist_ok=True)

print("=" * 70)
print("FETCHING NIFTY50 SPOT OHLCV DATA VIA DIRECT YAHOO REST API...")
print("Constraint: Post June 2025 (>= 2025-06-01)")
print("=" * 70)

# 1. 1m Data
print("\nFetching 1m data...")
df_1m = fetch_yahoo_direct(interval="1m", range_param="7d")
df_1m = df_1m[df_1m['timestamps'] >= '2025-06-01'].copy()
unique_days_1m = df_1m['timestamps'].dt.date.unique()
test_day_1m = unique_days_1m[-1] if len(unique_days_1m) > 0 else None
df_1m_selected = df_1m[df_1m['timestamps'].dt.date == test_day_1m].copy() if test_day_1m else df_1m

csv_1m = os.path.join(data_dir, "nifty_1m.csv")
df_1m_selected.to_csv(csv_1m, index=False)
print(f"[1 MINUTE DATA] Selected Date: {test_day_1m} | Rows: {len(df_1m_selected)}")
print(df_1m_selected.head(3))

# 2. 5m Data
print("\nFetching 5m data...")
df_5m = fetch_yahoo_direct(interval="5m", range_param="60d")
df_5m = df_5m[df_5m['timestamps'] >= '2025-06-01'].copy()
if test_day_1m:
    df_5m = df_5m[df_5m['timestamps'].dt.date != test_day_1m].copy()
unique_days_5m = df_5m['timestamps'].dt.date.unique()
test_days_5m = unique_days_5m[-3:-1] if len(unique_days_5m) >= 3 else unique_days_5m
df_5m_selected = df_5m[df_5m['timestamps'].dt.date.isin(test_days_5m)].copy()

csv_5m = os.path.join(data_dir, "nifty_5m.csv")
df_5m_selected.to_csv(csv_5m, index=False)
print(f"[5 MINUTE DATA] Selected Dates: {test_days_5m} | Rows: {len(df_5m_selected)}")
print(df_5m_selected.head(3))

# 3. 15m Data
print("\nFetching 15m data...")
df_15m = fetch_yahoo_direct(interval="15m", range_param="60d")
df_15m = df_15m[df_15m['timestamps'] >= '2025-06-01'].copy()
used_days = set([test_day_1m] if test_day_1m else [])
if len(test_days_5m) > 0:
    used_days.update(test_days_5m)

df_15m = df_15m[~df_15m['timestamps'].dt.date.isin(used_days)].copy()
unique_days_15m = df_15m['timestamps'].dt.date.unique()
test_days_15m = unique_days_15m[-5:-1] if len(unique_days_15m) >= 5 else unique_days_15m
df_15m_selected = df_15m[df_15m['timestamps'].dt.date.isin(test_days_15m)].copy()

csv_15m = os.path.join(data_dir, "nifty_15m.csv")
df_15m_selected.to_csv(csv_15m, index=False)
print(f"[15 MINUTE DATA] Selected Dates: {test_days_15m} | Rows: {len(df_15m_selected)}")
print(df_15m_selected.head(3))

# 4. 1d Data (From 2025-06-01 = 1748736000)
print("\nFetching 1d data...")
df_1d = fetch_yahoo_direct(interval="1d", period1=1748736000, period2=1999999999)
df_1d = df_1d[df_1d['timestamps'] >= '2025-06-01'].copy()

csv_1d = os.path.join(data_dir, "nifty_1d.csv")
df_1d.to_csv(csv_1d, index=False)
print(f"[DAILY DATA] Dates: {df_1d['timestamps'].dt.date.min()} to {df_1d['timestamps'].dt.date.max()} | Rows: {len(df_1d)}")
print(df_1d.head(3))

print("\nSUCCESS! ALL DISTINCT NON-OVERLAPPING NIFTY 50 DATASETS SAVED FOR VERIFICATION.")
