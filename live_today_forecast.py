import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import pytz
import json
import urllib.request
import torch

kronos_src_path = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\Kronos_src"
if kronos_src_path not in sys.path:
    sys.path.append(kronos_src_path)

from model import KronosTokenizer, Kronos, KronosPredictor

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_today_dir = os.path.join(base_dir, "plots", "today_live_forecast")
os.makedirs(plots_today_dir, exist_ok=True)

print("=" * 80)
print("FETCHING TODAY'S LIVE NIFTY 50 DATA (3:13 PM IST)...")
print("=" * 80)

def fetch_yahoo_direct(symbol="^NSEI", interval="1m", range_str="5d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_str}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'timestamps': pd.to_datetime(timestamps, unit='s', utc=True),
                'open': quote['open'],
                'high': quote['high'],
                'low': quote['low'],
                'close': quote['close'],
                'volume': quote['volume']
            }).dropna()
            
            # Convert to IST
            ist_tz = pytz.timezone("Asia/Kolkata")
            df['timestamps'] = df['timestamps'].dt.tz_convert(ist_tz).dt.tz_localize(None)
            
            # Filter active NSE hours
            df['time_only'] = df['timestamps'].dt.time
            start_t = datetime.strptime("09:15", "%H:%M").time()
            end_t = datetime.strptime("15:30", "%H:%M").time()
            df = df[(df['time_only'] >= start_t) & (df['time_only'] <= end_t)].copy()
            df['amount'] = df['close'] * df['volume']
            return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].sort_values('timestamps').reset_index(drop=True)
    except Exception as e:
        print(f"Direct API call for {interval} failed ({e}). Loading existing dataset...")
        return None

# Attempt live fetch with fallback to local files
df_1m = fetch_yahoo_direct(interval="1m", range_str="5d")
if df_1m is None or len(df_1m) == 0:
    df_1m = pd.read_csv(os.path.join(data_dir, "nifty_1m_ist.csv"))
    df_1m['timestamps'] = pd.to_datetime(df_1m['timestamps'])

df_5m = fetch_yahoo_direct(interval="5m", range_str="10d")
if df_5m is None or len(df_5m) == 0:
    df_5m = pd.read_csv(os.path.join(data_dir, "nifty_5m_ist.csv"))
    df_5m['timestamps'] = pd.to_datetime(df_5m['timestamps'])

df_15m = fetch_yahoo_direct(interval="15m", range_str="30d")
if df_15m is None or len(df_15m) == 0:
    df_15m = pd.read_csv(os.path.join(data_dir, "nifty_15m_ist.csv"))
    df_15m['timestamps'] = pd.to_datetime(df_15m['timestamps'])

print("\nModel Initializing (CPU Mode)...")
device = "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
model.eval()
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

def draw_ohlc_candlesticks(ax, df, width_ratio=0.5, alpha=0.9):
    dates = mdates.date2num(df['timestamps'])
    width = (dates[1] - dates[0]) * width_ratio if len(dates) > 1 else 0.0005

    for i in range(len(df)):
        open_p = df['open'].iloc[i]
        high_p = df['high'].iloc[i]
        low_p = df['low'].iloc[i]
        close_p = df['close'].iloc[i]
        date_val = dates[i]

        color = '#10b981' if close_p >= open_p else '#ef4444'

        ax.plot([date_val, date_val], [low_p, high_p], color=color, linewidth=1.2, alpha=alpha)

        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), 0.25)
        rect = Rectangle((date_val - width/2, body_bottom), width, body_height, facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)

def run_live_forecast(df, tf_label, pred_len=15, lookback=400):
    if df is None or len(df) < 50:
        return None

    if len(df) < lookback:
        lookback = len(df) - pred_len

    x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = df.iloc[-lookback:]['timestamps'].copy()

    last_ts = x_ts.iloc[-1]
    
    freq = "1min" if "1 Min" in tf_label else ("5min" if "5 Min" in tf_label else "15min")
    y_ts = pd.Series(pd.date_range(start=last_ts + pd.Timedelta(freq), periods=pred_len, freq=freq))

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )

    current_price = x_df['close'].iloc[-1]
    predicted_close = pred_df['close'].iloc[-1]
    predicted_move = predicted_close - current_price
    pct_move = (predicted_move / current_price) * 100

    signal = "BUY CALL (BULLISH)" if predicted_move > 5 else ("BUY PUT (BEARISH)" if predicted_move < -5 else "NEUTRAL / NO TRADE")

    print(f"\n" + "=" * 80)
    print(f"TODAY'S LIVE FORECAST ({tf_label}) - AS OF {last_ts.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"Current Nifty 50 Price: {current_price:.2f} INR")
    print(f"Predicted Market Close Price (3:30 PM IST): {predicted_close:.2f} INR")
    print(f"Predicted Remaining Movement: {predicted_move:+.2f} Pts ({pct_move:+.2f}%)")
    print(f"Actionable Signal: {signal}")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(14, 7))

    ctx_subset = df.iloc[-35:].copy()
    draw_ohlc_candlesticks(ax, ctx_subset, width_ratio=0.5, alpha=0.9)

    ax.plot(y_ts, pred_df['close'], label='Kronos AI Forecast (Remaining Session)', color='#f59e0b', linewidth=3.0, linestyle='--', marker='o', markersize=5)

    trigger_str = last_ts.strftime('%Y-%m-%d %H:%M IST')
    ax.axvline(x=mdates.date2num(last_ts), color='#374151', linestyle=':', linewidth=2, label=f'Current Time ({trigger_str})')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M IST'))
    fig.autofmt_xdate()

    ax.set_title(f"TODAY'S LIVE FORECAST | Nifty 50 ({tf_label}) | Current: {current_price:.2f} -> Pred Close: {predicted_close:.2f} ({predicted_move:+.2f} pts)\nSignal: {signal}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Time (Indian Standard Time - IST)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color='#10b981', lw=4, label='Today Live Bullish Candle'),
        Line2D([0], [0], color='#ef4444', lw=4, label='Today Live Bearish Candle'),
        Line2D([0], [0], color='#f59e0b', lw=3.0, linestyle='--', marker='o', label=f'Kronos Predicted Remaining Path ({predicted_move:+.2f} pts)'),
        Line2D([0], [0], color='#374151', lw=2, linestyle=':', label=f'Live Trigger Time: {trigger_str}')
    ]
    ax.legend(handles=custom_legend, loc='upper left', fontsize=10, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    plot_file = os.path.join(plots_today_dir, f"today_live_{tf_label.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved Today's Live Plot: {plot_file}")

    return {
        'timeframe': tf_label,
        'trigger_time': trigger_str,
        'current_price': current_price,
        'predicted_close': predicted_close,
        'predicted_move_pts': predicted_move,
        'signal': signal,
        'plot_file': plot_file
    }

res_15m = run_live_forecast(df_15m, "15 Minute", pred_len=2, lookback=300)
res_5m = run_live_forecast(df_5m, "5 Minute", pred_len=4, lookback=300)
res_1m = run_live_forecast(df_1m, "1 Minute", pred_len=17, lookback=300)

summary_today = [r for r in [res_15m, res_5m, res_1m] if r is not None]
summary_df = pd.DataFrame(summary_today)
summary_df.to_csv(os.path.join(plots_today_dir, "today_live_forecast_summary.csv"), index=False)
print("\nTODAY'S LIVE FORECAST COMPLETED SUCCESSFULLY!")
