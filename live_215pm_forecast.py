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
plots_live_dir = os.path.join(base_dir, "plots", "live_215pm_forecast")
os.makedirs(plots_live_dir, exist_ok=True)

print("=" * 80)
print("FETCHING TODAY'S LIVE NIFTY 50 DATA AS OF 2:15 PM IST (JULY 30, 2026)...")
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
            
            ist_tz = pytz.timezone("Asia/Kolkata")
            df['timestamps'] = df['timestamps'].dt.tz_convert(ist_tz).dt.tz_localize(None)
            
            df['time_only'] = df['timestamps'].dt.time
            start_t = datetime.strptime("09:15", "%H:%M").time()
            end_t = datetime.strptime("15:30", "%H:%M").time()
            df = df[(df['time_only'] >= start_t) & (df['time_only'] <= end_t)].copy()
            df['amount'] = df['close'] * df['volume']
            return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].sort_values('timestamps').reset_index(drop=True)
    except Exception as e:
        print(f"Direct API call for {interval} failed ({e}). Loading dataset...")
        return None

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

def run_live_215pm_forecast(df, tf_label, pred_len=15, lookback=350):
    if df is None or len(df) < 50:
        return None

    # Compute Moving Averages
    df['sma50'] = df['close'].rolling(window=50, min_periods=10).mean()
    df['sma100'] = df['close'].rolling(window=100, min_periods=20).mean()
    df['sma200'] = df['close'].rolling(window=200, min_periods=30).mean()

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
    
    # Immediate 15-min forecast
    next_15m_idx = min(len(pred_df) - 1, 15 if "1 Min" in tf_label else (3 if "5 Min" in tf_label else 1))
    next_15m_price = pred_df['close'].iloc[next_15m_idx]
    next_15m_move = next_15m_price - current_price
    
    # Session close forecast (3:30 PM)
    predicted_close = pred_df['close'].iloc[-1]
    predicted_move_close = predicted_close - current_price
    pct_move_close = (predicted_move_close / current_price) * 100

    signal = "BUY CALL (BULLISH)" if predicted_move_close > 5 else ("BUY PUT (BEARISH)" if predicted_move_close < -5 else "NEUTRAL / NO TRADE")

    print(f"\n" + "=" * 80)
    print(f"LIVE 2:15 PM FORECAST ({tf_label}) - AS OF {last_ts.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"Current Nifty 50 Price: {current_price:.2f} INR")
    print(f"Next 15-Min Forecast Price: {next_15m_price:.2f} INR ({next_15m_move:+.2f} Pts)")
    print(f"Predicted Market Close Price (3:30 PM IST): {predicted_close:.2f} INR ({predicted_move_close:+.2f} Pts / {pct_move_close:+.2f}%)")
    print(f"Actionable Signal: {signal}")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(14, 7))

    ctx_subset = df.iloc[-40:].copy()
    draw_ohlc_candlesticks(ax, ctx_subset, width_ratio=0.5, alpha=0.9)

    # Plot Moving Averages
    ax.plot(ctx_subset['timestamps'], ctx_subset['sma50'], label='50 SMA (Cyan)', color='#06b6d4', linewidth=1.8, linestyle='-')
    ax.plot(ctx_subset['timestamps'], ctx_subset['sma100'], label='100 SMA (Purple)', color='#8b5cf6', linewidth=1.8, linestyle='-')
    ax.plot(ctx_subset['timestamps'], ctx_subset['sma200'], label='200 SMA (Gold)', color='#eab308', linewidth=2.0, linestyle='-')

    # Plot Kronos Forecast Path
    ax.plot(y_ts, pred_df['close'], label='Kronos AI Predicted Path (2:15 PM -> 3:30 PM Close)', color='#f59e0b', linewidth=3.0, linestyle='--', marker='o', markersize=4)

    trigger_str = last_ts.strftime('%Y-%m-%d %H:%M IST')
    ax.axvline(x=mdates.date2num(last_ts), color='#374151', linestyle=':', linewidth=2, label=f'2:15 PM Trigger Time ({trigger_str})')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M IST'))
    fig.autofmt_xdate()

    ax.set_title(f"LIVE 2:15 PM IST FORECAST | Nifty 50 ({tf_label})\nCurrent: {current_price:.2f} -> Next 15m: {next_15m_price:.2f} ({next_15m_move:+.2f} pts) -> Pred Close: {predicted_close:.2f} ({predicted_move_close:+.2f} pts)\nSignal: {signal}", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Time (Indian Standard Time - IST)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color='#10b981', lw=4, label='Live Bullish Candle'),
        Line2D([0], [0], color='#ef4444', lw=4, label='Live Bearish Candle'),
        Line2D([0], [0], color='#06b6d4', lw=1.8, label='50 SMA'),
        Line2D([0], [0], color='#8b5cf6', lw=1.8, label='100 SMA'),
        Line2D([0], [0], color='#eab308', lw=2.0, label='200 SMA'),
        Line2D([0], [0], color='#f59e0b', lw=3.0, linestyle='--', marker='o', label=f'Kronos Predicted Path ({predicted_move_close:+.2f} pts)'),
        Line2D([0], [0], color='#374151', lw=2, linestyle=':', label=f'Trigger: {trigger_str}')
    ]
    ax.legend(handles=custom_legend, loc='upper left', fontsize=9, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    plot_file = os.path.join(plots_live_dir, f"live_215pm_{tf_label.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved Live 2:15 PM Plot: {plot_file}")

    return {
        'timeframe': tf_label,
        'trigger_time': trigger_str,
        'current_price': current_price,
        'next_15m_price': next_15m_price,
        'next_15m_move_pts': next_15m_move,
        'predicted_close': predicted_close,
        'predicted_move_close_pts': predicted_move_close,
        'signal': signal,
        'plot_file': plot_file
    }

# Execute Live 2:15 PM Forecasts
res_15m = run_live_215pm_forecast(df_15m, "15 Minute", pred_len=5, lookback=300)
res_5m = run_live_215pm_forecast(df_5m, "5 Minute", pred_len=15, lookback=300)
res_1m = run_live_215pm_forecast(df_1m, "1 Minute", pred_len=75, lookback=300)

summary_today = [r for r in [res_15m, res_5m, res_1m] if r is not None]
summary_df = pd.DataFrame(summary_today)
summary_df.to_csv(os.path.join(plots_live_dir, "live_215pm_forecast_summary.csv"), index=False)
print("\nLIVE 2:15 PM IST FORECAST COMPLETED SUCCESSFULLY!")
