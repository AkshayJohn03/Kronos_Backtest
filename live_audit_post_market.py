import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, time
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
plots_audit_dir = os.path.join(base_dir, "plots", "live_audit_post_market")
os.makedirs(plots_audit_dir, exist_ok=True)

print("=" * 80)
print("AUDITING TODAY'S 2:15 PM IST LIVE PREDICTIONS AGAINST FINAL 3:30 PM MARKET CLOSE...")
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

def audit_215pm_forecast(df, tf_label, lookback=350):
    if df is None or len(df) < 50:
        return None

    df['sma50'] = df['close'].rolling(window=50, min_periods=10).mean()
    df['sma100'] = df['close'].rolling(window=100, min_periods=20).mean()
    df['sma200'] = df['close'].rolling(window=200, min_periods=30).mean()

    df['date_str'] = df['timestamps'].dt.strftime('%Y-%m-%d')
    latest_date = df['date_str'].iloc[-1]
    
    day_df = df[df['date_str'] == latest_date].copy().reset_index(drop=True)
    
    # Locate 2:15 PM IST trigger index
    target_t = time(14, 15, 0)
    trigger_indices = day_df[day_df['timestamps'].dt.time >= target_t].index
    
    if len(trigger_indices) == 0:
        trigger_idx_in_day = len(day_df) - 15
    else:
        trigger_idx_in_day = trigger_indices[0]

    # Map back to main df global index
    trigger_ts = day_df.iloc[trigger_idx_in_day]['timestamps']
    global_idx = df[(df['date_str'] == latest_date) & (df['timestamps'] == trigger_ts)].index[0]

    if global_idx < lookback:
        lookback = global_idx

    x_df = df.iloc[global_idx - lookback : global_idx][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = df.iloc[global_idx - lookback : global_idx]['timestamps'].copy()

    # Remaining ground truth candles from 2:15 PM to 3:30 PM market close
    remaining_day = day_df.iloc[trigger_idx_in_day:].copy()
    pred_len = len(remaining_day) if len(remaining_day) > 0 else 15

    y_ts = remaining_day['timestamps'].copy()
    act_df = remaining_day[['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )

    current_price = x_df['close'].iloc[-1]
    predicted_close = pred_df['close'].iloc[-1]
    actual_close = act_df['close'].iloc[-1]

    pred_move = predicted_close - current_price
    act_move = actual_close - current_price

    is_win = (np.sign(pred_move) == np.sign(act_move))
    signal = "BUY CALL (BULLISH)" if pred_move > 5 else ("BUY PUT (BEARISH)" if pred_move < -5 else "NEUTRAL")
    mae = abs(actual_close - predicted_close)

    trigger_str = x_ts.iloc[-1].strftime('%Y-%m-%d %H:%M IST')

    print(f"\n" + "=" * 80)
    print(f"POST-MARKET AUDIT (2:15 PM -> 3:30 PM CLOSE) ({tf_label}) - Date: {latest_date}")
    print(f"Trigger Time: {trigger_str}")
    print(f"2:15 PM Price: {current_price:.2f} INR")
    print(f"Predicted Market Close (3:30 PM): {predicted_close:.2f} INR ({pred_move:+.2f} pts)")
    print(f"Actual Ground-Truth Close (3:30 PM): {actual_close:.2f} INR ({act_move:+.2f} pts)")
    win_str = "WIN [MATCH]" if is_win else "MISMATCH [FAIL]"
    print(f"Directional Win: {win_str} | MAE: {mae:.2f} pts | Signal: {signal}")
    print("=" * 80)

    # Render Visual Audit Plot (Actual Candles vs Kronos Prediction Path)
    fig, ax = plt.subplots(figsize=(14, 7))

    ctx_subset = df.iloc[max(0, global_idx - 35) : global_idx].copy()
    draw_ohlc_candlesticks(ax, ctx_subset, width_ratio=0.5, alpha=0.7)
    draw_ohlc_candlesticks(ax, act_df, width_ratio=0.5, alpha=1.0)

    # Plot 50, 100, 200 SMA Moving Averages
    full_plot_subset = df.iloc[max(0, global_idx - 35) : global_idx + pred_len].copy()
    ax.plot(full_plot_subset['timestamps'], full_plot_subset['sma50'], label='50 SMA (Cyan)', color='#06b6d4', linewidth=1.8, linestyle='-')
    ax.plot(full_plot_subset['timestamps'], full_plot_subset['sma100'], label='100 SMA (Purple)', color='#8b5cf6', linewidth=1.8, linestyle='-')
    ax.plot(full_plot_subset['timestamps'], full_plot_subset['sma200'], label='200 SMA (Gold)', color='#eab308', linewidth=2.0, linestyle='-')

    # Plot Kronos AI Prediction Path
    ax.plot(y_ts, pred_df['close'], label='Kronos AI Predicted Path (Gold Dashed)', color='#f59e0b', linewidth=3.0, linestyle='--', marker='o', markersize=4)

    # Vertical 2:15 PM Trigger Marker
    ax.axvline(x=mdates.date2num(x_ts.iloc[-1]), color='#374151', linestyle=':', linewidth=2, label=f'2:15 PM Trigger ({trigger_str})')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M IST'))
    fig.autofmt_xdate()

    win_text = "WIN ✅" if is_win else "MISMATCH ❌"
    ax.set_title(f"POST-MARKET AUDIT (2:15 PM - 3:30 PM IST Close) | Nifty 50 ({tf_label})\nPrice at 2:15 PM: {current_price:.2f} | Pred Close: {predicted_close:.2f} ({pred_move:+.2f} pts) | Actual Close: {actual_close:.2f} ({act_move:+.2f} pts) | {win_text}", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Date & Time (Indian Standard Time - IST)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color='#10b981', lw=4, label='Actual Bullish Candle'),
        Line2D([0], [0], color='#ef4444', lw=4, label='Actual Bearish Candle'),
        Line2D([0], [0], color='#06b6d4', lw=1.8, label='50 SMA'),
        Line2D([0], [0], color='#8b5cf6', lw=1.8, label='100 SMA'),
        Line2D([0], [0], color='#eab308', lw=2.0, label='200 SMA'),
        Line2D([0], [0], color='#f59e0b', lw=3.0, linestyle='--', marker='o', label=f'Kronos Forecast ({pred_move:+.2f} pts)'),
        Line2D([0], [0], color='#374151', lw=2, linestyle=':', label=f'Trigger: {trigger_str}')
    ]
    ax.legend(handles=custom_legend, loc='upper left', fontsize=9, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    plot_file = os.path.join(plots_audit_dir, f"audit_215pm_{tf_label.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved Post-Market Audit Plot: {plot_file}")

    return {
        'timeframe': tf_label,
        'date': latest_date,
        'trigger_time': trigger_str,
        'price_at_215pm': current_price,
        'predicted_close': predicted_close,
        'actual_close': actual_close,
        'predicted_move_pts': pred_move,
        'actual_move_pts': act_move,
        'is_win': is_win,
        'mae_pts': mae,
        'signal': signal,
        'plot_file': plot_file
    }

# Execute Post-Market Audit across 15m, 5m, 1m
res_15m = audit_215pm_forecast(df_15m, "15 Minute", lookback=300)
res_5m = audit_215pm_forecast(df_5m, "5 Minute", lookback=300)
res_1m = audit_215pm_forecast(df_1m, "1 Minute", lookback=300)

summary_today = [r for r in [res_15m, res_5m, res_1m] if r is not None]
summary_df = pd.DataFrame(summary_today)
summary_df.to_csv(os.path.join(plots_audit_dir, "post_market_audit_summary.csv"), index=False)

print("\n" + "=" * 80)
print("POST-MARKET AUDIT SUMMARY TABLE (JULY 30, 2026)")
print("=" * 80)
print(summary_df.to_string(index=False))
print("\nPOST-MARKET AUDIT COMPLETED SUCCESSFULLY!")
