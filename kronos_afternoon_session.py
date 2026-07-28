import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, time
import torch

kronos_src_path = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\Kronos_src"
if kronos_src_path not in sys.path:
    sys.path.append(kronos_src_path)

from model import KronosTokenizer, Kronos, KronosPredictor

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_pm_dir = os.path.join(base_dir, "plots", "afternoon_session_2pm_close")
os.makedirs(plots_pm_dir, exist_ok=True)

print("=" * 80)
print("RUNNING AFTERNOON SESSION FORECAST (02:00 PM IST TO 03:30 PM IST CLOSE)...")
print("Target Machine: CPU Execution Mode")
print("=" * 80)

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

def run_afternoon_session_forecast(file_name, tf_label, lookback=400):
    filepath = os.path.join(data_dir, file_name)
    if not os.path.exists(filepath):
        print(f"File {filepath} missing. Skipping.")
        return None

    df = pd.read_csv(filepath)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df['time_only'] = df['timestamps'].dt.time

    # Calculate Moving Averages
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['sma50'] = df['close'].rolling(window=50, min_periods=10).mean()

    # Find closest candle to 14:00:00 IST (02:00 PM IST)
    target_time = time(14, 0, 0)
    df['date_str'] = df['timestamps'].dt.strftime('%Y-%m-%d')
    unique_dates = df['date_str'].unique()
    latest_date = unique_dates[-1]

    day_df = df[df['date_str'] == latest_date].copy().reset_index(drop=True)
    pm_indices = day_df[day_df['time_only'] >= target_time].index

    if len(pm_indices) == 0:
        trigger_idx_in_day = len(day_df) - 10
    else:
        trigger_idx_in_day = pm_indices[0]

    # Map back to main df index
    global_idx = df[(df['date_str'] == latest_date) & (df['timestamps'] == day_df.iloc[trigger_idx_in_day]['timestamps'])].index[0]

    if global_idx < lookback:
        lookback = global_idx

    x_df = df.iloc[global_idx - lookback : global_idx][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = df.iloc[global_idx - lookback : global_idx]['timestamps'].copy()

    # Remaining candles in the day up to 15:30 IST
    remaining_day = day_df.iloc[trigger_idx_in_day:].copy()
    pred_len = len(remaining_day) if len(remaining_day) > 0 else 10

    y_ts = remaining_day['timestamps'].copy()
    act_df = remaining_day[['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )

    current_price = x_df['close'].iloc[-1]
    predicted_close = pred_df['close'].iloc[-1]
    actual_close = act_df['close'].iloc[-1] if len(act_df) > 0 else current_price

    pred_move = predicted_close - current_price
    act_move = actual_close - current_price

    is_win = (np.sign(pred_move) == np.sign(act_move))
    signal = "BUY CALL (BULLISH)" if pred_move > 5 else ("BUY PUT (BEARISH)" if pred_move < -5 else "NEUTRAL")

    trigger_str = x_ts.iloc[-1].strftime('%Y-%m-%d %H:%M IST')

    print(f"\n" + "=" * 80)
    print(f"AFTERNOON SESSION (02:00 PM -> CLOSE) ({tf_label}) - Date: {latest_date}")
    print(f"Trigger Time: {trigger_str}")
    print(f"Price at 02:00 PM IST: {current_price:.2f} INR")
    print(f"Predicted Close (03:30 PM): {predicted_close:.2f} INR ({pred_move:+.2f} pts)")
    print(f"Actual Close (03:30 PM): {actual_close:.2f} INR ({act_move:+.2f} pts)")
    print(f"Directional Win: {'WIN' if is_win else 'MISMATCH'} | Signal: {signal}")
    print("=" * 80)

    # Render Professional OHLC Candlestick Plot with Moving Averages
    fig, ax = plt.subplots(figsize=(14, 7))

    # Subset context candles
    ctx_subset = df.iloc[max(0, global_idx - 35) : global_idx].copy()
    draw_ohlc_candlesticks(ax, ctx_subset, width_ratio=0.5, alpha=0.7)
    draw_ohlc_candlesticks(ax, act_df, width_ratio=0.5, alpha=1.0)

    # Plot Moving Averages
    full_plot_subset = df.iloc[max(0, global_idx - 35) : global_idx + pred_len].copy()
    ax.plot(full_plot_subset['timestamps'], full_plot_subset['ema20'], label='20 EMA (Cyan)', color='#06b6d4', linewidth=1.8, linestyle='-')
    ax.plot(full_plot_subset['timestamps'], full_plot_subset['sma50'], label='50 SMA (Purple)', color='#8b5cf6', linewidth=1.8, linestyle='-')

    # Plot Kronos Prediction Path
    ax.plot(y_ts, pred_df['close'], label='Kronos AI Predicted Path (Gold Dashed)', color='#f59e0b', linewidth=3.0, linestyle='--', marker='o', markersize=4)

    # Vertical 2:00 PM IST Trigger Line
    ax.axvline(x=mdates.date2num(x_ts.iloc[-1]), color='#374151', linestyle=':', linewidth=2, label=f'2:00 PM Trigger: {trigger_str}')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M IST'))
    fig.autofmt_xdate()

    ax.set_title(f"Afternoon Session (02:00 PM - 03:30 PM IST) Forecast | Nifty 50 ({tf_label})\n02:00 PM Price: {current_price:.2f} -> Pred Close: {predicted_close:.2f} ({pred_move:+.2f} pts) | Signal: {signal}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date & Time (Indian Standard Time - IST)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color='#10b981', lw=4, label='Bullish Market Candle'),
        Line2D([0], [0], color='#ef4444', lw=4, label='Bearish Market Candle'),
        Line2D([0], [0], color='#06b6d4', lw=1.8, label='20 EMA (Cyan Trendline)'),
        Line2D([0], [0], color='#8b5cf6', lw=1.8, label='50 SMA (Purple Trendline)'),
        Line2D([0], [0], color='#f59e0b', lw=3.0, linestyle='--', marker='o', label=f'Kronos Predicted Path ({pred_move:+.2f} pts)'),
        Line2D([0], [0], color='#374151', lw=2, linestyle=':', label=f'Trigger: {trigger_str}')
    ]
    ax.legend(handles=custom_legend, loc='upper left', fontsize=10, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    plot_file = os.path.join(plots_pm_dir, f"afternoon_session_{tf_label.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved Afternoon Session Plot: {plot_file}")

    return {
        'timeframe': tf_label,
        'date': latest_date,
        'trigger_time': trigger_str,
        'price_at_2pm': current_price,
        'predicted_close': predicted_close,
        'actual_close': actual_close,
        'predicted_move_pts': pred_move,
        'actual_move_pts': act_move,
        'is_win': is_win,
        'signal': signal,
        'plot_file': plot_file
    }

# Execute Afternoon Session Forecast across 1m, 5m, 15m
timeframes = [
    ("nifty_15m_ist.csv", "15 Minute"),
    ("nifty_5m_ist.csv", "5 Minute"),
    ("nifty_1m_ist.csv", "1 Minute")
]

results_pm = []
for fname, tf_label in timeframes:
    res = run_afternoon_session_forecast(fname, tf_label)
    if res:
        results_pm.append(res)

summary_df = pd.DataFrame(results_pm)
print("\n" + "=" * 80)
print("AFTERNOON SESSION (02:00 PM TO 03:30 PM IST CLOSE) SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(plots_pm_dir, "afternoon_session_summary.csv"), index=False)
print("\nAFTERNOON SESSION FORECAST COMPLETED SUCCESSFULLY!")
