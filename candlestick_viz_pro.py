import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import torch

kronos_src_path = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\Kronos_src"
if kronos_src_path not in sys.path:
    sys.path.append(kronos_src_path)

from model import KronosTokenizer, Kronos, KronosPredictor

data_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\data"
plots_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\plots\pro_candlesticks"
os.makedirs(plots_dir, exist_ok=True)

print("=" * 80)
print("GENERATING PROFESSIONAL OHLC CANDLESTICK CHARTS WITH FULL LEGENDS...")
print("=" * 80)

device = "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

def draw_ohlc_candlesticks(ax, df, width_ratio=0.6, alpha=0.9):
    """Draws green (bullish) and red (bearish) OHLC candlestick bars on a Matplotlib axis."""
    dates = mdates.date2num(df['timestamps'])
    
    # Compute candle width based on timeframe diff
    if len(dates) > 1:
        width = (dates[1] - dates[0]) * width_ratio
    else:
        width = 0.0005

    for i in range(len(df)):
        open_p = df['open'].iloc[i]
        high_p = df['high'].iloc[i]
        low_p = df['low'].iloc[i]
        close_p = df['close'].iloc[i]
        date_val = dates[i]

        color = '#10b981' if close_p >= open_p else '#ef4444' # Emerald Green / Bright Red

        # High-Low Wick Line
        ax.plot([date_val, date_val], [low_p, high_p], color=color, linewidth=1.2, alpha=alpha)

        # Open-Close Real Body Rectangle
        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), 0.2) # Minimum height for flat candles
        rect = Rectangle((date_val - width/2, body_bottom), width, body_height, facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)

def plot_pro_candlestick(df, timeframe_label, lookback=120, pred_len=15):
    if len(df) < lookback + pred_len:
        lookback = int(len(df) * 0.6)
        pred_len = int(len(df) * 0.2)

    curr_idx = lookback
    x_df = df.iloc[curr_idx - lookback : curr_idx][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = df.iloc[curr_idx - lookback : curr_idx]['timestamps'].copy()
    y_ts = df.iloc[curr_idx : curr_idx + pred_len]['timestamps'].copy()
    act_df = df.iloc[curr_idx : curr_idx + pred_len][['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    # Display recent 40 context candles + 15 future candles
    context_subset = df.iloc[curr_idx - 40 : curr_idx].copy()
    draw_ohlc_candlesticks(ax, context_subset, width_ratio=0.6, alpha=0.7)
    draw_ohlc_candlesticks(ax, act_df, width_ratio=0.6, alpha=1.0)

    # Plot Kronos Forecasted Path Line
    ax.plot(y_ts, pred_df['close'], label='Kronos AI Predicted Trajectory (Gold Dashed)', color='#f59e0b', linewidth=2.8, linestyle='--', marker='o', markersize=4)

    # Trigger vertical line
    trigger_time_str = x_ts.iloc[-1].strftime('%Y-%m-%d %H:%M IST')
    ax.axvline(x=mdates.date2num(x_ts.iloc[-1]), color='#374151', linestyle=':', linewidth=2, label=f'Forecast Trigger Time ({trigger_time_str})')

    # Date / Time X-Axis Formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()

    start_p = x_df['close'].iloc[-1]
    act_end = act_df['close'].iloc[-1]
    pred_end = pred_df['close'].iloc[-1]
    act_move = act_end - start_p
    pred_move = pred_end - start_p
    is_win = (np.sign(act_move) == np.sign(pred_move))

    # Title & Formatting
    ax.set_title(f"Nifty 50 ({timeframe_label}) OHLC Candlestick Forecast | Date: {trigger_time_str} | Result: {'WIN (Direction Matched)' if is_win else 'MISMATCH'}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date & Time (Indian Standard Time - IST)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color='#10b981', lw=4, label='Bullish Market Candle (Close >= Open)'),
        Line2D([0], [0], color='#ef4444', lw=4, label='Bearish Market Candle (Close < Open)'),
        Line2D([0], [0], color='#f59e0b', lw=2.8, linestyle='--', marker='o', label='Kronos AI Forecast Path'),
        Line2D([0], [0], color='#374151', lw=2, linestyle=':', label=f'Trigger IST Time: {trigger_time_str}')
    ]
    ax.legend(handles=custom_legend, loc='upper left', fontsize=10, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    plot_file = os.path.join(plots_dir, f"pro_candlestick_{timeframe_label.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved pro candlestick plot to: {plot_file}")

# Run Pro Candlestick generator
timeframes = [
    ("nifty_1m_ist.csv", "1 Minute"),
    ("nifty_5m_ist.csv", "5 Minute"),
    ("nifty_15m_ist.csv", "15 Minute")
]

for fname, tf_label in timeframes:
    filepath = os.path.join(data_dir, fname)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        plot_pro_candlestick(df, tf_label)

print("\nPRO CANDLESTICK PLOTS WITH FULL IST DATE & TIME LEGENDS GENERATED SUCCESSFULLY!")
