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

data_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\data"
plots_dir_run3 = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\plots\run3_calm_zone_1030_1430"
os.makedirs(plots_dir_run3, exist_ok=True)

print("=" * 80)
print("RUN 3: EXECUTING CALM MARKET ZONE BACKTEST (10:30 AM - 02:30 PM IST)...")
print("Target Machine: CPU Execution Mode")
print("=" * 80)

device = "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

calm_start = time(10, 30, 0)
calm_end = time(14, 30, 0)

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

def run_calm_zone_backtest(file_name, tf_label, lookback=400, pred_len=15):
    filepath = os.path.join(data_dir, file_name)
    if not os.path.exists(filepath):
        print(f"File {filepath} missing. Skipping.")
        return None

    df = pd.read_csv(filepath)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df['time_only'] = df['timestamps'].dt.time

    total_bars = len(df)
    if total_bars < lookback + pred_len:
        lookback = min(200, int(total_bars * 0.6))
        pred_len = max(5, int(total_bars * 0.15))

    print(f"\n--- Backtesting {tf_label} in Calm Zone (10:30 AM - 02:30 PM IST) ---")

    results = []
    plot_samples = []

    start_idx = lookback
    step = pred_len

    for curr_idx in range(start_idx, total_bars - pred_len + 1, step):
        trigger_time = df.iloc[curr_idx - 1]['time_only']
        
        # STRICT FILTER: Trigger must occur between 10:30 AM IST and 02:30 PM IST
        if not (calm_start <= trigger_time <= calm_end):
            continue

        x_df = df.iloc[curr_idx - lookback : curr_idx][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_ts = df.iloc[curr_idx - lookback : curr_idx]['timestamps'].copy()
        y_ts = df.iloc[curr_idx : curr_idx + pred_len]['timestamps'].copy()
        act_df = df.iloc[curr_idx : curr_idx + pred_len][['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()

        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
        )

        start_p = x_df['close'].iloc[-1]
        act_end = act_df['close'].iloc[-1]
        pred_end = pred_df['close'].iloc[-1]

        act_move = act_end - start_p
        pred_move = pred_end - start_p

        act_dir = 1 if act_move > 0 else (-1 if act_move < 0 else 0)
        pred_dir = 1 if pred_move > 0 else (-1 if pred_move < 0 else 0)

        is_win = (act_dir == pred_dir)
        pnl_pts = act_move * pred_dir
        mae = abs(act_move - pred_move)

        results.append({
            'trigger_time': x_ts.iloc[-1].strftime('%Y-%m-%d %H:%M IST'),
            'start_price': start_p,
            'actual_end': act_end,
            'pred_end': pred_end,
            'actual_move': act_move,
            'pred_move': pred_move,
            'is_win': is_win,
            'pnl_pts': pnl_pts,
            'mae': mae
        })

        if len(plot_samples) < 2:
            plot_samples.append({
                'ctx_ts': x_ts, 'act_df': act_df, 'y_ts': y_ts,
                'pred_df': pred_df, 'is_win': is_win, 'trigger_time_str': x_ts.iloc[-1].strftime('%Y-%m-%d %H:%M IST')
            })

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        print(f"No calm zone windows found for {tf_label}.")
        return None

    win_rate = (res_df['is_win'].mean()) * 100
    total_pnl = res_df['pnl_pts'].sum()
    avg_pnl = res_df['pnl_pts'].mean()
    avg_mae = res_df['mae'].mean()

    # Generate OHLC Candlestick Plot
    if plot_samples:
        p = plot_samples[-1]
        fig, ax = plt.subplots(figsize=(14, 7))

        ctx_subset = df.iloc[max(0, curr_idx - lookback - 20) : curr_idx].copy()
        draw_ohlc_candlesticks(ax, ctx_subset.iloc[-30:], width_ratio=0.5, alpha=0.7)
        draw_ohlc_candlesticks(ax, p['act_df'], width_ratio=0.5, alpha=1.0)

        ax.plot(p['y_ts'], p['pred_df']['close'], label='Kronos AI Predicted Trajectory (Gold Dashed)', color='#f59e0b', linewidth=2.8, linestyle='--', marker='o', markersize=4)

        ax.axvline(x=mdates.date2num(p['ctx_ts'].iloc[-1]), color='#374151', linestyle=':', linewidth=2, label=f'Trigger IST: {p["trigger_time_str"]}')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()

        ax.set_title(f"Run 3 Calm Zone (10:30-14:30 IST) - Nifty 50 ({tf_label}) | Win Rate: {win_rate:.1f}%", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Date & Time (Indian Standard Time - IST)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)

        from matplotlib.lines import Line2D
        custom_legend = [
            Line2D([0], [0], color='#10b981', lw=4, label='Bullish Market Candle (Close >= Open)'),
            Line2D([0], [0], color='#ef4444', lw=4, label='Bearish Market Candle (Close < Open)'),
            Line2D([0], [0], color='#f59e0b', lw=2.8, linestyle='--', marker='o', label='Kronos AI Forecast Path'),
            Line2D([0], [0], color='#374151', lw=2, linestyle=':', label=f'Trigger IST: {p["trigger_time_str"]}')
        ]
        ax.legend(handles=custom_legend, loc='upper left', fontsize=10, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

        plot_file = os.path.join(plots_dir_run3, f"run3_calm_zone_{tf_label.lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved Run 3 Calm Zone plot to {plot_file}")

    return {
        'timeframe': tf_label,
        'windows': len(res_df),
        'win_rate_pct': win_rate,
        'total_pnl_pts': total_pnl,
        'avg_pnl_pts': avg_pnl,
        'mae_pts': avg_mae
    }

# Execute Run 3 Calm Zone Backtest across 1m, 5m, 15m IST datasets
timeframes = [
    ("nifty_1m_ist.csv", "1 Minute"),
    ("nifty_5m_ist.csv", "5 Minute"),
    ("nifty_15m_ist.csv", "15 Minute")
]

all_run3_summary = []
for fname, tf_label in timeframes:
    res = run_calm_zone_backtest(fname, tf_label)
    if res:
        all_run3_summary.append(res)

summary_df = pd.DataFrame(all_run3_summary)
print("\n" + "=" * 80)
print("RUN 3: CALM MARKET ZONE (10:30 AM - 02:30 PM IST) SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))

summary_csv = os.path.join(plots_dir_run3, "run3_calm_zone_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved Run 3 summary to {summary_csv}")
