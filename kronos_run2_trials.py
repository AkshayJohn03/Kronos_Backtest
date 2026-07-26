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
base_plots_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\plots\run2_trial_multiday"
os.makedirs(base_plots_dir, exist_ok=True)

print("=" * 80)
print("RUN 2: EXECUTING 3-DAY MULTI-TRIAL BACKTEST (UNSEEN IST DATES)...")
print("Target Machine: CPU Execution Mode")
print("=" * 80)

device = "cpu"

def get_fresh_predictor():
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=512)

def draw_ohlc_candlesticks(ax, df, width_ratio=0.5, alpha=0.9):
    """Draws clear green (bullish) and red (bearish) OHLC candlestick bars."""
    dates = mdates.date2num(df['timestamps'])
    width = (dates[1] - dates[0]) * width_ratio if len(dates) > 1 else 0.0005

    for i in range(len(df)):
        open_p = df['open'].iloc[i]
        high_p = df['high'].iloc[i]
        low_p = df['low'].iloc[i]
        close_p = df['close'].iloc[i]
        date_val = dates[i]

        color = '#10b981' if close_p >= open_p else '#ef4444'

        # Wick line
        ax.plot([date_val, date_val], [low_p, high_p], color=color, linewidth=1.2, alpha=alpha)

        # Body rectangle
        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), 0.25)
        rect = Rectangle((date_val - width/2, body_bottom), width, body_height, facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)

def run_single_day_trial(df_all, tf_label, target_date, trial_num, lookback=200, pred_len=15):
    tf_folder = tf_label.lower().replace(" ", "")
    save_dir = os.path.join(base_plots_dir, tf_folder)
    os.makedirs(save_dir, exist_ok=True)

    # Pre-load context prior to target_date
    df_all['date_str'] = df_all['timestamps'].dt.strftime('%Y-%m-%d')
    date_indices = df_all[df_all['date_str'] == target_date].index
    
    if len(date_indices) == 0:
        print(f"Target date {target_date} not found in {tf_label} dataset.")
        return None

    first_date_idx = date_indices[0]
    last_date_idx = date_indices[-1]

    # Ensure lookback context is available
    if first_date_idx < lookback:
        start_idx = lookback
    else:
        start_idx = first_date_idx

    print(f"[{tf_label} | Trial {trial_num}] Evaluating Date: {target_date} | Day Bars: {len(date_indices)}")

    predictor = get_fresh_predictor()
    results = []
    
    step = pred_len
    curr = start_idx
    eval_window_count = 0

    plot_data = None

    while curr + pred_len <= last_date_idx + 1:
        x_df = df_all.iloc[curr - lookback : curr][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_ts = df_all.iloc[curr - lookback : curr]['timestamps'].copy()
        y_ts = df_all.iloc[curr : curr + pred_len]['timestamps'].copy()
        act_df = df_all.iloc[curr : curr + pred_len][['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()

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
            'trial_num': trial_num,
            'date': target_date,
            'start_time': x_ts.iloc[-1].strftime('%H:%M IST'),
            'end_time': y_ts.iloc[-1].strftime('%H:%M IST'),
            'start_price': start_p,
            'actual_end': act_end,
            'pred_end': pred_end,
            'actual_move': act_move,
            'pred_move': pred_move,
            'is_win': is_win,
            'pnl_pts': pnl_pts,
            'mae': mae
        })

        if plot_data is None or eval_window_count == 1:
            plot_data = {
                'x_df': x_df, 'x_ts': x_ts, 'y_ts': y_ts,
                'act_df': act_df, 'pred_df': pred_df,
                'is_win': is_win, 'target_date': target_date
            }

        curr += step
        eval_window_count += 1

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        return None

    win_rate = (res_df['is_win'].mean()) * 100
    total_pnl = res_df['pnl_pts'].sum()
    avg_mae = res_df['mae'].mean()

    # Generate Professional OHLC Candlestick Plot
    if plot_data:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ctx_subset = df_all.iloc[max(0, curr - lookback - 20) : curr].copy()
        draw_ohlc_candlesticks(ax, ctx_subset.iloc[-30:], width_ratio=0.5, alpha=0.7)
        draw_ohlc_candlesticks(ax, plot_data['act_df'], width_ratio=0.5, alpha=1.0)

        ax.plot(plot_data['y_ts'], plot_data['pred_df']['close'], label='Kronos AI Predicted Trajectory (Gold Dashed)', color='#f59e0b', linewidth=2.8, linestyle='--', marker='o', markersize=4)

        trigger_time_str = plot_data['x_ts'].iloc[-1].strftime('%Y-%m-%d %H:%M IST')
        ax.axvline(x=mdates.date2num(plot_data['x_ts'].iloc[-1]), color='#374151', linestyle=':', linewidth=2, label=f'Trigger IST: {trigger_time_str}')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()

        ax.set_title(f"Run 2 Trial {trial_num} - Nifty 50 ({tf_label}) Date: {target_date} | Day Win Rate: {win_rate:.1f}%", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Date & Time (Indian Standard Time - IST)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)

        from matplotlib.lines import Line2D
        custom_legend = [
            Line2D([0], [0], color='#10b981', lw=4, label='Bullish Market Candle (Close >= Open)'),
            Line2D([0], [0], color='#ef4444', lw=4, label='Bearish Market Candle (Close < Open)'),
            Line2D([0], [0], color='#f59e0b', lw=2.8, linestyle='--', marker='o', label='Kronos AI Forecast Path'),
            Line2D([0], [0], color='#374151', lw=2, linestyle=':', label=f'Trigger IST: {trigger_time_str}')
        ]
        ax.legend(handles=custom_legend, loc='upper left', fontsize=10, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

        plot_file = os.path.join(save_dir, f"run2_trial{trial_num}_{target_date}.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved Run 2 plot: {plot_file}")

    return {
        'timeframe': tf_label,
        'trial_num': trial_num,
        'date': target_date,
        'windows': len(res_df),
        'day_win_rate_pct': win_rate,
        'day_pnl_pts': total_pnl,
        'day_mae_pts': avg_mae,
        'results_df': res_df
    }


# Define 3 Unseen Trial Dates per Timeframe
trial_schedule = [
    ("nifty_1m_ist.csv", "1 Minute", ['2026-07-22', '2026-07-23', '2026-07-24']),
    ("nifty_5m_ist.csv", "5 Minute", ['2026-07-17', '2026-07-20', '2026-07-21']),
    ("nifty_15m_ist.csv", "15 Minute", ['2026-07-14', '2026-07-15', '2026-07-16'])
]

all_trial_summary = []
overall_timeframe_summary = []

for fname, tf_label, trial_dates in trial_schedule:
    filepath = os.path.join(data_dir, fname)
    if not os.path.exists(filepath):
        continue

    df_all = pd.read_csv(filepath)
    df_all['timestamps'] = pd.to_datetime(df_all['timestamps'])

    tf_wins = 0
    tf_windows = 0
    tf_pnl = 0

    for i, tdate in enumerate(trial_dates, 1):
        res = run_single_day_trial(df_all, tf_label, tdate, trial_num=i)
        if res:
            all_trial_summary.append({
                'timeframe': tf_label,
                'trial': f"Trial {i}",
                'date': res['date'],
                'windows': res['windows'],
                'win_rate_pct': res['day_win_rate_pct'],
                'pnl_pts': res['day_pnl_pts'],
                'mae_pts': res['day_mae_pts']
            })
            
            tf_wins += res['results_df']['is_win'].sum()
            tf_windows += res['windows']
            tf_pnl += res['day_pnl_pts']

    overall_win_rate = (tf_wins / tf_windows * 100) if tf_windows > 0 else 0
    overall_timeframe_summary.append({
        'timeframe': tf_label,
        'total_trials': len(trial_dates),
        'total_windows': tf_windows,
        'overall_win_rate_pct': overall_win_rate,
        'overall_pnl_pts': tf_pnl
    })

# Save Summaries
trials_df = pd.DataFrame(all_trial_summary)
overall_df = pd.DataFrame(overall_timeframe_summary)

print("\n" + "=" * 80)
print("RUN 2: INDIVIDUAL TRIAL DAYS SUMMARY TABLE")
print("=" * 80)
print(trials_df.to_string(index=False))

print("\n" + "=" * 80)
print("RUN 2: OVERALL TIMEFRAME SUCCESS RATE SUMMARY TABLE")
print("=" * 80)
print(overall_df.to_string(index=False))

trials_df.to_csv(os.path.join(base_plots_dir, "run2_individual_trials_summary.csv"), index=False)
overall_df.to_csv(os.path.join(base_plots_dir, "run2_overall_summary.csv"), index=False)
print("\nRUN 2 MULTI-DAY TRIAL EVALUATION COMPLETED SUCCESSFULLY!")
