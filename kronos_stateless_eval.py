import os
import sys
import pandas as pd
import numpy as np
import torch

# Add Kronos_src to path
kronos_src_path = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\Kronos_src"
if kronos_src_path not in sys.path:
    sys.path.append(kronos_src_path)

from model import KronosTokenizer, Kronos, KronosPredictor

data_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\data"
report_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"

print("=" * 80)
print("INITIALIZING STATELESS CROSS-VALIDATION ENGINE (ZERO LEAKAGE)...")
print("Target Environment: CPU Execution Mode | Device: cpu")
print("=" * 80)

device = "cpu"

def get_fresh_predictor():
    """Stateless Factory: Re-instantiates model & predictor for 100% memory clearing."""
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=512)

def evaluate_timeframe_stateless(file_name, tf_label, target_dates, lookback=400, pred_len=15):
    filepath = os.path.join(data_dir, file_name)
    if not os.path.exists(filepath):
        print(f"File {filepath} missing. Skipping.")
        return None

    df = pd.read_csv(filepath)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # Filter for specific distinct target dates
    df['date_str'] = df['timestamps'].dt.strftime('%Y-%m-%d')
    if target_dates:
        df = df[df['date_str'].isin(target_dates)].copy().reset_index(drop=True)

    total_bars = len(df)
    if total_bars < lookback + pred_len:
        lookback = min(200, int(total_bars * 0.6))
        pred_len = max(5, int(total_bars * 0.15))

    print(f"\n--- Evaluating {tf_label} Statelessly ---")
    print(f"Dates Evaluated: {df['date_str'].unique()} | Total Bars: {total_bars}")

    results = []
    step = pred_len
    eval_count = (total_bars - lookback - pred_len) // step + 1
    eval_count = max(1, min(25, eval_count))

    for i in range(eval_count):
        curr_idx = lookback + i * step
        if curr_idx + pred_len > total_bars:
            break

        # Extract context and ground truth
        x_df = df.iloc[curr_idx - lookback : curr_idx][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_ts = df.iloc[curr_idx - lookback : curr_idx]['timestamps'].copy()
        y_ts = df.iloc[curr_idx : curr_idx + pred_len]['timestamps'].copy()
        act_df = df.iloc[curr_idx : curr_idx + pred_len][['open', 'high', 'low', 'close', 'volume']].copy()

        # STATELESS INFERENCE: Use fresh predictor instance for zero memory state leakage
        predictor = get_fresh_predictor()

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
            'start_time': x_ts.iloc[-1].strftime('%Y-%m-%d %H:%M IST'),
            'end_time': y_ts.iloc[-1].strftime('%Y-%m-%d %H:%M IST'),
            'start_price': start_p,
            'actual_end': act_end,
            'pred_end': pred_end,
            'actual_move_pts': act_move,
            'pred_move_pts': pred_move,
            'is_win': is_win,
            'pnl_pts': pnl_pts,
            'mae': mae
        })

    res_df = pd.DataFrame(results)
    win_rate = (res_df['is_win'].mean()) * 100 if len(res_df) > 0 else 0
    total_pnl = res_df['pnl_pts'].sum() if len(res_df) > 0 else 0
    avg_pnl = res_df['pnl_pts'].mean() if len(res_df) > 0 else 0
    avg_mae = res_df['mae'].mean() if len(res_df) > 0 else 0

    # Filter for strong moves >= 15 pts
    strong_moves = res_df[res_df['actual_move_pts'].abs() >= 15]
    strong_win_rate = (strong_moves['is_win'].mean()) * 100 if len(strong_moves) > 0 else win_rate

    print(f"[{tf_label}] Stateless Win Rate: {win_rate:.1f}% | Strong Move (>=15pt) Win Rate: {strong_win_rate:.1f}% | Total PnL: {total_pnl:+.2f} pts")

    return {
        'timeframe': tf_label,
        'dates': ", ".join(df['date_str'].unique()),
        'windows': len(res_df),
        'win_rate_pct': win_rate,
        'strong_win_rate_pct': strong_win_rate,
        'total_pnl_pts': total_pnl,
        'avg_pnl_pts': avg_pnl,
        'mae_pts': avg_mae
    }

# Run Stateless Cross-Validation on Distinct Trading Dates
evaluations = [
    ("nifty_1m_ist.csv", "1 Minute", ['2026-07-24']),
    ("nifty_5m_ist.csv", "5 Minute", ['2026-07-22']),
    ("nifty_15m_ist.csv", "15 Minute", ['2026-07-16', '2026-07-17', '2026-07-20'])
]

metrics_summary = []
for fname, tf_label, dates in evaluations:
    m = evaluate_timeframe_stateless(fname, tf_label, dates)
    if m:
        metrics_summary.append(m)

summary_df = pd.DataFrame(metrics_summary)
print("\n" + "=" * 80)
print("STATELESS CROSS-VALIDATION SUMMARY TABLE (DISTINCT IST DATES)")
print("=" * 80)
print(summary_df.to_string(index=False))

# Save Summary to File
summary_path = os.path.join(report_dir, "stateless_validation_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved stateless validation summary to {summary_path}")
