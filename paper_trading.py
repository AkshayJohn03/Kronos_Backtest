import os
import sys
import pandas as pd
import time
import torch

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
kronos_src_path = os.path.join(base_dir, "Kronos_src")
if kronos_src_path not in sys.path:
    sys.path.append(kronos_src_path)
data_dir = os.path.join(base_dir, "data")

from model import KronosTokenizer, Kronos, KronosPredictor
from data_fetcher_ist import fetch_nifty_ist

print("=" * 80)
print("INITIALIZING KRONOS AI PAPER TRADING AUTOMATION SCRIPT")
print("Target Asset: Nifty 50 Spot (^NSEI)")
print("Note: Slippage is NOT considered in this simulation.")
print("=" * 80)

# 1. Load Model
device = "cpu"
print("Loading model...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
print("Model loaded successfully!")

# 2. Fetch Latest Real Data
print("\nFetching latest 1m data for simulation...")
df = fetch_nifty_ist(interval="1m", range_param="7d")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
if len(df) < lookback + 1:
    print("Not enough data to run simulation. Minimum required:", lookback + 1)
    sys.exit()

print(f"\nStarting Rolling Paper Trading Simulation on Last 10 Candles...")

# 3. Simulate Paper Trading
total_bars = len(df)
steps_to_roll = 10
start_idx = total_bars - steps_to_roll - 1

total_pnl = 0.0
trades_won = 0

for step in range(steps_to_roll):
    curr_idx = start_idx + step

    # Historical Context (Past 'lookback' bars up to curr_idx)
    x_df = df.iloc[curr_idx - lookback : curr_idx][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = df.iloc[curr_idx - lookback : curr_idx]['timestamps'].copy()

    # The true 'next' bar (the one we are predicting)
    target_ts = df.iloc[curr_idx : curr_idx + 1]['timestamps'].copy()
    act_bar = df.iloc[curr_idx]

    start_p = x_df['close'].iloc[-1]

    # Kronos Predicts
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=target_ts,
        pred_len=1, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )

    pred_p = pred_df['close'].iloc[0]
    pred_move = pred_p - start_p
    pred_dir = 1 if pred_move > 0 else (-1 if pred_move < 0 else 0)

    action = "BUY CALL" if pred_dir == 1 else ("BUY PUT" if pred_dir == -1 else "HOLD")

    # Ground Truth Actual
    act_p = act_bar['close']
    act_move = act_p - start_p

    # Trade Outcome
    pnl = act_move * pred_dir
    is_win = pnl > 0
    total_pnl += pnl
    if is_win: trades_won += 1

    print("-" * 50)
    print(f"Time (IST): {target_ts.iloc[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"Context Close: {start_p:.2f}")
    print(f"Kronos Prediction: {pred_p:.2f} (Expected Move: {pred_move:+.2f} pts) -> Action: {action}")
    print(f"Actual Close: {act_p:.2f} (Actual Move: {act_move:+.2f} pts)")
    print(f"Trade Outcome: {'WIN' if is_win else 'LOSS'} | PnL: {pnl:+.2f} pts")
    time.sleep(1) # Simulate real-time delay

print("=" * 50)
print("PAPER TRADING SIMULATION COMPLETE")
print(f"Total Trades: {steps_to_roll}")
print(f"Win Rate: {(trades_won/steps_to_roll)*100:.1f}%")
print(f"Cumulative PnL: {total_pnl:+.2f} pts (Slippage Not Considered)")
print("=" * 50)
