"""
KRONOS LIVE PAPER TRADER — July 31, 2026
=========================================
Fetches live Nifty50 data from 1:10 PM IST onwards.
Uses Kronos AI to predict next 15 minutes + rolling forecast to 3:30 PM close.
Generates CE/PE paper trade signals with entry/exit tracking.
Outputs charts & rolling P&L ledger.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, time, timedelta
import pytz
import json
import urllib.request
import torch

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── SETUP ──────────────────────────────────────────────────────────────────
kronos_src_path = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\Kronos_src"
if kronos_src_path not in sys.path:
    sys.path.append(kronos_src_path)

from model import KronosTokenizer, Kronos, KronosPredictor

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_dir = os.path.join(base_dir, "plots", "live_paper_trade_20260731")
os.makedirs(plots_dir, exist_ok=True)

IST = pytz.timezone("Asia/Kolkata")
NOW_IST = datetime.now(IST)
TODAY_STR = NOW_IST.strftime("%Y-%m-%d")

print("=" * 90)
print(f"  KRONOS LIVE PAPER TRADER — {TODAY_STR}")
print(f"  Current IST Time: {NOW_IST.strftime('%H:%M:%S IST')}")
print(f"  Mode: Paper Trade (CE/PE Signal) + 15-min Prediction + Rolling to Close")
print("=" * 90)

# ─── FETCH LIVE DATA ────────────────────────────────────────────────────────
def fetch_yahoo_direct(symbol="^NSEI", interval="1m", range_str="5d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_str}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
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

            df['timestamps'] = df['timestamps'].dt.tz_convert(IST).dt.tz_localize(None)
            df['time_only'] = df['timestamps'].dt.time
            start_t = time(9, 15)
            end_t = time(15, 30)
            df = df[(df['time_only'] >= start_t) & (df['time_only'] <= end_t)].copy()
            df['amount'] = df['close'] * df['volume']
            return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].sort_values('timestamps').reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] Direct API call for {interval} failed: {e}")
        return None


print("\n[1/4] Fetching live Nifty50 data (1m, 5m, 15m intervals)...")

df_1m = fetch_yahoo_direct(interval="1m", range_str="5d")
df_5m = fetch_yahoo_direct(interval="5m", range_str="10d")
df_15m = fetch_yahoo_direct(interval="15m", range_str="30d")

for label, df in [("1m", df_1m), ("5m", df_5m), ("15m", df_15m)]:
    if df is not None and len(df) > 0:
        last_ts = df['timestamps'].iloc[-1]
        last_close = df['close'].iloc[-1]
        print(f"  {label}: {len(df)} candles, latest={last_ts.strftime('%Y-%m-%d %H:%M')} IST, close={last_close:.2f}")
    else:
        print(f"  {label}: FAILED — falling back to cached CSV")

# Fallback to cached data
if df_1m is None or len(df_1m) == 0:
    df_1m = pd.read_csv(os.path.join(data_dir, "nifty_1m_ist.csv"))
    df_1m['timestamps'] = pd.to_datetime(df_1m['timestamps'])
if df_5m is None or len(df_5m) == 0:
    df_5m = pd.read_csv(os.path.join(data_dir, "nifty_5m_ist.csv"))
    df_5m['timestamps'] = pd.to_datetime(df_5m['timestamps'])
if df_15m is None or len(df_15m) == 0:
    df_15m = pd.read_csv(os.path.join(data_dir, "nifty_15m_ist.csv"))
    df_15m['timestamps'] = pd.to_datetime(df_15m['timestamps'])


# ─── INIT KRONOS MODEL ─────────────────────────────────────────────────────
print("\n[2/4] Initializing Kronos AI Model (CPU Mode)...")
device = "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
model.eval()
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
print("  Kronos AI ready.")


# ─── HELPER: OHLC CANDLESTICKS ──────────────────────────────────────────────
def draw_ohlc_candlesticks(ax, df, width_ratio=0.5, alpha=0.9):
    dates = mdates.date2num(df['timestamps'])
    width = (dates[1] - dates[0]) * width_ratio if len(dates) > 1 else 0.0005
    for i in range(len(df)):
        o, h, l, c = df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
        d = dates[i]
        color = '#10b981' if c >= o else '#ef4444'
        ax.plot([d, d], [l, h], color=color, linewidth=1.2, alpha=alpha)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), 0.25)
        rect = Rectangle((d - width/2, body_bottom), width, body_height, facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)


# ─── HELPER: SELECT ATM STRIKE ──────────────────────────────────────────────
def get_atm_strike(price, step=50):
    """Round to nearest Nifty option strike (multiples of 50)."""
    return int(round(price / step) * step)


# ─── CORE: RUN KRONOS PREDICTION + PAPER TRADE ──────────────────────────────
def run_kronos_paper_trade(df, tf_label, pred_len=15, lookback=350):
    if df is None or len(df) < 50:
        print(f"  [SKIP] Not enough data for {tf_label}")
        return None

    # Compute SMAs
    df = df.copy()
    df['sma50'] = df['close'].rolling(window=50, min_periods=10).mean()
    df['sma100'] = df['close'].rolling(window=100, min_periods=20).mean()
    df['sma200'] = df['close'].rolling(window=200, min_periods=30).mean()

    if len(df) < lookback:
        lookback = len(df) - pred_len

    # Context (everything up to now)
    x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = df.iloc[-lookback:]['timestamps'].copy()

    last_ts = x_ts.iloc[-1]
    current_price = x_df['close'].iloc[-1]

    # Generate future timestamps
    freq = "1min" if "1 Min" in tf_label else ("5min" if "5 Min" in tf_label else "15min")
    y_ts = pd.Series(pd.date_range(start=last_ts + pd.Timedelta(freq), periods=pred_len, freq=freq))

    # Run prediction
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )

    # Next 15-min forecast
    next_15m_idx = min(len(pred_df) - 1, 15 if "1 Min" in tf_label else (3 if "5 Min" in tf_label else 1))
    next_15m_price = pred_df['close'].iloc[next_15m_idx]
    next_15m_move = next_15m_price - current_price

    # End-of-session forecast
    predicted_close = pred_df['close'].iloc[-1]
    predicted_move = predicted_close - current_price
    pct_move = (predicted_move / current_price) * 100

    # Signal & Paper Trade
    atm_strike = get_atm_strike(current_price)

    if predicted_move > 5:
        signal = "BUY CALL (BULLISH)"
        option_type = "CE"
        option_strike = atm_strike
    elif predicted_move < -5:
        signal = "BUY PUT (BEARISH)"
        option_type = "PE"
        option_strike = atm_strike
    else:
        signal = "NEUTRAL / NO TRADE"
        option_type = "—"
        option_strike = atm_strike

    # Estimate option premium (rough delta-based estimate)
    # ATM options have ~0.5 delta, so move in option ≈ 50% of spot move
    est_premium_entry = max(abs(current_price - atm_strike) * 0.5 + 80, 50)
    est_premium_change_15m = abs(next_15m_move) * 0.5
    est_premium_change_close = abs(predicted_move) * 0.5
    est_pnl_15m = est_premium_change_15m if (next_15m_move > 0 and option_type == "CE") or (next_15m_move < 0 and option_type == "PE") else -est_premium_change_15m
    est_pnl_close = est_premium_change_close if (predicted_move > 0 and option_type == "CE") or (predicted_move < 0 and option_type == "PE") else -est_premium_change_close

    trigger_str = last_ts.strftime('%Y-%m-%d %H:%M IST')

    print(f"\n{'=' * 90}")
    print(f"  KRONOS PAPER TRADE — {tf_label}")
    print(f"  Trigger Time       : {trigger_str}")
    print(f"  Current Nifty Price : {current_price:.2f} INR")
    print(f"  ATM Strike          : {atm_strike}")
    print(f"  ─── Next 15-Minute Forecast ───")
    print(f"  Predicted Price     : {next_15m_price:.2f} INR ({next_15m_move:+.2f} pts)")
    print(f"  ─── End-of-Day Forecast (3:30 PM) ───")
    print(f"  Predicted Close     : {predicted_close:.2f} INR ({predicted_move:+.2f} pts / {pct_move:+.2f}%)")
    print(f"  ─── Paper Trade Signal ───")
    print(f"  Signal              : {signal}")
    if option_type != "—":
        print(f"  Paper Trade         : BUY 1 LOT Nifty {option_strike} {option_type} (Weekly Expiry)")
        print(f"  Est. Entry Premium  : ~₹{est_premium_entry:.0f}")
        print(f"  Est. P&L (15 min)   : {est_pnl_15m:+.1f} pts/lot")
        print(f"  Est. P&L (to close) : {est_pnl_close:+.1f} pts/lot")
    print(f"{'=' * 90}")

    # ─── CHART ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))

    ctx_subset = df.iloc[-45:].copy()
    draw_ohlc_candlesticks(ax, ctx_subset, width_ratio=0.5, alpha=0.85)

    # SMAs
    ax.plot(ctx_subset['timestamps'], ctx_subset['sma50'], color='#06b6d4', linewidth=1.8, linestyle='-', label='50 SMA')
    ax.plot(ctx_subset['timestamps'], ctx_subset['sma100'], color='#8b5cf6', linewidth=1.8, linestyle='-', label='100 SMA')
    ax.plot(ctx_subset['timestamps'], ctx_subset['sma200'], color='#eab308', linewidth=2.0, linestyle='-', label='200 SMA')

    # Kronos Forecast Path
    ax.plot(y_ts, pred_df['close'], color='#f59e0b', linewidth=3.0, linestyle='--', marker='o', markersize=5, label=f'Kronos Predicted Path ({predicted_move:+.2f} pts)')

    # 15-min marker
    if next_15m_idx < len(y_ts):
        ax.axvline(x=mdates.date2num(y_ts.iloc[next_15m_idx]), color='#3b82f6', linestyle=':', linewidth=1.5, alpha=0.7, label=f'+15 Min Mark ({next_15m_price:.2f})')
        ax.scatter([mdates.date2num(y_ts.iloc[next_15m_idx])], [next_15m_price], color='#3b82f6', s=100, zorder=10)

    # Trigger line
    ax.axvline(x=mdates.date2num(last_ts), color='#dc2626', linestyle=':', linewidth=2, label=f'NOW ({trigger_str})')

    # ATM strike line
    ax.axhline(y=atm_strike, color='#a855f7', linestyle='-.', linewidth=1.2, alpha=0.6, label=f'ATM Strike: {atm_strike}')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    title_line1 = f"KRONOS PAPER TRADER | Nifty 50 ({tf_label}) — {TODAY_STR}"
    title_line2 = f"Now: {current_price:.2f} → +15m: {next_15m_price:.2f} ({next_15m_move:+.2f}) → Close: {predicted_close:.2f} ({predicted_move:+.2f})"
    title_line3 = f"Signal: {signal} | Paper: Nifty {option_strike} {option_type}"
    ax.set_title(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=12, fontweight='bold', pad=14)
    ax.set_xlabel("Time (IST)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nifty 50 Spot Price (INR)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    custom_legend = [
        Line2D([0], [0], color='#10b981', lw=4, label='Bullish Candle'),
        Line2D([0], [0], color='#ef4444', lw=4, label='Bearish Candle'),
        Line2D([0], [0], color='#06b6d4', lw=1.8, label='50 SMA'),
        Line2D([0], [0], color='#8b5cf6', lw=1.8, label='100 SMA'),
        Line2D([0], [0], color='#eab308', lw=2.0, label='200 SMA'),
        Line2D([0], [0], color='#f59e0b', lw=3.0, linestyle='--', marker='o', label=f'Kronos ({predicted_move:+.2f} pts)'),
        Line2D([0], [0], color='#dc2626', lw=2, linestyle=':', label=f'Trigger: {trigger_str}'),
        Line2D([0], [0], color='#3b82f6', lw=1.5, linestyle=':', label=f'+15m: {next_15m_price:.2f}'),
        Line2D([0], [0], color='#a855f7', lw=1.2, linestyle='-.', label=f'ATM: {atm_strike}')
    ]
    ax.legend(handles=custom_legend, loc='upper left', fontsize=9, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    plot_file = os.path.join(plots_dir, f"paper_trade_{tf_label.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  Chart saved: {plot_file}")

    return {
        'timeframe': tf_label,
        'trigger_time': trigger_str,
        'current_price': round(current_price, 2),
        'atm_strike': atm_strike,
        'next_15m_price': round(next_15m_price, 2),
        'next_15m_move_pts': round(next_15m_move, 2),
        'predicted_close': round(predicted_close, 2),
        'predicted_move_pts': round(predicted_move, 2),
        'pct_move': round(pct_move, 3),
        'signal': signal,
        'option_type': option_type,
        'option_strike': option_strike,
        'est_entry_premium': round(est_premium_entry, 1),
        'est_pnl_15m': round(est_pnl_15m, 1),
        'est_pnl_close': round(est_pnl_close, 1),
        'plot_file': plot_file
    }


# ─── EXECUTE ────────────────────────────────────────────────────────────────
print("\n[3/4] Running Kronos predictions across 3 timeframes...")

res_5m = run_kronos_paper_trade(df_5m, "5 Minute", pred_len=25, lookback=300)
res_1m = run_kronos_paper_trade(df_1m, "1 Minute", pred_len=75, lookback=300)
res_15m = run_kronos_paper_trade(df_15m, "15 Minute", pred_len=10, lookback=300)

# ─── SUMMARY ────────────────────────────────────────────────────────────────
print("\n[4/4] Generating Paper Trade Summary...")

all_results = [r for r in [res_5m, res_1m, res_15m] if r is not None]

if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(plots_dir, "paper_trade_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 90)
    print("  KRONOS PAPER TRADE SUMMARY — JULY 31, 2026")
    print("=" * 90)
    print(f"{'Timeframe':<12} {'Now':>10} {'→ +15m':>10} {'Move':>8} {'→ Close':>10} {'Move':>8} {'Signal':<22} {'Option':<18} {'Est P&L(Close)':>15}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['timeframe']:<12} {r['current_price']:>10.2f} {r['next_15m_price']:>10.2f} {r['next_15m_move_pts']:>+8.2f} {r['predicted_close']:>10.2f} {r['predicted_move_pts']:>+8.2f} {r['signal']:<22} {r['option_strike']} {r['option_type']:<12} {r['est_pnl_close']:>+15.1f}")
    print("=" * 90)

    # Consensus signal
    signals = [r['signal'] for r in all_results if r['signal'] != "NEUTRAL / NO TRADE"]
    if signals:
        bullish = sum(1 for s in signals if "CALL" in s)
        bearish = sum(1 for s in signals if "PUT" in s)
        if bullish > bearish:
            consensus = f"BULLISH CONSENSUS ({bullish}/{len(signals)} models agree) → BUY ATM CE"
        elif bearish > bullish:
            consensus = f"BEARISH CONSENSUS ({bearish}/{len(signals)} models agree) → BUY ATM PE"
        else:
            consensus = "SPLIT SIGNAL — WAIT FOR CONFIRMATION"
    else:
        consensus = "ALL NEUTRAL — NO TRADE"

    print(f"\n  🎯 CONSENSUS: {consensus}")
    print(f"  📊 Summary CSV: {summary_csv}")
    print(f"  📈 Charts saved to: {plots_dir}")

print("\n✅ KRONOS PAPER TRADE SESSION COMPLETE!")
