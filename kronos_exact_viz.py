import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add Kronos_src to path
kronos_src_path = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\Kronos_src"
if kronos_src_path not in sys.path:
    sys.path.append(kronos_src_path)

from model import KronosTokenizer, Kronos, KronosPredictor

data_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\data"
plots_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos\plots\exact_kronos_repo_viz"
os.makedirs(plots_dir, exist_ok=True)

print("=" * 80)
print("RUNNING EXACT KRONOS REPO VISUALIZATION (examples/prediction_example.py)...")
print("=" * 80)

device = "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

def plot_prediction_exact(kline_df, pred_df, tf_label):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Plot Close Price
    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.8)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.8, linestyle='--')
    ax1.set_title(f"Kronos Official Repo Visualization - Nifty 50 ({tf_label} IST)", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Close Price (INR)', fontsize=12)
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Volume
    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.8)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.8, linestyle='--')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"exact_kronos_viz_{tf_label.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved exact repo visualization chart to: {plot_path}")

# Run exact visualization for 1m, 5m, 15m IST datasets
timeframes = [
    ("nifty_1m_ist.csv", "1 Minute", 400, 30),
    ("nifty_5m_ist.csv", "5 Minute", 400, 30),
    ("nifty_15m_ist.csv", "15 Minute", 400, 30)
]

for fname, tf_label, lookback, pred_len in timeframes:
    filepath = os.path.join(data_dir, fname)
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    if len(df) < lookback + pred_len:
        lookback = int(len(df) * 0.7)
        pred_len = int(len(df) * 0.2)
        
    x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_ts = df.iloc[:lookback]['timestamps']
    y_ts = df.iloc[lookback : lookback + pred_len]['timestamps']
    
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )
    
    kline_df = df.iloc[:lookback + pred_len].copy()
    plot_prediction_exact(kline_df, pred_df, tf_label)

print("\nEXACT REPO VISUALIZATIONS GENERATED SUCCESSFULLY!")
