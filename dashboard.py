import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Kronos AI Nifty50 IST Dashboard", layout="wide", page_icon="📈")

st.title("📈 Kronos AI Nifty50 Option Trading Dashboard")
st.markdown("### Real-Time Out-of-Sample Backtesting & Directional Win Rate Analytics (IST)")

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_mode1 = os.path.join(base_dir, "plots", "mode1_static_horizon")
plots_mode2 = os.path.join(base_dir, "plots", "mode2_rolling_feed")
plots_exact = os.path.join(base_dir, "plots", "exact_kronos_repo_viz")

# Sidebar Selection
st.sidebar.header("🕹️ Backtest Controls")
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1 Minute", "5 Minute", "15 Minute"])
mode = st.sidebar.radio("Select Prediction Mode / Chart View:", [
    "Mode 1: Static Multi-Step Horizon (15 Candles Ahead)",
    "Mode 2: Sequential Rolling One-Step Feed (Candle-by-Candle)",
    "Official Kronos Repo Subplot View (Price & Volume - prediction_example.py)"
])

# Performance Summary Matrix
metrics_data = {
    "1 Minute": {
        "Mode 1": {"win_rate": 73.3, "pnl": 124.10, "avg_pnl": 8.27, "mae": 25.83},
        "Mode 2": {"win_rate": 45.0, "pnl": 2.25, "avg_pnl": 0.11, "mae": 8.67}
    },
    "5 Minute": {
        "Mode 1": {"win_rate": 46.7, "pnl": -217.75, "avg_pnl": -14.52, "mae": 70.12},
        "Mode 2": {"win_rate": 45.0, "pnl": 49.45, "avg_pnl": 2.47, "mae": 12.09}
    },
    "15 Minute": {
        "Mode 1": {"win_rate": 53.3, "pnl": 262.20, "avg_pnl": 17.48, "mae": 125.73},
        "Mode 2": {"win_rate": 60.0, "pnl": 60.60, "avg_pnl": 3.03, "mae": 24.67}
    }
}

mode_key = "Mode 1" if "Mode 1" in mode else ("Mode 2" if "Mode 2" in mode else "Repo View")
m_key = "Mode 1" if mode_key in ["Mode 1", "Repo View"] else "Mode 2"
m = metrics_data[timeframe][m_key]

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Directional Win Rate", f"{m['win_rate']:.1f}%", delta=f"{m['win_rate'] - 50.0:.1f}% vs Random")
col2.metric("Cumulative PnL (Pts)", f"{m['pnl']:+.2f} pts")
col3.metric("Avg PnL / Trade", f"{m['avg_pnl']:+.2f} pts")
col4.metric("Mean Absolute Error (MAE)", f"{m['mae']:.2f} pts")
st.markdown("---")

# Visual Forecast Image Display
st.subheader(f"🖼️ Kronos AI Visual Forecast ({mode_key})")
tf_file_key = timeframe.lower().replace(" ", "_")

if mode_key == "Mode 1":
    img_path = os.path.join(plots_mode1, f"mode1_{tf_file_key}_static_horizon.png")
elif mode_key == "Mode 2":
    img_path = os.path.join(plots_mode2, f"mode2_{tf_file_key}_rolling_feed.png")
else:
    img_path = os.path.join(plots_exact, f"exact_kronos_viz_{tf_file_key}.png")

if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img, use_column_width=True, caption=f"Kronos AI Forecast ({timeframe} - {mode_key})")
else:
    st.warning(f"Plot image not found at {img_path}")

# Load and Display Data Preview
st.subheader(f"📊 IST Market Data Sample ({timeframe})")
data_filename = f"nifty_{tf_file_key}_ist.csv"
csv_path = os.path.join(data_dir, data_filename)

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df.tail(20), use_container_width=True)
    st.caption("Timestamps formatted in Indian Standard Time (IST: 09:15 AM - 03:30 PM).")

st.markdown("---")
st.info("💡 **Zerodha Manual Scalping Tip:** Use Mode 2 (15-min, 60% win rate) or Mode 1 (1-min, 73.3% win rate) to filter trade direction before placing Zerodha ATM option orders.")
