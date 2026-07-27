import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Kronos AI Nifty50 IST Dashboard", layout="wide", page_icon="📈")

st.title("📈 Kronos AI Nifty50 Option Trading Dashboard")
st.markdown("### Audited Out-of-Sample Backtesting & Real-Time Live Session Analytics (IST)")

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_today = os.path.join(base_dir, "plots", "today_live_forecast")
plots_run3 = os.path.join(base_dir, "plots", "run3_calm_zone_1030_1430")
plots_run2 = os.path.join(base_dir, "plots", "run2_trial_multiday")
plots_run1_mode1 = os.path.join(base_dir, "plots", "mode1_static_horizon")
plots_run1_mode2 = os.path.join(base_dir, "plots", "mode2_rolling_feed")
plots_run1_exact = os.path.join(base_dir, "plots", "exact_kronos_repo_viz")
plots_run1_pro = os.path.join(base_dir, "plots", "pro_candlesticks")

# Sidebar Controls & Version Segregation
st.sidebar.header("📂 Version & Run Segregation")
run_version = st.sidebar.radio("Select Backtest Version / Live Run:", [
    "🔥 Today's Live Real-Time Session Forecast (July 27, 2026)",
    "Run 3: Calm Market Zone (10:30 AM - 02:30 PM IST)",
    "Run 2: 3-Day Multi-Trial Out-of-Sample Backtest",
    "Run 1: Initial Benchmark Backtest"
])

st.sidebar.header("🕹️ Timeframe Controls")
timeframe = st.sidebar.selectbox("Select Timeframe:", ["15 Minute", "5 Minute", "1 Minute"])

tf_file_key = timeframe.lower().replace(" ", "")

if "Today's Live" in run_version:
    st.subheader(f"⚡ Today's Live Session Forecast ({timeframe}) - Market Close 3:30 PM IST")
    today_csv = os.path.join(plots_today, "today_live_forecast_summary.csv")
    if os.path.exists(today_csv):
        df_today = pd.read_csv(today_csv)
        row_t = df_today[df_today['timeframe'] == timeframe]
        if not row_t.empty:
            r = row_t.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Nifty 50 Price", f"{r['current_price']:.2f} INR")
            col2.metric("Predicted Market Close (3:30 PM)", f"{r['predicted_close']:.2f} INR")
            col3.metric("Predicted Remaining Move", f"{r['predicted_move_pts']:+.2f} pts")
            col4.metric("Actionable Signal", f"{r['signal']}")
    
    img_file = os.path.join(plots_today, f"today_live_{tf_file_key}.png")
    if os.path.exists(img_file):
        st.image(Image.open(img_file), use_column_width=True, caption=f"Today Live Forecast ({timeframe})")

elif "Run 3" in run_version:
    st.subheader(f"🕊️ Run 3: Calm Market Zone (10:30 AM - 02:30 PM IST) ({timeframe})")
    run3_csv = os.path.join(plots_run3, "run3_calm_zone_summary.csv")
    if os.path.exists(run3_csv):
        df_r3 = pd.read_csv(run3_csv)
        st.dataframe(df_r3, use_container_width=True)
    img_r3 = os.path.join(plots_run3, f"run3_calm_zone_{timeframe.lower().replace(' ', '_')}.png")
    if os.path.exists(img_r3):
        st.image(Image.open(img_r3), use_column_width=True)

elif "Run 2" in run_version:
    st.subheader(f"📊 Run 2: {timeframe} 3-Day Multi-Trial Analytics")
    run2_summary_file = os.path.join(plots_run2, "run2_individual_trials_summary.csv")
    if os.path.exists(run2_summary_file):
        df_trials = pd.read_csv(run2_summary_file)
        st.dataframe(df_trials[df_trials['timeframe'] == timeframe], use_container_width=True)

else:
    st.subheader(f"📊 Run 1 Benchmark Analytics ({timeframe})")
    mode = st.sidebar.radio("Select View:", ["Pro Candlestick View", "Mode 1", "Mode 2"])
    tf_key_run1 = timeframe.lower().replace(" ", "_")
    img_path = os.path.join(plots_run1_pro, f"pro_candlestick_{tf_key_run1}.png")
    if os.path.exists(img_path):
        st.image(Image.open(img_path), use_column_width=True)

st.markdown("---")
st.info("💡 **Intraday Live Execution Rule:** Feed past 400 candles into Kronos at the close of every candle to predict the next candle and enter option trades immediately on Zerodha.")
