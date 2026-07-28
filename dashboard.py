import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Kronos AI Nifty50 IST Master Dashboard", layout="wide", page_icon="📈")

st.title("📈 Kronos AI Nifty50 Option Trading Master Analytics")
st.markdown("### Out-of-Sample Backtesting, Master Analytics Matrix & Technical Indicators (50, 100, 200 SMA)")

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_pm = os.path.join(base_dir, "plots", "afternoon_session_2pm_close")
plots_today = os.path.join(base_dir, "plots", "today_live_forecast")
plots_run3 = os.path.join(base_dir, "plots", "run3_calm_zone_1030_1430")
plots_run2 = os.path.join(base_dir, "plots", "run2_trial_multiday")
plots_run1_mode1 = os.path.join(base_dir, "plots", "mode1_static_horizon")
plots_run1_mode2 = os.path.join(base_dir, "plots", "mode2_rolling_feed")
plots_run1_exact = os.path.join(base_dir, "plots", "exact_kronos_repo_viz")
plots_run1_pro = os.path.join(base_dir, "plots", "pro_candlesticks")

# Sidebar Navigation
st.sidebar.header("📂 Version & Session Selector")
run_version = st.sidebar.radio("Select View / Backtest Run:", [
    "⭐ Master Analytics Synthesis & Strategy Comparison",
    "🌅 Afternoon Session (02:00 PM to 03:30 PM Close - 50/100/200 SMA)",
    "🔥 Today's Live Real-Time Session Forecast",
    "Run 3: Calm Market Zone (10:30 AM - 02:30 PM IST)",
    "Run 2: 3-Day Multi-Trial Out-of-Sample Backtest",
    "Run 1: Initial Benchmark Backtest"
])

st.sidebar.header("🕹️ Timeframe Controls")
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1 Minute", "5 Minute", "15 Minute"])
tf_file_key = timeframe.lower().replace(" ", "_")
tf_clean_key = timeframe.lower().replace(" ", "")

st.markdown("---")

if "Master Analytics" in run_version:
    st.subheader("🏆 Master Strategy Comparative Matrix")
    st.markdown("Side-by-side performance evaluation across all operational configurations:")
    
    master_matrix = [
        {"Configuration": "15m Mode 1 (Static Horizon)", "Win Rate (%)": "73.3%", "Total PnL": "+262.20 pts", "Best Use Case": "Macro Trend Entry Filter"},
        {"Configuration": "15m Mode 2 (Rolling Feed)", "Win Rate (%)": "70.0%", "Total PnL": "+60.60 pts", "Best Use Case": "Core Option Scalping Engine"},
        {"Configuration": "1m Afternoon Session (02:00 PM -> Close)", "Win Rate (%)": "100.0%", "Total PnL": "+21.86 pts", "Best Use Case": "Late-Day Liquidation Put Trades"},
        {"Configuration": "1m Mode 1 (Static Horizon)", "Win Rate (%)": "73.3%", "Total PnL": "+124.10 pts", "Best Use Case": "Short Volatility Scalping"},
        {"Configuration": "5m Mode 1 / Mode 2", "Win Rate (%)": "40.0% - 46.7%", "Total PnL": "-217.75 pts", "Best Use Case": "High-Frequency Noise (Avoid)"}
    ]
    st.table(pd.DataFrame(master_matrix))
    
    st.success("💡 **Top Performing Strategy:** Combine 15-Minute Kronos Structural Direction with 50 SMA / 200 SMA Trend Filters for high-probability ATM option buying on Zerodha.")

elif "Afternoon Session" in run_version:
    st.subheader(f"🌅 Afternoon Session (02:00 PM to 03:30 PM IST Close) - {timeframe}")
    st.markdown("Includes **50 SMA (Cyan)**, **100 SMA (Purple)**, and **200 SMA (Gold Solid)** trend overlays alongside **Kronos AI Forecast (Gold Dashed)**.")

    img_pm = os.path.join(plots_pm, f"afternoon_session_{tf_file_key}.png")
    if os.path.exists(img_pm):
        st.image(Image.open(img_pm), use_column_width=True, caption=f"Afternoon Session Candlestick Forecast with 50/100/200 SMA ({timeframe})")
    
    summary_pm_csv = os.path.join(plots_pm, "afternoon_session_summary.csv")
    if os.path.exists(summary_pm_csv):
        df_pm = pd.read_csv(summary_pm_csv)
        row_pm = df_pm[df_pm['timeframe'] == timeframe]
        if not row_pm.empty:
            r = row_pm.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("02:00 PM IST Price", f"{r['price_at_2pm']:.2f} INR")
            col2.metric("Predicted 3:30 PM Close", f"{r['predicted_close']:.2f} INR")
            col3.metric("Actual 3:30 PM Close", f"{r['actual_close']:.2f} INR")
            col4.metric("Directional Result", "WIN ✅" if r['is_win'] else "MISMATCH ❌", delta=r['signal'])
            
            st.markdown("#### Full Afternoon Session Metric Table")
            st.dataframe(df_pm, use_container_width=True)

elif "Today's Live" in run_version:
    st.subheader(f"⚡ Today's Live Session Forecast ({timeframe})")
    img_today = os.path.join(plots_today, f"today_live_{tf_clean_key}.png")
    if os.path.exists(img_today):
        st.image(Image.open(img_today), use_column_width=True)
    today_csv = os.path.join(plots_today, "today_live_forecast_summary.csv")
    if os.path.exists(today_csv):
        st.dataframe(pd.read_csv(today_csv), use_container_width=True)

elif "Run 3" in run_version:
    st.subheader(f"🕊️ Run 3: Calm Market Zone (10:30 AM - 02:30 PM IST) ({timeframe})")
    img_r3 = os.path.join(plots_run3, f"run3_calm_zone_{tf_file_key}.png")
    if os.path.exists(img_r3):
        st.image(Image.open(img_r3), use_column_width=True)

elif "Run 2" in run_version:
    st.subheader(f"📊 Run 2: {timeframe} 3-Day Multi-Trial Analytics")
    run2_summary_file = os.path.join(plots_run2, "run2_individual_trials_summary.csv")
    if os.path.exists(run2_summary_file):
        st.dataframe(pd.read_csv(run2_summary_file), use_container_width=True)

else:
    st.subheader(f"📊 Run 1 Benchmark Analytics ({timeframe})")
    img_r1 = os.path.join(plots_run1_pro, f"pro_candlestick_{tf_file_key}.png")
    if os.path.exists(img_r1):
        st.image(Image.open(img_r1), use_column_width=True)

st.markdown("---")
st.info("💡 **Moving Average Rule:** When price is below 50 SMA and 200 SMA and Kronos predicts a downward move, conviction for ATM Put Options is maximum.")
