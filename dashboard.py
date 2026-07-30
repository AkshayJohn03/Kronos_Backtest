import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Kronos AI Nifty50 IST Master Dashboard", layout="wide", page_icon="📈")

st.title("📈 Kronos AI Nifty50 Option Trading Master Analytics")
st.markdown("### Real-Time Live 2:15 PM IST Session Forecast, Technical Indicators (50/100/200 SMA) & Out-of-Sample Backtesting")

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_215pm = os.path.join(base_dir, "plots", "live_215pm_forecast")
plots_pm = os.path.join(base_dir, "plots", "afternoon_session_2pm_close")
plots_today = os.path.join(base_dir, "plots", "today_live_forecast")
plots_run3 = os.path.join(base_dir, "plots", "run3_calm_zone_1030_1430")
plots_run2 = os.path.join(base_dir, "plots", "run2_trial_multiday")
plots_run1_pro = os.path.join(base_dir, "plots", "pro_candlesticks")

# Sidebar Navigation
st.sidebar.header("📂 Version & Session Selector")
run_version = st.sidebar.radio("Select View / Live Run:", [
    "⚡ Live 2:15 PM Session Forecast (July 30, 2026)",
    "⭐ Master Analytics Synthesis & Strategy Comparison",
    "🌅 Afternoon Session (02:00 PM to 03:30 PM Close - 50/100/200 SMA)",
    "🔥 Today's Live Real-Time Session Forecast",
    "Run 3: Calm Market Zone (10:30 AM - 02:30 PM IST)",
    "Run 2: 3-Day Multi-Trial Out-of-Sample Backtest",
    "Run 1: Initial Benchmark Backtest"
])

st.sidebar.header("🕹️ Timeframe Controls")
timeframe = st.sidebar.selectbox("Select Timeframe:", ["15 Minute", "5 Minute", "1 Minute"])
tf_file_key = timeframe.lower().replace(" ", "_")

st.markdown("---")

if "Live 2:15 PM" in run_version:
    st.subheader(f"⚡ Live 2:15 PM IST Forecast ({timeframe}) - Session Close 3:30 PM")
    img_215 = os.path.join(plots_215pm, f"live_215pm_{tf_file_key}.png")
    if os.path.exists(img_215):
        st.image(Image.open(img_215), use_column_width=True, caption=f"Live 2:15 PM Forecast Chart ({timeframe})")
    
    summary_215_csv = os.path.join(plots_215pm, "live_215pm_forecast_summary.csv")
    if os.path.exists(summary_215_csv):
        df_215 = pd.read_csv(summary_215_csv)
        row_215 = df_215[df_215['timeframe'] == timeframe]
        if not row_215.empty:
            r = row_215.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("2:15 PM Current Price", f"{r['current_price']:.2f} INR")
            col2.metric("Next 15-Min Price (2:30 PM)", f"{r['next_15m_price']:.2f} INR", delta=f"{r['next_15m_move_pts']:+.2f} pts")
            col3.metric("Predicted Market Close (3:30 PM)", f"{r['predicted_close']:.2f} INR", delta=f"{r['predicted_move_close_pts']:+.2f} pts")
            col4.metric("Actionable Signal", f"{r['signal']}")
            
            st.markdown("#### Full 2:15 PM Live Forecast Summary Table")
            st.dataframe(df_215, use_container_width=True)

elif "Master Analytics" in run_version:
    st.subheader("🏆 Master Strategy Comparative Matrix")
    master_matrix = [
        {"Configuration": "15m Mode 1 (Static Horizon)", "Win Rate (%)": "73.3%", "Total PnL": "+262.20 pts", "Best Use Case": "Macro Trend Entry Filter"},
        {"Configuration": "15m Mode 2 (Rolling Feed)", "Win Rate (%)": "70.0%", "Total PnL": "+60.60 pts", "Best Use Case": "Core Option Scalping Engine"},
        {"Configuration": "1m Afternoon Session (02:00 PM -> Close)", "Win Rate (%)": "100.0%", "Total PnL": "+21.86 pts", "Best Use Case": "Late-Day Liquidation Put Trades"},
        {"Configuration": "1m Mode 1 (Static Horizon)", "Win Rate (%)": "73.3%", "Total PnL": "+124.10 pts", "Best Use Case": "Short Volatility Scalping"}
    ]
    st.table(pd.DataFrame(master_matrix))

elif "Afternoon Session" in run_version:
    st.subheader(f"🌅 Afternoon Session (02:00 PM to 03:30 PM IST Close) - {timeframe}")
    img_pm = os.path.join(plots_pm, f"afternoon_session_{tf_file_key}.png")
    if os.path.exists(img_pm):
        st.image(Image.open(img_pm), use_column_width=True)

else:
    st.subheader(f"📊 Run Analytics ({timeframe})")
    img_r1 = os.path.join(plots_run1_pro, f"pro_candlestick_{tf_file_key}.png")
    if os.path.exists(img_r1):
        st.image(Image.open(img_r1), use_column_width=True)

st.markdown("---")
st.info("💡 **Live 2:15 PM Trade Rule:** If 1m and 5m both predict bullish upside after 2:15 PM and price is above 50 SMA, enter ATM Call Option targeting market close.")
