import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Kronos AI Nifty50 IST Dashboard", layout="wide", page_icon="📈")

st.title("📈 Kronos AI Nifty50 Option Trading Dashboard")
st.markdown("### Audited Out-of-Sample Backtesting & Multi-Trial Analytics (IST)")

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
data_dir = os.path.join(base_dir, "data")
plots_run1_mode1 = os.path.join(base_dir, "plots", "mode1_static_horizon")
plots_run1_mode2 = os.path.join(base_dir, "plots", "mode2_rolling_feed")
plots_run1_exact = os.path.join(base_dir, "plots", "exact_kronos_repo_viz")
plots_run1_pro = os.path.join(base_dir, "plots", "pro_candlesticks")
plots_run2 = os.path.join(base_dir, "plots", "run2_trial_multiday")

# Sidebar Controls & Version Segregation
st.sidebar.header("📂 Version & Run Segregation")
run_version = st.sidebar.radio("Select Backtest Version:", [
    "Run 2: 3-Day Multi-Trial Out-of-Sample Backtest (New)",
    "Run 1: Initial Benchmark Backtest"
])

st.sidebar.header("🕹️ Timeframe & Trial Controls")
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1 Minute", "5 Minute", "15 Minute"])

tf_file_key = timeframe.lower().replace(" ", "")

if "Run 2" in run_version:
    st.sidebar.subheader("Run 2 Trial Selector")
    trial_sel = st.sidebar.selectbox("Select Trial Day:", ["Trial 1", "Trial 2", "Trial 3", "Overall Summary"])
    
    run2_summary_file = os.path.join(plots_run2, "run2_individual_trials_summary.csv")
    overall_summary_file = os.path.join(plots_run2, "run2_overall_summary.csv")
    
    st.subheader(f"📊 Run 2: {timeframe} 3-Day Multi-Trial Analytics")
    
    if os.path.exists(run2_summary_file):
        df_trials = pd.read_csv(run2_summary_file)
        df_tf = df_trials[df_trials['timeframe'] == timeframe]
        
        st.markdown("#### Individual Trial Days Performance")
        st.dataframe(df_tf, use_container_width=True)
        
        if trial_sel != "Overall Summary":
            t_num = trial_sel.replace("Trial ", "")
            row = df_tf[df_tf['trial'] == trial_sel]
            if not row.empty:
                r = row.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{trial_sel} Win Rate ({r['date']})", f"{r['win_rate_pct']:.1f}%")
                col2.metric(f"{trial_sel} Day PnL", f"{r['pnl_pts']:+.2f} pts")
                col3.metric(f"{trial_sel} MAE", f"{r['mae_pts']:.2f} pts")
                
                # Image Display
                img_file = os.path.join(plots_run2, tf_file_key, f"run2_trial{t_num}_{r['date']}.png")
                if os.path.exists(img_file):
                    st.subheader(f"🖼️ OHLC Candlestick Plot ({trial_sel} - {r['date']})")
                    st.image(Image.open(img_file), use_column_width=True)
        else:
            if os.path.exists(overall_summary_file):
                df_ov = pd.read_csv(overall_summary_file)
                row_ov = df_ov[df_ov['timeframe'] == timeframe]
                if not row_ov.empty:
                    ro = row_ov.iloc[0]
                    col1, col2 = st.columns(2)
                    col1.metric(f"Overall {timeframe} Cumulative Win Rate", f"{ro['overall_win_rate_pct']:.1f}%")
                    col2.metric(f"Overall {timeframe} Cumulative PnL", f"{ro['overall_pnl_pts']:+.2f} pts")

else:
    # Run 1 Display
    mode = st.sidebar.radio("Select Prediction Mode / Chart View:", [
        "Professional OHLC Candlestick View (Green/Red Candles + Full Date & Time Legends)",
        "Mode 1: Static Multi-Step Horizon (15 Candles Ahead)",
        "Mode 2: Sequential Rolling One-Step Feed (Candle-by-Candle)",
        "Official Kronos Repo Subplot View (Price & Volume - prediction_example.py)"
    ])
    
    st.subheader(f"📊 Run 1 Benchmark Analytics ({timeframe})")
    
    mode_key = "Mode 1" if "Mode 1" in mode else ("Mode 2" if "Mode 2" in mode else "Pro Candlestick")
    tf_key_run1 = timeframe.lower().replace(" ", "_")
    
    if "Professional OHLC" in mode:
        img_path = os.path.join(plots_run1_pro, f"pro_candlestick_{tf_key_run1}.png")
    elif "Mode 1" in mode:
        img_path = os.path.join(plots_run1_mode1, f"mode1_{tf_key_run1}_static_horizon.png")
    elif "Mode 2" in mode:
        img_path = os.path.join(plots_run1_mode2, f"mode2_{tf_key_run1}_rolling_feed.png")
    else:
        img_path = os.path.join(plots_run1_exact, f"exact_kronos_viz_{tf_key_run1}.png")

    if os.path.exists(img_path):
        st.image(Image.open(img_path), use_column_width=True, caption=f"Run 1 Forecast ({timeframe})")

st.markdown("---")
st.info("💡 **Intraday Timing Rule:** Do NOT wait for market close. In live trading, as each candle closes, feed preceding 400 candles into Kronos to forecast the next candle and enter option trades immediately.")
