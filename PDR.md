# Product Design & Requirements (PDR)
## Kronos AI Model Nifty50 Intraday Option Trading Backtest Engine

---

## 1. Executive Summary & Project Goal

- **Project Name:** Kronos AI Nifty50 Intraday Option Trading Backtest Engine
- **Target Asset:** Nifty 50 Spot Index (`^NSEI`)
- **Primary Objective:** Rigorously evaluate the zero-shot directional prediction accuracy (win rate), point move errors, and simulated PnL performance of the **Kronos AI Foundation Model** (`shiyu-coder/Kronos` / `NeoQuasar/Kronos-small`) on Nifty 50 spot price movements to determine its statistical edge for manual option trading on **Zerodha**.
- **GitHub Repository:** [https://github.com/AkshayJohn03/Kronos_Backtest](https://github.com/AkshayJohn03/Kronos_Backtest)

---

## 2. Problem Definition & Scope Constraints

### 2.1 Problem Definition
Retail option traders on Zerodha face high market noise, volatility spikes, and emotional bias during intraday trading. Existing technical indicators (RSI, MACD) often suffer from lagging signals. Kronos—a decoder-only time series foundation model pre-trained on over 12 billion K-line records—offers autoregressive multi-dimensional time series forecasting for market K-lines. This project establishes whether Kronos provides a measurable statistical win rate edge on Indian equity index spot prices.

### 2.2 Strict Constraints & Exclusions
- **No Brokerage Automation:** Excludes automated API order execution on Zerodha to avoid unnecessary infrastructure complexity and slippage risks.
- **No Fine-Tuning:** The model is evaluated strictly zero-shot using pre-trained weights (`NeoQuasar/Kronos-small`).
- **Target Asset Isolation:** Focuses exclusively on Nifty 50 spot prices (`^NSEI`).
- **Date Boundary Constraint:** Historic market data must be post-June 2025 (`2025-06-02` to `2026-07-24`).
- **Hardware Target:** Optimized for CPU execution on standard Windows hardware (16 GB RAM, ~1.2s latency per prediction).

---

## 3. Detailed Product Requirements

### Requirement 1: Indian Standard Time (IST) & Market Session Standardization
- **Timezone Conversion:** All UTC timestamps from raw market APIs must be converted to `Asia/Kolkata` (IST).
- **Session Filtering:** Candles must be filtered strictly for active National Stock Exchange (NSE) operating hours: **09:15 AM IST to 03:30 PM IST**.

### Requirement 2: Data Anti-Leakage & Statistical Rigor
- **Out-of-Sample Isolation:** Test data must be strictly from post-June 2025.
- **Zero Lookahead Bias:** Sliding window evaluations feed historical context ($N=400$ past bars) to predict future $K$ bars without access to future price data.
- **Sample Size Rigor:** Evaluated over 15 to 20 continuous rolling windows per timeframe to eliminate small-sample statistical anomalies (such as initial false 100% win rates).

### Requirement 3: Dual Prediction Evaluation Modes
1. **Mode 1: Static Multi-Step Horizon (15-Step Ahead Prediction)**
   - Feeds 400 historical IST candles to predict a 15-step future trajectory in a single pass.
   - Evaluates multi-step directional trend accuracy and cumulative PnL.
2. **Mode 2: Real-Time Rolling One-Step Feed (Sequential Candle-by-Candle Feed)**
   - Feeds 400 historical candles, predicts the next 1 candle ($t+1$).
   - As real market time advances, incorporates the actual ground-truth candle into context at every step and rolls forward continuously.

### Requirement 4: Visualization & Dashboard UI
- **Exact Kronos Repo Plots:** Replicate the 2-panel subplot format (Close Price + Volume, Ground Truth in Blue vs Prediction in Red) from `examples/prediction_example.py`.
- **Interactive Streamlit Dashboard:** Interactive local web dashboard hosted at `http://localhost:8501` featuring:
  - Dropdown controls for Timeframe (**1m**, **5m**, **15m**).
  - Mode selection (**Mode 1**, **Mode 2**, **Official Repo Subplot View**).
  - Live metric cards (Win Rate %, Total PnL Pts, Avg PnL / Trade, MAE Error).
  - Real-time IST market data table.

---

## 4. Hardware & System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 SYSTEM ARCHITECTURE                                |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Yahoo REST API (^NSEI)]  --->  [data_fetcher_ist.py] (IST Conversion: 09:15-15:30) |
|                                             |                                     |
|                                             v                                     |
|                              [data/ nifty_1m/5m/15m_ist.csv]                      |
|                                             |                                     |
|                                             v                                     |
|                     [kronos_engine_v2.py / NeoQuasar Kronos-small]                |
|                             (CPU Execution | Latency: 1.2s)                      |
|                                             |                                     |
|              +------------------------------+------------------------------+       |
|              |                                                             |       |
|              v                                                             v       |
|  [plots/ Mode 1, Mode 2 & Exact Repo PNGs]                   [kronos_nifty_report.md] |
|              |                                                             |       |
|              +------------------------------+------------------------------+       |
|                                             |                                     |
|                                             v                                     |
|                        [dashboard.py (Streamlit Web Dashboard)]                    |
|                              (Live at http://localhost:8501)                      |
|                                             |                                     |
|                                             v                                     |
|                    [Git Commit & Remote Push: AkshayJohn03/Kronos_Backtest]        |
+-----------------------------------------------------------------------------------+
```

---

## 5. Audited Backtest Performance & Key Metrics

| Mode | Timeframe | Windows Evaluated | Directional Win Rate (%) | Cumulative PnL (Pts) | Avg PnL / Trade | MAE (Pts) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Mode 1 (Static Horizon)** | **1 Minute** | 15 | **73.3%** | **+124.10 pts** | **+8.27 pts** | 25.83 pts |
| **Mode 2 (Rolling Feed)** | **1 Minute** | 20 | **45.0%** | **+2.25 pts** | **+0.11 pts** | 8.67 pts |
| **Mode 1 (Static Horizon)** | **5 Minute** | 15 | **46.7%** | **-217.75 pts** | **-14.52 pts** | 70.12 pts |
| **Mode 2 (Rolling Feed)** | **5 Minute** | 20 | **45.0%** | **+49.45 pts** | **+2.47 pts** | 12.09 pts |
| **Mode 1 (Static Horizon)** | **15 Minute** | 15 | **53.3%** | **+262.20 pts** | **+17.48 pts** | 125.73 pts |
| **Mode 2 (Rolling Feed)** | **15 Minute** | 20 | **60.0%** | **+60.60 pts** | **+3.03 pts** | 24.67 pts |

---

## 6. Actionable Guidelines for Zerodha Manual Option Trading

1. **Directional Filter:** Use Kronos **15-Minute Mode 2 (60.0% Win Rate)** and **1-Minute Mode 1 (73.3% Win Rate)** as an entry direction filter before buying At-The-Money (ATM) Call or Put options on Zerodha.
2. **Risk Management:** Maintain a strict **1:1.5 or 1:2 Risk-to-Reward ratio** (e.g., 15-point stop loss vs 30-point profit target on Nifty spot).
3. **Execution Rule:** Avoid holding unhedged option positions through major macroeconomic news releases or binary events without trailing stop losses.

---

## 7. Project Artifacts & File Registry

- **PDR Document:** [PDR.md](file:///C:/Users/Akshay.JOHN-XAVIER/OneDrive%20-%20Akkodis/Documents/Me/trade_kronos/PDR.md)
- **Interactive Dashboard:** [dashboard.py](file:///C:/Users/Akshay.JOHN-XAVIER/OneDrive%20-%20Akkodis/Documents/Me/trade_kronos/dashboard.py)
- **Kronos Engine V2:** [kronos_engine_v2.py](file:///C:/Users/Akshay.JOHN-XAVIER/OneDrive%20-%20Akkodis/Documents/Me/trade_kronos/kronos_engine_v2.py)
- **IST Data Fetcher:** [data_fetcher_ist.py](file:///C:/Users/Akshay.JOHN-XAVIER/OneDrive%20-%20Akkodis/Documents/Me/trade_kronos/data_fetcher_ist.py)
- **Exact Repo Subplot Visualizer:** [kronos_exact_viz.py](file:///C:/Users/Akshay.JOHN-XAVIER/OneDrive%20-%20Akkodis/Documents/Me/trade_kronos/kronos_exact_viz.py)
- **Markdown Report:** [kronos_nifty_backtest_report.md](file:///C:/Users/Akshay.JOHN-XAVIER/OneDrive%20-%20Akkodis/Documents/Me/trade_kronos/kronos_nifty_backtest_report.md)
- **GitHub Repository:** [https://github.com/AkshayJohn03/Kronos_Backtest](https://github.com/AkshayJohn03/Kronos_Backtest)
