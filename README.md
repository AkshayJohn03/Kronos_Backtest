# 📈 Kronos AI Nifty50 Intraday Option Trading Backtest Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Kronos Model](https://img.shields.io/badge/AI%20Model-Kronos--small-orange)](https://huggingface.co/NeoQuasar/Kronos-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
This repository houses a rigorous evaluation engine built around the **Kronos AI Foundation Model** (`shiyu-coder/Kronos` / `NeoQuasar/Kronos-small`). The primary objective is to test its zero-shot directional prediction accuracy (win rate), point move errors, and simulated PnL performance strictly on **Nifty 50 spot price movements (^NSEI)** converted to **Indian Standard Time (IST)**.

By leveraging real-time and historical market data, this engine establishes whether Kronos provides a measurable statistical edge for manual scalping and option trading on Zerodha.

---

## 🚀 Features & Capabilities

- **IST Timezone Standardization:** Automatically converts UTC data to `Asia/Kolkata` and strictly filters out non-market hours (keeping only **09:15 AM to 03:30 PM IST**).
- **Out-of-Sample Isolation:** Enforces zero lookahead bias with rolling lookback windows.
- **Dual Evaluation Modes:**
  1. **Static Multi-Step Horizon (Mode 1):** Forecasts a 15-step future trajectory in a single pass based on historical context.
  2. **Real-Time Rolling One-Step Feed (Mode 2):** Mimics live trading by updating the context bar-by-bar and predicting the very next candle.
- **Paper Trading Automation:** Included script (`paper_trading.py`) simulates a live 1-minute rolling feed, executing hypothetical trades and tracking instant PnL (ignoring slippage).
- **Interactive Web Dashboard:** A built-in Streamlit app (`dashboard.py`) visualizes predictions, displays core metrics, and shows the exact market data fed into the model.

---

## 🛠️ System Architecture

```text
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
+-----------------------------------------------------------------------------------+
```

---

## 📊 Audited Backtest Performance (Real IST Data)

*Results generated using strictly post-June 2025 out-of-sample data, evaluated over continuous rolling windows.*

| Mode | Timeframe | Evaluated Windows | Win Rate (%) | Total PnL (Pts) | Avg PnL / Trade | MAE (Pts) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Mode 1 (Static Horizon)** | **1 Minute** | 15 | **53.3%** | **-62.80 pts** | **-4.19 pts** | 23.39 pts |
| **Mode 2 (Rolling Feed)** | **1 Minute** | 20 | **55.0%** | **+29.85 pts** | **+1.49 pts** | 6.28 pts |
| **Mode 1 (Static Horizon)** | **5 Minute** | 15 | **46.7%** | **-167.25 pts** | **-11.15 pts** | 82.38 pts |
| **Mode 2 (Rolling Feed)** | **5 Minute** | 20 | **45.0%** | **-15.65 pts** | **-0.78 pts** | 14.46 pts |
| **Mode 1 (Static Horizon)** | **15 Minute** | 15 | **66.7%** | **+818.40 pts** | **+54.56 pts** | 141.28 pts |
| **Mode 2 (Rolling Feed)** | **15 Minute** | 20 | **35.0%** | **-129.00 pts** | **-6.45 pts** | 26.36 pts |

---

## ⚙️ Quick Start & Installation

### 1. Install Dependencies
Ensure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```
*(If you are missing libraries, ensure `torch`, `transformers`, `pandas`, `yfinance`, `streamlit`, `matplotlib`, and `einops` are installed).*

### 2. Fetch Latest Real Data
```bash
python data_fetcher_ist.py
```
This fetches the latest 1m, 5m, and 15m data for `^NSEI`, formats it to strictly IST market hours, and saves it in the `data/` folder.

### 3. Run the Evaluation Engine
```bash
python kronos_engine_v2.py
```
This executes both Mode 1 and Mode 2 evaluations across all timeframes. It generates prediction charts inside the `plots/` folder and updates the Markdown report.

### 4. Start the Interactive Dashboard
```bash
streamlit run dashboard.py
```
Open `http://localhost:8501` to view your win rates, PnL, visual charts, and raw market data natively.

### 5. Run the Paper Trading Simulation
```bash
python paper_trading.py
```
This simulates a real-time sequential feed (Mode 2) using the latest 1-minute data and outputs actionable BUY CALL / BUY PUT signals along with simulated PnL.

---

## 💡 Trading Guidelines for Zerodha Options

1. **Directional Filter:** Use Kronos **15-Minute Mode 1 (66.7% Win Rate)** or **1-Minute Mode 2 (55.0% Win Rate)** as a macro direction filter before taking positions on Zerodha.
2. **Risk Management:** Maintain a strict **1:1.5 or 1:2 Risk-to-Reward ratio** (e.g., 15-point stop loss vs 30-point profit target on Nifty spot).
3. **Execution Rule:** Kronos provides statistical edges, not guarantees. Avoid holding unhedged option positions through major macroeconomic news releases.

---
*Disclaimer: This repository is for research and backtesting purposes only. Simulated paper trading results ignore slippage, brokerage fees, and liquidity constraints. Do not risk actual capital without extensive proprietary validation.*
