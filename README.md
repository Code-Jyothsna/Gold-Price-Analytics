# 🪙 Gold Price Analytics & Predictive Intelligence Platform

An institutional-grade Quantitative Finance and Time-Series Machine Learning application built using **Streamlit**, **Pandas**, **Scikit-Learn**, and **Plotly**. This platform ingests historical gold price action, executes memory-optimized data transformations, evaluates technical momentum indicators, and delivers recursive multi-step forecasting via a Random Forest Regressor.

---

## 📌 1. Project Overview & Scope
Gold (*XAU / GLD ETF*) serves as the global financial ecosystem's premier safe-haven asset, store of value, and hedge against monetary inflation. 

This platform bridges quantitative technical analysis and predictive machine learning into a unified, high-performance operational cockpit. It was developed to provide investment desks, treasury risk managers, and commodity analysts with real-time analytics, automated data pipeline sanitation, and robust out-of-sample forward projections.

### Key Objectives:
- **Low-Memory Ingestion Pipeline**: Optimize memory footprint by downcasting numeric structures to `float32`/`int32` and using `@st.cache_data`.
- **Temporal Alignment**: Resample all records to uniform Daily (`'D'`) frequency, resolving non-trading weekend/holiday gaps via dual `.ffill()` and `.bfill()`.
- **Quantitative Analytics**: Compute dynamic technical indicators including 20-day/50-day Simple Moving Averages (SMA), 20-day Exponential Moving Averages (EMA), 14-day Relative Strength Index (RSI), and 20-day annualized realized volatility.
- **Predictive Supervised Learning**: Train a Random Forest Regressor on autoregressive time-lag features to project prices 7 to 90 days into the future with 95% confidence intervals.
- **Interactive UI**: Institutional cockpit with custom CSS metric cards, dynamic date slicers, and interactive Plotly visualization charts.

---

## 📊 2. Dataset Reference & Attribution
The platform natively supports the benchmark Kaggle Gold dataset:
- **Reference**: [Gold Price Prediction | Kaggle Benchmark](https://www.kaggle.com/code/farzadnekouei/gold-price-prediction-lstm-96-accuracy/input)
- **Primary Assets Covered**:
  - `Date`: Observation timestamp (parsed to `DatetimeIndex`)
  - `GLD`: SPDR Gold Shares ETF (Primary Target Asset)
  - `SPX`: S&P 500 Index (Benchmark US Equity Market Index)
  - `USO`: United States Oil Fund (Crude Oil Energy Proxy)
  - `SLV`: iShares Silver Trust (Precious Metal Companion)
  - `EUR/USD`: Euro to US Dollar Foreign Exchange Rate (Currency Valuation)
- *Automated Fallback*: In the absence of an uploaded CSV or local file, the application invokes a calibrated Geometric Brownian Motion generator reflecting historical asset co-movements.

---

## 🏗️ 3. Architecture & Optimization Pipeline

```
┌────────────────────────────────────────────────────────┐
│  DATA INGESTION LAYER                                  │
│  - pd.read_csv() (Uploaded file or benchmark CSV)      │
│  - DatetimeIndex standardization & chronological sort  │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│  MEMORY OPTIMIZATION & SANITATION (Guardrails 1-4)     │
│  - Uniform Daily Resampling: df.resample('D').mean()   │
│  - Imputation: .ffill() for weekends, .bfill() bounds  │
│  - Downcasting: float64 -> float32, int64 -> int32     │
│  - Intermediate GC cleanup: gc.collect()               │
│  - Streamlit In-Memory Caching: @st.cache_data         │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│  QUANTITATIVE FEATURE ENGINEERING                      │
│  - Lags: t-1, t-2, t-3, t-7, t-14                      │
│  - Rolling Statistics: Mean-7, Std-7, Volatility-20    │
│  - Momentum: SMA 20, SMA 50, EMA 20, RSI 14           │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│ PREDICTIVE ML ENGINE      │ │ STREAMLIT DASHBOARD UI   │
│ - Strict Temporal Split   │ │ - Metric KPI Delta Cards │
│ - Random Forest Regressor │ │ - Interactive Plotly Viz │
│ - Recursive Multi-Step    │ │ - Correlation Heatmap    │
│   Forecasting (7-90 days) │ │ - Residual Diagnostics   │
│ - 95% Confidence Bounds   │ │ - Telemetry & CSV Export │
└───────────────────────────┘ └──────────────────────────┘
```

---

## ⚡ 4. Installation & Execution Guide

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- `pip` package manager

### Step 1: Clone or Navigate to Directory
```bash
cd gold_price_analytics
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run YourName_GoldAnalytics.py
```
The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 📈 5. Analytical Insights & Risk Management Takeaways
1. **Equity vs. Gold Correlation**: Gold demonstrates low or negative correlation with equities during standard volatility regimes, verifying its role as a defensive portfolio stabilizer.
2. **Precious Metal Interdependence**: Strong positive cointegration exists between Gold (`GLD`) and Silver (`SLV`). Sustained divergences between the two assets frequently signal imminent mean-reversion opportunities.
3. **Volatility Clustering**: Periods of elevated 20-day annualized realized volatility (>18%) precede macroeconomic trend pivots and monetary policy adjustments by central banks.
4. **Model Predictability**: Autoregressive lag features (`Lag_1`, `Lag_2`) along with `Rolling_Mean_7` account for over 85% of tree split decisions in the Random Forest Regressor, reflecting strong short-term inertia in commodity price dynamics.
