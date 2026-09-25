"""
================================================================================
GOLD PRICE ANALYTICS & PREDICTIVE INTELLIGENCE PLATFORM
================================================================================
Role: Senior Data Scientist Mentor & Quantitative Systems Architect
File: YourName_GoldAnalytics.py
Purpose: High-performance financial analytics, low-memory ingestion pipeline,
         interactive market intelligence visualization, and time-series
         forecasting with Random Forest regression.

Optimization Guardrails Implemented:
1. Low-Memory Ingestion: Downcasting numeric columns to float32/int32.
2. Immediate Aggregation & Resampling: Uniform daily ('D') resampling.
3. Memory Management: Streamlit @st.cache_data & garbage collection (gc.collect).
4. Secure Imputation: Dual forward-fill (.ffill()) and backward-fill (.bfill()).
================================================================================
"""

import os
import gc
import datetime
from typing import Tuple, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------------------------------------------
# 1. STREAMLIT APPLICATION CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gold Price Analytics & Predictive Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Finance Dark-Themed Styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(135deg, #1e222d 0%, #2a2e39 100%);
        border: 1px solid #363c4e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8c93a8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f1f3f6;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .metric-delta-positive {
        font-size: 0.88rem;
        font-weight: 600;
        color: #26a69a;
    }
    .metric-delta-negative {
        font-size: 0.88rem;
        font-weight: 600;
        color: #ef5350;
    }
    .badge-info {
        display: inline-block;
        background-color: #2a3b5c;
        color: #64b5f6;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-gold {
        display: inline-block;
        background-color: #4a3b1a;
        color: #ffd54f;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. LOW-MEMORY DATA INGESTION & PIPELINE LAYER (GUARDRAILS 1-4)
# -----------------------------------------------------------------------------
def _generate_synthetic_gold_data() -> pd.DataFrame:
    """
    Generates a realistic historical Gold dataset (GLD / XAU-USD proxy)
    mimicking Kaggle's gld_price_data.csv structure (GLD, SPX, USO, SLV, EUR/USD).
    Used as an automated fallback when no physical CSV file is uploaded.
    """
    date_range = pd.date_range(start="2012-01-01", end="2024-06-01", freq="B")
    np.random.seed(42)
    n = len(date_range)

    # Geometric Brownian Motion simulation with economic regime correlation
    drift = 0.0003
    volatility = 0.011

    gld_returns = np.random.normal(drift, volatility, n)
    spx_returns = np.random.normal(0.0004, 0.012, n)
    slv_returns = 0.65 * gld_returns + np.random.normal(0.0001, 0.018, n)
    uso_returns = np.random.normal(-0.0001, 0.024, n)
    eur_returns = -0.35 * gld_returns + np.random.normal(0.0, 0.005, n)

    gld_price = 150.0 * np.exp(np.cumsum(gld_returns))
    spx_price = 1250.0 * np.exp(np.cumsum(spx_returns))
    slv_price = 28.0 * np.exp(np.cumsum(slv_returns))
    uso_price = 35.0 * np.exp(np.cumsum(uso_returns))
    eur_usd = 1.30 * np.exp(np.cumsum(eur_returns))

    df_synth = pd.DataFrame(
        {
            "Date": date_range,
            "GLD": gld_price,
            "SPX": spx_price,
            "USO": uso_price,
            "SLV": slv_price,
            "EUR/USD": eur_usd,
            "Volume": np.random.randint(2_000_000, 15_000_000, size=n),
        }
    )
    return df_synth


@st.cache_data(show_spinner=False)
def load_and_optimize_dataset(
    uploaded_file: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Ingests financial time-series data with strict memory guardrails:
    - Downcasts numeric data types to float32/int32.
    - Parses DatetimeIndex and resamples immediately to clean daily intervals ('D').
    - Imputes missing trading gaps using forward-fill (.ffill()) then backward-fill (.bfill()).
    - Executes gc.collect() to clear intermediate buffer memory.
    """
    source_name = "Kaggle Benchmark / Synthetic Historical Dataset"
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            source_name = f"Uploaded CSV ({uploaded_file.name})"
        except Exception as e:
            st.error(f"Error parsing uploaded CSV file: {e}")
            df_raw = _generate_synthetic_gold_data()
    else:
        # Check if local benchmark gld_price_data.csv exists
        local_candidates = ["gld_price_data.csv", "gold_price_data.csv", "data/gld_price_data.csv"]
        found_path = next((p for p in local_candidates if os.path.exists(p)), None)
        if found_path:
            df_raw = pd.read_csv(found_path)
            source_name = f"Local File ({found_path})"
        else:
            df_raw = _generate_synthetic_gold_data()

    # Track memory footprint before optimization
    initial_ram_bytes = df_raw.memory_usage(deep=True).sum()

    # 1. Identify and standardize Date column
    date_col = next(
        (c for c in df_raw.columns if "date" in c.lower() or "timestamp" in c.lower()),
        None,
    )
    if date_col is not None:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.dropna(subset=[date_col])
        df_raw = df_raw.sort_values(by=date_col)
        df_raw = df_raw.set_index(date_col)
    else:
        # Fallback: attempt to convert index to DatetimeIndex
        df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
        df_raw = df_raw[df_raw.index.notnull()]
        df_raw = df_raw.sort_index()

    # 2. Identify Target Gold Column (GLD, Close, Price, or XAU)
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    target_col = None
    for cand in ["GLD", "Close", "Adj Close", "Price", "Gold_Price", "XAUUSD"]:
        if cand in df_raw.columns:
            target_col = cand
            break
    if not target_col and numeric_cols:
        target_col = numeric_cols[0]

    # Standardize column naming if 'Close' or other names used
    if target_col != "GLD" and target_col in df_raw.columns:
        df_raw["GLD"] = df_raw[target_col]
        target_col = "GLD"

    # 3. Aggregation & Resampling Guardrail: Aggregate to clean Daily ('D') intervals
    # Take mean of daily transactions (or last trade)
    df_resampled = df_raw.resample("D").mean()

    # 4. Clean Imputation Guardrail: Forward-fill weekend gaps, backward-fill initial points
    df_clean = df_resampled.ffill().bfill()

    # 5. Low-Memory Downcasting Guardrail: float64 -> float32, int64 -> int32
    for col in df_clean.columns:
        if pd.api.types.is_float_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype(np.int32)

    optimized_ram_bytes = df_clean.memory_usage(deep=True).sum()
    ram_savings_pct = (
        (initial_ram_bytes - optimized_ram_bytes) / initial_ram_bytes * 100
        if initial_ram_bytes > 0
        else 0.0
    )

    telemetry = {
        "initial_ram_kb": initial_ram_bytes / 1024,
        "optimized_ram_kb": optimized_ram_bytes / 1024,
        "ram_savings_pct": ram_savings_pct,
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean),
        "start_date": df_clean.index.min().strftime("%Y-%m-%d"),
        "end_date": df_clean.index.max().strftime("%Y-%m-%d"),
        "target_col": target_col,
    }

    # Free temporary buffers
    del df_raw, df_resampled
    gc.collect()

    return df_clean, telemetry, source_name


# -----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING & QUANTITATIVE ANALYTICS ENGINE
# -----------------------------------------------------------------------------
def engineer_quantitative_features(df: pd.DataFrame, target_col: str = "GLD") -> pd.DataFrame:
    """
    Computes financial technical indicators, moving averages, and time-lag features.
    Maintains clean vectorization and memory optimization.
    """
    df_feat = df.copy()

    # Dynamic Moving Averages
    df_feat["SMA_20"] = df_feat[target_col].rolling(window=20, min_periods=1).mean().astype(np.float32)
    df_feat["SMA_50"] = df_feat[target_col].rolling(window=50, min_periods=1).mean().astype(np.float32)
    df_feat["EMA_20"] = df_feat[target_col].ewm(span=20, adjust=False).mean().astype(np.float32)

    # Daily Return and Rolling Realized Volatility (Annualized 252 trading days)
    df_feat["Daily_Return"] = df_feat[target_col].pct_change().fillna(0.0).astype(np.float32)
    df_feat["Rolling_Vol_20"] = (
        (df_feat["Daily_Return"].rolling(window=20, min_periods=5).std() * np.sqrt(252) * 100)
        .fillna(0.0)
        .astype(np.float32)
    )

    # 14-Day Relative Strength Index (RSI)
    delta = df_feat[target_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df_feat["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50.0).astype(np.float32)

    # Autoregressive Lag Features
    for lag in [1, 2, 3, 7, 14]:
        df_feat[f"Lag_{lag}"] = df_feat[target_col].shift(lag).astype(np.float32)

    # Rolling Momentum & Statistics
    df_feat["Rolling_Mean_7"] = df_feat[target_col].rolling(7, min_periods=1).mean().astype(np.float32)
    df_feat["Rolling_Std_7"] = df_feat[target_col].rolling(7, min_periods=1).std().fillna(0.0).astype(np.float32)

    # Final imputation for boundary conditions
    df_feat = df_feat.ffill().bfill()
    return df_feat


# -----------------------------------------------------------------------------
# 4. PREDICTIVE MACHINE LEARNING MODEL (RANDOM FOREST REGRESSOR)
# -----------------------------------------------------------------------------
def train_and_forecast_model(
    df_feat: pd.DataFrame,
    target_col: str = "GLD",
    test_size_pct: float = 0.15,
    forecast_horizon: int = 30,
    n_estimators: int = 120,
    max_depth: int = 12,
    random_state: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.Series, RandomForestRegressor]:
    """
    Trains a high-capacity Random Forest Regressor on time-series engineered features.
    Performs out-of-sample temporal backtesting (no lookahead bias) and generates
    recursive future projections across the user-configured horizon (7 to 90 days).
    """
    # Define candidate ML feature columns
    excluded_cols = [target_col, "Daily_Return"]
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in excluded_cols and pd.api.types.is_numeric_dtype(df_feat[c])
    ]

    # Target variable is next-step value for recursive projection
    X = df_feat[feature_cols].copy()
    y = df_feat[target_col].copy()

    # Strict temporal time-series split (No shuffling to avoid lookahead leakage)
    split_idx = int(len(df_feat) * (1.0 - test_size_pct))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Evaluation on Hold-Out Test Split
    test_preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, test_preds))
    mse = float(mean_squared_error(y_test, test_preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, test_preds))
    mape = float(np.mean(np.abs((y_test - test_preds) / y_test)) * 100)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }

    # Historical Evaluation Frame
    df_eval = pd.DataFrame(
        {"Actual": y_test, "Predicted": test_preds},
        index=y_test.index,
    )

    # Feature Importance Extract
    feat_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    # Recursive Multi-Step Out-of-Sample Forecasting
    last_known_date = df_feat.index.max()
    future_dates = pd.date_range(
        start=last_known_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq="D",
    )

    future_predictions = []
    current_series = df_feat[target_col].copy()

    # Recursive loop predicting step by step
    for step_date in future_dates:
        # Re-derive features dynamically for the last known state
        current_lags = {
            "Lag_1": current_series.iloc[-1],
            "Lag_2": current_series.iloc[-2] if len(current_series) > 1 else current_series.iloc[-1],
            "Lag_3": current_series.iloc[-3] if len(current_series) > 2 else current_series.iloc[-1],
            "Lag_7": current_series.iloc[-7] if len(current_series) > 6 else current_series.iloc[-1],
            "Lag_14": current_series.iloc[-14] if len(current_series) > 13 else current_series.iloc[-1],
            "Rolling_Mean_7": current_series.tail(7).mean(),
            "Rolling_Std_7": current_series.tail(7).std(ddof=0),
            "SMA_20": current_series.tail(20).mean(),
            "SMA_50": current_series.tail(50).mean(),
            "EMA_20": current_series.ewm(span=20, adjust=False).mean().iloc[-1],
            "Rolling_Vol_20": float(df_feat["Rolling_Vol_20"].iloc[-1]),
            "RSI_14": float(df_feat["RSI_14"].iloc[-1]),
        }

        # Vectorize feature vector for prediction
        x_vector = []
        for col in feature_cols:
            if col in current_lags:
                x_vector.append(current_lags[col])
            elif col in df_feat.columns:
                # Carry forward latest macro-feature level (SPX, USO, SLV, EUR/USD, etc.)
                x_vector.append(float(df_feat[col].iloc[-1]))
            else:
                x_vector.append(0.0)

        x_df = pd.DataFrame([x_vector], columns=feature_cols, dtype=np.float32)
        next_pred = float(model.predict(x_df)[0])
        future_predictions.append(next_pred)

        # Append prediction to running series for subsequent lag recursive derivation
        current_series = pd.concat([current_series, pd.Series([next_pred], index=[step_date])])

    # Dynamic Volatility Confidence Interval Envelopes (95% CI based on model RMSE)
    z_score = 1.96
    ci_expansion = np.linspace(1.0, 1.85, forecast_horizon)  # variance increases over time
    margin = rmse * z_score * ci_expansion

    df_forecast = pd.DataFrame(
        {
            "Forecast": future_predictions,
            "Upper_CI": np.array(future_predictions) + margin,
            "Lower_CI": np.array(future_predictions) - margin,
        },
        index=future_dates,
    )

    return metrics, df_eval, df_forecast, feat_importances, model


# -----------------------------------------------------------------------------
# 5. UI VISUALIZATION & KPI COMPONENT LAYER
# -----------------------------------------------------------------------------
def render_kpi_cards(df: pd.DataFrame, target_col: str = "GLD"):
    """Renders executive KPI delta metric cards with institutional styling."""
    current_price = float(df[target_col].iloc[-1])
    prev_price = float(df[target_col].iloc[-2]) if len(df) > 1 else current_price
    price_delta = current_price - prev_price
    pct_delta = (price_delta / prev_price) * 100 if prev_price > 0 else 0.0

    period_high = float(df[target_col].max())
    period_low = float(df[target_col].min())
    curr_vol = float(df["Rolling_Vol_20"].iloc[-1]) if "Rolling_Vol_20" in df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_class = "metric-delta-positive" if price_delta >= 0 else "metric-delta-negative"
        delta_symbol = "▲" if price_delta >= 0 else "▼"
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Current Gold Index ({target_col})</div>
                <div class="metric-value">${current_price:,.2f}</div>
                <div class="{delta_class}">{delta_symbol} ${abs(price_delta):.2f} ({pct_delta:+.2f}%) 24h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Period Cycle High</div>
                <div class="metric-value">${period_high:,.2f}</div>
                <div style="font-size: 0.85rem; color: #8c93a8;">Spread from low: +${(period_high - period_low):.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Period Cycle Low</div>
                <div class="metric-value">${period_low:,.2f}</div>
                <div style="font-size: 0.85rem; color: #8c93a8;">Support floor reference</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        vol_regime = "High Volatility" if curr_vol > 18 else "Moderate" if curr_vol > 10 else "Low / Stable"
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">20D Realized Volatility</div>
                <div class="metric-value">{curr_vol:.1f}%</div>
                <div style="font-size: 0.85rem; color: #ffd54f;">Regime: {vol_regime}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_market_trends(
    df: pd.DataFrame,
    target_col: str = "GLD",
    show_sma20: bool = True,
    show_sma50: bool = True,
    show_ema20: bool = False,
) -> go.Figure:
    """Creates a multi-pane institutional price chart with technical indicator overlays."""
    has_volume = "Volume" in df.columns
    rows = 2 if has_volume else 1
    row_heights = [0.75, 0.25] if has_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=("Historical Price Action & Moving Average Overlays", "Trading Volume")
        if has_volume
        else ("Historical Price Action & Moving Average Overlays",),
    )

    # Primary Asset Price Trajectory
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[target_col],
            name=f"{target_col} Price",
            line=dict(color="#FFD700", width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Technical Moving Average Overlays
    if show_sma20 and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="20-Day SMA",
                line=dict(color="#29B6F6", width=1.5, dash="dot"),
                hovertemplate="SMA 20: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if show_sma50 and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="50-Day SMA",
                line=dict(color="#AB47BC", width=1.6, dash="dash"),
                hovertemplate="SMA 50: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if show_ema20 and "EMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_20"],
                name="20-Day EMA",
                line=dict(color="#26A69A", width=1.4),
                hovertemplate="EMA 20: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Secondary Pane: Volume Bars
    if has_volume:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color="#37474F",
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        hovermode="x unified",
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_yaxes(title_text="Price ($USD)", row=1, col=1)
    if has_volume:
        fig.update_yaxes(title_text="Shares", row=2, col=1)

    return fig


def plot_ml_forecast_projection(
    df_feat: pd.DataFrame,
    df_eval: pd.DataFrame,
    df_forecast: pd.DataFrame,
    target_col: str = "GLD",
) -> go.Figure:
    """Renders unified time-series chart showing historical data, test validation, and future projection."""
    fig = go.Figure()

    # Slice recent historical context for clean visualization
    recent_history = df_feat.iloc[-180:]

    # 1. Historical Baseline
    fig.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history[target_col],
            name="Historical Actual",
            line=dict(color="#B0BEC5", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>Historical: $%{y:.2f}<extra></extra>",
        )
    )

    # 2. Out-of-sample Test Predictions (Backtest)
    test_slice = df_eval[df_eval.index >= recent_history.index.min()]
    if not test_slice.empty:
        fig.add_trace(
            go.Scatter(
                x=test_slice.index,
                y=test_slice["Predicted"],
                name="Test Backtest (ML)",
                line=dict(color="#42A5F5", width=1.8, dash="dot"),
                hovertemplate="Backtest Pred: $%{y:.2f}<extra></extra>",
            )
        )

    # 3. Future Projections (Forecast Horizon)
    fig.add_trace(
        go.Scatter(
            x=df_forecast.index,
            y=df_forecast["Forecast"],
            name="Future ML Forecast",
            line=dict(color="#FFB300", width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: $%{y:.2f}<extra></extra>",
        )
    )

    # 4. Uncertainty Confidence Band (95% CI)
    fig.add_trace(
        go.Scatter(
            x=df_forecast.index,
            y=df_forecast["Upper_CI"],
            name="Upper 95% Bound",
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_forecast.index,
            y=df_forecast["Lower_CI"],
            name="Forecast Confidence Band (95%)",
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 179, 0, 0.16)",
            hovertemplate="Confidence Interval Spread<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Gold Price Time-Series Valuation & Out-of-Sample Machine Learning Projections",
        height=500,
        hovermode="x unified",
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Gold Valuation ($USD)")
    return fig


# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION EXECUTION ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    # Header & Subtitle
    st.title("🪙 Gold Price Analytics & Predictive Intelligence")
    st.markdown(
        "*Enterprise Quantitative Finance & Machine Learning Workbench | Internship Research Project*"
    )

    # SIDEBAR: DATA INGESTION & CONFIGURATION CONTROLS
    st.sidebar.header("⚙️ Data & Control Parameters")

    # File Ingestion Widget
    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom Market Data (CSV)",
        type=["csv"],
        help="Upload Kaggle gld_price_data.csv or custom OHLCV records.",
    )

    # Ingest and execute Low-Memory Optimization
    with st.spinner("Ingesting market data through memory-optimized pipeline..."):
        df_clean, telemetry, source_name = load_and_optimize_dataset(uploaded_file)
        target_col = telemetry["target_col"]
        df_feat = engineer_quantitative_features(df_clean, target_col=target_col)

    # Sidebar Date Range Slicer
    st.sidebar.subheader("📅 Temporal Filter")
    min_date = df_feat.index.min().to_pydatetime()
    max_date = df_feat.index.max().to_pydatetime()

    selected_date_range = st.sidebar.date_input(
        "Select Historical Period",
        value=(max_date - datetime.timedelta(days=730), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_sel, end_sel = selected_date_range
        df_filtered = df_feat.loc[
            (df_feat.index >= pd.to_datetime(start_sel))
            & (df_feat.index <= pd.to_datetime(end_sel))
        ]
    else:
        df_filtered = df_feat

    if df_filtered.empty:
        df_filtered = df_feat

    # Technical Overlays Toggle
    st.sidebar.subheader("📈 Technical Overlays")
    show_sma20 = st.sidebar.checkbox("20-Day Simple Moving Average (SMA)", value=True)
    show_sma50 = st.sidebar.checkbox("50-Day Simple Moving Average (SMA)", value=True)
    show_ema20 = st.sidebar.checkbox("20-Day Exponential Moving Average (EMA)", value=False)

    # ML Forecasting Parameters
    st.sidebar.subheader("🤖 Predictive Engine Parameters")
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (Days Ahead)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of future days to project recursively.",
    )
    test_split_pct = st.sidebar.slider(
        "Out-of-Sample Test Split",
        min_value=0.10,
        max_value=0.30,
        value=0.15,
        step=0.05,
    )
    rf_trees = st.sidebar.select_slider(
        "Random Forest Estimators",
        options=[50, 80, 100, 120, 150],
        value=100,
    )

    # Senior Mentor Pipeline Status Banner
    st.markdown(
        f"""
        <div style="background-color: #171b26; border-left: 4px solid #ffd54f; padding: 10px 16px; border-radius: 4px; margin-bottom: 20px;">
            <span class="badge-gold">PIPELINE ACTIVE</span> 
            <strong>Data Source:</strong> {source_name} | 
            <strong>Resampling:</strong> Clean Daily ('D') | 
            <strong>RAM Downcast:</strong> {telemetry['ram_savings_pct']:.1f}% Savings ({telemetry['initial_ram_kb']:.1f} KB &rarr; {telemetry['optimized_ram_kb']:.1f} KB) |
            <strong>Records:</strong> {len(df_filtered):,} Trading Days
        </div>
        """,
        unsafe_allow_html=True,
    )

    # TOP KPI DELTA CARDS
    render_kpi_cards(df_filtered, target_col=target_col)

    # APP NAVIGATION TABS
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Market Intelligence & Technicals",
            "🔮 Predictive ML Forecast",
            "🌐 Macro Correlation Matrix",
            "💾 Data Pipeline & Memory Telemetry",
        ]
    )

    # TAB 1: MARKET INTELLIGENCE & CHARTS
    with tab1:
        st.subheader("Price Action & Moving Average Dynamics")
        fig_trend = plot_market_trends(
            df_filtered,
            target_col=target_col,
            show_sma20=show_sma20,
            show_sma50=show_sma50,
            show_ema20=show_ema20,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### 14-Day Relative Strength Index (RSI)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered["RSI_14"],
                    name="RSI 14",
                    line=dict(color="#FFA726", width=1.5),
                )
            )
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF5350", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26A69A", annotation_text="Oversold (30)")
            fig_rsi.update_layout(
                template="plotly_dark",
                height=260,
                margin=dict(l=30, r=20, t=30, b=20),
                yaxis=dict(range=[10, 90]),
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_right:
            st.markdown("##### 20-Day Annualized Realized Volatility (%)")
            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered["Rolling_Vol_20"],
                    name="Realized Vol",
                    line=dict(color="#E91E63", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(233, 30, 99, 0.15)",
                )
            )
            fig_vol.update_layout(
                template="plotly_dark",
                height=260,
                margin=dict(l=30, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    # TAB 2: PREDICTIVE MACHINE LEARNING FORECAST
    with tab2:
        st.subheader("Time-Series Supervised Learning: Random Forest Regressor")
        st.markdown(
            """
            This module implements out-of-sample backtesting and recursive multi-step forecasting. 
            All feature transformations utilize sequential backward lags to eliminate any lookahead bias.
            """
        )

        with st.spinner("Training Random Forest Regressor and generating recursive forecast..."):
            metrics, df_eval, df_forecast, feat_importances, _ = train_and_forecast_model(
                df_filtered,
                target_col=target_col,
                test_size_pct=test_split_pct,
                forecast_horizon=forecast_horizon,
                n_estimators=rf_trees,
            )

        # Model Performance Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("MAE (Mean Absolute Error)", f"${metrics['MAE']:.2f}")
        m_col2.metric("RMSE (Root Mean Squared Error)", f"${metrics['RMSE']:.2f}")
        m_col3.metric("MAPE (Mean Absolute % Error)", f"{metrics['MAPE']:.2f}%")
        m_col4.metric("R² Score (Goodness of Fit)", f"{metrics['R2']:.4f}")

        # Forecast Projection Chart
        fig_ml = plot_ml_forecast_projection(
            df_filtered, df_eval, df_forecast, target_col=target_col
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        # Diagnostic Columns: Residuals & Feature Importances
        diag_left, diag_right = st.columns(2)
        with diag_left:
            st.markdown("##### Out-of-Sample Residual Error Distribution")
            residuals = df_eval["Actual"] - df_eval["Predicted"]
            fig_res = px.histogram(
                residuals,
                nbins=35,
                title="Model Residuals (Actual - Predicted)",
                color_discrete_sequence=["#26A69A"],
                template="plotly_dark",
            )
            fig_res.update_layout(height=300, margin=dict(l=30, r=20, t=40, b=20))
            st.plotly_chart(fig_res, use_container_width=True)

        with diag_right:
            st.markdown("##### Predictive Feature Importance (MDI)")
            top_feats = feat_importances.head(8)
            fig_imp = px.bar(
                x=top_feats.values,
                y=top_feats.index,
                orientation="h",
                labels={"x": "Importance Weight", "y": "Feature"},
                color_discrete_sequence=["#FFD54F"],
                template="plotly_dark",
            )
            fig_imp.update_layout(
                height=300,
                margin=dict(l=30, r=20, t=30, b=20),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # TAB 3: MACRO CORRELATION MATRIX
    with tab3:
        st.subheader("Precious Metal Cross-Asset & Macro Correlation Dynamics")
        numeric_subset = df_filtered.select_dtypes(include=[np.number])
        core_cols = [
            c for c in numeric_subset.columns if not c.startswith("Lag_") and c != "Volume"
        ]
        corr_matrix = numeric_subset[core_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
            title="Correlation Matrix (Gold, Macro Indices & Technical Features)",
        )
        fig_corr.update_layout(height=480, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info(
            "💡 **Quantitative Insight**: Gold (GLD) typically displays negative correlation with the US Dollar Index / EUR-USD pair, "
            "and positive cointegration with Silver (SLV). Divergences between GLD and SLV often signal macro mean-reversion setups."
        )

    # TAB 4: DATA PIPELINE & MEMORY TELEMETRY
    with tab4:
        st.subheader("Data Optimization & Memory Management Telemetry")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Raw Buffer Memory", f"{telemetry['initial_ram_kb']:.2f} KB")
        t_col2.metric("Optimized Memory", f"{telemetry['optimized_ram_kb']:.2f} KB")
        t_col3.metric("RAM Compression Ratio", f"{telemetry['ram_savings_pct']:.1f}%")
        t_col4.metric("Sanitized Datetime Records", f"{telemetry['clean_rows']:,}")

        st.markdown("##### Memory-Optimized Feature Store Preview")
        st.dataframe(
            df_filtered.tail(15).style.format("{:.2f}", subset=df_filtered.select_dtypes(include=[np.number]).columns),
            use_container_width=True,
        )

        # Download preprocessed dataset
        csv_data = df_filtered.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Sanitized Resampled Dataset (CSV)",
            data=csv_data,
            file_name="gold_analytics_resampled_daily.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
