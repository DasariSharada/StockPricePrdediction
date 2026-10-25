"""
StopPricePrediction - Streamlit app (single-file)

Usage:
1) pip install -r requirements.txt
2) streamlit run app.py
3) Upload a Kaggle OHLCV CSV (must contain Date, Open, High, Low, Close, Volume columns or similar)

Notes:
- Designed for daily historical OHLCV CSV files from Kaggle (or yfinance downloads).
- If your CSV date format differs, the app will try to parse it automatically.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import io
import datetime
import tempfile
import os

# Optional improved TA library - used only if installed
try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

# TensorFlow/LSTM import lazily (only when training LSTM)
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="StopPricePrediction", layout="wide", initial_sidebar_state="expanded")

# ----------------------
# Helper functions
# ----------------------
def infer_columns(df):
    # Normalize column names: find Date, Open, High, Low, Close, Volume
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for expected in ['date', 'open', 'high', 'low', 'close', 'volume']:
        for k, orig in cols.items():
            if expected == 'date' and ('date' in k or 'time' in k):
                mapping[orig] = 'Date'
                break
            if expected == 'open' and k == 'open':
                mapping[orig] = 'Open'; break
            if expected == 'high' and k == 'high':
                mapping[orig] = 'High'; break
            if expected == 'low' and k == 'low':
                mapping[orig] = 'Low'; break
            if expected == 'close' and k == 'close':
                mapping[orig] = 'Close'; break
            if expected == 'volume' and 'volume' in k:
                mapping[orig] = 'Volume'; break
    return mapping

def load_csv(uploaded_file):
    # return dataframe with Date parsed and columns normalized
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return None
    mapping = infer_columns(df)
    df = df.rename(columns=mapping)
    if 'Date' not in df.columns:
        st.error("Could not find a Date column in the CSV. Please ensure your CSV has a date/time field.")
        return None
    # parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    # Ensure numeric for numeric columns
    for col in ['Open','High','Low','Close','Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Keep rows with Close not null
    df = df.dropna(subset=['Close']).sort_values('Date').reset_index(drop=True)
    return df

def compute_sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(close, short=12, long=26, signal=9):
    ema_short = compute_ema(close, short)
    ema_long = compute_ema(close, long)
    macd_line = ema_short - ema_long
    signal_line = compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # assuming first column is Close scaled
    return np.array(X), np.array(y)

def evaluate_preds(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'r2': float(r2), 'rmse': float(rmse), 'mape_pct': float(mape)}

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.title("StopPricePrediction")
st.sidebar.markdown("Upload Kaggle CSV (OHLCV) or download via yfinance using ticker.")
upload = st.sidebar.file_uploader("Upload historical CSV (Kaggle)", type=['csv'])
use_yfinance = st.sidebar.checkbox("Fetch from yfinance instead of upload", value=False)
ticker = st.sidebar.text_input("Ticker (if yfinance):", value="AAPL" if use_yfinance else "")
period = st.sidebar.selectbox("yfinance period (if fetch):", options=["1y","2y","5y","10y","max"], index=2)
interval = st.sidebar.selectbox("yfinance interval:", options=["1d","1wk","1mo"], index=0)
st.sidebar.markdown("---")
st.sidebar.subheader("Model & UI Options")
use_model = st.sidebar.selectbox("Model", options=["RandomForest (fast)","LSTM (deep)"], index=0)
seq_len = st.sidebar.slider("LSTM sequence length (days)", min_value=10, max_value=200, value=60, step=5)
test_size = st.sidebar.slider("Test size (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
train_epochs = st.sidebar.slider("LSTM epochs", min_value=5, max_value=200, value=30, step=5)
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")
btn_compute = st.sidebar.button("Compute Indicators")
btn_train = st.sidebar.button("Train Model")
btn_predict = st.sidebar.button("Predict Next Days")
download_preds = st.sidebar.button("Download Predictions CSV")
st.sidebar.markdown("Built for interactive use. Upload and click actions.")

# ----------------------
# Main area
# ----------------------
st.title("🔮 StockPricePrediction")
st.write("Interactive stock price prediction and analysis. Upload a Kaggle OHLCV CSV and use the controls on the left.")

# Load data - either uploaded or via yfinance
df = None
if use_yfinance and ticker.strip():
    import yfinance as yf
    with st.spinner(f"Downloading {ticker} from yfinance..."):
        try:
            hist = yf.download(ticker, period=period, interval=interval, progress=False)
            if hist.empty:
                st.error("No data returned from yfinance. Check ticker/period/interval.")
            else:
                hist = hist.reset_index()
                hist = hist.rename(columns={'Date':'Date', 'Adj Close':'Adj Close'})
                # Keep expected columns
                df = hist[['Date','Open','High','Low','Close','Volume']].copy()
                st.success(f"Downloaded {len(df)} rows for {ticker}.")
        except Exception as e:
            st.error(f"yfinance error: {e}")
elif upload is not None:
    with st.spinner("Loading CSV..."):
        df = load_csv(upload)
        if df is None:
            st.stop()
        else:
            st.success(f"Loaded {len(df)} rows from uploaded file.")

if df is None:
    st.info("No data loaded. Upload a Kaggle CSV or enable yfinance fetch.")
    st.stop()

# Show top-level info
col1, col2, col3, col4 = st.columns([1,1,1,1])
start_date = df['Date'].min().date()
end_date = df['Date'].max().date()
col1.metric("Start Date", str(start_date))
col2.metric("End Date", str(end_date))
col3.metric("Rows", len(df))
col4.metric("Latest Close", f"{float(df['Close'].iloc[-1]):.2f}")

# Compute indicators now or when button clicked
if btn_compute or True:
    df = df.copy()
    df['SMA20'] = compute_sma(df['Close'], 20)
    df['EMA20'] = compute_ema(df['Close'], 20)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    # Additional TA library indicators if available
    if TA_AVAILABLE:
        try:
            df['RSI14'] = ta.momentum.rsi(df['Close'], window=14)
            df['ATR14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        except Exception:
            pass

    low_val = float(df['Low'].min())
    high_val = float(df['High'].max())
    latest_close = float(df['Close'].iloc[-1])
    macd_latest = float(df['MACD'].iloc[-1])
    macd_signal_latest = float(df['MACD_Signal'].iloc[-1])
    st.subheader("Summary")
    st.write(f"- Low (all-time in dataset): **{low_val:.4f}**")
    st.write(f"- High (all-time in dataset): **{high_val:.4f}**")
    st.write(f"- Latest Close: **{latest_close:.4f}**")
    st.write(f"- Latest MACD: **{macd_latest:.6f}**, Signal: **{macd_signal_latest:.6f}**")

    # Main chart (Close + SMA/EMA)
    st.subheader("Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], mode='lines', name='SMA20', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], mode='lines', name='EMA20', line=dict(dash='dot')))
    fig.update_layout(height=400, margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MACD")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(dash='dash')))
    fig2.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
    fig2.update_layout(height=300, margin=dict(t=20,l=10,r=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------
# Model training
# ----------------------
model = None
predictions_df = None
metrics_summary = None

if btn_train:
    st.info("Preparing data and training the selected model...")
    with st.spinner("Preparing features..."):
        # We'll build features as previous n returns and technical indicators
        data = df.copy().reset_index(drop=True)
        data['Return'] = data['Close'].pct_change().fillna(0)
        # Feature set: Close, Return, SMA20, EMA20, MACD, ATR14(if present)
        feature_cols = ['Close','Return','SMA20','EMA20','MACD']
        if 'ATR14' in data.columns:
            feature_cols.append('ATR14')
        feat_df = data[feature_cols].fillna(method='ffill').fillna(0)

    if "RandomForest" in use_model:
        # Simple RandomForest predicting next day's Close using lag features
        lags = 5
        Xrows = []
        yrows = []
        for i in range(lags, len(feat_df)-1):
            row = feat_df.iloc[i-lags:i].values.flatten()
            Xrows.append(row)
            yrows.append(data['Close'].iloc[i+1])  # predict next day's close
        X = np.array(Xrows)
        y = np.array(yrows)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        metrics_summary = evaluate_preds(y_test, y_pred)
        st.success("RandomForest trained.")
        st.write("Performance on test set:")
        st.write(metrics_summary)
        model = rf
        # Prepare predictions_df for display and download
        test_index = data.index[-len(y_test)-1:-1] if len(y_test)>0 else []
        # Ensure y_pred is 1D
        import numpy as np

        # Use X_test index if y_test has no index
        import numpy as np
        import pandas as pd

        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        # Handle both cases (Series or array)
        try:
            dates = X_test.index
        except AttributeError:
            dates = range(len(y_test))

        import numpy as np
        import pandas as pd

# Ensure y_pred and y_test are 1D
        y_pred = np.ravel(y_pred)
        y_test = np.ravel(y_test)

# Use X_test index if available, else create simple range
        try:
            dates = X_test.index
        except AttributeError:
            dates = range(len(y_test))

# Build DataFrame safely
        predictions_df = pd.DataFrame({
            "Date": dates,
            "Actual": y_test.tolist(),
            "Predicted": y_pred.tolist()
        })


    else:
        # LSTM path
        if not TF_AVAILABLE:
            st.error("TensorFlow is not installed; LSTM training is not available. Install tensorflow and retry.")
        else:
            # Scale features
            sequence_features = feat_df.values
            scaler = MinMaxScaler()
            seq_scaled = scaler.fit_transform(sequence_features)
            # create sequences using seq_len from sidebar
            X_all, y_all = create_sequences(seq_scaled, seq_len=seq_len)
            # y_all is scaled close column (index 0), but create_sequences assumed that
            # split into train/test without shuffling
            split_idx = int((1-test_size)*len(X_all))
            X_train, X_test = X_all[:split_idx], X_all[split_idx:]
            y_train, y_test = y_all[:split_idx], y_all[split_idx:]
            st.write(f"Training sequences: {X_train.shape}, Test sequences: {X_test.shape}")

            # build model
            model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            with st.spinner("Training LSTM... this may take time"):
                history = model.fit(X_train, y_train, validation_split=0.1, epochs=train_epochs, batch_size=32, verbose=0)
            st.success("LSTM training completed.")

            # predict (scaled), then invert scaling for close
            y_pred_scaled = model.predict(X_test).reshape(-1)
            # invert predictions by reconstructing feature vectors (replace scaled close with predicted)
            inv_preds = []
            inv_truth = []
            for i in range(len(y_pred_scaled)):
                last_row = X_test[i, -1, :].copy()
                last_row[0] = y_pred_scaled[i]   # set close scaled
                inv = scaler.inverse_transform(last_row.reshape(1,-1))[0][0]
                inv_preds.append(inv)
                # true
                true_row = X_test[i, -1, :].copy()
                true_row[0] = y_test[i]
                inv_true = scaler.inverse_transform(true_row.reshape(1,-1))[0][0]
                inv_truth.append(inv_true)

            metrics_summary = evaluate_preds(np.array(inv_truth), np.array(inv_preds))
            st.write("Performance on test set:")
            st.write(metrics_summary)

            predictions_df = pd.DataFrame({
                'Date': data['Date'].iloc[-len(inv_preds):].reset_index(drop=True),
                'TrueClose': inv_truth,
                'PredClose': inv_preds
            })
            # Persist model and scaler into memory (note: Streamlit ephemeral; for production save to file)
            st.session_state['lstm_model'] = model
            st.session_state['lstm_scaler'] = scaler

# ----------------------
# Predict next days iteratively
# ----------------------
if btn_predict:
    days = st.sidebar.number_input("Days to predict (iterative):", min_value=1, max_value=90, value=7)
    st.info(f"Predicting next {days} days...")
    if model is None:
        st.warning("No trained model found. Click 'Train Model' first.")
    else:
        if "RandomForest" in use_model:
            # For RF, we take last available lags and iteratively predict next day using last features
            data = df.copy().reset_index(drop=True)
            data['Return'] = data['Close'].pct_change().fillna(0)
            feat_df = data[['Close','Return','SMA20','EMA20','MACD']].fillna(method='ffill').fillna(0)
            lags = 5
            last_seq = feat_df.iloc[-lags:].values.flatten().reshape(1,-1)
            preds = []
            cur_seq = last_seq.copy()
            for _ in range(days):
                p = model.predict(cur_seq)[0]
                preds.append(p)
                # shift sequence: drop oldest day's features and append new day features
                # For simplicity, set new day's Close = predicted; Return=0 (approx); SMA/EMA/MACD repeat last
                last_features = cur_seq[0].reshape(lags, -1)[-1]
                new_row = last_features.copy()
                new_row[0] = p
                new_row[1] = 0.0
                cur_seq = np.hstack([cur_seq.reshape(1,-1)[0][len(new_row):].reshape(1,-1), new_row.reshape(1,-1)])
                cur_seq = cur_seq.reshape(1,-1)
            future_dates = [df['Date'].iloc[-1] + datetime.timedelta(days=i+1) for i in range(days)]
            predictions_future_df = pd.DataFrame({'Date': future_dates, 'PredClose': preds})
            st.write(predictions_future_df)
            predictions_df = predictions_future_df
        else:
            # LSTM iterative - needs scaler & model
            if not TF_AVAILABLE or 'lstm_model' not in st.session_state:
                st.error("LSTM model/scaler not available. Train LSTM in this session first.")
            else:
                model = st.session_state['lstm_model']
                scaler = st.session_state['lstm_scaler']
                # prepare last seq_len rows
                data = df.copy().reset_index(drop=True)
                data['Return'] = data['Close'].pct_change().fillna(0)
                feat_df = data[['Close','Return','SMA20','EMA20','MACD']].fillna(method='ffill').fillna(0)
                last_seq = scaler.transform(feat_df.values)[-seq_len:].copy()
                cur_seq = last_seq.copy()
                preds = []
                for _ in range(days):
                    X_in = cur_seq.reshape(1, seq_len, cur_seq.shape[1])
                    p_scaled = model.predict(X_in)[0,0]
                    # form next row by copying last row and replacing scaled close with p_scaled
                    nxt = cur_seq[-1].copy()
                    nxt[0] = p_scaled
                    # shift
                    cur_seq = np.vstack([cur_seq[1:], nxt])
                    # invert to actual price
                    inv = scaler.inverse_transform(nxt.reshape(1,-1))[0][0]
                    preds.append(inv)
                future_dates = [df['Date'].iloc[-1] + datetime.timedelta(days=i+1) for i in range(days)]
                predictions_future_df = pd.DataFrame({'Date': future_dates, 'PredClose': preds})
                st.write(predictions_future_df)
                predictions_df = predictions_future_df

# ----------------------
# Show test predictions and metrics
# ----------------------
if predictions_df is not None:
    st.subheader("Predictions (sample)")
    st.dataframe(predictions_df.head(50))

    # Plot predicted vs true if both present
    if 'TrueClose' in predictions_df.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['TrueClose'], mode='lines+markers', name='True'))
        fig3.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['PredClose'], mode='lines+markers', name='Predicted'))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # Metrics area
    if metrics_summary is not None:
        st.subheader("Metrics")
        st.write(metrics_summary)

# ----------------------
# Natural-language-like mini assistant
# ----------------------
st.subheader("Ask the model / quick requests")
user_q = st.text_input("Type requests like: 'low', 'high', 'macd', 'r2', 'predict 5', 'show sma', 'download predictions'")

if user_q:
    q = user_q.strip().lower()
    if q in ['low','what is low','show low']:
        st.info(f"Low (dataset): {low_val:.4f}")
    elif q in ['high','what is high','show high']:
        st.info(f"High (dataset): {high_val:.4f}")
    elif 'macd' in q:
        st.info(f"Latest MACD: {macd_latest:.6f}, Signal: {macd_signal_latest:.6f}")
    elif q in ['r2','what is r2','show r2']:
        if metrics_summary is not None:
            st.info(f"R² on test set: {metrics_summary.get('r2'):.4f}")
        else:
            st.warning("Model not trained yet; no R² available.")
    elif q.startswith('predict'):
        parts = q.split()
        num = 7
        try:
            num = int(parts[1])
        except Exception:
            num = 7
        st.info(f"Try clicking Predict Next Days and set days={num} in sidebar and run.")
    elif 'sma' in q:
        st.info("SMA20 shown on the main price chart.")
    elif 'download' in q and predictions_df is not None:
        towrite = io.StringIO()
        predictions_df.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button("Download predictions CSV", data=towrite.getvalue(), file_name="predictions.csv", mime="text/csv")
    else:
        st.info("I didn't catch that. Try queries like: 'low', 'high', 'macd', 'r2', 'predict 7', 'download predictions'.")

# ----------------------
# Allow downloading the full processed dataset
# ----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Export")
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
st.sidebar.download_button("Download processed dataset CSV", data=csv_buffer.getvalue(), file_name="processed_dataset.csv", mime="text/csv")

st.sidebar.markdown("Made with ❤️ — StopPricePrediction")
