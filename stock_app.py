# stock_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("📈 Stock Price Predictor (Simple ML)")

# User input
ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA)", "AAPL")

if st.button("Predict"):
    with st.spinner("Fetching data and training model..."):
        # 1. Download stock data
        df = yf.download(ticker, start="2018-01-01", end="2024-12-31")
        df = df[['Close']]

        # 2. Create features
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['Return'] = df['Close'].pct_change()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # 3. Prepare data
        features = ['Close', 'SMA_5', 'SMA_10', 'Return']
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 4. Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # 5. Evaluate
        mse = mean_squared_error(y_test, predictions)
        st.success(f"Model trained. Mean Squared Error: {mse:.2f}")

        # 6. Plot
        st.subheader("📊 Actual vs Predicted Closing Prices")
        fig,ax=plt.subplots(figsize=(12, 5))
        ax.plot(y_test.values, label='Actual Prices')
        ax.plot(predictions, label='Predicted Prices')
        ax.legend()
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        st.pyplot(fig)
