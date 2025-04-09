import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report
import joblib
import streamlit as st

# Set plot style
plt.style.use('fivethirtyeight')

# Download data
def download_data(stock, start, end):
    df = yf.download(stock, start, end)
    return df

# Calculate RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0).flatten()  # Convert to 1D array
    loss = np.where(delta < 0, -delta, 0).flatten()  # Convert to 1D array

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Classify trend
def classify_trend(today, tomorrow):
    if pd.isna(tomorrow):  # Handle the last row where shift(-1) will be NaN
        return 'Neutral'
    pct_change = (tomorrow - today) / today * 100
    if pct_change > 1:
        return 'Bullish'
    elif pct_change < -1:
        return 'Bearish'
    else:
        return 'Neutral'

# Main function
def main():
    st.title("Stock Price Analysis and Prediction")
    st.sidebar.header("User Input")

    # User inputs
    stock = st.sidebar.text_input("Enter Stock Ticker (e.g., POWERGRID.NS)", "POWERGRID.NS")
    start = st.sidebar.date_input("Start Date", dt.datetime(2000, 1, 1))
    end = st.sidebar.date_input("End Date", dt.datetime(2024, 11, 1))

    # Download data
    df = download_data(stock, start, end)

    if df.empty:
        st.error("No data found for the given stock ticker and date range.")
        return

    # Display raw data
    st.subheader("Raw Data")
    st.write(df.tail())

    # Basic data exploration
    st.subheader("Data Exploration")
    st.write("Shape of the dataset:", df.shape)
    st.write("Missing values:", df.isnull().sum())
    st.write("Descriptive statistics:", df.describe())

    # Reset index and save to CSV
    df = df.reset_index()
    df.to_csv("powergrid.csv", index=False)

    # Load data from CSV (optional, for consistency)
    try:
        # Debug: Print the first few lines of the CSV file
        with open("powergrid.csv", "r") as f:
            for _ in range(5):  # Print the first 5 lines
                st.write(f.readline())

        # Read the CSV file with error handling
        data01 = pd.read_csv("powergrid.csv")

        # Convert numeric columns to float, coercing errors to NaN
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            data01[col] = pd.to_numeric(data01[col], errors='coerce')

        # Drop rows with NaN values in critical columns
        data01.dropna(subset=numeric_columns, inplace=True)

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    # Create 'Next_Close' column
    data01['Next_Close'] = data01['Close'].shift(-1)

    # Convert 'Close' and 'Next_Close' to numeric
    data01['Close'] = pd.to_numeric(data01['Close'], errors='coerce')
    data01['Next_Close'] = pd.to_numeric(data01['Next_Close'], errors='coerce')

    # Drop rows with missing values in 'Close' or 'Next_Close'
    data01.dropna(subset=['Close', 'Next_Close'], inplace=True)

    # Calculate technical indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['ema100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Plot closing prices
    st.subheader("Closing Prices Over Time")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label=f'{stock} Closing Price', linewidth=1)
    plt.title(f'{stock} Closing prices over time')
    plt.legend()
    st.pyplot(plt)

    # Plot moving averages
    st.subheader("Moving Averages")
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label=f'{stock} Close Price', linewidth=1)
    plt.plot(ma100, label=f'{stock} Moving Average 100 Price', linewidth=1)
    plt.plot(ma200, label=f'{stock} Moving Average 200 Price', linewidth=1)
    plt.legend()
    st.pyplot(plt)

    # Prepare data for training
    features = ['Open', 'High', 'Low', 'Volume', 'Close', 'ema100', 'ema200', 'RSI']
    available_features = [col for col in features if col in df.columns]

    X = df[available_features].shift(1).dropna()  # Use previous day's data
    y = df['Close'].shift(-1).dropna()  # Target: Next day's Close price

    # Align indices
    common_index = X.index.intersection(y.index)
    X, y = X.loc[common_index], y.loc[common_index]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train regression model
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train_scaled, y_train)

    # Predictions and evaluation
    y_pred = regressor.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    st.subheader("Regression Model Performance")
    st.write(f'Mean Absolute Error: {mae}')

    # Classification: Trend prediction
    data01['Trend'] = data01.apply(lambda row: classify_trend(row['Close'], row['Next_Close']), axis=1)
    data01.drop(columns=['Next_Close'], inplace=True)

    y_classification = data01['Trend'].loc[X.index]

    # Train classification model
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_scaled, y_classification.loc[X_train.index])

    # Predictions and evaluation
    y_pred_class = classifier.predict(X_test_scaled)
    st.subheader("Classification Model Performance")
    st.write("Classification Report:")
    st.text(classification_report(y_classification.loc[X_test.index], y_pred_class))

    # Save models
    joblib.dump(regressor, "regressor.pkl")
    joblib.dump(classifier, "classifier.pkl")
    joblib.dump(scaler, "scaler.pkl")
    st.success("✅ Models saved successfully!")

# Run main function
if __name__ == "__main__":
    main()