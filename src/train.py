import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def preprocess_data():
    df = pd.read_csv('data/supermarket_sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    daily_sales = df.groupby('Date')['Total'].sum().reset_index()
    daily_sales.set_index('Date', inplace=True)
    daily_sales = daily_sales.asfreq('D').ffill()
    
    print(f"✅ Total days: {len(daily_sales)}")
    return daily_sales

def create_sequences(data, seq_length=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])
    return np.array(X), np.array(y), scaler

def train_model():
    daily_sales = preprocess_data()
    data = daily_sales['Total'].values
    
    X, y, scaler = create_sequences(data, seq_length=30)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("\n🚀 Training model...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, 
              validation_data=(X_test, y_test), verbose=1)
    
    # Clean Save
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_sales_model.h5', save_format='h5')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n🎉 Model saved successfully!")
    return model

if __name__ == "__main__":
    train_model()