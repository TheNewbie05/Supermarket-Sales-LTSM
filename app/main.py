from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import timedelta
import uvicorn
from loguru import logger
from prometheus_client import Counter, Histogram, make_asgi_app
import time
import os

# Logging Configuration
logger.add("logs/app_{time:YYYY-MM-DD}.log", rotation="1 day", retention="7 days", level="INFO")

# Prometheus Metrics
PREDICTIONS = Counter('sales_forecast_total', 'Total forecasts', ['days'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

app = FastAPI(title="Supermarket Sales LSTM Forecaster", version="1.0.0")

# Mount Prometheus metrics
app.mount("/metrics", make_asgi_app())

# Load Model
try:
    model = load_model('models/lstm_sales_model.h5', compile=False)
    scaler = joblib.load('models/scaler.pkl')
    SEQ_LENGTH = 30
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise

class ForecastResponse(BaseModel):
    forecast: list[float]
    dates: list[str]
    message: str

@app.get("/forecast")
async def predict_sales(days: int = 30):
    start_time = time.time()
    try:
        if not 1 <= days <= 90:
            raise HTTPException(400, "Days must be between 1 and 90")

        # Data Preprocessing for inference
        df = pd.read_csv('data/supermarket_sales.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        daily = df.groupby('Date')['Total'].sum().asfreq('D').ffill()

        last_seq = daily.values[-SEQ_LENGTH:].reshape(-1, 1)
        scaled = scaler.transform(last_seq)
        current = scaled.copy()
        predictions = []

        for _ in range(days):
            pred = model.predict(current.reshape(1, SEQ_LENGTH, 1), verbose=0)
            predictions.append(pred[0, 0])
            current = np.append(current[1:], pred, axis=0)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        last_date = daily.index[-1]
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]

        # Monitoring
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTIONS.labels(days=days).inc()

        logger.info(f"Forecast generated for {days} days | Latency: {latency:.3f}s")

        return ForecastResponse(
            forecast=[round(float(x), 2) for x in predictions.flatten()],
            dates=future_dates,
            message=f"Daily sales forecast for next {days} days"
        )

    except Exception as e:
        logger.error(f"Error in /forecast: {str(e)}")
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "LSTM Time Series"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)