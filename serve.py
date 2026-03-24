# serve.py

import os
import sys
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Setup ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.engineer import build_features, get_feature_columns
from models.train import load_models, HORIZONS
from models.monitor import get_active_alerts, init_monitor_tables, compute_rolling_mae

try:
    from models.ensemble import ensemble_predict
    ENSEMBLE_AVAILABLE = True
except Exception:
    ENSEMBLE_AVAILABLE = False

from config import WARDS, DB_PATH

# Ensure DB directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ── App ──
app = FastAPI(
    title="AirGuardian Forecast API",
    description="72-hour ward-level AQI forecast",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

WARD_MAP = {w["id"]: w for w in WARDS}

# ── Model Cache ──
MODEL_CACHE = {}

def get_models_cached(ward_id):
    if ward_id not in MODEL_CACHE:
        MODEL_CACHE[ward_id] = load_models(ward_id)
    return MODEL_CACHE[ward_id]

# ── Startup ──
@app.on_event("startup")
def startup():
    logger.info(f"[startup] DB_PATH = {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""CREATE TABLE IF NOT EXISTS aqi_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ward_id TEXT,
        ward_name TEXT,
        aqi REAL,
        pm25 REAL,
        pm10 REAL,
        no2 REAL,
        so2 REAL,
        co REAL,
        o3 REAL,
        source TEXT
    )""")

    conn.execute("""CREATE TABLE IF NOT EXISTS weather_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ward_id TEXT,
        temperature REAL,
        humidity REAL,
        wind_speed REAL,
        wind_direction REAL,
        boundary_layer_h REAL,
        precipitation REAL
    )""")

    conn.execute("""CREATE TABLE IF NOT EXISTS iot_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ward_id TEXT,
        sensor_id TEXT,
        pm25 REAL,
        pm10 REAL,
        temperature REAL,
        humidity REAL
    )""")

    conn.commit()
    conn.close()

    init_monitor_tables()
    logger.info("Startup complete")

# ── Helpers ──
def aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def aqi_color(aqi: float) -> str:
    if aqi <= 50: return "#4ade80"
    if aqi <= 100: return "#f0a020"
    if aqi <= 150: return "#f97316"
    if aqi <= 200: return "#f05151"
    if aqi <= 300: return "#9d7df5"
    return "#7e22ce"

def confidence_from_horizon(horizon: int) -> int:
    return {6: 92, 12: 86, 24: 78, 48: 68, 72: 57}.get(horizon, 70)

# ── Core Forecast ──
def get_forecast(ward_id: str):
    if ward_id not in WARD_MAP:
        raise HTTPException(status_code=404, detail="Ward not found")

    df = build_features(ward_id)
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Not enough data")

    models = get_models_cached(ward_id)
    if not models:
        raise HTTPException(status_code=503, detail="Models not trained")

    now = datetime.now(timezone.utc)
    feature_cols = get_feature_columns()
    latest = df[feature_cols].iloc[-1].values.reshape(1, -1)
    current_aqi = float(df["aqi"].iloc[-1])

    forecast = [{
        "horizon_hours": 0,
        "timestamp": now.isoformat(),
        "aqi": round(current_aqi),
        "category": aqi_category(current_aqi),
        "color": aqi_color(current_aqi),
        "confidence": 95,
        "model_used": "observed"
    }]

    for h in HORIZONS:
        if h not in models:
            continue

        pred = float(np.clip(models[h].predict(latest)[0], 0, 500))
        ts = now + timedelta(hours=h)

        forecast.append({
            "horizon_hours": h,
            "timestamp": ts.isoformat(),
            "aqi": round(pred),
            "category": aqi_category(pred),
            "color": aqi_color(pred),
            "confidence": confidence_from_horizon(h),
            "model_used": "xgb"
        })

    return {
        "ward_id": ward_id,
        "generated_at": now.isoformat(),
        "current_aqi": round(current_aqi),
        "forecast": forecast
    }

# ── Routes ──
@app.get("/")
def root():
    return {
        "service": "AirGuardian API",
        "wards": len(WARDS),
        "horizons": HORIZONS
    }

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {"status": "ok"}

@app.get("/forecast/{ward_id}")
def forecast(ward_id: str):
    return get_forecast(ward_id)

@app.get("/forecast/ensemble/{ward_id}")
def ensemble(ward_id: str):
    if not ENSEMBLE_AVAILABLE:
        return get_forecast(ward_id)

    result = ensemble_predict(ward_id)
    if not result:
        raise HTTPException(status_code=503, detail="Not enough data")

    return result

@app.get("/alerts")
def alerts():
    return {"alerts": get_active_alerts()}

@app.get("/accuracy")
def accuracy():
    report = {}
    for ward in WARDS:
        wid = ward["id"]
        report[wid] = {}
        for h in HORIZONS:
            mae = compute_rolling_mae(wid, h)
            report[wid][f"+{h}h"] = round(mae, 1) if mae else "NA"
    return report

# ── Run Server (Render Fix) ──
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("serve:app", host="0.0.0.0", port=port)