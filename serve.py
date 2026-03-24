# serve.py — top section — correct order
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

# ── App created FIRST ──
app = FastAPI(
    title="AirGuardian Forecast API",
    description="72-hour ward-level AQI forecast for Delhi NCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WARD_MAP = {w["id"]: w for w in WARDS}

# ── Startup event AFTER app is created ──
@app.on_event("startup")
def startup():
    import sqlite3, logging
    log = logging.getLogger(__name__)
    log.info(f"[startup] DB_PATH = {DB_PATH}")
    # Ensure core tables exist before anything else
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS aqi_readings (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ward_id   TEXT,
        ward_name TEXT,
        aqi       REAL,
        pm25      REAL,
        pm10      REAL,
        no2       REAL,
        so2       REAL,
        co        REAL,
        o3        REAL,
        source    TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS weather_readings (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT,
        ward_id          TEXT,
        temperature      REAL,
        humidity         REAL,
        wind_speed       REAL,
        wind_direction   REAL,
        boundary_layer_h REAL,
        precipitation    REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS iot_readings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT,
        ward_id     TEXT,
        sensor_id   TEXT,
        pm25        REAL,
        pm10        REAL,
        temperature REAL,
        humidity    REAL
    )""")
    conn.commit()
    # Debug: list all tables
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    log.info(f"[startup] Tables in DB: {tables}")
    # Log row counts per ward
    for ward in WARDS:
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM aqi_readings WHERE ward_id=?", (ward["id"],)
            ).fetchone()[0]
            log.info(f"[startup] aqi_readings[{ward['id']}] = {count} rows")
        except Exception:
            pass
    conn.close()
    init_monitor_tables()

# ── Helper functions ──
def aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

def aqi_color(aqi: float) -> str:
    if aqi <= 50:   return "#4ade80"
    if aqi <= 100:  return "#f0a020"
    if aqi <= 150:  return "#f97316"
    if aqi <= 200:  return "#f05151"
    if aqi <= 300:  return "#9d7df5"
    return "#7e22ce"

def confidence_from_horizon(horizon: int) -> int:
    return {6: 92, 12: 86, 24: 78, 48: 68, 72: 57}.get(horizon, 70)


def get_forecast(ward_id: str):
    """XGBoost-only forecast — used as fallback when ensemble is unavailable."""
    if ward_id not in WARD_MAP:
        raise HTTPException(status_code=404, detail=f"Ward '{ward_id}' not found")

    df = build_features(ward_id)
    if df is None or df.empty:
        raise HTTPException(status_code=503,
                            detail="Not enough data. Run collector + backfill first.")

    models = load_models(ward_id)
    if not models:
        raise HTTPException(status_code=503,
                            detail="Models not trained yet. Run training first.")

    ward = WARD_MAP[ward_id]
    now  = datetime.now(timezone.utc)
    feature_cols = get_feature_columns()
    latest = df[feature_cols].iloc[-1].values.reshape(1, -1)
    current_aqi = float(df["aqi"].iloc[-1])

    forecast_points = [{
        "horizon_hours": 0,
        "timestamp":     now.isoformat(),
        "aqi":           round(current_aqi),
        "category":      aqi_category(current_aqi),
        "color":         aqi_color(current_aqi),
        "confidence":    95,
        "model_used":    "observed",
    }]

    for horizon in HORIZONS:
        if horizon not in models:
            continue
        import numpy as np
        pred = float(np.clip(models[horizon].predict(latest)[0], 0, 500))
        ts   = now + timedelta(hours=horizon)
        forecast_points.append({
            "horizon_hours": horizon,
            "timestamp":     ts.isoformat(),
            "aqi":           round(pred),
            "category":      aqi_category(pred),
            "color":         aqi_color(pred),
            "confidence":    confidence_from_horizon(horizon),
            "model_used":    "xgb_only",
        })

    t0  = current_aqi
    t24 = next((p["aqi"] for p in forecast_points if p["horizon_hours"] == 24), t0)
    if   t24 > t0 + 15: trend = "worsening"
    elif t24 < t0 - 15: trend = "improving"
    else:                trend = "stable"

    return {
        "ward_id":          ward_id,
        "ward_name":        ward["name"],
        "generated_at":     now.isoformat(),
        "current_aqi":      round(current_aqi),
        "current_category": aqi_category(current_aqi),
        "trend":            trend,
        "forecast":         forecast_points,
        "model_version":    "xgb_v1",
    }

# ── All routes below ──
@app.get("/")
def root():
    return {
        "service":  "AirGuardian Forecast API",
        "version":  "1.0.0",
        "wards":    len(WARDS),
        "horizons": HORIZONS,
    }

app.get("/health", (req, res) => {
  res.send("OK");
});

@app.get("/forecast/ensemble/{ward_id}")
def get_ensemble_forecast(ward_id: str):
    """
    Enhanced forecast using LSTM + XGBoost ensemble with confidence bands.
    Falls back to XGBoost-only if LSTM not trained yet.
    """
    if not ENSEMBLE_AVAILABLE:
        return get_forecast(ward_id)

    if ward_id not in WARD_MAP:
        raise HTTPException(status_code=404, detail=f"Ward '{ward_id}' not found")

    result = ensemble_predict(ward_id)
    if not result:
        raise HTTPException(status_code=503,
                            detail="Not enough data. Run collector + retrain first.")

    ward = WARD_MAP[ward_id]
    now  = datetime.now(timezone.utc)

    forecast_points = [{
        "horizon_hours": 0,
        "timestamp":     now.isoformat(),
        "aqi":           result["current_aqi"],
        "aqi_lower":     result["current_aqi"],
        "aqi_upper":     result["current_aqi"],
        "category":      aqi_category(result["current_aqi"]),
        "color":         aqi_color(result["current_aqi"]),
        "confidence":    95,
        "model_used":    "observed",
    }]

    for horizon, data in result["horizons"].items():
        ts = now + timedelta(hours=horizon)
        forecast_points.append({
            "horizon_hours": horizon,
            "timestamp":     ts.isoformat(),
            "aqi":           data["aqi"],
            "aqi_lower":     data["aqi_lower"],
            "aqi_upper":     data["aqi_upper"],
            "category":      aqi_category(data["aqi"]),
            "color":         aqi_color(data["aqi"]),
            "confidence":    confidence_from_horizon(horizon),
            "model_used":    data["model_used"],
            "xgb_pred":      data.get("xgb_pred"),
            "lstm_pred":     data.get("lstm_pred"),
        })

    t0  = result["current_aqi"]
    t24 = result["horizons"].get(24, {}).get("aqi", t0)
    if   t24 > t0 + 15: trend = "worsening"
    elif t24 < t0 - 15: trend = "improving"
    else:                trend = "stable"

    return {
        "ward_id":          ward_id,
        "ward_name":        ward["name"],
        "generated_at":     now.isoformat(),
        "current_aqi":      result["current_aqi"],
        "current_category": aqi_category(result["current_aqi"]),
        "trend":            trend,
        "forecast":         forecast_points,
        "model_version":    "ensemble_v1",
    }


@app.get("/forecast/{ward_id}")
def forecast_xgb(ward_id: str):
    """XGBoost-only forecast endpoint."""
    return get_forecast(ward_id)

@app.get("/alerts")
def get_alerts():
    """Return all active unacknowledged AQI alerts."""
    return {"alerts": get_active_alerts()}


@app.get("/accuracy")
def get_accuracy():
    """Rolling MAE per ward per horizon from logged forecasts."""
    from models.monitor import compute_rolling_mae
    report = {}
    for ward in WARDS:
        wid = ward["id"]
        report[wid] = {}
        for horizon in [6, 12, 24, 48, 72]:
            mae = compute_rolling_mae(wid, horizon)
            report[wid][f"+{horizon}h"] = round(mae, 1) if mae else "insufficient data"
    return report