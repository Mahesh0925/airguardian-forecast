# features/engineer.py — builds ML-ready feature rows from raw collected data
import sqlite3
import pandas as pd
import numpy as np
from math import sin, cos, radians
from datetime import datetime, timezone
from config import DB_PATH, WARDS

def load_raw(ward_id: str, hours_back: int = 2000) -> pd.DataFrame:
    import os
    db = os.environ.get("DB_PATH", "storage/airguardian.db")
    conn = sqlite3.connect(db)

    aqi_df = pd.read_sql_query(f"""
        SELECT timestamp, aqi, pm25, pm10, no2, so2, co, o3
        FROM aqi_readings
        WHERE ward_id = '{ward_id}'
        ORDER BY timestamp ASC
    """, conn, parse_dates=["timestamp"])

    weather_df = pd.read_sql_query(f"""
        SELECT timestamp, temperature, humidity,
               wind_speed, wind_direction, boundary_layer_h, precipitation
        FROM weather_readings
        WHERE ward_id = '{ward_id}'
        ORDER BY timestamp ASC
    """, conn, parse_dates=["timestamp"])

    iot_df = pd.read_sql_query(f"""
        SELECT timestamp, AVG(pm25) as iot_pm25, AVG(pm10) as iot_pm10
        FROM iot_readings
        WHERE ward_id = '{ward_id}'
        GROUP BY timestamp
        ORDER BY timestamp ASC
    """, conn, parse_dates=["timestamp"])

    conn.close()

    if aqi_df.empty:
        return pd.DataFrame()

    # Normalize all timestamps to UTC-naive floored to hour
    for df in [aqi_df, weather_df, iot_df]:
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        ).dt.tz_localize(None).dt.floor("h")

    # Force numeric on all columns
    for df in [aqi_df, weather_df, iot_df]:
        if df.empty:
            continue
        num_cols = [c for c in df.columns if c != "timestamp"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Deduplicate AQI rows by hour — keep mean
    aqi_df = aqi_df.groupby("timestamp", as_index=False).mean(numeric_only=True)
    aqi_df = aqi_df.sort_values("timestamp").reset_index(drop=True)

    # ── KEY FIX: don't join weather — synthesize it from AQI data ──
    # Weather fetch is failing/empty on Render so we build fallback values
    if weather_df.empty or len(weather_df) < 5:
        # Use neutral defaults — model still trains on AQI lag features
        aqi_df["temperature"]    = 22.0
        aqi_df["humidity"]       = 60.0
        aqi_df["wind_speed"]     = 3.0
        aqi_df["wind_direction"] = 180.0
        aqi_df["boundary_layer_h"] = 800.0
        aqi_df["precipitation"]  = 0.0
    else:
        weather_df = weather_df.groupby("timestamp", as_index=False).mean(numeric_only=True)
        aqi_df = aqi_df.merge(weather_df, on="timestamp", how="left")
        weather_cols = ["temperature","humidity","wind_speed",
                        "wind_direction","boundary_layer_h","precipitation"]
        aqi_df[weather_cols] = aqi_df[weather_cols].ffill().bfill().fillna({
            "temperature": 22.0, "humidity": 60.0, "wind_speed": 3.0,
            "wind_direction": 180.0, "boundary_layer_h": 800.0, "precipitation": 0.0
        })

    # IoT fusion
    if not iot_df.empty:
        iot_df = iot_df.groupby("timestamp", as_index=False).mean(numeric_only=True)
        aqi_df = aqi_df.merge(iot_df, on="timestamp", how="left")
        aqi_df["iot_pm25"] = aqi_df.get("iot_pm25", aqi_df["pm25"])
        aqi_df["iot_pm10"] = aqi_df.get("iot_pm10", aqi_df["pm10"])
    else:
        aqi_df["iot_pm25"] = aqi_df["pm25"]
        aqi_df["iot_pm10"] = aqi_df["pm10"]

    # Fill remaining NaNs
    aqi_df = aqi_df.fillna(aqi_df.median(numeric_only=True))
    aqi_df = aqi_df.sort_values("timestamp").reset_index(drop=True)

    return aqi_df

def build_features(ward_id: str) -> pd.DataFrame | None:
    """
    Build the full feature matrix for a ward.
    Returns one row per hour with all features needed by the ML model.
    """
    df = load_raw(ward_id)

    if df.empty or len(df) < 2:
        return None

    # ── 1. LAG FEATURES (most predictive) ──
    df["aqi_lag_1h"]   = df["aqi"].shift(1)
    df["aqi_lag_6h"]   = df["aqi"].shift(6)
    df["aqi_lag_24h"]  = df["aqi"].shift(24)
    df["aqi_lag_168h"] = df["aqi"].shift(168)   # same time last week

    df["pm25_lag_1h"]  = df["pm25"].shift(1)
    df["pm25_lag_6h"]  = df["pm25"].shift(6)

    # ── 2. ROLLING STATISTICS ──
    df["aqi_roll_mean_6h"]  = df["aqi"].rolling(6,  min_periods=1).mean()
    df["aqi_roll_mean_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
    df["aqi_roll_std_6h"]   = df["aqi"].rolling(6,  min_periods=1).std().fillna(0)
    df["aqi_trend_3h"]      = df["aqi"].diff(3)   # rising or falling

    # ── 3. METEOROLOGICAL FEATURES ──
    # Cyclical encoding for wind direction (0-360 is circular, not linear)
    df["wind_dir_sin"] = df["wind_direction"].apply(lambda x: sin(radians(x)))
    df["wind_dir_cos"] = df["wind_direction"].apply(lambda x: cos(radians(x)))

    # Temperature inversion proxy — low boundary layer = pollution trap
    df["inversion_risk"] = (df["boundary_layer_h"] < 500).astype(int)

    # Wind categories
    df["calm_wind"] = (df["wind_speed"] < 2).astype(int)   # < 2 km/h = stagnant

    # ── 4. TEMPORAL FEATURES (cyclical) ──
    df["hour"]     = df["timestamp"].dt.hour
    df["hour_sin"] = df["hour"].apply(lambda h: sin(2 * np.pi * h / 24))
    df["hour_cos"] = df["hour"].apply(lambda h: cos(2 * np.pi * h / 24))

    df["dow"]      = df["timestamp"].dt.dayofweek
    df["dow_sin"]  = df["dow"].apply(lambda d: sin(2 * np.pi * d / 7))
    df["dow_cos"]  = df["dow"].apply(lambda d: cos(2 * np.pi * d / 7))

    df["month_sin"] = df["timestamp"].dt.month.apply(lambda m: sin(2 * np.pi * m / 12))
    df["month_cos"] = df["timestamp"].dt.month.apply(lambda m: cos(2 * np.pi * m / 12))

    # Weekend flag (higher biomass burning on weekends in Delhi)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # ── 5. IOT FUSION FEATURES ──
    # Blend AQICN and IoT readings (IoT is noisier but more granular)
    df["pm25_fused"] = (df["pm25"] * 0.6 + df["iot_pm25"] * 0.4).round(2)
    df["pm25_iot_divergence"] = (df["iot_pm25"] - df["pm25"]).abs()

    # ── 6. INTERACTION FEATURES ──
    # High humidity + high PM = worse apparent AQI
    df["humidity_pm_interaction"] = df["humidity"] * df["pm25_fused"] / 100
    # Calm wind + high AQI = accumulation risk
    df["stagnation_score"] = df["calm_wind"] * df["aqi_roll_mean_6h"]

    # ── 7. ADD WARD ID ──
    df["ward_id"] = ward_id

    # Drop rows with NaN in critical columns
    critical_cols = ["aqi", "wind_speed"]
    df = df.dropna(subset=critical_cols)

    return df


def build_all_wards() -> pd.DataFrame:
    """Build features for all wards and stack into one training DataFrame."""
    frames = []
    for ward in WARDS:
        df = build_features(ward["id"])
        if df is not None:
            frames.append(df)
            print(f"Features built: {ward['name']} — {len(df)} rows, {len(df.columns)} features")

    if not frames:
        print("No data yet — run collector first for a few cycles")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal feature matrix: {combined.shape[0]} rows × {combined.shape[1]} columns")
    return combined


def get_feature_columns() -> list:
    """Returns the exact list of feature columns the ML model expects."""
    return [
        "aqi_lag_1h", "aqi_lag_6h", "aqi_lag_24h", "aqi_lag_168h",
        "pm25_lag_1h", "pm25_lag_6h",
        "aqi_roll_mean_6h", "aqi_roll_mean_24h", "aqi_roll_std_6h", "aqi_trend_3h",
        "wind_speed", "wind_dir_sin", "wind_dir_cos",
        "temperature", "humidity", "boundary_layer_h", "precipitation",
        "inversion_risk", "calm_wind",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend",
        "stagnation_score",
    ]


if __name__ == "__main__":
    df = build_all_wards()
    if not df.empty:
        print("\nSample row:")
        print(df[get_feature_columns()].iloc[-1].to_string())
        # Save to CSV for inspection
        df.to_csv("storage/features.csv", index=False)
        print("\nSaved to storage/features.csv")