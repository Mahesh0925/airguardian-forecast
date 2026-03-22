# ingestion/openmeteo.py — fetches hourly weather per ward (free, no API key)
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timezone
import logging
from config import WARDS

logger = logging.getLogger(__name__)

# Setup cached + retry session (cache=1h to avoid hammering)
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
client = openmeteo_requests.Client(session=retry_session)

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",  # critical for AQI forecasting
    "precipitation",
]

def fetch_ward_weather(ward: dict) -> dict | None:
    """Fetch current hour weather for a single ward."""
    try:
        responses = client.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  ward["lat"],
                "longitude": ward["lon"],
                "hourly":    VARIABLES,
                "forecast_days": 4,  # gives us 96h — enough for 72h forecast
                "timezone":  "Asia/Kolkata",
            }
        )
        r = responses[0]
        hourly = r.Hourly()

        # Build a dataframe of all hourly values
        times = pd.date_range(
            start=pd.Timestamp(hourly.Time(), unit="s", tz="Asia/Kolkata"),
            end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="Asia/Kolkata"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        df = pd.DataFrame({
            "timestamp":        times,
            "temperature_2m":   hourly.Variables(0).ValuesAsNumpy(),
            "humidity":         hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed":       hourly.Variables(2).ValuesAsNumpy(),
            "wind_direction":   hourly.Variables(3).ValuesAsNumpy(),
            "boundary_layer_h": hourly.Variables(4).ValuesAsNumpy(),
            "precipitation":    hourly.Variables(5).ValuesAsNumpy(),
        })

        # Get only the current hour row
        now = pd.Timestamp.now(tz="Asia/Kolkata").floor("h")
        row = df[df["timestamp"] == now]
        if row.empty:
            row = df.iloc[0]  # fallback to first row
        else:
            row = row.iloc[0]

        return {
            "ward_id":        ward["id"],
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "temperature":    round(float(row["temperature_2m"]), 2),
            "humidity":       round(float(row["humidity"]), 2),
            "wind_speed":     round(float(row["wind_speed"]), 2),
            "wind_direction": round(float(row["wind_direction"]), 2),
            "boundary_layer_h": round(float(row["boundary_layer_h"]), 2),
            "precipitation":  round(float(row["precipitation"]), 4),
            "source":         "open-meteo",
            # Store full forecast dataframe as JSON for feature engineering later
            "forecast_json":  df.to_json(orient="records", date_format="iso"),
        }

    except Exception as e:
        logger.error(f"Open-Meteo fetch failed for {ward['name']}: {e}")
        return None


def fetch_all_wards() -> list[dict]:
    results = []
    for ward in WARDS:
        reading = fetch_ward_weather(ward)
        if reading:
            results.append(reading)
            logger.info(
                f"Weather {ward['name']}: "
                f"{reading['temperature']}°C  "
                f"wind {reading['wind_speed']}km/h"
            )
    return results