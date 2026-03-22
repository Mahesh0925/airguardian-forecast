# ingestion/collector.py — runs all 4 sources and saves to storage
import logging
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from ingestion.aqicn     import fetch_all_wards as fetch_aqi
from ingestion.openmeteo import fetch_all_wards as fetch_weather
from ingestion.sentinel  import fetch_all_wards as fetch_satellite
from ingestion.iot_sim   import fetch_all_sensors
from config import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def init_db():
    """Create SQLite tables if they don't exist."""
    Path("storage").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS aqi_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ward_id TEXT, ward_name TEXT, timestamp TEXT,
        aqi REAL, pm25 REAL, pm10 REAL,
        no2 REAL, so2 REAL, co REAL, o3 REAL,
        source TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS weather_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ward_id TEXT, timestamp TEXT,
        temperature REAL, humidity REAL,
        wind_speed REAL, wind_direction REAL,
        boundary_layer_h REAL, precipitation REAL,
        forecast_json TEXT, source TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS satellite_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ward_id TEXT, date TEXT, timestamp TEXT,
        no2_tropospheric_column REAL, source TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS iot_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sensor_id TEXT, ward_id TEXT, timestamp TEXT,
        pm25 REAL, pm10 REAL, temperature REAL,
        humidity REAL, battery_pct REAL,
        status TEXT, source TEXT
    )""")

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def save(table: str, rows: list[dict]):
    if not rows:
        return
    conn = sqlite3.connect(DB_PATH)
    for row in rows:
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            list(row.values())
        )
    conn.commit()
    conn.close()
    logger.info(f"Saved {len(rows)} rows → {table}")


def run_collection():
    """Run one full collection cycle across all 4 sources."""
    logger.info("=== Collection cycle started ===")

    # 1. AQICN — real-time AQI per ward
    aqi_data = fetch_aqi()
    save("aqi_readings", aqi_data)

    # 2. Open-Meteo — weather per ward
    weather_data = fetch_weather()
    save("weather_readings", weather_data)

    # 3. Sentinel-5P — daily NO2 (runs once per day is enough)
    hour = datetime.now().hour
    if hour == 6:  # run at 6AM only
        satellite_data = fetch_satellite()
        save("satellite_readings", satellite_data)

    # 4. IoT sensors — every cycle
    iot_data = fetch_all_sensors()
    save("iot_readings", iot_data)

    logger.info("=== Collection cycle complete ===")


if __name__ == "__main__":
    init_db()
    run_collection()