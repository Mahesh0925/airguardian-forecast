# launch.py — runs collector + retrain in background, API in foreground
import threading
import time
import logging
import os
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Use /data on Render (persistent disk), local storage/ in dev
_default_db = "/data/airguardian.db" if os.path.isdir("/data") else "storage/airguardian.db"
os.environ.setdefault("DB_PATH", _default_db)
DB_PATH = os.environ["DB_PATH"]

log.info(f"[launch] DB_PATH = {DB_PATH}")

# Ensure directories exist
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)
os.makedirs("models/saved", exist_ok=True)
os.makedirs("models/saved_lstm", exist_ok=True)


def init_db():
    """Create all required tables if they don't exist."""
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
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    log.info(f"[init_db] Tables: {tables}")
    conn.close()


def run_collector():
    """Runs data collection every 15 minutes forever."""
    import schedule
    from datetime import datetime

    try:
        from ingestion.collector import init_db as collector_init_db
        collector_init_db()
    except Exception as e:
        log.warning(f"[collector] collector init_db failed ({e}), using local init_db")
        init_db()

    def run_collection():
        log.info("=== Collection cycle started ===")

        # AQICN — every cycle
        try:
            from ingestion.aqicn import fetch_all_wards as fetch_aqi
            aqi_data = fetch_aqi()
            if aqi_data:
                conn = sqlite3.connect(DB_PATH)
                for row in aqi_data:
                    cols = ", ".join(row.keys())
                    ph = ", ".join(["?"] * len(row))
                    conn.execute(f"INSERT INTO aqi_readings ({cols}) VALUES ({ph})", list(row.values()))
                conn.commit()
                conn.close()
                log.info(f"Saved {len(aqi_data)} rows → aqi_readings")
        except Exception as e:
            log.error(f"AQI fetch failed: {e}")

        # Open-Meteo — once per hour only
        try:
            if datetime.now().minute < 15:
                from ingestion.openmeteo import fetch_all_wards as fetch_weather
                weather_data = fetch_weather()
                if weather_data:
                    conn = sqlite3.connect(DB_PATH)
                    for row in weather_data:
                        # Remove forecast_json — too large to store
                        row.pop("forecast_json", None)
                        row.pop("source", None)
                        cols = ", ".join(row.keys())
                        ph = ", ".join(["?"] * len(row))
                        try:
                            conn.execute(f"INSERT INTO weather_readings ({cols}) VALUES ({ph})", list(row.values()))
                        except Exception:
                            continue
                    conn.commit()
                    conn.close()
                    log.info(f"Saved {len(weather_data)} rows → weather_readings")
            else:
                log.info("Skipping weather fetch — runs once per hour only")
        except Exception as e:
            log.error(f"Weather fetch failed: {e}")

        # IoT sensors — every cycle
        try:
            from ingestion.iot_sim import fetch_all_sensors
            iot_data = fetch_all_sensors()
            if iot_data:
                conn = sqlite3.connect(DB_PATH)
                for row in iot_data:
                    row.pop("status", None)
                    row.pop("source", None)
                    row.pop("battery_pct", None)
                    cols = ", ".join(row.keys())
                    ph = ", ".join(["?"] * len(row))
                    try:
                        conn.execute(f"INSERT INTO iot_readings ({cols}) VALUES ({ph})", list(row.values()))
                    except Exception:
                        continue
                conn.commit()
                conn.close()
                log.info(f"Saved {len(iot_data)} rows → iot_readings")
        except Exception as e:
            log.error(f"IoT fetch failed: {e}")

        log.info("=== Collection cycle complete ===")

    # Run immediately then schedule
    run_collection()
    schedule.every(15).minutes.do(run_collection)

    while True:
        schedule.run_pending()
        time.sleep(30)

def run_retrain():
    """Runs backfill on first boot then retrains daily at 3AM."""
    from pathlib import Path

    # Wait for collector to seed some data first
    time.sleep(60)

    # Check if first boot (table missing or < 100 rows)
    first_boot = True
    if Path(DB_PATH).exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='aqi_readings'"
            ).fetchone()
            if table_exists:
                count = conn.execute("SELECT COUNT(*) FROM aqi_readings").fetchone()[0]
                log.info(f"[retrain] aqi_readings has {count} rows")
                first_boot = count < 100
            conn.close()
        except Exception as e:
            log.warning(f"[retrain] DB check failed: {e}")
            first_boot = True

    if first_boot:
        log.info("First boot — ensuring tables exist...")
        init_db()
        log.info("First boot — running backfill...")
        from backfill import backfill_ward
        from config import WARDS
        total = 0
        for ward in WARDS:
            inserted = backfill_ward(ward)
            total += inserted
            log.info(f"[backfill] {ward['id']}: {inserted} rows inserted")
        log.info(f"[backfill] Total rows inserted: {total}")

        if total > 0:
            log.info("Backfill complete — training models...")
            from models.train import train_all
            train_all()
            log.info("Initial training complete")
        else:
            log.error("[backfill] Zero rows inserted — skipping training. Check API connectivity.")
    else:
        log.info("Data already exists — skipping backfill")

    # Schedule daily retraining at 3AM
    import schedule
    schedule.every().day.at("03:00").do(_retrain_job)

    while True:
        schedule.run_pending()
        time.sleep(60)


def _retrain_job():
    log.info("Scheduled retrain starting...")
    from models.train import train_all
    train_all()
    log.info("Scheduled retrain complete")


# REPLACE everything from "# Small wait..." to the end with this:

if __name__ == "__main__":
    log.info("AirGuardian starting all services...")

    # Step 1: Initialize DB tables FIRST
    init_db()

    # Step 2: Start collector in background thread
    t1 = threading.Thread(target=run_collector, daemon=True)
    t1.start()
    log.info("Collector started")

    # Step 3: Start retrain in background thread
    t2 = threading.Thread(target=run_retrain, daemon=True)
    t2.start()
    log.info("Retrain pipeline started")

    # ❌ REMOVE: time.sleep(10)  ← this was delaying port binding
    # ❌ REMOVE: the manual "import serve" test block

    # Step 4: Start FastAPI IMMEDIATELY (Render needs port bound fast)
    log.info("Starting API server...")
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # ← match Render's default
   uvicorn.run("serve:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), log_level="info")