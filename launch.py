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
    try:
        from ingestion.collector import init_db as collector_init_db
        collector_init_db()
    except Exception as e:
        log.warning(f"[collector] collector init_db failed ({e}), using local init_db")
        init_db()

    try:
        from ingestion.collector import run_collection
        run_collection()  # run immediately on start
        schedule.every(15).minutes.do(run_collection)
    except Exception as e:
        log.error(f"[collector] run_collection failed: {e}")

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


if __name__ == "__main__":
    log.info("AirGuardian starting all services...")

    # Step 1: Initialize DB tables FIRST (before any thread touches the DB)
    init_db()

    # Step 2: Start collector in background thread
    t1 = threading.Thread(target=run_collector, daemon=True)
    t1.start()
    log.info("Collector started")

    # Step 3: Start retrain in background thread
    t2 = threading.Thread(target=run_retrain, daemon=True)
    t2.start()
    log.info("Retrain pipeline started")

    # Small wait to let collector init DB before API starts
    time.sleep(10)

    # Step 4: Start FastAPI in foreground (this blocks)
    log.info("Starting API server...")
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))