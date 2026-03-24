# startup.py — runs on first boot to seed data and train models
import os
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_default_db = "/data/airguardian.db" if os.path.isdir("/data") else "storage/airguardian.db"
DB_PATH = os.getenv("DB_PATH", _default_db)
log.info(f"[startup] DB_PATH = {DB_PATH}")

_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)


def init_tables():
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
    log.info(f"[startup] Tables: {tables}")
    conn.close()


def is_first_boot() -> bool:
    """Check if DB has enough data to skip seeding."""
    if not Path(DB_PATH).exists():
        return True
    conn = sqlite3.connect(DB_PATH)
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='aqi_readings'"
    ).fetchone()
    count = 0
    if table_exists:
        count = conn.execute("SELECT COUNT(*) FROM aqi_readings").fetchone()[0]
    conn.close()
    log.info(f"[startup] aqi_readings row count = {count}")
    return count < 100


if __name__ == "__main__":
    # Step 1: Always create tables first
    init_tables()

    if is_first_boot():
        log.info("First boot detected — seeding data...")

        from backfill import backfill_ward
        from config import WARDS
        total = 0
        for ward in WARDS:
            inserted = backfill_ward(ward)
            total += inserted
            log.info(f"  {ward['id']}: {inserted} rows")
        log.info(f"Total rows inserted: {total}")

        if total > 0:
            from models.train import train_all
            train_all()
            log.info("Startup complete — API ready")
        else:
            log.error("Zero rows inserted — check API connectivity. Skipping training.")
    else:
        log.info("Data exists — skipping seed")
