# startup.py — runs on first boot to seed data and train models
import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "storage/airguardian.db")

def is_first_boot() -> bool:
    """Check if DB exists and has data."""
    if not Path(DB_PATH).exists():
        return True
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(
        "SELECT COUNT(*) FROM aqi_readings"
    ).fetchone()[0] if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='aqi_readings'"
    ).fetchone() else 0
    conn.close()
    return count < 100


if __name__ == "__main__":
    if is_first_boot():
        print("First boot detected — seeding data...")
        from ingestion.collector import init_db
        init_db()

        from backfill import backfill_ward
        from config import WARDS
        for ward in WARDS:
            backfill_ward(ward)

        from models.train import train_all
        train_all()
        print("Startup complete — API ready")
    else:
        print("Data exists — skipping seed")