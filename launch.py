# launch.py — runs collector + retrain in background, API in foreground
import threading
import time
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Set DB path to local storage (inside container)
os.environ.setdefault("DB_PATH", "storage/airguardian.db")
os.makedirs("storage", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)
os.makedirs("models/saved_lstm", exist_ok=True)


def run_collector():
    """Runs data collection every 15 minutes forever."""
    import schedule
    from ingestion.collector import init_db, run_collection
    init_db()
    run_collection()  # run immediately on start
    schedule.every(15).minutes.do(run_collection)
    while True:
        schedule.run_pending()
        time.sleep(30)


def run_retrain():
    """Runs backfill on first boot then retrains daily at 3AM."""
    import sqlite3
    from pathlib import Path

    db = os.environ.get("DB_PATH", "storage/airguardian.db")

    # Wait for collector to seed some data first
    time.sleep(60)

    # Check if first boot
    first_boot = True
    if Path(db).exists():
        try:
            conn = sqlite3.connect(db)
            count = conn.execute(
                "SELECT COUNT(*) FROM aqi_readings"
            ).fetchone()[0]
            conn.close()
            first_boot = count < 100
        except Exception:
            first_boot = True

    if first_boot:
        log.info("First boot — running backfill...")
        from backfill import backfill_ward
        from config import WARDS
        for ward in WARDS:
            backfill_ward(ward)
        log.info("Backfill complete — training models...")
        from models.train import train_all
        train_all()
        log.info("Initial training complete")

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

    # Start collector in background thread
    t1 = threading.Thread(target=run_collector, daemon=True)
    t1.start()
    log.info("Collector started")

    # Start retrain in background thread
    t2 = threading.Thread(target=run_retrain, daemon=True)
    t2.start()
    log.info("Retrain pipeline started")

    # Small wait to let collector init DB before API starts
    time.sleep(10)

    # Start FastAPI in foreground (this blocks)
    log.info("Starting API server...")
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))