# run.py — entry point, runs collection + forecast logging every 15 minutes
import schedule
import time
import logging
from datetime import datetime, timezone

from ingestion.collector import init_db, run_collection
from models.monitor import init_monitor_tables, fill_actuals, check_alerts
from models.ensemble import ensemble_predict
from models.monitor import log_forecast
from config import WARDS, FETCH_INTERVAL_MINUTES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


def log_current_forecasts():
    """After each collection cycle, run ensemble and log predictions."""
    for ward in WARDS:
        try:
            result = ensemble_predict(ward["id"])
            if not result:
                continue
            for horizon, data in result["horizons"].items():
                log_forecast(
                    ward_id    = ward["id"],
                    horizon    = horizon,
                    predicted  = data["aqi"],
                    model_used = data["model_used"],
                )
        except Exception as e:
            log.warning(f"Could not log forecast for {ward['name']}: {e}")


def check_forecast_alerts():
    """Check if any forecast exceeds alert thresholds."""
    for ward in WARDS:
        try:
            result = ensemble_predict(ward["id"])
            if not result:
                continue
            check_alerts(ward["id"], ward["name"], result["horizons"])
        except Exception as e:
            log.warning(f"Alert check failed for {ward['name']}: {e}")


def full_cycle():
    """One complete cycle: collect → forecast → log → alerts → fill actuals."""
    log.info("=== Cycle started ===")
    run_collection()
    log_current_forecasts()
    check_forecast_alerts()
    fill_actuals()
    log.info("=== Cycle complete ===")


if __name__ == "__main__":
    log.info("AirGuardian pipeline starting...")
    init_db()
    init_monitor_tables()

    full_cycle()  # run immediately on start

    schedule.every(FETCH_INTERVAL_MINUTES).minutes.do(full_cycle)
    log.info(f"Scheduled every {FETCH_INTERVAL_MINUTES} minutes")

    while True:
        schedule.run_pending()
        time.sleep(30)