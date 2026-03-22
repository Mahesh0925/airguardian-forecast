# models/retrain.py — scheduled retraining of both XGBoost and LSTM
import schedule
import time
import logging
from datetime import datetime

from models.train import train_all as train_xgb_all
from models.lstm_model import train_lstm_ward, save_lstm_models
from models.monitor import (init_monitor_tables, fill_actuals,
                             detect_drift, print_accuracy_report)
from features.engineer import build_features
from config import WARDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Baseline MAEs from initial training (update after first real evaluation)
BASELINE_MAE = {6: 18.0, 12: 25.0, 24: 33.0, 48: 43.0, 72: 49.0}


def retrain_all():
    log.info("=== Scheduled Retraining Started ===")

    # Retrain XGBoost only (LSTM disabled — tensorflow not installed)
    log.info("Retraining XGBoost models...")
    train_all_xgb()

    log.info("=== Retraining Complete ===")

def monitoring_cycle():
    """Run every collection cycle — fill actuals + drift check."""
    fill_actuals()
    for ward in WARDS:
        for horizon in [6, 12, 24, 48, 72]:
            detect_drift(ward["id"], horizon,
                         baseline_mae=BASELINE_MAE.get(horizon, 30.0))


if __name__ == "__main__":
    init_monitor_tables()
    log.info("Layer 5 — Intelligence Pipeline starting...")

    # Run immediately on start
    retrain_all()
    print_accuracy_report()

    # Schedule retraining every 24 hours at 3AM
    schedule.every().day.at("03:00").do(retrain_all)

    # Run monitoring every 15 minutes (same cadence as collector)
    schedule.every(15).minutes.do(monitoring_cycle)

    log.info("Scheduled: retrain daily at 03:00, monitor every 15 min")

    while True:
        schedule.run_pending()
        time.sleep(60)