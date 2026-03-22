# models/monitor.py — accuracy logging, drift detection, alert triggers
import sqlite3
import json
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from config import DB_PATH, WARDS

ALERT_THRESHOLDS = {
    "unhealthy":          150,
    "very_unhealthy":     200,
    "hazardous":          300,
}

def init_monitor_tables():
    """Create monitoring tables in the existing DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS forecast_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ward_id TEXT, horizon_hours INTEGER,
        predicted_aqi REAL, actual_aqi REAL,
        absolute_error REAL,
        model_used TEXT,
        logged_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ward_id TEXT, ward_name TEXT,
        alert_type TEXT, threshold INTEGER,
        forecast_aqi REAL, horizon_hours INTEGER,
        triggered_at TEXT, acknowledged INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS drift_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ward_id TEXT, horizon_hours INTEGER,
        rolling_mae REAL, baseline_mae REAL,
        drift_detected INTEGER,
        logged_at TEXT
    )""")
    conn.commit()
    conn.close()


def log_forecast(ward_id: str, horizon: int,
                 predicted: float, model_used: str):
    """Store a forecast prediction — actual gets filled in later."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO forecast_log
        (ward_id, horizon_hours, predicted_aqi, model_used, logged_at)
        VALUES (?, ?, ?, ?, ?)""",
        (ward_id, horizon, predicted, model_used,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def fill_actuals():
    """
    Match logged forecasts with real AQI readings that have now come in.
    Called on every collection cycle.
    """
    conn = sqlite3.connect(DB_PATH)

    # Find forecasts that have no actual yet but whose target time has passed
    pending = conn.execute("""
        SELECT id, ward_id, horizon_hours, predicted_aqi, logged_at
        FROM forecast_log
        WHERE actual_aqi IS NULL
    """).fetchall()

    filled = 0
    for row_id, ward_id, horizon, predicted, logged_at in pending:
        logged_dt = datetime.fromisoformat(logged_at)
        target_dt = logged_dt + timedelta(hours=horizon)

        if datetime.now(timezone.utc) < target_dt:
            continue  # target time hasn't arrived yet

        # Find closest actual reading to target time
        actual = conn.execute("""
            SELECT aqi FROM aqi_readings
            WHERE ward_id = ?
              AND ABS(JULIANDAY(timestamp) - JULIANDAY(?)) < 0.042
              AND aqi IS NOT NULL
            ORDER BY ABS(JULIANDAY(timestamp) - JULIANDAY(?))
            LIMIT 1
        """, (ward_id, target_dt.isoformat(), target_dt.isoformat())).fetchone()

        if actual:
            error = abs(predicted - float(actual[0]))
            conn.execute("""UPDATE forecast_log
                SET actual_aqi = ?, absolute_error = ?
                WHERE id = ?""",
                (float(actual[0]), error, row_id))
            filled += 1

    conn.commit()
    conn.close()
    if filled:
        print(f"  Monitor: filled {filled} actuals")


def compute_rolling_mae(ward_id: str, horizon: int,
                        window: int = 48) -> float | None:
    """Compute rolling MAE over last `window` matched forecasts."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT absolute_error FROM forecast_log
        WHERE ward_id = ? AND horizon_hours = ?
          AND absolute_error IS NOT NULL
        ORDER BY logged_at DESC
        LIMIT ?
    """, (ward_id, horizon, window)).fetchall()
    conn.close()

    if len(rows) < 5:
        return None
    return float(np.mean([r[0] for r in rows]))


def detect_drift(ward_id: str, horizon: int,
                 baseline_mae: float, threshold_pct: float = 0.30):
    """
    Drift = rolling MAE has grown > threshold_pct above baseline.
    Logs result and prints warning.
    """
    rolling = compute_rolling_mae(ward_id, horizon)
    if rolling is None:
        return

    drift = rolling > baseline_mae * (1 + threshold_pct)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO drift_log
        (ward_id, horizon_hours, rolling_mae, baseline_mae,
         drift_detected, logged_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (ward_id, horizon, rolling, baseline_mae,
         int(drift), datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

    if drift:
        print(f"  DRIFT DETECTED {ward_id} +{horizon}h: "
              f"rolling MAE={rolling:.1f} vs baseline={baseline_mae:.1f}")


def check_alerts(ward_id: str, ward_name: str, forecasts: dict):
    """
    Trigger ONE alert per forecast point — only at the highest threshold breached.
    """
    conn = sqlite3.connect(DB_PATH)

    for horizon, data in forecasts.items():
        aqi = data["aqi"]

        # Find the HIGHEST threshold breached only
        triggered_type = None
        triggered_threshold = None
        for alert_type, threshold in sorted(
            ALERT_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if aqi >= threshold:
                triggered_type = alert_type
                triggered_threshold = threshold
                break  # stop at highest match

        if not triggered_type:
            continue

        # Deduplicate — skip if same alert fired in last 6h
        recent = conn.execute("""
            SELECT id FROM alerts
            WHERE ward_id = ? AND alert_type = ?
              AND horizon_hours = ?
              AND JULIANDAY('now') - JULIANDAY(triggered_at) < 0.25
        """, (ward_id, triggered_type, horizon)).fetchone()

        if recent:
            continue

        conn.execute("""INSERT INTO alerts
            (ward_id, ward_name, alert_type, threshold,
             forecast_aqi, horizon_hours, triggered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ward_id, ward_name, triggered_type, triggered_threshold,
             aqi, horizon, datetime.now(timezone.utc).isoformat()))

        print(f"  ALERT [{triggered_type.upper()}] {ward_name} "
              f"+{horizon}h: AQI {aqi} >= {triggered_threshold}")

    conn.commit()
    conn.close()

def get_active_alerts() -> list:
    """Return all unacknowledged alerts from last 24 hours."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT ward_id, ward_name, alert_type, threshold,
               forecast_aqi, horizon_hours, triggered_at
        FROM alerts
        WHERE acknowledged = 0
          AND JULIANDAY('now') - JULIANDAY(triggered_at) < 1.0
        ORDER BY triggered_at DESC
    """).fetchall()
    conn.close()
    return [
        {
            "ward_id":      r[0], "ward_name":    r[1],
            "alert_type":   r[2], "threshold":    r[3],
            "forecast_aqi": r[4], "horizon_hours":r[5],
            "triggered_at": r[6],
        }
        for r in rows
    ]


def print_accuracy_report():
    """Print a quick accuracy summary across all wards and horizons."""
    conn = sqlite3.connect(DB_PATH)
    print("\n=== Model Accuracy Report ===")
    for horizon in [6, 12, 24, 48, 72]:
        rows = conn.execute("""
            SELECT ward_id, AVG(absolute_error), COUNT(*)
            FROM forecast_log
            WHERE horizon_hours = ? AND absolute_error IS NOT NULL
            GROUP BY ward_id
        """, (horizon,)).fetchall()
        if rows:
            avg_mae = np.mean([r[1] for r in rows])
            print(f"  +{horizon}h avg MAE: {avg_mae:.1f} AQI  "
                  f"({rows[0][2]} samples per ward)")
    conn.close()
    print()