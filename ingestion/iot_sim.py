# ingestion/iot_sim.py — IoT sensor simulator
# Replace read_sensor() with real serial/MQTT read when hardware is ready
import random
import logging
from datetime import datetime, timezone
from config import WARDS

logger = logging.getLogger(__name__)

# Simulated sensors — map ward_id to sensor hardware ID
SENSORS = {
    "wazirpur":    ["S-01", "S-14"],
    "shahdara":    ["S-05", "S-06"],
    "rohini":      ["S-08", "S-09", "S-20"],
    "dwarka":      ["S-11", "S-12"],
    "saket":       ["S-17", "S-18", "S-31"],
    "okhla":       ["S-40", "S-41", "S-42"],
    "karolbagh":   ["S-25", "S-26"],
    "mehrauli":    ["S-30", "S-31", "S-32"],
    "pitampura":   ["S-35", "S-36"],
    "lajpatnagar": ["S-22", "S-23"],
}

def read_sensor(sensor_id: str, ward: dict) -> dict:
    """
    Simulate a sensor reading.
    TODO: replace this with actual MQTT/serial read:
        import serial
        ser = serial.Serial('/dev/ttyUSB0', 9600)
        raw = ser.readline().decode().strip()
        pm25, pm10 = parse_raw(raw)
    """
    # Simulate realistic Delhi AQI range with noise
    base_pm25 = random.uniform(60, 200)
    return {
        "sensor_id":  sensor_id,
        "ward_id":    ward["id"],
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "pm25":       round(base_pm25 + random.gauss(0, 8), 2),
        "pm10":       round(base_pm25 * 1.7 + random.gauss(0, 12), 2),
        "temperature":round(random.uniform(18, 42), 2),
        "humidity":   round(random.uniform(30, 85), 2),
        "battery_pct":round(random.uniform(15, 100), 1),
        "status":     "online",
        "source":     "iot_sensor",
    }


def fetch_all_sensors() -> list[dict]:
    readings = []
    ward_map = {w["id"]: w for w in WARDS}
    for ward_id, sensor_ids in SENSORS.items():
        ward = ward_map.get(ward_id)
        if not ward:
            continue
        for sid in sensor_ids:
            r = read_sensor(sid, ward)
            readings.append(r)
            logger.info(f"IoT {sid} ({ward['name']}): PM2.5={r['pm25']}")
    return readings