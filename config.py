import os
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
GEE_PROJECT  = os.getenv("GEE_PROJECT")

# Use /data on Render (persistent disk), local storage/ in development
_default_db = "/data/airguardian.db" if os.path.isdir("/data") else "storage/airguardian.db"
DB_PATH = os.getenv("DB_PATH", _default_db)

# Auto-create parent dir if it doesn't exist
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

import logging as _logging
_logging.getLogger(__name__).info(f"[config] DB_PATH = {DB_PATH}")

WARDS = [
    {"id": "wazirpur",    "name": "Wazirpur",         "aqicn": "@7021",       "lat": 28.6942, "lon": 77.1630},
    {"id": "shahdara",    "name": "Shahdara",          "aqicn": "@8684",       "lat": 28.6726, "lon": 77.2944},
    {"id": "rohini",      "name": "Rohini Sector 36",  "aqicn": "rohini",      "lat": 28.7495, "lon": 77.0590},
    {"id": "dwarka",      "name": "Dwarka Sector 12",  "aqicn": "@9025",       "lat": 28.5921, "lon": 77.0460},
    {"id": "saket",       "name": "Saket",             "aqicn": "@9550",       "lat": 28.5245, "lon": 77.2066},
    {"id": "okhla",       "name": "Okhla",             "aqicn": "@9551",       "lat": 28.5355, "lon": 77.2730},
    {"id": "karolbagh",   "name": "Karol Bagh",        "aqicn": "karol-bagh",  "lat": 28.6514, "lon": 77.1908},
    {"id": "mehrauli",    "name": "Mehrauli",          "aqicn": "@9024",       "lat": 28.5245, "lon": 77.1855},
    {"id": "pitampura",   "name": "Pitampura",         "aqicn": "@9549",       "lat": 28.7020, "lon": 77.1310},
    {"id": "lajpatnagar", "name": "Lajpat Nagar",      "aqicn": "@9548",       "lat": 28.5677, "lon": 77.2437},
]

FETCH_INTERVAL_MINUTES = 15