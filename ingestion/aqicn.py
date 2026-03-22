# ingestion/aqicn.py — fetches real-time AQI from AQICN for each ward
import requests
import logging
from datetime import datetime, timezone
from config import AQICN_TOKEN, WARDS

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waqi.info/feed"

def fetch_ward_aqi(ward: dict) -> dict | None:
    """Fetch current AQI — tries config name first, falls back to geo search."""
    slug = ward["aqicn"]
    result = _fetch_by_slug(slug, ward)

    # If name-based lookup fails, try finding nearest station by coordinates
    if result is None:
        logger.info(f"Trying geo fallback for {ward['name']}...")
        slug = find_nearest_station(ward)
        if slug:
            result = _fetch_by_slug(slug, ward)

    return result


def _fetch_by_slug(slug: str, ward: dict) -> dict | None:
    """Internal: fetch AQI by a specific slug or @uid."""
    url = f"{BASE_URL}/{slug}/?token={AQICN_TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            return None
        d = data["data"]
        iaqi = d.get("iaqi", {})
        return {
            "ward_id":   ward["id"],
            "ward_name": ward["name"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
           "aqi": float(d.get("aqi")) if str(d.get("aqi", "")).lstrip("-").isdigit() else None,
            "pm25":      iaqi.get("pm25", {}).get("v"),
            "pm10":      iaqi.get("pm10", {}).get("v"),
            "no2":       iaqi.get("no2",  {}).get("v"),
            "so2":       iaqi.get("so2",  {}).get("v"),
            "co":        iaqi.get("co",   {}).get("v"),
            "o3":        iaqi.get("o3",   {}).get("v"),
            "source":    "aqicn",
        }
    except Exception:
        return None

def fetch_all_wards() -> list[dict]:
    """Fetch AQI for all configured wards."""
    results = []
    for ward in WARDS:
        reading = fetch_ward_aqi(ward)
        if reading:
            results.append(reading)
            logger.info(f"AQI {ward['name']}: {reading['aqi']}")
    return results

# Add this function to ingestion/aqicn.py

def find_nearest_station(ward: dict) -> str | None:
    """Find the nearest AQICN station to a ward using geo search."""
    url = (
        f"https://api.waqi.info/map/bounds/"
        f"?latlng={ward['lat']-0.1},{ward['lon']-0.1},"
        f"{ward['lat']+0.1},{ward['lon']+0.1}"
        f"&token={AQICN_TOKEN}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") != "ok" or not data["data"]:
            return None
        # Pick station with highest AQI reading (most active)
        stations = sorted(data["data"], key=lambda x: x.get("aqi", 0), reverse=True)
        uid = stations[0]["uid"]
        logger.info(f"Found station uid={uid} near {ward['name']}")
        return f"@{uid}"
    except Exception as e:
        logger.error(f"Geo search failed for {ward['name']}: {e}")
        return None