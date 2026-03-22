# ingestion/sentinel.py — daily Sentinel-5P NO2 from Google Earth Engine
import ee
import logging
from datetime import datetime, timezone, timedelta
from config import GEE_PROJECT, WARDS

logger = logging.getLogger(__name__)

def init_gee():
    """Authenticate and init GEE — run earthengine authenticate once first."""
    try:
        ee.Initialize(project=GEE_PROJECT)
        logger.info("Google Earth Engine initialized")
    except Exception as e:
        logger.error(f"GEE init failed: {e}")
        raise


def fetch_ward_no2(ward: dict, date: str = None) -> dict | None:
    """
    Fetch daily average NO2 column for a ward from Sentinel-5P.
    date format: 'YYYY-MM-DD'. Defaults to yesterday (latest available).
    """
    if date is None:
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime("%Y-%m-%d")

    next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        # Sentinel-5P NRTI NO2 collection
        collection = (
            ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
            .filterDate(date, next_day)
            .select("NO2_column_number_density")
        )

        # Point for the ward
        point = ee.Geometry.Point([ward["lon"], ward["lat"]])

        # 3km buffer around the ward centre
        region = point.buffer(3000)

        # Reduce to mean value in the region
        image = collection.mean()
        result = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            maxPixels=1e6,
        ).getInfo()

        no2_val = result.get("NO2_column_number_density")

        return {
            "ward_id":   ward["id"],
            "date":      date,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "no2_tropospheric_column": round(no2_val * 1e6, 4) if no2_val else None,
            # Unit: µmol/m² (multiplied from mol/m²)
            "source": "sentinel-5p",
        }

    except Exception as e:
        logger.error(f"Sentinel fetch failed for {ward['name']}: {e}")
        return None


def fetch_all_wards(date: str = None) -> list[dict]:
    init_gee()
    results = []
    for ward in WARDS:
        reading = fetch_ward_no2(ward, date)
        if reading:
            results.append(reading)
            logger.info(
                f"NO2 {ward['name']}: "
                f"{reading['no2_tropospheric_column']} µmol/m²"
            )
    return results