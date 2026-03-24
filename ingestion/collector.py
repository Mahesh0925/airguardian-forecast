import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_collection():
    """Run one full collection cycle across all 4 sources."""
    logger.info("=== Collection cycle started ===")

    try:
        # 1. AQICN — every 15 min
        try:
            aqi_data = fetch_aqi()
            save("aqi_readings", aqi_data)
        except Exception as e:
            logger.error(f"AQI fetch failed: {e}")

        # 2. Weather — once per hour
        current_time = datetime.now()
        if current_time.minute < 15:
            try:
                weather_data = fetch_weather()
                save("weather_readings", weather_data)
            except Exception as e:
                logger.error(f"Weather fetch failed: {e}")
        else:
            logger.info("Skipping weather fetch — runs once per hour only")

        # 3. Satellite — once per day at 6 AM
        if current_time.hour == 6 and current_time.minute < 15:
            try:
                satellite_data = fetch_satellite()
                save("satellite_readings", satellite_data)
            except Exception as e:
                logger.error(f"Satellite fetch failed: {e}")

        # 4. IoT — always
        try:
            iot_data = fetch_all_sensors()
            save("iot_readings", iot_data)
        except Exception as e:
            logger.error(f"IoT fetch failed: {e}")

    except Exception as e:
        logger.error(f"run_collection failed: {e}")

    logger.info("=== Collection cycle complete ===")