def run_collection():
    """Run one full collection cycle across all 4 sources."""
    logger.info("=== Collection cycle started ===")

    # 1. AQICN — every 15 min (real-time AQI changes fast)
    aqi_data = fetch_aqi()
    save("aqi_readings", aqi_data)

    # 2. Open-Meteo — only once per hour (weather changes slowly)
    current_minute = datetime.now().minute
    if current_minute < 15:  # only run in first 15 min of each hour
        weather_data = fetch_weather()
        save("weather_readings", weather_data)
    else:
        logger.info("Skipping weather fetch — runs once per hour only")

    # 3. Sentinel-5P — once per day at 6AM
    if datetime.now().hour == 6 and current_minute < 15:
        satellite_data = fetch_satellite()
        save("satellite_readings", satellite_data)

    # 4. IoT sensors — every cycle
    iot_data = fetch_all_sensors()
    save("iot_readings", iot_data)

    logger.info("=== Collection cycle complete ===")