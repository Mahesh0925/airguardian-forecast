# backfill.py — pulls 60 days of real hourly AQI history from Open-Meteo Air Quality API
import requests
import sqlite3
from datetime import datetime, timedelta
from config import WARDS, DB_PATH

def backfill_ward(ward: dict):
    """
    Fetch 60 days of hourly historical air quality from Open-Meteo.
    Completely free, no API key needed.
    """
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":        ward["lat"],
        "longitude":       ward["lon"],
        "hourly":          ["pm2_5", "pm10", "nitrogen_dioxide",
                            "sulphur_dioxide", "carbon_monoxide", "ozone",
                            "european_aqi"],
        "start_date":      start,
        "end_date":        end,
        "timezone":        "Asia/Kolkata",
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})

        times  = hourly.get("time", [])
        aqi    = hourly.get("european_aqi", [])
        pm25   = hourly.get("pm2_5", [])
        pm10   = hourly.get("pm10", [])
        no2    = hourly.get("nitrogen_dioxide", [])
        so2    = hourly.get("sulphur_dioxide", [])
        co     = hourly.get("carbon_monoxide", [])
        o3     = hourly.get("ozone", [])

        rows = []
        for i, t in enumerate(times):
            aqi_val = aqi[i] if i < len(aqi) else None
            if aqi_val is None:
                continue
            rows.append({
                "ward_id":   ward["id"],
                "ward_name": ward["name"],
                "timestamp": t + ":00+05:30",
                "aqi":       float(aqi_val),
                "pm25":      float(pm25[i])  if i < len(pm25)  and pm25[i]  is not None else None,
                "pm10":      float(pm10[i])  if i < len(pm10)  and pm10[i]  is not None else None,
                "no2":       float(no2[i])   if i < len(no2)   and no2[i]   is not None else None,
                "so2":       float(so2[i])   if i < len(so2)   and so2[i]   is not None else None,
                "co":        float(co[i])    if i < len(co)    and co[i]    is not None else None,
                "o3":        float(o3[i])    if i < len(o3)    and o3[i]    is not None else None,
                "source":    "openmeteo_backfill",
            })

        if not rows:
            print(f"  {ward['name']}: 0 rows returned from API")
            return 0

        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Ensure table exists before inserting
        conn.execute("""CREATE TABLE IF NOT EXISTS aqi_readings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ward_id   TEXT,
            ward_name TEXT,
            aqi       REAL,
            pm25      REAL,
            pm10      REAL,
            no2       REAL,
            so2       REAL,
            co        REAL,
            o3        REAL,
            source    TEXT
        )""")
        conn.commit()
        inserted = 0
        skipped  = 0
        for row in rows:
            cols         = ", ".join(row.keys())
            placeholders = ", ".join(["?"] * len(row))
            try:
                conn.execute(
                    f"INSERT INTO aqi_readings ({cols}) VALUES ({placeholders})",
                    list(row.values())
                )
                inserted += 1
            except Exception:
                skipped += 1
                continue
        conn.commit()
        conn.close()
        print(f"  {ward['name']}: inserted {inserted} rows (skipped {skipped})")
        return inserted

    except Exception as e:
        print(f"  {ward['name']}: error — {e}")
        return 0


if __name__ == "__main__":
    print("Backfilling 60 days of real hourly AQI data from Open-Meteo...\n")
    total = 0
    for ward in WARDS:
        total += backfill_ward(ward)
    print(f"\nTotal rows inserted: {total}")
    print("Now run: python -m features.engineer && python -m models.train")