import sqlite3

conn = sqlite3.connect("storage/airguardian.db")

print("=== Active Alerts ===\n")
rows = conn.execute("""
    SELECT ward_name, alert_type, forecast_aqi,
           horizon_hours, triggered_at
    FROM alerts
    ORDER BY triggered_at DESC
    LIMIT 20
""").fetchall()

for r in rows:
    ward, atype, aqi, horizon, ts = r
    print(f"  [{atype.upper():20}] {ward:20} +{horizon}h → AQI {aqi:.0f}  ({ts[:16]})")

print(f"\nTotal alerts in DB: {conn.execute('SELECT COUNT(*) FROM alerts').fetchone()[0]}")

print("\n=== Alert Summary by Ward ===\n")
summary = conn.execute("""
    SELECT ward_name, alert_type, COUNT(*) as cnt
    FROM alerts
    GROUP BY ward_name, alert_type
    ORDER BY cnt DESC
""").fetchall()
for r in summary:
    print(f"  {r[0]:20} {r[1]:25} × {r[2]}")

conn.close()