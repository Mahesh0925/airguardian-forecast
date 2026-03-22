# Add this to check.py and run once
import sqlite3
conn = sqlite3.connect("storage/airguardian.db")

# Keep only the highest-severity alert per ward+horizon combination
conn.execute("""
    DELETE FROM alerts
    WHERE id NOT IN (
        SELECT MIN(id) FROM alerts
        GROUP BY ward_id, horizon_hours,
        DATE(triggered_at), CAST(JULIANDAY(triggered_at)*4 AS INT)
    )
""")
deleted = conn.total_changes
conn.commit()
conn.close()
print(f"Cleaned {deleted} duplicate alerts")


## Full pipeline status summary
