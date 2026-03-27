# seed_csv.py — loads the competition training CSV into the DB
import sqlite3, os, sys
import pandas as pd
from config import DB_PATH

CSV_PATH = os.environ.get("SEED_CSV", "data/delhi_aqi_training_data.csv")

if not os.path.exists(CSV_PATH):
    print(f"❌ CSV not found at {CSV_PATH}. Upload it first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# Only keep columns that match aqi_readings table
COLS = ["timestamp", "ward_id", "ward_name", "aqi", "pm25", "pm10",
        "no2", "so2", "co", "o3", "source"]
df = df[COLS]

conn = sqlite3.connect(DB_PATH)
conn.execute("""CREATE TABLE IF NOT EXISTS aqi_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, ward_id TEXT, ward_name TEXT,
    aqi REAL, pm25 REAL, pm10 REAL, no2 REAL,
    so2 REAL, co REAL, o3 REAL, source TEXT
)""")
conn.commit()

df.to_sql("aqi_readings", conn, if_exists="append", index=False)
count = conn.execute("SELECT COUNT(*) FROM aqi_readings").fetchone()[0]
conn.close()

print(f"✅ Seeded {len(df):,} rows. Total in DB: {count:,}")
