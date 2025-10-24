import os
import sqlite3
import pandas as pd

files = os.listdir("./Data/RawData/")
patients = [os.path.splitext(f)[0].replace("_01", "") for f in files]
# path to your text file
xls_file = "./Data/demographics.xls"   # <-- replace with the actual filename
db_file = "Gait_Database.sqlite"

df = pd.read_excel(xls_file)
first_col = df.columns[0]
df = df[df[first_col].isin(patients)]

# --- 4) Write to SQLite ---
conn = sqlite3.connect(db_file)
# Use a fixed schema/table name
df.to_sql("demographics", conn, if_exists="replace", index=False)

# Quick sanity check
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM demographics;")
print("Rows inserted:", cur.fetchone()[0])

# ==== USER SETTINGS ====
for i, file in enumerate(files):
    headers = [
        "time", "sensor_l1", "sensor_l2", "sensor_l3", "sensor_l4", "sensor_l5",
        "sensor_l6", "sensor_l7", "sensor_l8", "sensor_r1", "sensor_r2", "sensor_r3",
        "sensor_r4", "sensor_r5", "sensor_r6", "sensor_r7", "sensor_r8", "total_left", "total_right"
    ]

    df = pd.read_csv(f"./Data/RawData/{file}", sep="\t", header=None)
    df.columns = headers

    df.to_sql(patients[i], conn, if_exists="replace", index=False)
    print(f"Imported {len(df)} rows into table '{patients[i]}'")
conn.close()


