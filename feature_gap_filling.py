import sqlite3
import numpy as np

conn = sqlite3.connect("Gait_Database.sqlite")
cur = conn.cursor()

# --- pull gender + metrics ---
rows = cur.execute("""
    SELECT
        ID,
        CASE WHEN LOWER(Gender) = 'male' THEN 1 ELSE 0 END AS is_male,
        "Height (meters)",
        "Weight (kg)",
        "Speed_01 (m/sec)"
    FROM demographics
""").fetchall()

arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)

male_mask   = arr[:, 0] == 1
female_mask = arr[:, 0] == 0

# --- compute rounded means ---
means = {
    1: {  # male
        "Height (meters)": round(np.nanmean(arr[male_mask, 1]), 2),
        "Weight (kg)":     round(np.nanmean(arr[male_mask, 2]), 2),
        "Speed_01 (m/sec)":round(np.nanmean(arr[male_mask, 3]), 2)
    },
    0: {  # female
        "Height (meters)": round(np.nanmean(arr[female_mask, 1]), 2),
        "Weight (kg)":     round(np.nanmean(arr[female_mask, 2]), 2),
        "Speed_01 (m/sec)":round(np.nanmean(arr[female_mask, 3]), 2)
    }
}

print("Gender means:", means, "\n--- Filling missing values ---")

# --- iterate and fill gaps ---
for pid, is_male, h, w, s in rows:
    updates = []
    gender_means = means[int(is_male)]

    for col, val in [("Height (meters)", h), ("Weight (kg)", w), ("Speed_01 (m/sec)", s)]:
        if val is None:
            cur.execute(f'UPDATE demographics SET "{col}"=? WHERE ID=?',
                        (gender_means[col], pid))
            updates.append(f"{col.split()[0].lower()}={gender_means[col]}")

    if updates:
        print(f"Filled {pid}: {', '.join(updates)}")

conn.commit()
conn.close()
print("\n✅ Finished filling missing values (rounded to 2 decimals).")
