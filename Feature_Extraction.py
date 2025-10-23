import sqlite3
import numpy as np
from scipy.signal import find_peaks

def feature_extraction():
    fs = 100
    features = np.eye(80, 8)
    classification = np.arange(80)

    conn = sqlite3.connect("Gait_Database.sqlite")
    cursor = conn.cursor()

    cursor.execute("SELECT ID FROM demographics")
    patients = [row[0] for row in cursor.fetchall()]

    cursor.execute(f"PRAGMA table_info({patients[0]})")
    all_cols = [row[1] for row in cursor.fetchall()]
    cols = [c for c in all_cols if c.startswith("sensor_") or c.startswith("total_")]

    for i, patient in enumerate(patients):
        cursor.execute(f"SELECT {', '.join(cols)} FROM {patient}")
        data = np.array(cursor.fetchall(), dtype=float)

        sensor_data = {col: data[:, j] for j, col in enumerate(cols)}

        total_left  = sensor_data["total_left"]
        total_right = sensor_data["total_right"]

        steps_left,  _ = find_peaks(total_left,  prominence=400, height=0.5*np.max(total_left))
        steps_right, _ = find_peaks(total_right, prominence=400, height=0.5*np.max(total_right))
        stride_left  = np.diff(steps_left)  / fs
        stride_right = np.diff(steps_right) / fs

        avg_stride_left  = np.mean(stride_left)
        avg_stride_right = np.mean(stride_right)

        features[i, 0] = len(steps_left) + len(steps_right)
        features[i, 1] = np.nanmean([avg_stride_left, avg_stride_right])
        features[i, 2] = abs(avg_stride_left - avg_stride_right)
        features[i, 3] = 100 * np.std(stride_left)  / avg_stride_left
        features[i, 4] = 100 * np.std(stride_right) / avg_stride_right

        max_total_left  = np.max(total_left)
        max_total_right = np.max(total_right)
        double_support = np.mean((total_left > 0.05 * max_total_left) & (total_right > 0.05 * max_total_right))
        features[i, 5] = double_support
        # hi
        heel_L = (sensor_data["sensor_l1"] + sensor_data["sensor_l2"] + sensor_data["sensor_l3"]) / 3.0
        toe_L  = (sensor_data["sensor_l6"] + sensor_data["sensor_l7"] + sensor_data["sensor_l8"]) / 3.0
        heel_R = (sensor_data["sensor_r1"] + sensor_data["sensor_r2"] + sensor_data["sensor_r3"]) / 3.0
        toe_R  = (sensor_data["sensor_r6"] + sensor_data["sensor_r7"] + sensor_data["sensor_r8"]) / 3.0

        stance_L = (heel_L + toe_L) > 0
        stance_R = (heel_R + toe_R) > 0

        heel_toe_ratio_l = np.mean(heel_L[stance_L]) / np.mean(toe_L[stance_L])
        heel_toe_ratio_r = np.mean(heel_R[stance_R]) / np.mean(toe_R[stance_R])
        features[i, 6] = heel_toe_ratio_l
        features[i, 7] = heel_toe_ratio_r

        classification[i] = 1 if "Pt" in patient else 0

    conn.close()
    return features, classification



