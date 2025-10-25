import sqlite3
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

def feature_extraction():
    fs = 100  # Hz

    conn = sqlite3.connect("Gait_Database.sqlite")
    cursor = conn.cursor()

    # Patient table names
    cursor.execute("SELECT ID FROM demographics")
    patients = [row[0] for row in cursor.fetchall()]
    n = len(patients)

    # Get all sensor columns from first table
    cursor.execute(f"PRAGMA table_info({patients[0]})")
    all_cols = [row[1] for row in cursor.fetchall()]
    cols = [c for c in all_cols if c.startswith("sensor_") or c.startswith("total_")]

    # Allocate arrays
    features = np.zeros((n, 15), dtype=float)
    classification = np.zeros(n, dtype=int)

    for i, patient in enumerate(patients):
        cursor.execute(f"SELECT {', '.join(cols)} FROM {patient}")
        data = np.array(cursor.fetchall(), dtype=float)
        sensor_data = {col: data[:, j] for j, col in enumerate(cols)}

        total_left, total_right  = sensor_data["total_left"], sensor_data["total_right"]

        steps_left,  _ = find_peaks(total_left,  prominence=400, height=0.5*np.max(total_left))
        steps_right, _ = find_peaks(total_right, prominence=400, height=0.5*np.max(total_right))

        stride_left, stride_right  = np.diff(steps_left)  / fs, np.diff(steps_right) / fs

        mean_stride_L, mean_stride_R = np.mean(stride_left), np.mean(stride_right)

        heel_L = (sensor_data["sensor_l1"] + sensor_data["sensor_l2"] + sensor_data["sensor_l3"]) / 3.0
        toe_L  = (sensor_data["sensor_l6"] + sensor_data["sensor_l7"] + sensor_data["sensor_l8"]) / 3.0
        heel_R = (sensor_data["sensor_r1"] + sensor_data["sensor_r2"] + sensor_data["sensor_r3"]) / 3.0
        toe_R  = (sensor_data["sensor_r6"] + sensor_data["sensor_r7"] + sensor_data["sensor_r8"]) / 3.0

        stance_L, stance_R = (heel_L + toe_L) > 0, (heel_R + toe_R) > 0

        stance_pct_L = [100.0 * np.mean(stance_L[a:b]) for a, b in zip(steps_left[:-1], steps_left[1:])]
        stance_pct_R = [100.0 * np.mean(stance_R[a:b]) for a, b in zip(steps_right[:-1], steps_right[1:])]

        mean_stance_L, mean_stance_R = np.mean(stance_pct_L), np.mean(stance_pct_R)

        idx_R_after_L = np.searchsorted(steps_right, steps_left)
        valid_LR = idx_R_after_L < len(steps_right)
        step_LR = (steps_right[idx_R_after_L[valid_LR]] - steps_left[valid_LR]) / fs

        idx_L_after_R = np.searchsorted(steps_left, steps_right)
        valid_RL = idx_L_after_R < len(steps_left)
        step_RL = (steps_left[idx_L_after_R[valid_RL]] - steps_right[valid_RL]) / fs

        mean_step_LR, mean_step_RL = np.mean(step_LR), np.mean(step_RL)

        step_all = np.concatenate([step_LR, step_RL])

        swing_pct_L = [100.0 * (1.0 - np.mean(stance_L[a:b])) for a, b in zip(steps_left[:-1], steps_left[1:])]
        swing_pct_R = [100.0 * (1.0 - np.mean(stance_R[a:b])) for a, b in zip(steps_right[:-1], steps_right[1:])]

        mean_swing_L, mean_swing_R = np.mean(swing_pct_L), np.mean(swing_pct_R)

        stride_all = np.concatenate([stride_left, stride_right])
        swing_all = np.concatenate([swing_pct_L, swing_pct_R])

        thrL, thrR = 0.02 * np.max(total_left), 0.02 * np.max(total_right)
        contact_L, contact_R = total_left  > thrL, total_right > thrR
        n_min = min(contact_L.size, contact_R.size)

        htr_L = np.mean(heel_L[stance_L]) / np.mean(toe_L[stance_L])
        htr_R = np.mean(heel_R[stance_R]) / np.mean(toe_R[stance_R])

        # Demographics
        query = """SELECT Gender, Age, "Speed_01 (m/sec)", "Height (meters)", "TUAG"
                   FROM demographics WHERE ID = ?"""
        gender, age, speed, height, TUAG = cursor.execute(query, (patient,)).fetchone()
        gender_m = 1.0 if gender.lower() == "male" else 0.0

        features[i, :] = [
            np.std(stride_all, ddof=1) / np.mean(stride_all),                                   # 0 stride_time_var
            np.std(swing_all, ddof=1) / np.mean(swing_all),                                     # 1 swing_time_var
            np.std(step_all, ddof=1) / np.mean(step_all),                                       # 2 step_time_var
            mean_stride_L / mean_stride_R,                                                      # 3 stride_time_asym
    mean_swing_L / mean_swing_R,                                                                # 4 swing_time_asym
            mean_step_LR / mean_step_RL,                                                        # 5 step_time_asym
            mean_stance_L / mean_stance_R,                                                      # 6 stance_time_asym
            np.mean(contact_L[:n_min] & contact_R[:n_min]),                                     # 7 double_support
            0.5 * (htr_L + htr_R),                                                              # 8 heel_toe_ratio
            htr_L / htr_R,                                                                      # 9 heel_toe_asym
            gender_m,                                                                           # 10 Gender (M=1)
            float(age),                                                                         # 11 Age
            float(speed),                                                                       # 12 Speed (m/s)
            float(height),                                                                      # 13 Height (m)
            float(TUAG)                                                                       # 14 Weight (kg)
        ]

        # Label (1=PD, 0=Control)
        classification[i] = 1 if "Pt" in patient else 0

    conn.close()

    feature_names = np.array([
        "Stride Time Var.",
        "Swing Time Var.",
        "Step Time Var.",
        "Stride Time Asym.",
        "Swing Time Asym.",
        "Step Time Asym.",
        "Stance Time Asym.",
        "Double Time Support",
        "Heel-Toe Pressure",
        "Heel-Toe Pressure Asym.",
        "Gender",
        "Age",
        "Speed",
        "Height",
        "Weight"
    ])

    return features, classification, feature_names





def print_feature_stats(features, feature_names):
    for name, col in zip(feature_names, features.T):
        print(f"{name} & "
              f"{np.min(col):.2f} & "
              f"{np.max(col):.2f} & "
              f"{np.mean(col):.2f} & "
              f"{np.std(col, ddof=1):.2f} & "
              f"{np.median(col):.2f}\\\\")


def minmax_scale_features(X_train):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    return X_train_scaled


