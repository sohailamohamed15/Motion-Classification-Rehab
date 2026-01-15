import joblib
import pandas as pd
import numpy as np

# =============================
# Load trained artifacts
# =============================
model = joblib.load("best_mpu_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_list.joblib")

# =============================
# Parameters (MATCH TRAINING)
# =============================
WINDOW_SIZE = 5
STEP = 2
sensor_cols = ['Pitch', 'Roll', 'AccelX', 'AccelY', 'AccelZ']

# =============================
# MANUAL INPUT (EDIT FREELY)
# =============================

movement_name = "Stretching"   # Stretching / Lift up

manual_input = {
    "Shoulder": [
        [-0.25, -0.01, -0.02,  0.01, 9.78],
        [ 0.30, -0.05, -0.01,  0.01, 9.76],
        [ 0.40, -0.06, -0.02,  0.02, 9.77],
        [ 0.35, -0.04, -0.01,  0.01, 9.78],
        [ 0.45, -0.07, -0.02,  0.03, 9.79],
    ],

    "Elbow": [
        [ 0.91, -0.17, -0.03, 0.00, 9.77],
        [ 0.57, -0.12, -0.02, 0.03, 9.75],
        [ 0.71, -0.11, -0.04, 0.05, 9.72],
        [ 0.25, -0.07, -0.02, 0.02, 9.77],
        [ 0.30, -0.05, -0.01, 0.01, 9.76],
    ],

    "Wrist": [
        [0.20, -0.03, -0.01, 0.01, 9.74],
        [0.25, -0.04, -0.02, 0.02, 9.75],
        [0.30, -0.05, -0.01, 0.01, 9.76],
        [0.28, -0.04, -0.02, 0.02, 9.77],
        [0.35, -0.06, -0.03, 0.03, 9.78],
    ],

    "Hand": [
        [0.10, -0.02, -0.01, 0.00, 9.73],
        [0.15, -0.03, -0.01, 0.01, 9.74],
        [0.18, -0.04, -0.02, 0.02, 9.75],
        [0.20, -0.03, -0.01, 0.01, 9.76],
        [0.22, -0.04, -0.02, 0.02, 9.77],
    ]
}

# =============================
# Feature Extraction
# =============================
features = []

for joint_label, readings in manual_input.items():

    df_joint = pd.DataFrame(readings, columns=sensor_cols)

    for i in range(0, len(df_joint) - WINDOW_SIZE + 1, STEP):
        window = df_joint.iloc[i:i + WINDOW_SIZE]

        feat = {}
        for col in sensor_cols:
            feat[f"{col}_mean"] = window[col].mean()
            feat[f"{col}_std"]  = window[col].std()
            feat[f"{col}_max"]  = window[col].max()
            feat[f"{col}_min"]  = window[col].min()

        # Manual categorical encoding
        for fname in feature_names:
            if fname.startswith("Movement_Name_"):
                feat[fname] = int(fname == f"Movement_Name_{movement_name}")

            if fname.startswith("Joint_Label_"):
                feat[fname] = int(fname == f"Joint_Label_{joint_label}")

        features.append(feat)

# =============================
# Prediction
# =============================
X = pd.DataFrame(features)

for col in feature_names:
    if col not in X.columns:
        X[col] = 0

X = X[feature_names].fillna(0)
X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)[:, 1]

final_prediction = int(np.round(preds.mean()))
confidence = probs.mean()

# =============================
# Output
# =============================
print("=" * 55)
print("Movement Name :", movement_name)
print("Joints Used   : Shoulder | Elbow | Wrist | Hand")
print("Prediction    :", "Correct ✅" if final_prediction == 1 else "Incorrect ❌")
print(f"Confidence    : {confidence:.2f}")
print("=" * 55)
