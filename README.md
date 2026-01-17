# Motion Correctness Classification for Rehabilitation 🦾🤖

## 📌 Project Overview

**Motion Correctness Classification** is an AI-based system designed to support **physical therapy and rehabilitation** by automatically evaluating the quality of human movements using **IMU (MPU6050) sensors**.

The system analyzes motion data collected from **four upper-limb joints** and classifies each performed movement as:

- ✅ **Correct**
- ❌ **Incorrect**

This project is developed as part of a **Biomedical Engineering Graduation Project**, with a strong focus on **AI-driven healthcare and rehabilitation systems**.

---

## 🎯 Project Objectives

- Analyze raw IMU sensor data collected during rehabilitation exercises  
- Extract meaningful motion features from time-series signals  
- Train machine learning models to assess movement correctness  
- Provide objective, data-driven feedback to support physiotherapists  

---

## 📁 Repository Structure

Motion-Correctness-Classification/
│
├── models/
│   ├── best_mpu_model.joblib
│   ├── scaler.joblib
│   └── feature_list.joblib
│
├── notebooks/
│   └── classification_model.ipynb   # EDA, preprocessing, and training
│
├── scripts/
│   └── predict.py                   # Inference on new movement sessions
│
├── data/
│   └── mpuData.xlsx                 # Collected IMU dataset
│
└── README.md
🦾 Supported Movements & Joints
Movements
Stretching

Lift Up

Joints Monitored
IMU sensors are mounted on four upper-limb joints:

🦴 Shoulder

🦴 Elbow

🦴 Wrist

🦴 Hand

The final prediction considers all joints together, not a single joint in isolation.

🧠 AI & Machine Learning Pipeline
1️⃣ Data Preprocessing
Sorting and cleaning raw IMU data

Reconstructing movement sessions

Handling time-series inconsistencies

2️⃣ Feature Engineering
For each sliding window:

Statistical features:

Mean

Standard Deviation

Minimum

Maximum

Extracted from:

Pitch

Roll

Acceleration (X, Y, Z)

3️⃣ Machine Learning Models
The following models were trained and evaluated:

Logistic Regression (baseline)

Random Forest

AdaBoost

XGBoost

The best-performing model is selected based on F1-score, which is critical for medical and rehabilitation applications.

📊 Model Performance
Model	Accuracy	F1 Score
Random Forest	0.97	0.97
XGBoost	0.95	0.95
AdaBoost	0.89	0.86
Logistic Regression	0.70	0.63

⚠️ Due to the limited dataset size, performance is expected to further improve as more rehabilitation sessions are collected.

🚀 How to Use
1️⃣ Install Dependencies
bash
Copy code
pip install numpy pandas scikit-learn xgboost joblib openpyxl
2️⃣ Run Inference
bash
Copy code
python scripts/predict.py
The script outputs:

Movement type

Correct / Incorrect classification

Confidence score

🧩 Future Work
Expand dataset with more patients and rehabilitation sessions

Real-time inference from live IMU sensor streams

Integration with mobile or VR rehabilitation platforms

Sequence-level modeling using LSTM or HMM

### 🙌 Contributors
* [Sohaila Mohamed](https://github.com/sohailamohamed15)

* Nadin Awad

### 📜 License
This project is released under the MIT License.

### ⭐ Support the Project
If you find this project useful for rehabilitation technology, give it a star on GitHub!