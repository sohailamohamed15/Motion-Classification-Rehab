# Motion Correctness Classification for Rehabilitation 🚀

<p align="center">
  <img src="https://img.icons8.com/fluency/144/physical-therapy.png" alt="Motion AI Logo" width="150" />
</p>

## 📌 Project Overview

**Motion Correctness AI** is a machine-learning-powered system designed to support physical therapy and rehabilitation. By analyzing data from IMU sensors (MPU6050), the model classifies patient movements as **Correct** or **Incorrect**, providing real-time feedback to ensure safe and effective recovery.

This repository contains the full end-to-end pipeline:
- **Raw Sensor Processing** (Accel, Pitch, Roll)
- **Feature Engineering** (Physics-based & Statistical features)
- **Machine Learning Models** (XGBoost, Logistic Regression, etc.)
- **Inference & Logging Scripts** for real-time deployment

---

## 📁 Repository Structure

Motion-Classification-Rehab/ ├── models/ # Final trained models & scalers │ ├── final_motion_classifier.joblib │ └── motion_scaler.joblib │ ├── code/ # Training & analysis notebooks │ └── mpudata.ipynb # EDA, feature extraction, and training │ ├── scripts/ # Utility scripts for deployment │ ├── predict.py # Real-time inference script │ └── logger.py # Script to save new sensor data to Excel │ ├── data/

│ └── MPU data.csv # Training dataset │ └── README.md


---

## 🦾 Supported Exercises & Joints

The model is trained to monitor various rehabilitation movements across key joints:
- **Exercises:** Stretching, and more.
- **Joints Covered:** - 🦴 Shoulder
  - 🦴 Elbow
  - 🦴 Wrist
  - 🦴 Hand

---

## 🧠 Technical Architecture

The system transforms raw time-series sensor data into a format understandable by Machine Learning models:

### 1. Feature Engineering
- **Physical Features:** Calculates `Accel_Mag` (Resultant Acceleration) and `Angle_Diff` (Pitch vs Roll).
- **Statistical Aggregation:** Converts 100+ rows of raw data into a single row of **Mean** and **Standard Deviation** to capture movement stability.

### 2. Algorithms Used
- **XGBoost:** Best performing model for handling non-linear patterns in motion.
- **Logistic Regression:** Used for baseline comparison and linear classification.
- **Random Forest & AdaBoost:** Evaluated for ensemble robustness.

---

## 📊 Performance Summary

The pipeline automatically selects the best model based on **Cross-Validation (CV) Accuracy** to ensure the model generalizes well to new patients.

| Model Name | Test Accuracy | CV Accuracy (Mean) | F1 Score |
| :--- | :---: | :---: | :---: |
| **XGBoost** | **0.50** | **0.75** | **0.33** |
| Logistic Regression | 1.00 | 0.50 | 1.00 |
| AdaBoost | 0.75 | 0.42 | 0.73 |

> 💡 *Note: The high Test Accuracy vs. lower CV Accuracy in some models is due to the small sample size (16 sessions), which will stabilize as more data is collected.*

---

## 📦 How to Use

### 1️⃣ Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost joblib openpyxl
2️⃣ Test the Model (Inference)
To predict the correctness of a new movement session:

Bash

python scripts/predict.py
3️⃣ Log New Data
To record new sensor data into a separate Excel file for future training:

Bash

python scripts/logger.py
🧩 Future Roadmap
[ ] Expand Dataset: Collect 100+ sessions for more robust training.

[ ] Mobile Integration: Export models to TFLite for Android/iOS apps.

[ ] Real-time Visualization: Dashboard to show patient progress over time.

[ ] Portion Detection: Detect if the patient performed the full range of motion.

🙌 Contributors
* [Sohaila Mohamed](https://github.com/sohailamohamed15)

* Nadin Awad

📜 License
This project is released under the MIT License.

⭐ Support the Project
If you find this project useful for rehabilitation technology, give it a star on GitHub!