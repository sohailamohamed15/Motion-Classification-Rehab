import joblib
import pandas as pd
import numpy as np

# 1. Load the saved model and scaler
model = joblib.load('models/final_motion_classifier.joblib')
scaler = joblib.load('models/motion_scaler.joblib')
feature_names = scaler.feature_names_in_

# 2. Read the original dataset to test a full "Correct" session
df_raw = pd.read_excel(r'D:\الكليه\Project 3\Motion-Correctness-Classification\data\mpuData.xlsx')

# Extract all rows belonging to a specific correct Stretching session for the Shoulder
sample_session = df_raw[(df_raw['Movement_Name'] == 'Stretching') & 
                        (df_raw['Label_Value'] == 1) & 
                        (df_raw['Joint_Label'] == 'Shoulder')]

def test_real_session(session_data, joint_name='Shoulder'):
    """
    Simulates the processing of a full movement session and returns the prediction.
    """
    # Apply the same feature engineering as during training
    session_data = session_data.copy()
    session_data['Accel_Mag'] = np.sqrt(session_data['AccelX']**2 + session_data['AccelY']**2 + session_data['AccelZ']**2)
    session_data['Angle_Diff'] = np.abs(session_data['Pitch'] - session_data['Roll'])
    
    # Calculate statistical features (Mean and Standard Deviation)
    agg = session_data.agg({
        'AccelX': ['mean', 'std'], 'AccelY': ['mean', 'std'], 'AccelZ': ['mean', 'std'],
        'Pitch': ['mean', 'std'], 'Roll': ['mean', 'std'],
        'Accel_Mag': ['mean', 'std'], 'Angle_Diff': ['mean', 'std']
    })
    
    # Flatten the statistics into a single feature list
    features = []
    for col in ['AccelX', 'AccelY', 'AccelZ', 'Pitch', 'Roll', 'Accel_Mag', 'Angle_Diff']:
        features.append(agg.loc['mean', col])
        features.append(agg.loc['std', col])
    
    # Manual One-Hot Encoding mapping for the Joint labels
    joint_map = {'Elbow': [1,0,0,0], 'Hand': [0,1,0,0], 'Shoulder': [0,0,1,0], 'Wrist': [0,0,0,1]}
    joint_encoded = joint_map.get(joint_name, [0,0,0,0])
    
    # Construct final DataFrame and apply scaling
    final_df = pd.DataFrame([features + joint_encoded], columns=feature_names)
    final_scaled = scaler.transform(final_df)
    
    # Perform prediction and get confidence scores
    prediction = model.predict(final_scaled)
    prob = model.predict_proba(final_scaled)
    
    result = "✅ Correct" if prediction[0] == 1 else "❌ Incorrect"
    return result, np.max(prob)

# Run the inference test
print("🚀 Testing with a FULL real session from CSV...")
result, confidence = test_real_session(sample_session, 'Shoulder')

print(f"Result: {result}")
print(f"Confidence: {confidence*100:.2f}%")