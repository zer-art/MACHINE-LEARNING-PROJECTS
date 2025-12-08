import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- Constants & Mappings (Copied from app.py to ensure consistency) ---
crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
                 6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 
                 11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 
                 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",19: "Pigeonpeas", 
                 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# Reverse crop dict for evaluation
crop_name_to_id = {v: k for k, v in crop_dict.items()}

fert_dict = {
'Urea':1,
'DAP':2,
'14-35-14':3,
'28-28':4,
'17-17-17':5,
'20-20':6,
'10-26-26':7,
}

soil_type_dict = {
    'Sandy': 1,
    'Loamy': 2,
    'Black': 3,
    'Red': 4,
    'Clayey': 5
}

crop_type_dict = {
    'Maize': 1,
    'Sugarcane': 2,
    'Cotton': 3,
    'Tobacco': 4,
    'Paddy': 5,
    'Barley': 6,
    'Wheat': 7,
    'Millets': 8,
    'Oil seeds': 9,
    'Pulses': 10,
    'Ground Nuts': 11
}

def evaluate_crop_model():
    print("\n--- Evaluating Crop Recommendation Model ---")
    try:
        # Load Data
        df = pd.read_csv('data/Crop_recommendation.csv')
        
        # Prepare Features and Target
        # The CSV probably has labels as strings, we need to map them if the model predicts IDs
        # Let's check model output content. app.py says: crop_dict[prediction[0]].
        # So model predicts an ID (1-22).
        
        # We need to map the label column in CSV to IDs
        if 'label' in df.columns:
            # Normalize crop names to title case just in case
            df['label_id'] = df['label'].apply(lambda x: crop_name_to_id.get(x.title(), -1))
            # specific fix for 'rice' vs 'Rice' etc if needed. 
            # Looking at dict, keys are Title Case. 
        
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y_true = df['label_id']

        # Load Model & Scaler
        model = pickle.load(open('model/crop_model.sav', 'rb'))
        scaler = pickle.load(open('model/crop_scaler.sav', 'rb'))

        # Predict
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=[crop_dict[i] for i in sorted(crop_dict.keys()) if i in y_true.unique()]))
        
        return {"accuracy": acc}

    except Exception as e:
        print(f"Error evaluating Crop Model: {e}")
        return None

def evaluate_fertilizer_model():
    print("\n--- Evaluating Fertilizer Recommendation Model ---")
    try:
        # Load Data
        df = pd.read_csv('data/Fertilizer Prediction.csv')
        
        # Features: 'Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'
        # Note: CSV headers might differ slightly from app.py inputs.
        # app.py inputs: temperature_f, humidity_f, moisture_f, soil_type, crop_type, nitrogen_f, potassium_f, phosphorous_f
        # app.py array order: [temp, hum, moisture, soil, crop, N, K, P]
        
        # Clean col names
        df.columns = df.columns.str.strip()
        
        # Map Categorical Variables
        df['Soil Type_Num'] = df['Soil Type'].map(soil_type_dict)
        df['Crop Type_Num'] = df['Crop Type'].map(crop_type_dict)
        df['Fertilizer Name_Num'] = df['Fertilizer Name'].map(fert_dict)

        # Drop rows with unmapped values if any
        df = df.dropna(subset=['Soil Type_Num', 'Crop Type_Num', 'Fertilizer Name_Num'])

        # Order matches app.py: [Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous]
        # BUT app.py has: [temp, hum, moisture, soil, crop, N, K, P]
        # Wait, app.py: input_data = np.array([[temperature_f, humidity_f, moisture_f, soil_type_dict[soil_type], crop_type_dict[crop_type], nitrogen_f, potassium_f, phosphorous_f]])
        # Let's verify CSV column names for N, K, P match app.py expectations
        
        X = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type_Num', 'Crop Type_Num', 'Nitrogen', 'Potassium', 'Phosphorous']].values
        y_true = df['Fertilizer Name_Num'].values

        # Load Model & Scaler
        model = pickle.load(open('model/fertilizer_model.sav', 'rb'))
        scaler = pickle.load(open('model/fertilizer_scaler.sav', 'rb'))

        # Predict
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        # print(classification_report(y_true, y_pred))
        
        return {"accuracy": acc}

    except Exception as e:
        print(f"Error evaluating Fertilizer Model: {e}")
        # print columns to help debug
        try:
             print(f"Available columns: {df.columns.tolist()}")
        except: pass
        return None

if __name__ == "__main__":
    evaluate_crop_model()
    evaluate_fertilizer_model()
