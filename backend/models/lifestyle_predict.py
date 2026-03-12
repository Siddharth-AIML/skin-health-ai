import joblib
import numpy as np

# load trained model
model = joblib.load("models/lifestyle_model.pkl")

def predict_risk(sleep, stress, water, exercise, screen_time, skin_condition):

    stress_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    skin_map = {
        "healthy": 0,
        "mild": 1,
        "moderate": 2
    }

    input_data = np.array([[
        float(sleep),
        stress_map[stress],
        float(water),
        float(exercise),
        float(screen_time),
        skin_map[skin_condition]
    ]])

    prediction = model.predict(input_data)[0]

    risk_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return risk_map[prediction]