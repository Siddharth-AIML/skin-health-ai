import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("C:/Users/siddh/Skin_Health_System/backend/data/lifestyle_dataset.csv")

# Encode categorical
df["skin_condition"] = df["skin_condition"].map({
    "healthy":0,
    "mild":1,
    "moderate":2
})

df["risk_level"] = df["risk_level"].map({
    "Low":0,
    "Medium":1,
    "High":2
})

X = df[[
    "sleep",
    "stress",
    "water",
    "exercise",
    "screen_time",
    "skin_condition"
]]

y = df["risk_level"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10
)

model.fit(X_train,y_train)

joblib.dump(model,"C:/Users/siddh/Skin_Health_System/backend/models/lifestyle_model.pkl")

print("Model trained")
print("Accuracy:",model.score(X_test,y_test))