import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report

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
    X,y,test_size=0.2,random_state=42,stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced"
)


model1 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    reg_lambda=1,
    reg_alpha=0.5,
    scale_pos_weight=1
)

model2 = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function='MultiClass',
    verbose=0
)

model.fit(X_train,y_train)
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)

joblib.dump(model,"C:/Users/siddh/Skin_Health_System/backend/models/lifestyle_model.pkl")
joblib.dump(model1,"C:/Users/siddh/Skin_Health_System/backend/models/lifestyle_model1.pkl")
joblib.dump(model2,"C:/Users/siddh/Skin_Health_System/backend/models/lifestyle_model2.pkl")

print("Model trained")
#Applied Cross validation to check for overfitting and generalization
scores = cross_val_score(model, X, y, cv=5)
print("CV Accuracy:", scores.mean())
print("Accuracy:",model.score(X_test,y_test))
print("Accuracy (XGB):",model1.score(X_test,y_test))
print("Accuracy (CatBoost):",model2.score(X_test,y_test))

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

y_pred_cb = model2.predict(X_test)
cm_cb = confusion_matrix(y_test, y_pred_cb)
print("Confusion Matrix (CatBoost):\n", cm_cb)
print("\nClassification Report (CatBoost):\n", classification_report(y_test, y_pred_cb, zero_division=0))