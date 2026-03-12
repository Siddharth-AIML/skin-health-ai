import pandas as pd
import numpy as np

# Load raw dataset
df = pd.read_csv("C:/Users/siddh/Skin_Health_System/backend/data/lifestyle_raw.csv")

# Keep required columns
df = df[[
    "Sleep Duration",
    "Stress Level",
    "Physical Activity Level"
]]

# Rename columns
df.columns = [
    "sleep",
    "stress",
    "exercise"
]

# Convert stress level to numeric
stress_map = {
    "Low":0,
    "Medium":1,
    "High":2
}

df["stress"] = df["stress"].map(stress_map)

# Add additional lifestyle features
df["water"] = np.random.uniform(1,4,len(df))
df["screen_time"] = np.random.uniform(2,10,len(df))

# Placeholder for CNN output
df["skin_condition"] = np.random.choice(
    ["healthy","mild","moderate"],
    len(df)
)

# Risk calculation
def calculate_risk(row):

    score = 0

    if row["sleep"] < 6:
        score += 2

    if row["stress"] == 2:
        score += 2

    if row["water"] < 2:
        score += 1

    if row["exercise"] < 60:
        score += 1

    if row["screen_time"] > 7:
        score += 1

    if row["skin_condition"] == "moderate":
        score += 2

    if score >=5:
        return "High"

    elif score >=3:
        return "Medium"

    else:
        return "Low"


df["risk_level"] = df.apply(calculate_risk, axis=1)

# Expand dataset
df_expanded = pd.concat([df]*5, ignore_index=True)

df_expanded["sleep"] += np.random.normal(0,0.5,len(df_expanded))
df_expanded["water"] += np.random.normal(0,0.2,len(df_expanded))

# Save dataset
df_expanded.to_csv("C:/Users/siddh/Skin_Health_System/backend/data/lifestyle_dataset.csv",index=False)

print("Dataset created:",len(df_expanded))