import pandas as pd
import numpy as np
import random

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
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["stress"] = df["stress"].map(stress_map)

# Add additional lifestyle features
df["water"] = np.random.uniform(1, 4, len(df))
df["screen_time"] = np.random.uniform(2, 10, len(df))

# Placeholder for CNN output (more realistic distribution)
df["skin_condition"] = np.random.choice(
    ["healthy", "mild", "moderate"],
    len(df),
    p=[0.4, 0.4, 0.2]
)

# ----------- IMPROVED RISK CALCULATION -----------

def calculate_risk(row):
    score = 0

    # Softer scoring
    if row["sleep"] < 6:
        score += 1

    if row["stress"] == 2:
        score += 2

    if row["water"] < 2:
        score += 1

    if row["exercise"] < 60:
        score += 1

    if row["screen_time"] > 7:
        score += 1

    if row["skin_condition"] == "moderate":
        score += random.choice([1, 2])

    # -------- Probabilistic labeling (KEY FIX) --------
    if score >= 5:
        return random.choices(["High", "Medium"], weights=[0.7, 0.3])[0]

    elif score >= 3:
        return random.choices(["Medium", "Low"], weights=[0.7, 0.3])[0]

    else:
        return random.choices(["Low", "Medium"], weights=[0.8, 0.2])[0]


df["risk_level"] = df.apply(calculate_risk, axis=1)

# -------- Expand dataset --------
df_expanded = pd.concat([df] * 5, ignore_index=True)

# -------- Add stronger noise (IMPORTANT) --------
df_expanded["sleep"] += np.random.normal(0, 1, len(df_expanded))
df_expanded["water"] += np.random.normal(0, 0.5, len(df_expanded))
df_expanded["exercise"] += np.random.normal(0, 15, len(df_expanded))
df_expanded["screen_time"] += np.random.normal(0, 2, len(df_expanded))

# Clip values to realistic ranges
df_expanded["sleep"] = df_expanded["sleep"].clip(0, 12)
df_expanded["water"] = df_expanded["water"].clip(0, 5)
df_expanded["exercise"] = df_expanded["exercise"].clip(0, 300)
df_expanded["screen_time"] = df_expanded["screen_time"].clip(0, 16)

# Shuffle dataset
df_expanded = df_expanded.sample(frac=1).reset_index(drop=True)

# Save dataset
df_expanded.to_csv(
    "C:/Users/siddh/Skin_Health_System/backend/data/lifestyle_dataset.csv",
    index=False
)

print("Dataset created:", len(df_expanded))