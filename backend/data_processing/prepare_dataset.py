import pandas as pd
import numpy as np
import random

df = pd.read_csv("C:/Users/siddh/Skin_Health_System/backend/data/lifestyle_raw.csv")

df = df[[
    "Sleep Duration",
    "Stress Level",
    "Physical Activity Level"
]]

df.columns = [
    "sleep",
    "stress",
    "exercise"
]

stress_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["stress"] = df["stress"].map(stress_map)

# Added new features that too with noise 
df["water"] = np.random.uniform(1, 4, len(df))
df["screen_time"] = np.random.uniform(2, 10, len(df))

# Placeholder for CNN output (more realistic distribution)
df["skin_condition"] = np.random.choice(
    ["healthy", "mild", "moderate"],
    len(df),
    p=[0.4, 0.4, 0.2]
)


def calculate_risk(row):
    score = 0

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

    # 🔥 IMPROVED CLASS SEPARATION (FINAL)

    if score >= 4:
        return random.choices(["High", "Medium"], weights=[0.7, 0.3])[0]

    elif score >= 3 and score < 5:
        return random.choices(["Medium", "Low"], weights=[0.85, 0.15])[0]

    else:
        return random.choices(["Low", "Medium"], weights=[0.8, 0.2])[0]


df["risk_level"] = df.apply(calculate_risk, axis=1)

# Expanded dataset 
df_expanded = pd.concat([df] * 5, ignore_index=True)

# Added Noise to make it more realistic
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

# ============================
# 🔥 HIGH-RISK AUGMENTATION (UNCHANGED)
# ============================

target_high = 200

high_df = df_expanded[df_expanded["risk_level"] == "High"]
current_high = len(high_df)

if current_high > 0 and current_high < target_high:

    needed = target_high - current_high

    augmented_list = []
    total_added = 0

    while total_added < needed:

        temp = high_df.copy()

        temp["sleep"] -= np.random.uniform(0.5, 1.5, len(temp))
        temp["water"] -= np.random.uniform(0.2, 0.8, len(temp))
        temp["exercise"] -= np.random.uniform(10, 30, len(temp))
        temp["screen_time"] += np.random.uniform(1, 3, len(temp))

        temp["sleep"] = temp["sleep"].clip(0, 12)
        temp["water"] = temp["water"].clip(0, 5)
        temp["exercise"] = temp["exercise"].clip(0, 300)
        temp["screen_time"] = temp["screen_time"].clip(0, 16)

        augmented_list.append(temp)
        total_added += len(temp)

    augmented_high = pd.concat(augmented_list).head(needed)

    df_expanded = pd.concat([df_expanded, augmented_high], ignore_index=True)

# Final shuffle
df_expanded = df_expanded.sample(frac=1).reset_index(drop=True)

# ============================

# Save dataset
df_expanded.to_csv(
    "C:/Users/siddh/Skin_Health_System/backend/data/lifestyle_dataset.csv",
    index=False
)

print("Dataset created:", len(df_expanded))
print(df_expanded["risk_level"].value_counts())