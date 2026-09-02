import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = "weather.csv"

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET ==========")
print(df.head())

# Convert date
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

print("\n========== DATA INFORMATION ==========")
print(df.info())

# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ============================================================
# 3. FOG DISTRIBUTION
# ============================================================

print("\n========== FOG DISTRIBUTION ==========")

print(df["fog"].value_counts())

print("\nFog percentage:")
print(df["fog"].value_counts(normalize=True) * 100)

# ============================================================
# 4. CLEAN DATA
# ============================================================

# Numeric columns
numeric_columns = [
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "temperature",
    "dew_point_temperature",
    "dew_point_depression",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "station_level_pressure"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Fill missing numeric values using median
for column in numeric_columns:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].median())

# Remove rows where important values are missing
df = df.dropna(subset=["DATE", "fog"])

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

# ============================================================
# 5. CREATE TIME FEATURES
# ============================================================

df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month
df["day"] = df["DATE"].dt.day
df["hour"] = df["DATE"].dt.hour

df["day_of_year"] = df["DATE"].dt.dayofyear

# Cyclic month features
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Cyclic hour features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Temperature/dew-point relationship
df["temp_dew_difference"] = (
    df["temperature"] - df["dew_point_temperature"]
)

# Humidity-temperature interaction
df["humidity_temp"] = (
    df["relative_humidity"] * df["temperature"]
)

# ============================================================
# 6. FEATURES
# ============================================================

# IMPORTANT:
# visibility is NOT included because it can directly reveal fog.
# Station_name is also not directly used as a numeric feature.

features = [
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "temperature",
    "dew_point_temperature",
    "dew_point_depression",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "station_level_pressure",
    "month",
    "hour",
    "year",
    "day",
    "day_of_year",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "temp_dew_difference",
    "humidity_temp"
]

X = df[features]
y = df["fog"]

print("\n========== FEATURES ==========")
print(X.head())

print("\nNumber of features:", len(features))

print("\n========== TARGET ==========")
print(y.head())

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n========== TRAINING MODEL ==========")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
# 9. MODEL EVALUATION
# ============================================================

y_probability = model.predict_proba(X_test)[:, 1]

# Use 0.50 threshold
threshold = 0.50

y_pred = (y_probability >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Fog", "Fog"],
        zero_division=0
    )
)

# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")

print("                 Predicted")
print("                 No Fog    Fog")
print(f"Actual No Fog    {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"Actual Fog       {cm[1][0]:5d}   {cm[1][1]:5d}")

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.to_string(index=False))

# ============================================================
# 13. KODAIKANAL DECEMBER DATA
# ============================================================

print("\n")
print("=" * 60)
print("       KODAIKANAL DECEMBER FOG ANALYSIS")
print("=" * 60)

kodaikanal_december = df[
    (df["Station_name"].str.upper() == "KODAIKANAL") &
    (df["month"] == 12)
].copy()

print("\nStation: KODAIKANAL")
print("Month: December")

print(
    "\nHistorical December records:",
    len(kodaikanal_december)
)

# ============================================================
# 14. HISTORICAL FOG STATISTICS
# ============================================================

if len(kodaikanal_december) > 0:

    historical_fog = kodaikanal_december["fog"].sum()

    historical_total = len(kodaikanal_december)

    historical_percentage = (
        historical_fog / historical_total
    ) * 100

    print(
        "Historical fog occurrences:",
        historical_fog
    )

    print(
        f"Historical fog percentage: "
        f"{historical_percentage:.2f}%"
    )

    print("\nHistorical December distribution:")

    print(
        kodaikanal_december["fog"].value_counts()
    )

    # ========================================================
    # 15. MODEL PREDICTION FOR HISTORICAL DECEMBER CONDITIONS
    # ========================================================

    X_december = kodaikanal_december[features]

    december_probability = model.predict_proba(
        X_december
    )[:, 1]

    december_prediction = (
        december_probability >= threshold
    ).astype(int)

    # Add predictions to dataframe
    kodaikanal_december["fog_probability"] = (
        december_probability
    )

    kodaikanal_december["predicted_fog"] = (
        december_prediction
    )

    # ========================================================
    # 16. DECEMBER MODEL RESULTS
    # ========================================================

    predicted_fog_days = december_prediction.sum()

    total_days = len(december_prediction)

    predicted_percentage = (
        predicted_fog_days / total_days
    ) * 100

    average_probability = (
        december_probability.mean() * 100
    )

    maximum_probability = (
        december_probability.max() * 100
    )

    minimum_probability = (
        december_probability.min() * 100
    )

    print("\n========== MODEL DECEMBER RESULTS ==========")

    print(
        f"Average fog probability: "
        f"{average_probability:.2f}%"
    )

    print(
        f"Maximum fog probability: "
        f"{maximum_probability:.2f}%"
    )

    print(
        f"Minimum fog probability: "
        f"{minimum_probability:.2f}%"
    )

    print(
        f"Predicted fog records: "
        f"{predicted_fog_days}"
    )

    print(
        f"Predicted no-fog records: "
        f"{total_days - predicted_fog_days}"
    )

    print(
        f"Predicted fog percentage: "
        f"{predicted_percentage:.2f}%"
    )

    # ========================================================
    # 17. MOST LIKELY FOG CONDITIONS
    # ========================================================

    print("\n========== HIGHEST FOG PROBABILITY RECORDS ==========")

    highest = kodaikanal_december.sort_values(
        by="fog_probability",
        ascending=False
    ).head(10)

    print(
        highest[
            [
                "DATE",
                "temperature",
                "dew_point_temperature",
                "relative_humidity",
                "wind_speed",
                "fog_probability",
                "fog"
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # 18. OVERALL DECEMBER RISK
    # ========================================================

    print("\n========== DECEMBER FOG RISK ==========")

    if average_probability >= 70:
        print("🔴 VERY HIGH FOG RISK")

    elif average_probability >= 50:
        print("🟠 HIGH FOG RISK")

    elif average_probability >= 30:
        print("🟡 MODERATE FOG RISK")

    elif average_probability >= 10:
        print("🟢 LOW FOG RISK")

    else:
        print("🟢 VERY LOW FOG RISK")

else:

    print(
        "\nNo Kodaikanal December records found."
    )

# ============================================================
# 19. SAMPLE NEW WEATHER PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("          SAMPLE WEATHER PREDICTION")
print("=" * 60)

# Example:
# Change these values to predict a particular
# December weather condition.

sample_weather = {
    "LATITUDE": 10.2333,
    "LONGITUDE": 77.4667,
    "ELEVATION": 2343.0,

    "temperature": 10.0,
    "dew_point_temperature": 9.5,
    "dew_point_depression": 0.5,

    "relative_humidity": 95.0,

    "wind_speed": 2.0,
    "wind_direction": 90.0,

    "station_level_pressure":
        df["station_level_pressure"].median(),

    "month": 12,
    "hour": 3,
    "year": 2026,
    "day": 15,
    "day_of_year": 349,

    "month_sin":
        np.sin(2 * np.pi * 12 / 12),

    "month_cos":
        np.cos(2 * np.pi * 12 / 12),

    "hour_sin":
        np.sin(2 * np.pi * 3 / 24),

    "hour_cos":
        np.cos(2 * np.pi * 3 / 24),

    "temp_dew_difference":
        10.0 - 9.5,

    "humidity_temp":
        95.0 * 10.0
}

sample = pd.DataFrame(
    [sample_weather],
    columns=features
)

sample_probability = model.predict_proba(
    sample
)[0][1]

sample_no_fog_probability = 1 - sample_probability

sample_prediction = (
    sample_probability >= threshold
)

print("\n========== NEW WEATHER PREDICTION ==========")

print(
    f"Fog probability: "
    f"{sample_probability * 100:.2f}%"
)

print(
    f"No Fog probability: "
    f"{sample_no_fog_probability * 100:.2f}%"
)

print(
    f"Decision threshold: {threshold}"
)

if sample_prediction:
    print("\n🌫️ FOG PREDICTED")
else:
    print("\n☀️ NO FOG PREDICTED")

# ============================================================
# 20. SAMPLE RISK
# ============================================================

print("\n========== FOG RISK ==========")

if sample_probability >= 0.70:
    print("🔴 VERY HIGH FOG RISK")

elif sample_probability >= 0.50:
    print("🟠 HIGH FOG RISK")

elif sample_probability >= 0.30:
    print("🟡 MODERATE FOG RISK")

elif sample_probability >= 0.10:
    print("🟢 LOW FOG RISK")

else:
    print("🟢 VERY LOW FOG RISK")

print("\n========== PROGRAM FINISHED ==========")