import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("weather.csv")

print("\n========== DATASET ==========")
print(df.head())

print("\n========== DATA INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ============================================================
# 2. CLEAN DATA
# ============================================================

df["DATE"] = pd.to_datetime(df["DATE"])

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
    "station_level_pressure",
    "visibility"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].fillna(df[column].median())


print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())


# ============================================================
# 3. FOG DISTRIBUTION
# ============================================================

print("\n========== FOG DISTRIBUTION ==========")
print(df["fog"].value_counts())

print("\nFog percentage:")
print(df["fog"].value_counts(normalize=True) * 100)


# ============================================================
# 4. CREATE TIME FEATURES
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


# ============================================================
# 5. CREATE WEATHER FEATURES
# ============================================================

df["temp_dew_difference"] = (
    df["temperature"] - df["dew_point_temperature"]
)

df["humidity_temp"] = (
    df["relative_humidity"] * df["temperature"]
)


# ============================================================
# 6. SELECT FEATURES
# ============================================================

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
    "year",
    "month",
    "day",
    "day_of_year",
    "hour",
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
# 8. TRAIN RANDOM FOREST
# ============================================================

print("\n========== TRAINING MODEL ==========")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
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

probabilities = model.predict_proba(X_test)[:, 1]

threshold = 0.50

predictions = (probabilities >= threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Fog", "Fog"],
        zero_division=0
    )
)


# ============================================================
# 10. FEATURE IMPORTANCE
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
# 11. KODAIKANAL DECEMBER PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("       KODAIKANAL DECEMBER FOG PREDICTION")
print("=" * 60)


# ------------------------------------------------------------
# DATE INPUT
# ------------------------------------------------------------

date_input = input(
    "\nEnter December date (YYYY-MM-DD): "
)

time_input = input(
    "Enter time (HH:MM): "
)


date_time = pd.to_datetime(
    date_input + " " + time_input
)


# Make sure month is December

if date_time.month != 12:

    print("\n❌ Please enter a December date.")

    exit()


# ------------------------------------------------------------
# WEATHER INPUT
# ------------------------------------------------------------

print("\nEnter the weather conditions for Kodaikanal:")

temperature = float(
    input("Temperature (°C): ")
)

dew_point_temperature = float(
    input("Dew point temperature (°C): ")
)

relative_humidity = float(
    input("Relative humidity (%): ")
)

wind_speed = float(
    input("Wind speed: ")
)

wind_direction = float(
    input("Wind direction (degrees): ")
)

station_level_pressure = float(
    input("Station level pressure: ")
)


# ------------------------------------------------------------
# KODAIKANAL LOCATION
# ------------------------------------------------------------

latitude = 10.2333
longitude = 77.4667
elevation = 2343.0


# ------------------------------------------------------------
# CALCULATE DERIVED FEATURES
# ------------------------------------------------------------

month = date_time.month
day = date_time.day
hour = date_time.hour
year = date_time.year

day_of_year = date_time.dayofyear


month_sin = np.sin(
    2 * np.pi * month / 12
)

month_cos = np.cos(
    2 * np.pi * month / 12
)


hour_sin = np.sin(
    2 * np.pi * hour / 24
)

hour_cos = np.cos(
    2 * np.pi * hour / 24
)


dew_point_depression = (
    temperature - dew_point_temperature
)


temp_dew_difference = (
    temperature - dew_point_temperature
)


humidity_temp = (
    relative_humidity * temperature
)


# ============================================================
# 12. CREATE INPUT DATAFRAME
# ============================================================

new_weather = pd.DataFrame({

    "LATITUDE": [latitude],

    "LONGITUDE": [longitude],

    "ELEVATION": [elevation],

    "temperature": [temperature],

    "dew_point_temperature": [
        dew_point_temperature
    ],

    "dew_point_depression": [
        dew_point_depression
    ],

    "relative_humidity": [
        relative_humidity
    ],

    "wind_speed": [wind_speed],

    "wind_direction": [wind_direction],

    "station_level_pressure": [
        station_level_pressure
    ],

    "year": [year],

    "month": [month],

    "day": [day],

    "day_of_year": [day_of_year],

    "hour": [hour],

    "month_sin": [month_sin],

    "month_cos": [month_cos],

    "hour_sin": [hour_sin],

    "hour_cos": [hour_cos],

    "temp_dew_difference": [
        temp_dew_difference
    ],

    "humidity_temp": [
        humidity_temp
    ]
})


# ============================================================
# 13. PREDICT FOG
# ============================================================

fog_probability = model.predict_proba(
    new_weather
)[0][1]


no_fog_probability = 1 - fog_probability


prediction = (
    fog_probability >= threshold
)


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

print("\n========== KODAIKANAL WEATHER ==========")

print(
    "Date:",
    date_time.strftime("%Y-%m-%d")
)

print(
    "Time:",
    date_time.strftime("%H:%M")
)

print(
    "Temperature:",
    temperature,
    "°C"
)

print(
    "Dew Point:",
    dew_point_temperature,
    "°C"
)

print(
    "Humidity:",
    relative_humidity,
    "%"
)


print("\n========== FOG PREDICTION ==========")

print(
    f"Fog probability: {fog_probability * 100:.2f}%"
)

print(
    f"No Fog probability: {no_fog_probability * 100:.2f}%"
)

print(
    f"Decision threshold: {threshold}"
)


if prediction:

    print("\n🌫️ FOG PREDICTED")

else:

    print("\n☀️ NO FOG PREDICTED")


# ============================================================
# 15. FOG RISK LEVEL
# ============================================================

print("\n========== FOG RISK ==========")


if fog_probability >= 0.75:

    print("🔴 VERY HIGH FOG RISK")

elif fog_probability >= 0.50:

    print("🟠 HIGH FOG RISK")

elif fog_probability >= 0.30:

    print("🟡 MODERATE FOG RISK")

elif fog_probability >= 0.10:

    print("🟢 LOW FOG RISK")

else:

    print("🟢 VERY LOW FOG RISK")