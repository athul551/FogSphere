import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/weather_dataset.csv"
OUTPUT_FILE = "data/processed/fog_training.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))

# --------------------------------------------------
# 1. Combine present-weather columns
# --------------------------------------------------

weather_columns = [
    "pres_wx_MW1",
    "pres_wx_MW2",
    "pres_wx_MW3",
    "pres_wx_AU1",
    "pres_wx_AU2",
    "pres_wx_AU3"
]

available_weather_columns = [
    col for col in weather_columns if col in df.columns
]

df["weather_codes"] = (
    df[available_weather_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.upper()
)

# --------------------------------------------------
# 2. Detect fog
# --------------------------------------------------

# GHCNh present-weather codes beginning with FG
df["fog"] = df["weather_codes"].str.contains(
    r"(^|[\s;])FG[:\s]",
    regex=True,
    na=False
).astype(int)

# --------------------------------------------------
# 3. Detect other weather conditions
# --------------------------------------------------

df["rain"] = df["weather_codes"].str.contains(
    r"(^|[\s;])(RA)[:\s]",
    regex=True,
    na=False
).astype(int)

df["drizzle"] = df["weather_codes"].str.contains(
    r"(^|[\s;])(DZ)[:\s]",
    regex=True,
    na=False
).astype(int)

df["mist"] = df["weather_codes"].str.contains(
    r"(^|[\s;])(BR)[:\s]",
    regex=True,
    na=False
).astype(int)

df["dust"] = df["weather_codes"].str.contains(
    r"(^|[\s;])(DU)[:\s]",
    regex=True,
    na=False
).astype(int)

# --------------------------------------------------
# 4. Date/time features
# --------------------------------------------------

df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month
df["day"] = df["DATE"].dt.day
df["hour"] = df["DATE"].dt.hour

# --------------------------------------------------
# 5. Temperature-dew point difference
# --------------------------------------------------

df["dew_point_depression"] = (
    df["temperature"] -
    df["dew_point_temperature"]
)

# --------------------------------------------------
# 6. Select ML features
# --------------------------------------------------

features = [
    "DATE",
    "Station_name",
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
    "fog",
    "visibility"
]

# Keep only columns that exist
features = [col for col in features if col in df.columns]

result = df[features].copy()

# --------------------------------------------------
# 7. Remove rows without essential weather data
# --------------------------------------------------

essential = [
    "temperature",
    "dew_point_temperature",
    "relative_humidity"
]

essential = [col for col in essential if col in result.columns]

result = result.dropna(subset=essential)

# --------------------------------------------------
# 8. Sort chronologically
# --------------------------------------------------

result = result.sort_values("DATE")

# --------------------------------------------------
# 9. Save
# --------------------------------------------------

result.to_csv(OUTPUT_FILE, index=False)

print()
print("========================================")
print("FOG DATASET CREATED")
print("========================================")

print("Rows:", len(result))
print("Fog observations:", result["fog"].sum())
print("Non-fog observations:", (result["fog"] == 0).sum())

print()
print("Fog percentage:")
print(round(result["fog"].mean() * 100, 2), "%")

print()
print("Fog by station:")
print(
    result.groupby("Station_name")["fog"]
    .agg(["count", "sum"])
)

print()
print("Saved to:")
print(OUTPUT_FILE)