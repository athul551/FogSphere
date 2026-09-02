import pandas as pd

FILE = "data/processed/weather_dataset.csv"

df = pd.read_csv(FILE)

print("Total observations:", len(df))

print("\nVisibility statistics:")
print(df["visibility"].describe())

print("\nSmallest visibility values:")
print(
    df["visibility"]
    .dropna()
    .sort_values()
    .head(50)
    .to_string(index=False)
)

print("\nVisibility value counts:")
print(
    df["visibility"]
    .value_counts()
    .sort_index()
    .head(50)
)

print("\nObservations with very low visibility:")
print(
    df[
        df["visibility"].notna()
        & (df["visibility"] <= 100)
    ][
        [
            "DATE",
            "Station_name",
            "temperature",
            "dew_point_temperature",
            "relative_humidity",
            "wind_speed",
            "visibility"
        ]
    ]
    .head(50)
    .to_string(index=False)
)