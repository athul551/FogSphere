import pandas as pd

FILE = "data/processed/weather_dataset.csv"

df = pd.read_csv(FILE)

print("Total rows:", len(df))

print("\nMissing values:")
print(
    df[
        [
            "temperature",
            "dew_point_temperature",
            "relative_humidity",
            "station_level_pressure",
            "wind_speed",
            "visibility"
        ]
    ].isna().sum()
)

print("\nVisibility values:")
print(
    df["visibility"]
    .value_counts(dropna=False)
    .head(30)
)

for column in [
    "pres_wx_MW1",
    "pres_wx_MW2",
    "pres_wx_MW3"
]:

    print(f"\n{column} values:")
    print(
        df[column]
        .value_counts(dropna=False)
        .head(30)
    )

print("\nSample rows with visibility:")
print(
    df[
        [
            "DATE",
            "Station_name",
            "temperature",
            "dew_point_temperature",
            "relative_humidity",
            "wind_speed",
            "visibility",
            "pres_wx_MW1",
            "pres_wx_MW2",
            "pres_wx_MW3"
        ]
    ]
    .dropna(subset=["visibility"])
    .head(30)
    .to_string(index=False)
)