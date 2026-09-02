import os
import pandas as pd

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

USEFUL_COLUMNS = [
    "STATION",
    "Station_name",
    "DATE",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "temperature",
    "dew_point_temperature",
    "station_level_pressure",
    "wind_direction",
    "wind_speed",
    "relative_humidity",
    "visibility",
    "pres_wx_MW1",
    "pres_wx_MW2",
    "pres_wx_MW3"
]


def process_file(filepath):

    print(f"\nProcessing: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="|",
        usecols=lambda column: column in USEFUL_COLUMNS
    )

    print("Rows:", len(df))

    # Convert date
    df["DATE"] = pd.to_datetime(
        df["DATE"],
        errors="coerce"
    )

    # Convert numerical columns
    numeric_columns = [
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "temperature",
        "dew_point_temperature",
        "station_level_pressure",
        "wind_direction",
        "wind_speed",
        "relative_humidity",
        "visibility"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Create time features
    df["hour"] = df["DATE"].dt.hour
    df["month"] = df["DATE"].dt.month
    df["day_of_year"] = df["DATE"].dt.dayofyear

    # Temperature - dew point difference
    df["dew_point_depression"] = (
        df["temperature"] -
        df["dew_point_temperature"]
    )

    return df


def main():

    files = [
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".psv")
    ]

    print("Files found:", len(files))

    datasets = []

    for filename in files:

        filepath = os.path.join(
            RAW_DIR,
            filename
        )

        try:

            df = process_file(filepath)

            datasets.append(df)

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

    if not datasets:

        print("No data found.")
        return

    final_df = pd.concat(
        datasets,
        ignore_index=True
    )

    output = os.path.join(
        OUTPUT_DIR,
        "weather_dataset.csv"
    )

    final_df.to_csv(
        output,
        index=False
    )

    print("\n================================")
    print("DATASET CREATED")
    print("================================")
    print("Rows:", len(final_df))
    print("Columns:", len(final_df.columns))
    print("Saved:", output)

    print("\nColumns:")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    main()