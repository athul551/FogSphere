import os
import requests
import pandas as pd
from io import StringIO
from tqdm import tqdm

STATION_URL = (
    "https://www.ncei.noaa.gov/"
    "oa/global-historical-climatology-network/"
    "hourly/doc/ghcnh-station-list.csv"
)

DATA_URL = (
    "https://www.ncei.noaa.gov/"
    "oa/global-historical-climatology-network/"
    "hourly/access/by-station/"
)

OUTPUT_DIR = "data/raw"
STATION_DIR = "data/stations"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATION_DIR, exist_ok=True)


def download_station_list():

    print("Downloading NOAA GHCNh station list...")

    response = requests.get(STATION_URL, timeout=60)
    response.raise_for_status()

    text = response.text

    with open(
        os.path.join(STATION_DIR, "ghcnh-station-list.csv"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(text)

    print("Station list downloaded.")

    df = pd.read_csv(StringIO(text))

    print("\nColumns:")
    print(df.columns.tolist())

    return df


def find_indian_stations(df):

    # GHCNh station IDs beginning with IN are Indian stations
    stations = df[
        df["GHCN_ID"].astype(str).str.startswith("IN")
    ].copy()

    print("\nIndian stations found:", len(stations))

    return stations


def filter_kerala(stations):

    # Approximate Kerala bounding box
    kerala = stations[
        (stations["LATITUDE"] >= 8.0) &
        (stations["LATITUDE"] <= 13.0) &
        (stations["LONGITUDE"] >= 74.5) &
        (stations["LONGITUDE"] <= 77.5)
    ].copy()

    print("Kerala/nearby stations:", len(kerala))

    return kerala


def download_station(station_id):

    filename = f"GHCNh_{station_id}_por.psv"

    url = DATA_URL + filename

    output_file = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(output_file):
        print(f"Already exists: {filename}")
        return

    print(f"\nDownloading {station_id}...")

    response = requests.get(
        url,
        stream=True,
        timeout=120
    )

    if response.status_code == 404:
        print(f"Not available: {station_id}")
        return

    response.raise_for_status()

    with open(output_file, "wb") as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                f.write(chunk)

    print(f"Saved: {output_file}")


def main():

    stations = download_station_list()

    # Find all Indian stations
    india = find_indian_stations(stations)

    india.to_csv(
        os.path.join(
            STATION_DIR,
            "india_ghcnh_stations.csv"
        ),
        index=False
    )

    # For our project, start with Kerala
    kerala = filter_kerala(india)

    kerala.to_csv(
        os.path.join(
            STATION_DIR,
            "kerala_ghcnh_stations.csv"
        ),
        index=False
    )

    print("\nKerala stations:")
    print(
        kerala[
            [
                "GHCN_ID",
                "LATITUDE",
                "LONGITUDE",
                "ELEVATION",
                "NAME"
            ]
        ].to_string(index=False)
    )

    print("\nStarting downloads...")

    for station_id in tqdm(
        kerala["GHCN_ID"],
        desc="Stations"
    ):
        download_station(station_id)

    print("\nDone.")


if __name__ == "__main__":
    main()