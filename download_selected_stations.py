import os
import requests

OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = (
    "https://www.ncei.noaa.gov/"
    "oa/global-historical-climatology-network/"
    "hourly/access/by-station/"
)

STATIONS = {
    "INI0000VOCI": "Cochin",
    "INI0000VOCL": "Calicut",
    "INI0000VOKN": "Kannur",
    "INI0000VOTV": "Thiruvananthapuram",
    "INM00043335": "Palakkad",
    "INM00043352": "Alappuzha",
    "INM00043339": "Kodaikanal",
    "INM00043341": "Valparai",
}


def download_station(station_id, name):

    filename = f"GHCNh_{station_id}_por.psv"

    url = BASE_URL + filename

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print("\n====================================")
    print(f"Station: {name}")
    print(f"ID:      {station_id}")
    print("====================================")

    # Don't download again if already present
    if os.path.exists(output_path):

        size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"Already downloaded: {size:.2f} MB")

        return True

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=120
        )

        print("HTTP:", response.status_code)

        if response.status_code == 404:

            print("❌ File not found")

            return False

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        downloaded = 0

        with open(
            output_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                file.write(chunk)

                downloaded += len(chunk)

                downloaded_mb = (
                    downloaded /
                    (1024 * 1024)
                )

                if total:

                    total_mb = (
                        total /
                        (1024 * 1024)
                    )

                    percent = (
                        downloaded /
                        total
                    ) * 100

                    print(
                        f"\r{downloaded_mb:.1f} / "
                        f"{total_mb:.1f} MB "
                        f"({percent:.1f}%)",
                        end=""
                    )

                else:

                    print(
                        f"\r{downloaded_mb:.1f} MB",
                        end=""
                    )

        print("\n✅ Download complete")

        return True

    except Exception as e:

        print("\n❌ Error:", e)

        if os.path.exists(output_path):

            os.remove(output_path)

        return False


def main():

    print("Selected stations:", len(STATIONS))

    successful = 0

    for station_id, name in STATIONS.items():

        if download_station(
            station_id,
            name
        ):

            successful += 1

    print("\n====================================")
    print("DOWNLOAD FINISHED")
    print("====================================")
    print(f"Successful: {successful}")
    print(f"Total:      {len(STATIONS)}")


if __name__ == "__main__":
    main()