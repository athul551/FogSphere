python tools for collecting hourly weather observations from the NOAA Global Historical Climatology Network hourly dataset (GHCNh), preparing fog labels, and developing a fog-prediction model for Kerala and nearby stations.

## Status

The data collection, inspection, feature generation, and exploratory model scripts are available. The reusable modules in `training/train.py`, `training/preprocess.py`, `prediction/predict.py`, and `api/main.py` are placeholders.

## Quick Start

Run commands from the repository root.

### 1. Install dependencies

```powershell
python -m pip install pandas numpy requests tqdm scikit-learn
```

Python 3.9 or newer is recommended.

### 2. Download weather data

Download the configured station set:

```powershell
python download_selected_stations.py
```

This saves files to `data/raw/`. To discover Indian stations and download stations within the Kerala bounding box instead, run:

```powershell
python download_ghcnh.py
```

Internet access is required for these commands.

### 3. Inspect and prepare the data

```powershell
python inspect_weather_data.py
python create_fog_dataset.py
```

`create_fog_dataset.py` reads `data/processed/weather_dataset.csv` and writes `data/processed/fog_training.csv`. It identifies fog from GHCNh present-weather codes beginning with `FG` and derives rain, drizzle, mist, and dust indicators.

### 4. Analyze and evaluate

```powershell
python training/check_fog_data.py
python training/analyze_visibility.py
python data/processed/analysis.py
python data/processed/particular.py
python training/create_dataset.py
python training/evaluate.py
```

The current experimental Random Forest workflow excludes visibility from model features because visibility can directly reveal fog and cause target leakage. `training/train.py` is currently a placeholder.

## Data Pipeline

1. Download raw GHCNh station files into `data/raw/`.
2. Inspect and combine the weather observations into the processed CSV files.
3. Generate fog labels and derived weather features.
4. Analyze the resulting observations.
5. Train and evaluate the experimental model.

The checked-in processed files include `data/processed/weather.csv` and `data/processed/weather_dataset.csv`.

## Project Structure

```text
api/                    API entry point (not implemented yet)
data/raw/               Downloaded NOAA GHCNh PSV files
data/stations/          NOAA station lists and filtered station metadata
data/processed/         Combined datasets and data-analysis scripts
model/                  Reserved for saved model artifacts
prediction/             Prediction entry point (not implemented yet)
training/               Dataset preparation, training, evaluation, and analysis
create_fog_dataset.py   Generate fog labels and training features
download_ghcnh.py       Discover and download Kerala-area stations
download_selected_stations.py
                        Download the configured station set
inspect_weather_data.py Inspect downloaded or processed weather data
```

## Data Source

Weather observations are downloaded from the [NOAA GHCNh hourly dataset](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-hourly). The raw station files are pipe-separated (`.psv`) and may be large, so avoid committing newly downloaded files unless they are intentionally part of a reproducible sample.

## Notes

- Run commands from the repository root so the relative paths in the scripts resolve correctly.
- Generated files may overwrite existing processed outputs; review the configured input and output paths before running a pipeline step.
- The project does not currently provide automated tests, a packaged command-line interface, or a running prediction API.
