import os
import pandas as pd

DATA_DIR = "data/raw"

files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".psv")
]

print("PSV files found:", len(files))

for filename in files:

    path = os.path.join(DATA_DIR, filename)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    try:
        df = pd.read_csv(
            path,
            sep="|",
            nrows=50
        )

        print("\nColumns:")
        for column in df.columns:
            print(" -", column)

        print("\nFirst rows:")
        print(df.head(2).to_string())

    except Exception as e:
        print("ERROR:", e)