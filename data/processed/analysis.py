import pandas as pd

df = pd.read_csv("fog_training.csv")

df["DATE"] = pd.to_datetime(df["DATE"])

print(df.head())

print(df.info())

print(df.isnull().sum())

print(df.columns)