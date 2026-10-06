
# Diabetes Prediction System
# Step 1: Load and explore the dataset

import pandas as pd

df = pd.read_csv("diabetes.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nOutcome distribution:")
print(df["Outcome"].value_counts())