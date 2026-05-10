"""
download_data.py
Downloads the UCI Student Performance dataset (id=320) using ucimlrepo,
combines features and targets, adds an 'at_risk' label, and saves to data/students.csv.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_and_prepare():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    print("Fetching UCI Student Performance dataset (id=320)...")
    dataset = fetch_ucirepo(id=320)

    X = dataset.data.features   # Feature columns (demographics, study habits, etc.)
    y = dataset.data.targets    # Target columns: G1, G2, G3 (period grades)

    # Combine into one DataFrame
    df = pd.concat([X, y], axis=1)

    # Add binary at_risk label: 1 if final grade G3 < 10 (failing), else 0
    df["at_risk"] = (df["G3"] < 10).astype(int)

    output_path = os.path.join("data", "students.csv")
    df.to_csv(output_path, index=False)

    total = len(df)
    at_risk_count = df["at_risk"].sum()
    print(f"Dataset saved to {output_path}")
    print(f"Total students: {total}")
    print(f"At-risk students (G3 < 10): {at_risk_count} ({100*at_risk_count/total:.1f}%)")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    download_and_prepare()
