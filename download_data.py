"""Download UCI Student Performance dataset and save with at_risk label."""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_prepare():
    os.makedirs("data", exist_ok=True)

    print("Fetching UCI Student Performance dataset (id=320)...")
    dataset = fetch_ucirepo(id=320)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # G3 is out of 20 in Portuguese grading — below 10 is a failing grade
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
