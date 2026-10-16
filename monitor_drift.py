
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "adult.csv"
OUTPUT_DIR = BASE_DIR / "monitoring"
OUTPUT_DIR.mkdir(exist_ok=True)

# Training/reference dataset
df = pd.read_csv(DATA_PATH)

# Features to monitor
features = ["age", "education.num", "hours.per.week", "capital.gain"]

print("===== DATA DRIFT MONITORING =====")

for feature in features:
    reference = pd.to_numeric(df[feature], errors="coerce").dropna()

    # Demonstration: compare two portions of the existing dataset.
    # Replace current with real production data when available.
    current = reference.sample(
        frac=0.3,
        random_state=42
    )

    reference_mean = reference.mean()
    current_mean = current.mean()

    reference_std = reference.std()

    if pd.isna(reference_std) or reference_std == 0:
        drift_score = 0.0
    else:
        drift_score = abs(current_mean - reference_mean) / reference_std

    print(f"\nFeature: {feature}")
    print(f"Reference mean: {reference_mean:.3f}")
    print(f"Current mean:   {current_mean:.3f}")
    print(f"Standardized mean difference: {drift_score:.3f}")

    if drift_score > 0.2:
        print("Status: CHECK FOR DRIFT")
    else:
        print("Status: No large mean shift detected")

    plt.figure(figsize=(7, 4))
    plt.hist(reference, bins=30, alpha=0.6, label="Reference")
    plt.hist(current, bins=30, alpha=0.6, label="Current sample")
    plt.title(f"Distribution comparison: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{feature}_drift.png")
    plt.close()

print("\nMonitoring complete.")
print(f"Reports saved in: {OUTPUT_DIR}")