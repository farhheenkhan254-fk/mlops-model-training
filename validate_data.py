
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "adult.csv"

REQUIRED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
    "income",
]


def validate_dataset():
    # 1. Dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # 2. Load dataset
    df = pd.read_csv(DATA_PATH)

    # 3. Dataset must not be empty
    if df.empty:
        raise ValueError("Dataset is empty!")

    # 4. Required columns must exist
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # 5. Target must contain valid values
    income = df["income"].astype("string").str.strip()

    valid_income = income.dropna()
    valid_income = valid_income[valid_income != ""]

    if valid_income.empty:
        raise ValueError("Income target has no valid values!")

    # 6. Report missing values
    missing_count = int(df.isna().sum().sum())

    # Adult dataset may represent missing values as '?'
    unknown_count = int(
        df.astype("string").eq("?").sum().sum()
    )

    print("===== DATA VALIDATION =====")
    print(f"Dataset shape: {df.shape}")
    print("Required columns: PASSED")
    print("Dataset non-empty: PASSED")
    print("Income target check: PASSED")
    print(f"Missing values (NaN): {missing_count}")
    print(f"Unknown values ('?'): {unknown_count}")
    print("DATA VALIDATION PASSED")

    return df


if __name__ == "__main__":
    validate_dataset()