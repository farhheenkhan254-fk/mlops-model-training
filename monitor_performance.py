
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "adult.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
OUTPUT_DIR = BASE_DIR / "monitoring"
OUTPUT_DIR.mkdir(exist_ok=True)

print("===== MODEL PERFORMANCE MONITORING =====")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python train.py first."
    )

df = pd.read_csv(DATA_PATH)

target_col = "income"
if target_col not in df.columns:
    raise ValueError("Target column 'income' not found.")

X = df.drop(columns=[target_col])
y = df[target_col].astype(str).str.strip()

valid = y.notna() & (y != "") & (y.str.lower() != "nan")
X, y = X.loc[valid].copy(), y.loc[valid].copy()

# Match the preprocessing used by train.py
X = pd.get_dummies(X, dummy_na=True).fillna(0)

counts = y.value_counts()
keep = y.isin(counts[counts >= 2].index)
X, y = X.loc[keep], y.loc[keep]

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

# Ensure columns and their order match the trained model
X = X.reindex(columns=features, fill_value=0)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

# Illustrative thresholds; adjust for your use case
MIN_ACCURACY = 0.80
MIN_F1 = 0.80

status = (
    "PASS"
    if accuracy >= MIN_ACCURACY and f1 >= MIN_F1
    else "ALERT"
)

report = {
    "accuracy": float(accuracy),
    "weighted_f1": float(f1),
    "minimum_accuracy": MIN_ACCURACY,
    "minimum_weighted_f1": MIN_F1,
    "status": status,
    "evaluation_rows": int(len(y_test))
}

report_path = OUTPUT_DIR / "performance_report.json"
report_path.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1: {f1:.4f}")
print(f"Evaluation rows: {len(y_test)}")
print(f"Status: {status}")
print(f"Report saved: {report_path}")