
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "adult.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

def train_model():

    # 1. Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # 2. Find target column
    target_col = next(
        (
            c for c in df.columns
            if c.strip().lower() in [
                "income", "class", "target", "income_bracket"
            ]
        ),
        None
    )

    if target_col is None:
        raise ValueError(
            "Income target column not found. Check dataset columns."
        )

    # 3. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).str.strip()

    # Remove missing target values
    valid = (
        y.notna()
        & (y != "")
        & (y.str.lower() != "nan")
    )

    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    # 4. Encode categorical columns
    X = pd.get_dummies(X, dummy_na=True)
    X = X.fillna(0)

    # Remove classes with fewer than 2 samples
    counts = y.value_counts()
    keep = y.isin(counts[counts >= 2].index)

    X = X.loc[keep]
    y = y.loc[keep]

    # 5. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 6. Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # 7. Start MLflow experiment
    mlflow.set_experiment("adult-income-classification")

    with mlflow.start_run():

        # Train
        model.fit(X_train, y_train)

        # Predict on held-out test set
        predictions = model.predict(X_test)

        # 8. Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        # 9. Print metrics
        print("\n===== MODEL EVALUATION =====")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\n===== CLASSIFICATION REPORT =====")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print("\n===== CONFUSION MATRIX =====")
        labels = sorted(y.unique())
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=labels
        )

        print("Class labels:", labels)
        print(matrix)

        # 10. Log parameters to MLflow
        mlflow.log_param(
            "model_type", "RandomForestClassifier"
        )
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("testing_rows", len(X_test))

        # 11. Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1)
        })

        # Log confusion matrix as an artifact
        report_path = BASE_DIR / "classification_report.txt"

        report_path.write_text(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            ),
            encoding="utf-8"
        )

        mlflow.log_artifact(str(report_path))

        # 12. Log trained model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="adult-income-model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # 13. Save model and feature names for Streamlit
        joblib.dump(
            {
                "model": model,
                "features": X.columns.tolist()
            },
            MODEL_PATH
        )

        print("\nModel saved:", MODEL_PATH)
        print("MLflow metrics and model logged successfully.")


if __name__ == "__main__":
    train_model()