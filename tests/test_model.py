
from pathlib import Path
import joblib
import pandas as pd
import pytest

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "adult.csv"
MODEL_PATH = BASE_DIR / "model.joblib"


def test_dataset_exists():
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"


def test_trained_model_exists():
    assert MODEL_PATH.exists(), (
        f"Model not found: {MODEL_PATH}. Run python train.py first."
    )


def test_model_prediction():
    if not MODEL_PATH.exists():
        pytest.skip("Run python train.py first")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    sample = pd.DataFrame(
        [[0] * len(features)],
        columns=features
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert str(prediction[0]).strip() in {
        "<=50K", ">50K", "<=50K.", ">50K."
    }