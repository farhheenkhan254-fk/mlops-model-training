
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


def predict_income(person):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model nahi mila. Pehle python train.py run karo."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    # Input ko DataFrame mein convert karo
    person_df = pd.DataFrame([person])

    # Training ke same feature columns rakho
    person_df = pd.get_dummies(person_df, dummy_na=True)
    person_df = person_df.reindex(columns=features, fill_value=0)

    prediction = model.predict(person_df)[0]
    print("Predicted income:", prediction)
    return prediction


if __name__ == "__main__":
    sample_person = {
        "age": 39,
        "workclass": "Private",
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    predict_income(sample_person)