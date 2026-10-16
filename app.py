
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

st.set_page_config(
    page_title="Adult Census Income Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Census Income Predictor")
st.write(
    "Enter the individual's details below to predict their income category."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Please run python train.py first.")
    st.stop()


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["features"]


model, features = load_model()

# Available values based on the Adult Census Income dataset
options = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov",
        "Without-pay", "Never-worked", "?"
    ],
    "education": [
        "Bachelors", "HS-grad", "11th", "Masters",
        "Some-college", "Assoc-acdm", "Assoc-voc",
        "Doctorate", "Prof-school", "5th-6th", "?"
    ],
    "marital.status": [
        "Never-married", "Married-civ-spouse", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse"
    ],
    "occupation": [
        "Prof-specialty", "Craft-repair", "Exec-managerial",
        "Adm-clerical", "Sales", "Other-service",
        "Machine-op-inspct", "Transport-moving",
        "Tech-support", "?"
    ],
    "relationship": [
        "Not-in-family", "Husband", "Wife", "Own-child",
        "Unmarried", "Other-relative"
    ],
    "race": [
        "White", "Black", "Asian-Pac-Islander",
        "Amer-Indian-Eskimo", "Other"
    ],
    "sex": ["Male", "Female"],
    "native.country": [
        "United-States", "Mexico", "India", "Canada",
        "Germany", "Philippines", "Other", "?"
    ]
}

with st.form("income_form"):
    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 17, 90, 35)
        workclass = st.selectbox("Work Class", options["workclass"])
        education = st.selectbox("Education Level", options["education"])
        education_num = st.number_input(
            "Education Number", 1, 16, 10
        )
        marital = st.selectbox(
            "Marital Status", options["marital.status"]
        )
        occupation = st.selectbox(
            "Occupation", options["occupation"]
        )
        relationship = st.selectbox(
            "Relationship", options["relationship"]
        )

    with col2:
        race = st.selectbox("Race", options["race"])
        sex = st.selectbox("Sex", options["sex"])
        country = st.selectbox(
            "Native Country", options["native.country"]
        )
        hours = st.number_input(
            "Hours Worked Per Week", 1, 99, 40
        )
        fnlwgt = st.number_input(
            "Final Weight (FNLWGT)", min_value=1, value=189000
        )
        gain = st.number_input(
            "Capital Gain", min_value=0, value=0
        )
        loss = st.number_input(
            "Capital Loss", min_value=0, value=0
        )

    submitted = st.form_submit_button(
        "Predict Income",
        type="primary",
        use_container_width=True
    )

if submitted:
    person = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education.num": education_num,
        "marital.status": marital,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": gain,
        "capital.loss": loss,
        "hours.per.week": hours,
        "native.country": country,
    }

    # input_df = pd.DataFrame([person])
    # input_df = pd.get_dummies(input_df, dummy_na=True)
    # input_df = input_df.reindex(
    #     columns=features, fill_value=0
    # ).fillna(0)
    input_df = pd.DataFrame([person])

    # Check that input columns match the training features
    input_df = pd.get_dummies(input_df, dummy_na=True)
    input_df = input_df.reindex(
        columns=features,
        fill_value=0
    ).fillna(0)

    prediction = str(model.predict(input_df)[0])

    st.subheader("Prediction Result")
    st.write("Age entered:", age)
    st.write("Model prediction:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        st.write("Prediction probabilities:")
        for label, probability in zip(model.classes_, probabilities):
            st.write(f"{label}: {probability:.2%}")

    prediction = str(model.predict(input_df)[0])

    st.subheader("Prediction Result")

    if ">50K" in prediction:
        st.success(f"Predicted Income Category: {prediction}")
    else:
        st.info(f"Predicted Income Category: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame({
            "Income Category": model.classes_,
            "Estimated Probability": probabilities
        })

        st.subheader("Prediction Probabilities")
        st.bar_chart(
            probability_df.set_index("Income Category")
        )

    st.caption(
        "This is a model estimate, not proof of an individual's actual income."
    )