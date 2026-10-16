
# Adult Census Income Prediction - MLOps Project

## About

This project predicts income categories using the Adult Census Income dataset and a Random Forest classifier.

## Technologies

- Python
- Pandas
- Scikit-learn
- MLflow
- Pytest
- GitHub Actions (CI/CD)

## Project Structure

```text
mlops_model+training/
├── data/
│   └── adult.csv
├── train.py
├── predict.py
├── test_model.py
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── ml-pipeline.yml
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

Training metrics are logged to MLflow. The trained model is saved as `model.joblib`.

## Run Predictions

```bash
python predict.py
```

## Run Tests

```bash
python -m pytest -v
```

## View MLflow

```bash
python -m mlflow ui
```

Open http://127.0.0.1:5000 in your browser.

## CI/CD

GitHub Actions runs model training and automated tests on pushes and pull requests to the main branch.
