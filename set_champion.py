
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "AdultIncomeClassifier"
MODEL_VERSION = "2"

client = MlflowClient()

# Confirm that the model version exists
version = client.get_model_version(
    name=MODEL_NAME,
    version=MODEL_VERSION,
)

# Assign the production alias
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=MODEL_VERSION,
)

print("Model registry updated successfully")
print(f"Registered model: {MODEL_NAME}")
print(f"Champion version: {version.version}")
print("Alias: champion")