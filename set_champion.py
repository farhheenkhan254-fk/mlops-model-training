from pathlib import Path
import json
import sys

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "AdultIncomeClassifier"
BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "monitoring" / "performance_report.json"

client = MlflowClient()


def promote_model():
    # 1. Verify performance report exists
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Performance report not found: {REPORT_PATH}. "
            "Run python monitor_performance.py first."
        )

    with REPORT_PATH.open(encoding="utf-8") as file:
        report = json.load(file)

    # 2. Reject models that fail performance checks
    if report.get("status") != "PASS":
        raise RuntimeError(
            "Model promotion blocked: performance check failed."
        )

    # 3. Get the requested version, or use the latest registered version
    if len(sys.argv) > 1:
        version_number = sys.argv[1]
    else:
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
        if not versions:
            raise RuntimeError(
                f"No registered versions found for {MODEL_NAME}."
            )
        version_number = str(
            max(int(v.version) for v in versions)
        )

    # 4. Verify that the selected version exists
    version = client.get_model_version(
        name=MODEL_NAME,
        version=version_number,
    )

    # 5. Promote only when the checks pass
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version.version,
    )

    print("MODEL PROMOTION SUCCESSFUL")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Champion version: {version.version}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Weighted F1: {report['weighted_f1']:.4f}")


if __name__ == "__main__":
    promote_model()