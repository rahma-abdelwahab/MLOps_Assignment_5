import argparse
import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser(description="Train an SVM classifier and log to MLflow.")
    parser.add_argument(
        "--accuracy-override",
        type=float,
        default=None,
        help="If set, override the real accuracy with this value (for demo purposes).",
    )
    args = parser.parse_args()

    # 1. Load data
    import pandas as pd
    df = pd.read_csv("data/breast_cancer.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train
    model = SVC(kernel="rbf", C=1.0, random_state=42)
    model.fit(X_train, y_train)

    # 3. Evaluate
    predictions = model.predict(X_test)
    real_accuracy = accuracy_score(y_test, predictions)

    # Use override if provided (useful for producing fail / pass screenshots)
    accuracy = args.accuracy_override if args.accuracy_override is not None else real_accuracy
    print(f"Real accuracy : {real_accuracy:.4f}")
    print(f"Logged accuracy: {accuracy:.4f}")

    # 4. Log to MLflow
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("breast-cancer-svm")

    with mlflow.start_run() as run:
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"MLflow Run ID : {run_id}")

    # 5. Export Run ID
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print("model_info.txt written successfully.")


if __name__ == "__main__":
    main()
