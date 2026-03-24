import os
import sys
import mlflow

THRESHOLD = 0.85


def main():
    # 1. Read Run ID
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("model_info.txt not found.")
        sys.exit(1)

    print(f"Run ID: {run_id}")

    # 2. Connect to MLflow
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 3. Query MLflow for accuracy
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print("'accuracy' metric not found in the MLflow run.")
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD}")

    # 4. Check threshold
    if accuracy < THRESHOLD:
        msg = f"FAILED — accuracy {accuracy:.4f} is below threshold {THRESHOLD}."
        print(msg)
        if "GITHUB_STEP_SUMMARY" in os.environ:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(f"### ❌ {msg}\n")
        sys.exit(1)
    else:
        msg = f"PASSED — accuracy {accuracy:.4f} meets threshold {THRESHOLD}."
        print(msg)
        if "GITHUB_STEP_SUMMARY" in os.environ:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(f"### ✅ {msg}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
