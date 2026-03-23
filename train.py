import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
 
# ── MLflow setup ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("breast-cancer-svm")
 
# ── Hyperparameters (can be overridden via env vars) ─────────────────────────
C          = float(os.environ.get("SVM_C",       "1.0"))
KERNEL     = os.environ.get("SVM_KERNEL",        "rbf")
GAMMA      = os.environ.get("SVM_GAMMA",         "scale")
TEST_SIZE  = float(os.environ.get("TEST_SIZE",   "0.2"))
RANDOM_STATE = 42
 
# ── Load dataset ──────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
 
print(f"[train.py] Dataset    : Breast Cancer Wisconsin")
print(f"[train.py] Samples    : {X.shape[0]}  |  Features: {X.shape[1]}")
print(f"[train.py] Classes    : {list(data.target_names)}")
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
 
# ── Build pipeline ────────────────────────────────────────────────────────────
clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(C=C, kernel=KERNEL, gamma=GAMMA, random_state=RANDOM_STATE)),
])
 
# ── Train & evaluate inside an MLflow run ─────────────────────────────────────
with mlflow.start_run() as run:
 
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)
 
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
 
    # Log hyperparameters
    mlflow.log_param("algorithm",   "SVM")
    mlflow.log_param("dataset",     "breast_cancer")
    mlflow.log_param("C",           C)
    mlflow.log_param("kernel",      KERNEL)
    mlflow.log_param("gamma",       GAMMA)
    mlflow.log_param("test_size",   TEST_SIZE)
 
    # Log all metrics
    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1",        f1)
 
    # Log the full sklearn pipeline as a model artifact
    mlflow.sklearn.log_model(clf_pipeline, "model")
 
    run_id = run.info.run_id
 
    print(f"\n[train.py] ── Results ──────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print(f"[train.py] Accuracy  : {accuracy:.4f}")
    print(f"[train.py] Precision : {precision:.4f}")
    print(f"[train.py] Recall    : {recall:.4f}")
    print(f"[train.py] F1 Score  : {f1:.4f}")
    print(f"[train.py] Run ID    : {run_id}")
 
# ── Write Run ID for the deploy job ───────────────────────────────────────────
with open("model_info.txt", "w") as f:
    f.write(run_id)
