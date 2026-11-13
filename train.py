"""Train and compare machine-failure classifiers on the UCI AI4I 2020 dataset."""
from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import certifi
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATASET_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
CSV_NAME = "ai4i2020.csv"
TARGET = "Machine failure"
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def get_dataset(csv_path: Path | None = None) -> Path:
    """Use a supplied CSV or download and extract the official UCI dataset."""
    if csv_path:
        path = csv_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        return path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / CSV_NAME
    if destination.exists():
        return destination

    archive = DATA_DIR / "ai4i_dataset.zip"
    print("Downloading UCI AI4I 2020 dataset...")
    # Use the regularly updated certifi CA bundle instead of relying only on
    # the Python installation's bundled certificate store (common on Windows).
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PredictiveMaintenanceProject/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, context=ssl_context, timeout=45) as response:
            archive.write_bytes(response.read())
    except (urllib.error.URLError, ssl.SSLError) as exc:
        raise RuntimeError(
            "Could not securely download the UCI dataset because SSL certificate "
            "verification failed. First check that Windows date/time is correct, "
            "then run `python -m pip install --upgrade certifi` and retry. "
            "If it still fails, download the dataset ZIP in your browser from "
            "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset, "
            "extract its CSV, and run `python train.py --csv path\to\ai4i2020.csv`. "
            "SSL verification is intentionally not disabled. Original error: " + str(exc)
        ) from exc
    with zipfile.ZipFile(archive) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError("The downloaded archive did not contain a CSV file.")
        # The archive contains the AI4I dataset CSV.
        chosen = next((n for n in candidates if "ai4i" in n.lower()), candidates[0])
        with zf.open(chosen) as src, destination.open("wb") as dst:
            dst.write(src.read())
    archive.unlink(missing_ok=True)
    return destination


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Path to an AI4I-compatible CSV. Downloads UCI data if omitted.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    csv_path = get_dataset(args.csv)
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found: {list(df.columns)}")

    # Exclude component-level failure flags: they are derived from the failure
    # event and would leak the answer into the model.
    drop_columns = [TARGET] + [c for c in ["UDI", "Product ID", *LEAKAGE_COLUMNS] if c in df.columns]
    X = df.drop(columns=drop_columns)
    y = df[TARGET].astype(int)

    if y.nunique() != 2:
        raise ValueError("The target must contain both classes (0 = no failure, 1 = failure).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=args.random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=args.random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=args.random_state, n_jobs=-1
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, random_state=args.random_state
        ),
    }

    results = []
    best_name, best_pipeline, best_score = None, None, -1.0
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, probabilities)
        except ValueError:
            auc = float("nan")
        ap = average_precision_score(y_test, probabilities)
        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(auc),
            "average_precision": float(ap),
        }
        results.append(metrics)
        print(f"\n{name}")
        print(json.dumps(metrics, indent=2))
        if ap > best_score:
            best_name, best_pipeline, best_score = name, pipeline, ap

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_machine_failure_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    joblib.dump(best_pipeline, model_path)

    best_predictions = best_pipeline.predict(X_test)
    best_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    metadata = {
        "best_model": best_name,
        "target": TARGET,
        "features": X.columns.tolist(),
        "categorical_features": X.select_dtypes(include=["object", "category"]).columns.tolist(),
        "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        "metrics": next(row for row in results if row["model"] == best_name),
        "all_model_results": results,
        "confusion_matrix": confusion_matrix(y_test, best_predictions).tolist(),
        "classification_report": classification_report(y_test, best_predictions, output_dict=True, zero_division=0),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "dataset_file": csv_path.name,
        "leakage_columns_excluded": [c for c in LEAKAGE_COLUMNS if c in df.columns],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, best_predictions))
    print("\nClassification report:")
    print(classification_report(y_test, best_predictions, zero_division=0))


if __name__ == "__main__":
    main()
