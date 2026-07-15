import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "modeling" / "artifacts" / "splits"

EXPERIMENT_NAME = "5g-nidd-attack-classification"


def load_splits():
    """Load the parquet splits produced by splits.py."""
    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    X_val = pd.read_parquet(SPLITS_DIR / "X_val.parquet")
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()
    y_val = pd.read_parquet(SPLITS_DIR / "y_val.parquet").squeeze()
    return X_train, X_val, y_train, y_val


def train_logreg():
    X_train, X_val, y_train, y_val = load_splits()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline_logreg_scaled"):
        # Log hyperparameters
        params = {
            "model": "LogisticRegression",
            "scaler": "StandardScaler",
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        }
        mlflow.log_params(params)

        # Fit
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        # Evaluate on val
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)

        # Log overall + per-class metrics
        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
        mlflow.log_metric("accuracy", report["accuracy"])
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in (
                "macro avg",
                "weighted avg",
            ):
                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])

        # Log the model itself
        mlflow.sklearn.log_model(model, name="model")

        # Log confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred, ax=ax, xticks_rotation=45
        )
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Feature importance
        logreg = model.named_steps["logreg"]
        importance = np.abs(logreg.coef_).mean(axis=0)
        top20 = (
            pd.Series(importance, index=X_train.columns)
            .sort_values(ascending=False)
            .head(20)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        top20.plot.barh(ax=ax)
        ax.set_title("Top 20 feature importances (mean |coef|)")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # Print for immediate feedback
        print(classification_report(y_val, y_pred))
        print("\nTop 10 features by importance:")
        print(top20.head(10))


def train_logreg_no_toolprint():
    """Train LogReg without sMeanPktSz to test robustness against tool fingerprinting."""
    X_train, X_val, y_train, y_val = load_splits()
    X_train = X_train.drop(columns=["sMeanPktSz"])
    X_val = X_val.drop(columns=["sMeanPktSz"])

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="logreg_scaled_no_sMeanPktSz"):
        # Log hyperparameters
        params = {
            "model": "LogisticRegression",
            "scaler": "StandardScaler",
            "dropped_feature": "sMeanPktSz",
            "class_weight": "balanced",
            "max_iter": 5000,
            "solver": "lbfgs",
            "random_state": 42,
        }
        mlflow.log_params(params)

        # Fit
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        # Evaluate on val
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)

        # Log overall + per-class metrics
        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
        mlflow.log_metric("accuracy", report["accuracy"])
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in (
                "macro avg",
                "weighted avg",
            ):
                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])

        # Log the model itself
        mlflow.sklearn.log_model(model, name="model")

        # Log confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred, ax=ax, xticks_rotation=45
        )
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Feature importance
        logreg = model.named_steps["logreg"]
        importance = np.abs(logreg.coef_).mean(axis=0)
        top20 = (
            pd.Series(importance, index=X_train.columns)
            .sort_values(ascending=False)
            .head(20)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        top20.plot.barh(ax=ax)
        ax.set_title("Top 20 feature importances (mean |coef|)")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # Print for immediate feedback
        print(classification_report(y_val, y_pred))
        print("\nTop 10 features by importance:")
        print(top20.head(10))




def train_random_forest():
    X_train, X_val, y_train, y_val = load_splits()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="random_forest_baseline"):
        params = {
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "class_weight": "balanced",
            "max_depth": None,
            "n_jobs": -1,
            "random_state": 42,
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)

        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
        mlflow.log_metric("accuracy", report["accuracy"])
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ("macro avg", "weighted avg"):
                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])

        mlflow.sklearn.log_model(model, name="model")

        # Feature importance — RF gives this natively, no coef math needed
        importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        # ... same feature_importance + confusion_matrix logging blocks as before ...
        
        # Log confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred, ax=ax, xticks_rotation=45
        )
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        
        # Feature importance
        top20 = importance.head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        top20.plot.barh(ax=ax)
        ax.set_title("Top 20 feature importances (mean |coef|)")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        
        print(classification_report(y_val, y_pred))
        print("\nTop 10 features by importance:")
        print(top20.head(10))
        
if __name__ == "__main__":
    train_logreg_no_toolprint()
