import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import mlflow.xgboost


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
        
def train_xgboost():
    X_train, X_val, y_train, y_val = load_splits()

    # XGBoost needs integer labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    # Handle class imbalance via sample weights (XGBoost's equivalent to class_weight='balanced')
    sample_weights = compute_sample_weight("balanced", y_train_enc)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_engineered_features"):
        params = {
            "model": "XGBClassifier",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "class_weighting": "balanced_sample_weights",
            "n_jobs": -1,
            "random_state": 42,
            "features": "baseline + engineered (ratios + logs)"
        }
        mlflow.log_params(params)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",   # fast histogram-based training
        )
        model.fit(X_train, y_train_enc, sample_weight=sample_weights)

        y_pred_enc = model.predict(X_val)
        y_pred = le.inverse_transform(y_pred_enc)

        report = classification_report(y_val, y_pred, output_dict=True)
        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
        mlflow.log_metric("accuracy", report["accuracy"])
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ("macro avg", "weighted avg"):
                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])

        mlflow.xgboost.log_model(model, name="model")

        importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        # ... same confusion matrix + feature importance logging as before ...
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


def threshold_analysis(run_id: str):
    """Sweep UDPFlood threshold on a previously-trained model loaded from MLflow."""
    X_train, X_val, y_train, y_val = load_splits()

    # Load the trained model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)
    print(f"Loaded model from {model_uri}")

    # The model expects encoded labels — refit the encoder on training labels
    # (deterministic given the same y_train, so this reproduces the training-time encoding)
    le = LabelEncoder()
    le.fit(y_train)

    y_proba = model.predict_proba(X_val)
    udpflood_idx = list(le.classes_).index("UDPFlood")

    thresholds = np.arange(0.30, 0.96, 0.05)
    results = []

    for t in thresholds:
        udpflood_prob = y_proba[:, udpflood_idx]
        default_preds = np.argmax(y_proba, axis=1)
        alt_preds = np.argsort(y_proba, axis=1)[:, -2]

        # Only override when the model was going to predict UDPFlood AND prob is below threshold
        override_mask = (default_preds == udpflood_idx) & (udpflood_prob <= t)
        preds_enc = np.where(override_mask, alt_preds, default_preds)

        y_pred = le.inverse_transform(preds_enc)

        results.append({
            "threshold": round(t, 2),
            "benign_recall": recall_score(y_val, y_pred, labels=["Benign"], average="macro"),
            "udpflood_recall": recall_score(y_val, y_pred, labels=["UDPFlood"], average="macro"),
            "macro_f1": f1_score(y_val, y_pred, average="macro"),
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(results_df["threshold"], results_df["benign_recall"], marker="o", label="Benign recall")
    ax.plot(results_df["threshold"], results_df["udpflood_recall"], marker="o", label="UDPFlood recall")
    ax.plot(results_df["threshold"], results_df["macro_f1"], marker="o", label="Macro-F1", linestyle="--")
    ax.set_xlabel("UDPFlood prediction threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Trade-off: Benign vs UDPFlood recall under threshold tuning")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("threshold_analysis.png")
    plt.close()
    print("Saved threshold_analysis.png")

    return results_df


def train_xgboost_binary():
    """Train a binary XGBoost: Benign vs Malicious.

    Framed as an operational-first model: for a real 5G security system, the
    primary question is often 'is this flow malicious?' rather than 'which
    specific attack type is this?'. This model answers the binary question
    directly, sidestepping the Benign↔UDPFlood multiclass confusion.
    """
    X_train, X_val, y_train, y_val = load_splits()

    # Derive binary labels from Attack Type
    y_train_bin = (y_train != "Benign").astype(int)  # 1 = malicious, 0 = benign
    y_val_bin = (y_val != "Benign").astype(int)

    print(f"Train class balance: malicious={y_train_bin.sum()}, benign={(1-y_train_bin).sum()}")
    print(f"Val class balance:   malicious={y_val_bin.sum()}, benign={(1-y_val_bin).sum()}")

    # scale_pos_weight is XGBoost's binary imbalance handling
    n_neg = (y_train_bin == 0).sum()
    n_pos = (y_train_bin == 1).sum()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_binary"):
        params = {
            "model": "XGBClassifier",
            "task": "binary",
            "target": "Benign vs Malicious",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42,
            "tree_method": "hist",
        }
        mlflow.log_params(params)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        )
        model.fit(X_train, y_train_bin)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Binary-specific metrics
        from sklearn.metrics import (
            classification_report, f1_score, precision_score,
            recall_score, roc_auc_score
        )

        report = classification_report(
            y_val_bin, y_pred,
            target_names=["Benign", "Malicious"],
            output_dict=True,
        )

        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("f1_binary", f1_score(y_val_bin, y_pred))
        mlflow.log_metric("precision_malicious", precision_score(y_val_bin, y_pred))
        mlflow.log_metric("recall_malicious", recall_score(y_val_bin, y_pred))
        mlflow.log_metric("precision_benign", precision_score(1 - y_val_bin, 1 - y_pred))
        mlflow.log_metric("recall_benign", recall_score(1 - y_val_bin, 1 - y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_val_bin, y_proba))

        mlflow.xgboost.log_model(model, name="model")

        # Confusion matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_val_bin, y_pred,
            display_labels=["Benign", "Malicious"],
            ax=ax,
        )
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Feature importance
        importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(8, 6))
        importance.plot.barh(ax=ax)
        ax.set_title("Top 20 feature importances (binary XGBoost)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        print("\n" + classification_report(
            y_val_bin, y_pred,
            target_names=["Benign", "Malicious"],
        ))
        print(f"\nROC-AUC: {roc_auc_score(y_val_bin, y_proba):.4f}")
        print("\nTop 10 features:")
        print(importance.head(10))

def probe_benign_vs_udpflood():
    """Quick probe: what's the best possible Benign vs UDPFlood classifier?
    Not a production model — just measures the ceiling of the two-class problem
    to decide whether a cascade approach is worth building.
    """
    X_train, X_val, y_train, y_val = load_splits()
    X_train = X_train.drop(columns=["sMeanPktSz"])
    X_val = X_val.drop(columns=["sMeanPktSz"])

    # Subset: only Benign and UDPFlood
    train_mask = y_train.isin(["Benign", "UDPFlood"])
    val_mask = y_val.isin(["Benign", "UDPFlood"])

    X_train_sub = X_train[train_mask]
    y_train_sub = y_train[train_mask]
    X_val_sub = X_val[val_mask]
    y_val_sub = y_val[val_mask]

    print(f"Train: {X_train_sub.shape}, balance: {y_train_sub.value_counts().to_dict()}")
    print(f"Val:   {X_val_sub.shape}, balance: {y_val_sub.value_counts().to_dict()}")

    # Binary encoding
    y_train_bin = (y_train_sub == "UDPFlood").astype(int)
    y_val_bin = (y_val_sub == "UDPFlood").astype(int)

    from sklearn.utils.class_weight import compute_sample_weight
    weights = compute_sample_weight("balanced", y_train_bin)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train_sub, y_train_bin, sample_weight=weights)

    y_pred = model.predict(X_val_sub)
    y_proba = model.predict_proba(X_val_sub)[:, 1]

    from sklearn.metrics import (
        classification_report, roc_auc_score, f1_score,
    )
    print("\nClassification report:")
    print(classification_report(
        y_val_bin, y_pred,
        target_names=["Benign", "UDPFlood"],
    ))
    print(f"ROC-AUC: {roc_auc_score(y_val_bin, y_proba):.4f}")
    print(f"F1: {f1_score(y_val_bin, y_pred):.4f}")

    # Importance — where does this specialist focus?
    importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False).head(10)
    print("\nTop 10 features:")
    print(importance)


def train_xgboost_manual_weights():
    """XGBoost multiclass with manually tuned class weights.

    The default sample_weight='balanced' weights each class inversely to its
    frequency, which was found to over-correct for UDPFlood (causing the
    Benign->UDPFlood false-positive tendency documented in prior runs). Here
    we deliberately ease off the UDPFlood weight while preserving heavy
    weighting for the truly rare attack classes (ICMPFlood especially).
    """
    X_train, X_val, y_train, y_val = load_splits()

    # Encode labels for XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    # Manual weights — reasoning per class:
    #  Benign: slightly up-weighted vs balanced (we want to reduce Benign FPs)
    #  UDPFlood: slightly down-weighted (was over-predicted with 'balanced')
    #  ICMPFlood: heavy weight — rarest class, only 808 train rows
    #  SYNFlood: elevated — moderate rare class
    #  Scans (SYNScan, TCPConnectScan, UDPScan): elevated to keep recall high
    #  HTTPFlood, SlowrateDoS: baseline (they were already fine at balanced)
    class_weights = {
        "Benign": 1.5,
        "UDPFlood": 0.7,
        "HTTPFlood": 1.0,
        "ICMPFlood": 10.0,
        "SYNFlood": 3.0,
        "SYNScan": 3.0,
        "SlowrateDoS": 1.0,
        "TCPConnectScan": 3.0,
        "UDPScan": 3.0,
    }

    # Map each training row's label to its weight
    sample_weights = np.array([class_weights[cls] for cls in y_train])

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_manual_weights"):
        params = {
            "model": "XGBClassifier",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "class_weighting": "manual_dict",
            "n_jobs": -1,
            "random_state": 42,
        }
        mlflow.log_params(params)

        # Log the actual weight dict as a param for future reference
        for cls, w in class_weights.items():
            mlflow.log_param(f"weight_{cls}", w)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        )
        model.fit(X_train, y_train_enc, sample_weight=sample_weights)

        y_pred_enc = model.predict(X_val)
        y_pred = le.inverse_transform(y_pred_enc)

        report = classification_report(y_val, y_pred, output_dict=True)
        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
        mlflow.log_metric("accuracy", report["accuracy"])
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ("macro avg", "weighted avg"):
                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])

        mlflow.xgboost.log_model(model, name="model")

        importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        print(classification_report(y_val, y_pred))
        print("\nTop 10 features:")
        print(importance.head(10))


if __name__ == "__main__":
    #RUN_ID = "ff2b46e8ac4a4ff0b9a054a57743de6e"
    #threshold_analysis(RUN_ID)
    train_xgboost_manual_weights()