"""
Model training + evaluation.

Trains three models in sequence, logs metrics, and saves the best one:
  1. Logistic regression baseline (interpretable floor)
  2. Random forest (no scaling, decent default)
  3. XGBoost (usually wins on tabular data)

Evaluation metrics:
  - ROC-AUC, PR-AUC (PR-AUC is more honest on imbalanced data)
  - KS statistic (banking-standard)
  - Confusion matrices at thresholds 0.3 / 0.5 / 0.7
  - Profit curve: optimal threshold for a given gain/loss ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class ModelMetrics:
    name: str
    roc_auc: float
    pr_auc: float
    ks_statistic: float

    def __str__(self) -> str:
        return (
            f"{self.name:18s}  "
            f"ROC-AUC: {self.roc_auc:.4f}  "
            f"PR-AUC: {self.pr_auc:.4f}  "
            f"KS: {self.ks_statistic:.4f}"
        )


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic — max gap between cumulative
    distributions of scores for defaulters vs non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        name=name,
        roc_auc=roc_auc_score(y_true, y_score),
        pr_auc=average_precision_score(y_true, y_score),
        ks_statistic=ks_statistic(y_true, y_score),
    )


def train_logistic(X_train, y_train, X_test, y_test) -> tuple[object, ModelMetrics, object]:
    """Logistic regression baseline. Scaling is required for stable convergence."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xte)[:, 1]
    return model, evaluate("LogisticRegression", y_test, proba), scaler


def train_random_forest(X_train, y_train, X_test, y_test) -> tuple[object, ModelMetrics]:
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, evaluate("RandomForest", y_test, proba)


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple[object, ModelMetrics]:
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    model = XGBClassifier(
        n_estimators=150,          # was 400
        max_depth=3,               # was 5
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=10,       # NEW — prevents tiny leaves
        reg_alpha=0.1,             # NEW — L1 regularization
        reg_lambda=1.0,            # NEW — L2 regularization
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, evaluate("XGBoost", y_test, proba)


def confusion_at_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float] = (0.3, 0.5, 0.7)
) -> pd.DataFrame:
    """Tabular view of TN/FP/FN/TP at several thresholds."""
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "threshold": t,
            "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "approval_rate": (y_pred == 0).mean(),
        })
    return pd.DataFrame(rows)


def profit_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    profit_per_good: float = 1000.0,
    loss_per_bad: float = 4000.0,
) -> pd.DataFrame:
    """
    Profit at each decision threshold. APPROVE if score < threshold,
    DECLINE if score >= threshold (lower score = lower default risk).

    Returns dataframe with columns: threshold, approval_rate, expected_profit.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        approved = y_score < t
        n_good = int(((y_true == 0) & approved).sum())
        n_bad = int(((y_true == 1) & approved).sum())
        profit = n_good * profit_per_good - n_bad * loss_per_bad
        rows.append({
            "threshold": float(t),
            "approval_rate": float(approved.mean()),
            "n_good_approved": n_good,
            "n_bad_approved": n_bad,
            "expected_profit": float(profit),
        })
    return pd.DataFrame(rows)


def train_all(X_train, y_train, X_test, y_test) -> dict:
    """Train all three models, return a dict with everything you need."""
    print("Training models...")
    logreg, m_logreg, scaler = train_logistic(X_train, y_train, X_test, y_test)
    rf, m_rf = train_random_forest(X_train, y_train, X_test, y_test)
    xgb, m_xgb = train_xgboost(X_train, y_train, X_test, y_test)

    print()
    print(m_logreg)
    print(m_rf)
    print(m_xgb)

    return {
        "logreg": {"model": logreg, "scaler": scaler, "metrics": m_logreg},
        "rf":     {"model": rf, "metrics": m_rf},
        "xgb":    {"model": xgb, "metrics": m_xgb},
    }


def save_artifacts(results: dict, feature_columns: list[str], out_dir: str | Path = "models") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best model = highest ROC-AUC
    best_name = max(
        [("logreg", results["logreg"]["metrics"].roc_auc),
         ("rf",     results["rf"]["metrics"].roc_auc),
         ("xgb",    results["xgb"]["metrics"].roc_auc)],
        key=lambda x: x[1],
    )[0]

    payload = {
        "best_name": best_name,
        "feature_columns": feature_columns,
        "logreg_model": results["logreg"]["model"],
        "logreg_scaler": results["logreg"]["scaler"],
        "rf_model": results["rf"]["model"],
        "xgb_model": results["xgb"]["model"],
        "metrics": {
            "logreg": vars(results["logreg"]["metrics"]),
            "rf":     vars(results["rf"]["metrics"]),
            "xgb":    vars(results["xgb"]["metrics"]),
        },
    }
    joblib.dump(payload, out_dir / "model_bundle.joblib")
    print(f"\nSaved bundle → {out_dir / 'model_bundle.joblib'}")
    print(f"Best model: {best_name}")


def load_bundle(path: str | Path = "models/model_bundle.joblib") -> dict:
    return joblib.load(path)


def predict_proba(bundle: dict, X: pd.DataFrame, model_name: str | None = None) -> np.ndarray:
    """Predict default probability using a chosen model from the bundle."""
    model_name = model_name or bundle["best_name"]
    if model_name == "logreg":
        Xs = bundle["logreg_scaler"].transform(X)
        return bundle["logreg_model"].predict_proba(Xs)[:, 1]
    if model_name == "rf":
        return bundle["rf_model"].predict_proba(X)[:, 1]
    if model_name == "xgb":
        return bundle["xgb_model"].predict_proba(X)[:, 1]
    raise ValueError(f"Unknown model {model_name!r}")
