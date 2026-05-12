"""
Feature engineering.

Two surfaces:
  - `build_features(df)`: returns a feature matrix ready for sklearn/xgboost,
    using one-hot encoding for categoricals.
  - `FEATURE_DESCRIPTIONS`: human-readable column names used in LLM prompts
    and dashboard tooltips.

Engineered features:
  - loan_to_income: loan_amnt / annual_inc
  - log_annual_inc: log-transform of income (handles right skew)
  - emp_length_bucket: 0, 1-3, 4-6, 7-9, 10+
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLS = [
    "grade",
    "home_ownership",
    "purpose",
    "verification_status",
    "addr_state",
    "emp_length_bucket",
]

NUMERICAL_COLS = [
    "loan_amnt",
    "term",
    "int_rate",
    "annual_inc",
    "log_annual_inc",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "revol_util",
    "total_acc",
    "emp_length",
    "loan_to_income",
    # Real-data-only (present when loaded via download_lending_club.py)
    "installment",
    "fico",
    "revol_bal",
    "inq_last_6mths",
    "pub_rec",
]

# Friendly labels for LLM explanations and dashboard tooltips
FEATURE_DESCRIPTIONS = {
    "loan_amnt": "Loan amount (USD)",
    "term": "Loan term (months)",
    "int_rate": "Interest rate (%)",
    "grade": "Loan grade (A–G)",
    "annual_inc": "Annual income (USD)",
    "log_annual_inc": "Log of annual income",
    "dti": "Debt-to-income ratio (%)",
    "delinq_2yrs": "Delinquencies in past 2 years",
    "open_acc": "Open credit lines",
    "revol_util": "Revolving credit utilization (%)",
    "total_acc": "Total credit accounts",
    "emp_length": "Employment length (years)",
    "emp_length_bucket": "Employment tenure bucket",
    "home_ownership": "Home ownership status",
    "purpose": "Loan purpose",
    "verification_status": "Income verification status",
    "addr_state": "Borrower state",
    "loan_to_income": "Loan amount as fraction of income",
    # Real-data-only columns
    "installment": "Monthly installment (USD)",
    "fico": "FICO credit score",
    "revol_bal": "Revolving credit balance (USD)",
    "inq_last_6mths": "Credit inquiries in last 6 months",
    "pub_rec": "Public records (bankruptcies, foreclosures)",
}


def _bucket_emp_length(years: float) -> str:
    if years == 0:
        return "0"
    if years <= 3:
        return "1-3"
    if years <= 6:
        return "4-6"
    if years <= 9:
        return "7-9"
    return "10+"


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Doesn't drop anything."""
    df = df.copy()
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].clip(lower=1)
    df["emp_length_bucket"] = df["emp_length"].apply(_bucket_emp_length)
    return df


def build_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a model-ready feature matrix.

    On the training pass, call with feature_columns=None — the function
    returns the column list to reuse for the test/inference pass so the
    one-hot schema matches.
    """
    df = add_engineered(df)

    keep = [c for c in (NUMERICAL_COLS + CATEGORICAL_COLS) if c in df.columns]
    X = df[keep]

    X = pd.get_dummies(X, columns=[c for c in CATEGORICAL_COLS if c in X.columns],
                       drop_first=False)

    if feature_columns is not None:
        # Align to the training schema: add any missing dummies as 0,
        # drop any extras (rare unseen categories).
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
    else:
        feature_columns = X.columns.tolist()

    # XGBoost wants numeric dtypes; pd.get_dummies sometimes leaves bool
    X = X.astype(float)

    return X, feature_columns
