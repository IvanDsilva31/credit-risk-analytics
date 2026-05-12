"""
Smoke tests for the pipeline. Run with:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as data_module
from src import features as features_module
from src.ai import FeatureContribution, _templated_explanation


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Minimal in-memory dataframe matching the schema."""
    return pd.DataFrame({
        "loan_amnt": [10000.0, 25000.0, 5000.0, 30000.0],
        "term": [36, 60, 36, 60],
        "int_rate": [10.0, 18.0, 8.0, 22.0],
        "grade": ["A", "C", "A", "E"],
        "emp_length": [5.0, 1.0, np.nan, 10.0],
        "home_ownership": ["MORTGAGE", "RENT", "OWN", "RENT"],
        "annual_inc": [70000.0, 35000.0, 100000.0, 45000.0],
        "verification_status": ["Verified", "Not Verified", "Verified", "Source Verified"],
        "purpose": ["debt_consolidation", "credit_card", "home_improvement", "small_business"],
        "dti": [12.0, 28.0, 8.0, 35.0],
        "delinq_2yrs": [0, 1, 0, 2],
        "open_acc": [8, 5, 12, 4],
        "revol_util": [40.0, 85.0, 20.0, np.nan],
        "total_acc": [20, 15, 25, 10],
        "addr_state": ["CA", "TX", "NY", "FL"],
        "issue_d": ["Jan-2020", "Feb-2020", "Mar-2020", "Apr-2020"],
        "is_default": [0, 1, 0, 1],
    })


def test_clean_fills_nulls(tiny_df: pd.DataFrame) -> None:
    cleaned = data_module.clean(tiny_df)
    assert cleaned["emp_length"].notna().all()
    assert cleaned["revol_util"].notna().all()


def test_split_is_stratified(tiny_df: pd.DataFrame) -> None:
    # Need a larger sample for meaningful stratification; replicate
    df = pd.concat([tiny_df] * 50, ignore_index=True)
    X_tr, X_te, y_tr, y_te = data_module.split(df)
    # Both splits should have at least one of each class
    assert y_tr.nunique() == 2
    assert y_te.nunique() == 2


def test_feature_engineering_adds_columns(tiny_df: pd.DataFrame) -> None:
    cleaned = data_module.clean(tiny_df)
    X, cols = features_module.build_features(cleaned.drop(columns=["is_default"]))
    # Engineered features present
    assert "loan_to_income" in cols
    assert "log_annual_inc" in cols
    # All numeric
    assert X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()


def test_feature_schema_alignment(tiny_df: pd.DataFrame) -> None:
    """When given an explicit column list, the output should match exactly."""
    cleaned = data_module.clean(tiny_df)
    X1, cols = features_module.build_features(cleaned.drop(columns=["is_default"]))
    # Pretend a new row comes in
    new_row = cleaned.drop(columns=["is_default"]).iloc[[0]]
    X2, _ = features_module.build_features(new_row, feature_columns=cols)
    assert list(X2.columns) == cols


def test_templated_explanation_returns_string() -> None:
    contribs = [
        FeatureContribution(name="dti", value=35.0, shap_value=0.4,
                            description="Debt-to-income ratio (%)"),
        FeatureContribution(name="annual_inc", value=45000, shap_value=-0.1,
                            description="Annual income (USD)"),
    ]
    out = _templated_explanation(0.65, contribs)
    assert "DECLINE" in out
    assert "Debt-to-income" in out
