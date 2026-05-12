"""
Data loading, cleaning, and splitting.

The clean step is intentionally explicit — every null-handling decision is
in code, not hidden in a pipeline, because that's what an interviewer wants
to walk through.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "is_default"

# Columns that would leak future information if used as features.
# (In LendingClub these are columns recorded AFTER loan origination —
#  total_pymnt, last_pymnt_d, recoveries, etc. Our synthetic data is
#  already clean, but the constant is here for documentation.)
LEAKAGE_COLS: list[str] = []


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV. Fail loudly if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found at {path}. Run:\n"
            f"  python scripts/generate_sample_data.py --out {path}"
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe.

    Decisions, all documented:
      - emp_length nulls → 0 (treating "unknown" as "no employment data")
      - revol_util nulls → median (small fraction, neutral fill)
      - Drop leakage columns if present
      - Coerce dtypes
    """
    df = df.copy()

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    # Null handling
    df["emp_length"] = df["emp_length"].fillna(0)
    df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())

    # Dtypes
    df["term"] = df["term"].astype(int)
    df["delinq_2yrs"] = df["delinq_2yrs"].astype(int)
    df["open_acc"] = df["open_acc"].astype(int)
    df["total_acc"] = df["total_acc"].astype(int)

    return df


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split on the target.

    Note: for credit data in production, prefer a TIME-BASED split (train on
    older issue dates, test on newer). We use stratified here because the
    synthetic data is generated uniformly across the time window. The
    `issue_d` column is preserved if you want to switch later.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def load_clean_split(
    path: str | Path = "data/loans.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Convenience: full pipeline in one call."""
    df = clean(load_raw(path))
    return split(df, test_size=test_size, random_state=random_state)
