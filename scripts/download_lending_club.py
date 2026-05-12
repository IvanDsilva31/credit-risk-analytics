"""
Download a real LendingClub-style dataset from Hugging Face.

Source: AnguloM/loan_data — 9,578 real loans, 223 KB, redistributable under
the Database Contents License (DbCL) v1.0.

Schema mapping from the HF dataset to this project's internal schema:
  - not.fully.paid    → is_default (target)
  - int.rate          → int_rate (× 100, HF stores as decimal)
  - log.annual.inc    → annual_inc (np.exp to invert)
  - dti               → dti
  - delinq.2yrs       → delinq_2yrs
  - revol.util        → revol_util
  - purpose           → purpose (normalize "all_other" → "other")
  - installment       → kept as new column; also used to derive loan_amnt
  - fico              → kept as new column (real strong predictor)
  - revol.bal         → kept as new column
  - inq.last.6mths    → kept as new column
  - pub.rec           → kept as new column

Columns the project's synthetic schema has that aren't in the HF data
(term, grade, emp_length, home_ownership, etc.) are filled with sensible
mode values so the existing pipeline works unchanged. Since those filled
columns have no variance, the model simply ignores them — no harm done.

Usage:
    python scripts/download_lending_club.py
    python scripts/download_lending_club.py --out data/loans.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HF_DATASET_ID = "AnguloM/loan_data"


def derive_loan_amount(installment: float, int_rate_pct: float, term: int = 36) -> float:
    """
    Solve the amortizing-loan formula for principal P given monthly installment M:
        P = M * (1 - (1+i)^-N) / i
    where i = APR/12 (monthly rate) and N = term in months.
    """
    apr = int_rate_pct / 100.0
    i = apr / 12.0
    if i <= 0:
        return installment * term
    multiplier = (1 - (1 + i) ** (-term)) / i
    return installment * multiplier


def derive_grade(int_rate_pct: float) -> str:
    """Map interest rate to LendingClub-style A-G grade."""
    if int_rate_pct < 8:
        return "A"
    if int_rate_pct < 11:
        return "B"
    if int_rate_pct < 14:
        return "C"
    if int_rate_pct < 17:
        return "D"
    if int_rate_pct < 20:
        return "E"
    if int_rate_pct < 24:
        return "F"
    return "G"


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the HF schema to this project's internal schema."""
    df = pd.DataFrame()

    # --- Direct mappings ---
    df["is_default"] = raw["not.fully.paid"].astype(int)
    df["int_rate"] = (raw["int.rate"] * 100).round(2)
    df["dti"] = raw["dti"]
    df["delinq_2yrs"] = raw["delinq.2yrs"].astype(int)
    df["revol_util"] = raw["revol.util"]

    # purpose: normalize "all_other" → "other" so it matches the synthetic vocabulary
    df["purpose"] = raw["purpose"].replace({"all_other": "other"})

    # Derive annual income from the log-transformed value
    df["annual_inc"] = np.exp(raw["log.annual.inc"]).round(2)

    # --- Real-data-only columns (kept as new features) ---
    df["installment"] = raw["installment"]
    df["fico"] = raw["fico"].astype(int)
    df["revol_bal"] = raw["revol.bal"]
    df["inq_last_6mths"] = raw["inq.last.6mths"].astype(int)
    df["pub_rec"] = raw["pub.rec"].astype(int)

    # --- Derived columns the project expects ---
    df["term"] = 36  # this HF dataset doesn't store term; 36 is by far the modal value
    df["loan_amnt"] = df.apply(
        lambda r: derive_loan_amount(r["installment"], r["int_rate"], r["term"]),
        axis=1,
    ).round(-2)  # round to nearest $100
    df["grade"] = df["int_rate"].apply(derive_grade)

    # --- Constant fills for columns the HF dataset doesn't have ---
    # (Pipeline expects them; constants mean they get ignored by the model)
    df["emp_length"] = 5  # median
    df["home_ownership"] = "MORTGAGE"  # mode in real LendingClub
    df["verification_status"] = "Verified"
    df["addr_state"] = "CA"
    df["issue_d"] = "Jan-2014"  # this dataset is from ~2010-2014
    df["open_acc"] = 10
    df["total_acc"] = 25

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/loans.csv", help="Output CSV path")
    parser.add_argument("--dataset", default=HF_DATASET_ID,
                        help="Hugging Face dataset ID (default: AnguloM/loan_data)")
    args = parser.parse_args()

    # Late import so the synthetic-data path doesn't require the `datasets` lib
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The `datasets` library is required for the real-data path.\n"
            "Install it with:\n"
            "    pip install datasets"
        )

    print(f"Downloading {args.dataset} from Hugging Face...")
    ds = load_dataset(args.dataset, split="train")
    raw = ds.to_pandas()
    print(f"  Got {len(raw):,} rows, {len(raw.columns)} columns")

    print("Transforming to project schema...")
    df = transform(raw)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} rows → {out_path}")
    print(f"  Default rate: {df['is_default'].mean():.2%}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nNext step:")
    print(f"  python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
