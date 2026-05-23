"""
Load the full LendingClub Kaggle dataset and transform to project schema.

Source: kaggle.com/datasets/wordsforthewise/lending-club
File expected at: data/accepted_2007_to_2018Q4.csv.gz

Drops leakage columns (recorded post-origination), keeps strong predictors
(FICO, installment, revolving balance, inquiries, public records).

Usage:
    python scripts/download_full_lendingclub.py
    python scripts/download_full_lendingclub.py --sample 200000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Columns recorded AFTER loan origination — these leak the outcome.
# Including them would inflate AUC artificially. NEVER train on these.
LEAKAGE_COLS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "out_prncp", "out_prncp_inv", "funded_amnt", "funded_amnt_inv",
    "loan_status",  # this is the source of our target, drop after extracting
    "settlement_amount", "settlement_percentage", "settlement_status",
    "settlement_date", "settlement_term", "debt_settlement_flag",
    "debt_settlement_flag_date", "hardship_flag", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "disbursement_method", "pymnt_plan",
]

# Columns we want to keep as features
FEATURE_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "open_acc", "revol_bal", "revol_util",
    "total_acc", "addr_state", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "pub_rec", "issue_d",
]

TARGET_BAD = {"Charged Off", "Default", "Late (31-120 days)",
              "Does not meet the credit policy. Status:Charged Off"}
TARGET_GOOD = {"Fully Paid",
               "Does not meet the credit policy. Status:Fully Paid"}


def clean_percent(series: pd.Series) -> pd.Series:
    """Convert '10.65%' string to 10.65 float."""
    return pd.to_numeric(series.astype(str).str.rstrip("%").str.strip(),
                          errors="coerce")


def clean_term(series: pd.Series) -> pd.Series:
    """Convert ' 36 months' string to 36 int."""
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0],
                          errors="coerce").astype("Int64")


def clean_emp_length(series: pd.Series) -> pd.Series:
    """Convert '5 years' / '10+ years' / '< 1 year' to int."""
    s = series.astype(str)
    s = s.replace({"< 1 year": "0", "10+ years": "10", "n/a": np.nan})
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the Kaggle schema to this project's internal schema."""
    print(f"  Starting with {len(raw):,} rows and {len(raw.columns)} columns")

    # Filter to definitive outcomes (drop in-flight loans)
    raw = raw[raw["loan_status"].isin(TARGET_BAD | TARGET_GOOD)].copy()
    print(f"  After filtering to definitive outcomes: {len(raw):,} rows")

    # Build target
    df = pd.DataFrame()
    df["is_default"] = raw["loan_status"].isin(TARGET_BAD).astype(int)

    # Clean percent columns
    df["int_rate"] = clean_percent(raw["int_rate"])
    df["revol_util"] = clean_percent(raw["revol_util"])

    # Clean term
    df["term"] = clean_term(raw["term"]).fillna(36).astype(int)

    # Clean employment length
    df["emp_length"] = clean_emp_length(raw["emp_length"]).fillna(0)

    # Direct numeric copies
    df["loan_amnt"] = pd.to_numeric(raw["loan_amnt"], errors="coerce")
    df["installment"] = pd.to_numeric(raw["installment"], errors="coerce")
    df["annual_inc"] = pd.to_numeric(raw["annual_inc"], errors="coerce")
    df["dti"] = pd.to_numeric(raw["dti"], errors="coerce")
    df["delinq_2yrs"] = pd.to_numeric(raw["delinq_2yrs"], errors="coerce").fillna(0).astype(int)
    df["open_acc"] = pd.to_numeric(raw["open_acc"], errors="coerce").fillna(0).astype(int)
    df["revol_bal"] = pd.to_numeric(raw["revol_bal"], errors="coerce")
    df["total_acc"] = pd.to_numeric(raw["total_acc"], errors="coerce").fillna(0).astype(int)
    df["inq_last_6mths"] = pd.to_numeric(raw["inq_last_6mths"], errors="coerce").fillna(0).astype(int)
    df["pub_rec"] = pd.to_numeric(raw["pub_rec"], errors="coerce").fillna(0).astype(int)

    # FICO: average of low + high range
    df["fico"] = ((pd.to_numeric(raw["fico_range_low"], errors="coerce") +
                   pd.to_numeric(raw["fico_range_high"], errors="coerce")) / 2)

    # Categoricals
    df["grade"] = raw["grade"]
    df["home_ownership"] = raw["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})
    df["verification_status"] = raw["verification_status"]
    df["purpose"] = raw["purpose"]
    df["addr_state"] = raw["addr_state"]
    df["issue_d"] = raw["issue_d"]

    # Drop rows with critical NaN
    critical = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"  After dropping rows with NaN in critical fields: {len(df):,} (-{before - len(df):,})")

    # Cap extreme annual_inc to reduce outlier effect
    df["annual_inc"] = df["annual_inc"].clip(upper=df["annual_inc"].quantile(0.99))

    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/accepted_2007_to_2018Q4.csv.gz",
                        help="Path to Kaggle CSV (gzipped or unzipped)")
    parser.add_argument("--out", default="data/loans.csv",
                        help="Output cleaned CSV path")
    parser.add_argument("--sample", type=int, default=200_000,
                        help="Sample N rows for faster training (default: 200000, use 0 for full)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(
            f"Input file not found at {in_path}.\n"
            f"Download from Kaggle:\n"
            f"  kaggle datasets download -d wordsforthewise/lending-club -p data/\n"
            f"Then unzip:\n"
            f"  cd data && unzip lending-club.zip"
        )

    print(f"Loading {in_path}...")
    # Read with explicit dtype handling
    cols_to_load = FEATURE_COLS + ["loan_status"]
    raw = pd.read_csv(in_path, usecols=cols_to_load, low_memory=False)

    print("Transforming to project schema...")
    df = transform(raw)

    if args.sample > 0 and len(df) > args.sample:
        # Stratified sample to preserve default rate
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df,
            train_size=args.sample,
            stratify=df["is_default"],
            random_state=42,
        )
        df = df.reset_index(drop=True)
        print(f"  Sampled to {len(df):,} rows")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} rows → {out_path}")
    print(f"  Default rate: {df['is_default'].mean():.2%}")
    print(f"  Features: {df.shape[1] - 1} columns")
    print(f"\nNext step:")
    print(f"  python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()