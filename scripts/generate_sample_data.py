"""
Generate a synthetic LendingClub-style loan dataset.

Produces ~10k loans with realistic feature distributions and a default
target that's predictable from features (with noise) — calibrated so a
well-tuned XGBoost model lands around 0.70-0.75 AUC.

Usage:
    python scripts/generate_sample_data.py --n 10000 --out data/loans.csv

The data is fully synthetic. Patterns mimic public credit datasets
(LendingClub, Home Credit) but no real borrowers are represented.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate(n: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Borrower financial profile ---
    annual_inc = rng.lognormal(mean=11.0, sigma=0.6, size=n)  # median ~$60k
    annual_inc = np.clip(annual_inc, 15_000, 500_000)

    loan_amnt = rng.uniform(1_000, 40_000, size=n).round(-2)  # nearest $100

    dti = np.clip(rng.beta(2, 5, size=n) * 50, 0, 50)  # debt-to-income %

    # Interest rate correlates with DTI (riskier borrowers → higher rates)
    int_rate = 5 + dti / 5 + rng.normal(0, 2, size=n)
    int_rate = np.clip(int_rate, 5, 30).round(2)

    # Grade derived from int_rate (A=cheapest, G=most expensive)
    grade_idx = np.clip(((int_rate - 5) / 4).astype(int), 0, 6)
    grade = np.array(list("ABCDEFG"))[grade_idx]

    term = rng.choice([36, 60], size=n, p=[0.7, 0.3])

    emp_length = rng.choice(
        list(range(11)), size=n,
        p=[0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05, 0.32],
    )

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"], size=n, p=[0.40, 0.50, 0.10]
    )

    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "small_business", "medical", "other"],
        size=n, p=[0.50, 0.22, 0.10, 0.05, 0.04, 0.03, 0.06],
    )

    verification_status = rng.choice(
        ["Verified", "Source Verified", "Not Verified"],
        size=n, p=[0.35, 0.35, 0.30],
    )

    # --- Credit history ---
    delinq_2yrs = rng.poisson(0.3, size=n)
    open_acc = rng.poisson(10, size=n) + 1
    total_acc = open_acc + rng.poisson(15, size=n)
    revol_util = np.clip(rng.beta(2, 3, size=n) * 100, 0, 100).round(1)

    addr_state = rng.choice(
        ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
         "NJ", "VA", "WA", "MA", "AZ"],
        size=n,
    )

    # --- Issue date (string format matching LendingClub) ---
    issue_dates = pd.date_range("2018-01-01", "2020-12-31", periods=n)
    issue_d = issue_dates.strftime("%b-%Y").to_numpy()

    # --- Build the dataframe before computing target so it's easier to debug ---
    df = pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "grade": grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc.round(2),
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti.round(2),
        "delinq_2yrs": delinq_2yrs,
        "open_acc": open_acc,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "addr_state": addr_state,
        "issue_d": issue_d,
    })

    # --- Target: probability of default as a function of features + noise ---
    log_odds = (
        -2.0
        + 0.12 * (df["int_rate"] - 12)
        + 0.05 * (df["dti"] - 15)
        - 1.2e-5 * (df["annual_inc"] - 60_000)
        + 3e-5 * (df["loan_amnt"] - 15_000)
        + 0.50 * df["delinq_2yrs"]
        + 0.015 * (df["revol_util"] - 50)
        - 0.06 * df["emp_length"]
        + 0.45 * (df["home_ownership"] == "RENT").astype(int)
        + 0.40 * (df["purpose"] == "small_business").astype(int)
        + 0.55 * (df["term"] == 60).astype(int)
        + rng.normal(0, 0.35, size=n)  # noise — keeps AUC realistic (~0.72)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    df["is_default"] = (rng.random(n) < prob_default).astype(int)

    # Inject a small fraction of nulls so cleaning code does meaningful work
    null_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[null_idx, "emp_length"] = np.nan
    null_idx2 = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[null_idx2, "revol_util"] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="Number of rows")
    parser.add_argument("--out", type=str, default="data/loans.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} rows → {out_path}")
    print(f"  Default rate: {df['is_default'].mean():.2%}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
