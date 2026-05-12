"""
SQL-based exploratory data analysis using DuckDB.

DuckDB is an in-process analytical database — it can run SQL directly
against pandas dataframes and parquet/csv files with zero setup. This
lets you write real SQL in a portfolio project without needing a
Postgres/MySQL server.
"""

from pathlib import Path

import duckdb
import pandas as pd


def query(sql: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Run a SQL query. If `df` is provided, it's registered as the
    table `loans` (so queries can reference it directly).
    """
    con = duckdb.connect(":memory:")
    if df is not None:
        con.register("loans", df)
    return con.execute(sql).fetchdf()


# ---------- Canonical EDA queries ----------

DEFAULT_RATE_BY_GRADE = """
SELECT
    grade,
    COUNT(*) AS n_loans,
    SUM(is_default) AS n_defaults,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_int_rate
FROM loans
GROUP BY grade
ORDER BY grade
"""

DEFAULT_RATE_BY_PURPOSE = """
SELECT
    purpose,
    COUNT(*) AS n_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(loan_amnt), 0) AS avg_loan_amnt
FROM loans
GROUP BY purpose
ORDER BY default_rate_pct DESC
"""

DEFAULT_RATE_BY_HOME = """
SELECT
    home_ownership,
    COUNT(*) AS n_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(annual_inc), 0) AS avg_annual_inc
FROM loans
GROUP BY home_ownership
ORDER BY default_rate_pct DESC
"""

VINTAGE_ANALYSIS = """
SELECT
    SUBSTR(issue_d, 5) AS year,
    COUNT(*) AS n_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct
FROM loans
GROUP BY year
ORDER BY year
"""

DTI_BUCKETS = """
SELECT
    CASE
        WHEN dti < 10 THEN '0-10'
        WHEN dti < 20 THEN '10-20'
        WHEN dti < 30 THEN '20-30'
        WHEN dti < 40 THEN '30-40'
        ELSE '40+'
    END AS dti_bucket,
    COUNT(*) AS n_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct
FROM loans
GROUP BY dti_bucket
ORDER BY dti_bucket
"""


def run_all(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all canonical EDA queries and return them as a dict."""
    return {
        "default_rate_by_grade": query(DEFAULT_RATE_BY_GRADE, df),
        "default_rate_by_purpose": query(DEFAULT_RATE_BY_PURPOSE, df),
        "default_rate_by_home_ownership": query(DEFAULT_RATE_BY_HOME, df),
        "vintage_analysis": query(VINTAGE_ANALYSIS, df),
        "dti_buckets": query(DTI_BUCKETS, df),
    }


def print_report(df: pd.DataFrame) -> None:
    """Pretty-print all EDA tables."""
    results = run_all(df)
    for name, table in results.items():
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))
