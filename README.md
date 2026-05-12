# Credit Risk Default Prediction Pipeline

End-to-end ML pipeline that predicts loan defaults, explains decisions with SHAP and an LLM-generated risk memo, and surfaces the most similar historical borrowers via embedding-based search — all in an interactive Streamlit dashboard.

**Stack:** Python · scikit-learn · XGBoost · DuckDB SQL · SHAP · Streamlit · Google Gemini (free tier)

## What it does

- **Trains and compares three models** — logistic regression baseline, random forest, and XGBoost — on loan-level data
- **Evaluates with banking-grade metrics** — ROC-AUC, PR-AUC, KS statistic, and threshold-tuned profit curves (not just accuracy)
- **Explains every prediction** using SHAP values, with the top contributing features visualized and the top 8 sent to an LLM for a plain-English memo
- **Finds similar historical borrowers** through text-embedding cosine similarity on natural-language borrower descriptions
- **Provides interactive dashboards** for portfolio overview, applicant scoring, and threshold/profit tradeoff exploration

## Why these design choices

A few decisions worth calling out (this is what an interviewer will ask):

- **PR-AUC alongside ROC-AUC** — credit data is imbalanced, and ROC-AUC paints an optimistic picture; PR-AUC tells the real story
- **Profit curve, not just accuracy** — every credit decision has asymmetric costs (a missed default loses 3–4x what an approved good loan earns); the optimal threshold isn't 0.5
- **DuckDB for SQL** — runs SQL directly on dataframes/parquet with zero setup, mirroring how modern analytics teams operate
- **Graceful fallbacks for AI features** — works fully offline; LLM and embedding features upgrade automatically when a key is set, never break

## Setup (macOS, Python 3.11)

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> credit-risk-analytics
cd credit-risk-analytics

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Set up Gemini for LLM memos and embedding similarity
cp .env.example .env
# Open .env and paste your free key from https://aistudio.google.com/app/apikey
```

Without a Gemini key, everything still runs — the LLM memo falls back to a templated explanation, and similarity falls back to scaled-feature cosine similarity.

## Run

You have two data options. Pick one.

**Option A — Synthetic data (default, instant, works offline):**

```bash
python scripts/run_pipeline.py
```

That's it — the pipeline auto-generates a calibrated synthetic dataset on first run, then trains.

**Option B — Real LendingClub data (~10-second one-time download):**

```bash
python scripts/run_pipeline.py --source real
```

Pulls the [AnguloM/loan_data](https://huggingface.co/datasets/AnguloM/loan_data) dataset from Hugging Face — 9,578 real LendingClub loans (223 KB), released under the Database Contents License (DbCL) v1.0 which permits commercial use and redistribution.

Either way, then launch the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard opens at [http://localhost:8501](http://localhost:8501).

To run tests:

```bash
python -m pytest tests/ -v
```

### Which dataset should I use?

Both are valid project deliverables. The differences:

| | Synthetic | Real LendingClub |
|---|---|---|
| **First-run time** | Instant | ~10 seconds (downloads ~250 KB) |
| **Rows** | 10,000 (configurable) | 9,578 |
| **Reproducibility** | Identical across machines (seed=42) | Identical (dataset is fixed) |
| **Strong predictors** | DTI, int rate, delinquencies, home ownership | FICO score, DTI, int rate, inquiries |
| **AUC achieved (XGBoost)** | ~0.72 | ~0.66–0.68 (real data has more noise) |
| **Best for interviews** | "I generated calibrated synthetic data and validated my pipeline against known patterns" | "I trained on real LendingClub loans" |

If you're going to publish the GitHub repo, run with real data — the project narrative is stronger. The synthetic path stays useful for tests and offline development.

## What you'll see

After step 2, expected console output:

```
Loading data from data/loans.csv...
  Loaded 10,000 rows. Default rate: 11.61%

=== default_rate_by_grade ===  ← real SQL run via DuckDB
grade  n_loans  n_defaults  default_rate_pct  avg_int_rate
    A     2456          43              1.75          7.42
    B     2837         163              5.75         10.96
    ...

Training models...

LogisticRegression  ROC-AUC: 0.7340  PR-AUC: 0.2909  KS: 0.3583
RandomForest        ROC-AUC: 0.7164  PR-AUC: 0.2539  KS: 0.3441
XGBoost             ROC-AUC: 0.7171  PR-AUC: 0.2359  KS: 0.3440

Saved bundle → models/model_bundle.joblib
```

In the dashboard:
- **Portfolio Overview** — default rates by grade, purpose, DTI bucket
- **Risk Scoring** — fill in an applicant form → P(default), SHAP chart, LLM memo, similar borrowers
- **Threshold & Profit Tuning** — adjust profit/loss parameters → see optimal decision threshold

## Project structure

```
credit-risk-analytics/
├── README.md
├── requirements.txt
├── .env.example
├── data/                          # CSV outputs (gitignored)
├── models/                        # Trained model bundle (gitignored)
├── src/
│   ├── data.py                    # Load, clean, split
│   ├── features.py                # Feature engineering + one-hot encoding
│   ├── model.py                   # Train, evaluate, save, predict
│   ├── ai.py                      # Gemini LLM memos + embedding similarity
│   └── sql_eda.py                 # DuckDB-powered EDA queries
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── scripts/
│   ├── generate_sample_data.py    # Synthetic LendingClub-style data
│   ├── download_lending_club.py   # Real LendingClub data from Hugging Face
│   └── run_pipeline.py            # End-to-end orchestrator (--source flag)
└── tests/
    └── test_basic.py              # pytest smoke tests
```

## Concepts to internalize before interviews

If you're asked about this project, lead with the *business framing* — predicting defaults to optimize lending profit — and have these talking points ready:

- **Class imbalance handling** — stratified sampling for train/test, class weights in logistic regression and `scale_pos_weight` in XGBoost
- **Why AUC alone is misleading** — explain the cost asymmetry between false positives and false negatives; this is where the profit curve earns its keep
- **Data leakage** — the `LEAKAGE_COLS` constant in `src/data.py` is empty for this synthetic data, but explain how on real LendingClub data you'd drop `total_pymnt`, `recoveries`, `last_pymnt_d`, etc. before training
- **Time-based vs random splits** — credit models in production split by `issue_d`; the code is set up for both
- **SHAP vs feature importance** — SHAP gives signed, per-prediction contributions; tree feature importance is global and biased toward high-cardinality features
- **Calibration** — model probabilities aren't necessarily calibrated; `CalibratedClassifierCV` and Platt scaling are the standard fixes

## Using a different LendingClub source

The default real-data source is `AnguloM/loan_data` on Hugging Face (~10k rows, the classic mid-2010s LendingClub sample). If you want to point at a different Hugging Face dataset, pass `--dataset`:

```bash
python scripts/download_lending_club.py --dataset some-org/lending-club-larger
```

You'll need to extend `transform()` in `scripts/download_lending_club.py` to handle any schema differences. For the full 2007-2018 LendingClub dump (~2.2M rows, ~150 columns) from Kaggle, you'd want to add a `LEAKAGE_COLS` list to `src/data.py` to drop post-origination columns before training.

## Stretch ideas (when you want to extend the project)

- **Time-based validation** — train on 2018 issues, validate on 2019, test on 2020; report performance drift
- **Calibration** — add `CalibratedClassifierCV` and a calibration plot
- **Cloud extension** — move the pipeline to Databricks/AWS; ingest data from S3, train on Spark, serve predictions via Lambda
- **MLflow tracking** — log all experiments to a local MLflow store

## License

MIT (or whatever you prefer — pick one before publishing).
