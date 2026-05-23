# Credit Risk Default Prediction Pipeline

End-to-end ML pipeline that predicts loan defaults on real LendingClub data, with an interactive Streamlit dashboard for applicant risk scoring and portfolio analytics.

**Stack:** Python · scikit-learn · XGBoost · DuckDB SQL · SHAP · Streamlit

## Setup

Requires Python 3.11+ and macOS, Linux, or Windows.

```bash
git clone https://github.com/IvanDsilva31/credit-risk-analytics.git
cd credit-risk-analytics

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Pick one of three data sources.

**Synthetic data (instant, offline):**

```bash
python scripts/run_pipeline.py
```

**Sample LendingClub data from Hugging Face:**

```bash
python scripts/run_pipeline.py --source real
```

**Full LendingClub data from Kaggle:**

```bash
kaggle datasets download -d wordsforthewise/lending-club -p data/
cd data && unzip lending-club.zip && cd ..
python scripts/download_full_lendingclub.py --sample 200000
python scripts/run_pipeline.py
```

Requires a Kaggle account and API token — see [Kaggle's API docs](https://www.kaggle.com/docs/api).

## Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

## Tests

```bash
python -m pytest tests/ -v
```

## Project structure

```
credit-risk-analytics/
├── data/                              # Data files (gitignored)
├── models/                            # Trained models (gitignored)
├── src/
│   ├── data.py                        # Load, clean, split
│   ├── features.py                    # Feature engineering
│   ├── model.py                       # Train, evaluate, predict
│   └── sql_eda.py                     # DuckDB SQL queries
├── dashboard/
│   └── app.py                         # Streamlit dashboard
├── scripts/
│   ├── generate_sample_data.py
│   ├── download_lending_club.py
│   ├── download_full_lendingclub.py
│   └── run_pipeline.py
└── tests/
    └── test_basic.py
```

## License

MIT — see [LICENSE](LICENSE).
