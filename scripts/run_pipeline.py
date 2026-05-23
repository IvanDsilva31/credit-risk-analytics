"""
Run the full pipeline: load → clean → SQL EDA → features → train → save.

Usage:
    # Synthetic data (default — works offline, instant)
    python scripts/run_pipeline.py

    # Real LendingClub data (downloads from Hugging Face on first use)
    python scripts/run_pipeline.py --source real

    # Custom path
    python scripts/run_pipeline.py --data data/loans.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path
from sklearn.model_selection import cross_val_score
import pandas as pd

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as data_module
from src import features as features_module
from src import model as model_module
from src import sql_eda


def ensure_data(data_path: Path, source: str) -> None:
    """If data isn't on disk, generate (synthetic) or download (real)."""
    if data_path.exists():
        return

    project_root = Path(__file__).resolve().parent.parent
    if source == "synthetic":
        print(f"Data not found at {data_path}. Generating synthetic data...")
        subprocess.check_call([
            sys.executable, str(project_root / "scripts/generate_sample_data.py"),
            "--out", str(data_path),
        ])
    elif source == "real":
        print(f"Data not found at {data_path}. Downloading real LendingClub data...")
        subprocess.check_call([
            sys.executable, str(project_root / "scripts/download_lending_club.py"),
            "--out", str(data_path),
        ])
    else:
        raise ValueError(f"Unknown source {source!r}. Use 'synthetic' or 'real'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/loans.csv",
                        help="Path to the loans CSV")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "real"],
                        help="Auto-generate the data file if missing")
    parser.add_argument("--models-dir", default="models",
                        help="Where to save the trained model bundle")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip SQL EDA report (faster)")
    args = parser.parse_args()

    data_path = Path(args.data)
    ensure_data(data_path, args.source)

    print(f"\nLoading data from {data_path}...")
    df = data_module.clean(data_module.load_raw(data_path))
    print(f"  Loaded {len(df):,} rows. Default rate: {df['is_default'].mean():.2%}")

    if not args.skip_eda:
        print("\nRunning SQL EDA...")
        sql_eda.print_report(df)

    print("\nBuilding features...")
    X_train, X_test, y_train, y_test = data_module.split(df)
    X_train_f, feature_cols = features_module.build_features(X_train)
    X_test_f, _ = features_module.build_features(X_test, feature_columns=feature_cols)
    print(f"  Feature matrix: {X_train_f.shape[1]} columns")

    print()
    results = model_module.train_all(X_train_f, y_train, X_test_f, y_test)

    print("\nRunning 5-fold cross-validation on best model...")
    best_name = max(
        [("logreg", results["logreg"]["metrics"].roc_auc),
        ("rf",     results["rf"]["metrics"].roc_auc),
        ("xgb",    results["xgb"]["metrics"].roc_auc)],
        key=lambda x: x[1]
    )[0]

    if best_name == "xgb":
        from xgboost import XGBClassifier
    
        best_model = results["xgb"]["model"]
        X_full = pd.concat([X_train_f, X_test_f])
        y_full = pd.concat([y_train, y_test])
    
        # Clone XGBoost params WITHOUT early stopping (cross_val_score doesn't pass eval_set)
        cv_params = best_model.get_params()
        cv_params.pop("early_stopping_rounds", None)
        cv_params.pop("callbacks", None)
        cv_model = XGBClassifier(**cv_params)
    
        cv_scores = cross_val_score(cv_model, X_full, y_full,
                                  cv=5, scoring="roc_auc", n_jobs=-1)
        print(f"  XGBoost 5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model_module.save_artifacts(results, feature_cols, args.models_dir)

    print("\nDone. Launch the dashboard with:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
