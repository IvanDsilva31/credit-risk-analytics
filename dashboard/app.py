"""
Credit Risk Analytics Dashboard.

Three tabs:
  1. Portfolio Overview — aggregate stats, charts, SQL-driven views
  2. Risk Scoring — score a hypothetical applicant; LLM-generated decision
     memo + embedding-based similar-borrower lookup
  3. Threshold & Profit Tuning — explore profit/recall tradeoffs

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from dotenv import load_dotenv

# Make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as data_module
from src import features as features_module
from src import model as model_module
from src import sql_eda
from src.ai import (
    BorrowerSimilarity,
    FeatureContribution,
    explain_decision,
)
from src.features import FEATURE_DESCRIPTIONS

load_dotenv()

# --- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="💰",
    layout="wide",
)

DATA_PATH = "data/loans.csv"
MODEL_PATH = "models/model_bundle.joblib"


# --- Cached loaders ----------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    return data_module.clean(data_module.load_raw(DATA_PATH))


@st.cache_resource
def load_model_bundle() -> dict:
    return model_module.load_bundle(MODEL_PATH)


@st.cache_resource
def get_shap_explainer(_bundle: dict):
    """SHAP TreeExplainer over the XGBoost model. Underscore prefix tells
    Streamlit not to hash the bundle (joblib objects aren't hashable)."""
    return shap.TreeExplainer(_bundle["xgb_model"])


@st.cache_resource
def get_similarity_engine(df: pd.DataFrame) -> BorrowerSimilarity:
    """Embedding-based borrower similarity. Cached across sessions."""
    # Sample of historical defaulted + non-defaulted loans
    defaults = df[df["is_default"] == 1].sample(min(150, len(df[df["is_default"] == 1])),
                                                 random_state=42)
    non_defaults = df[df["is_default"] == 0].sample(min(150, len(df[df["is_default"] == 0])),
                                                     random_state=42)
    corpus = pd.concat([defaults, non_defaults]).reset_index(drop=True)
    return BorrowerSimilarity(corpus, max_corpus=300)


# --- Bootstrap ---------------------------------------------------------------

# Gracefully handle missing data/model
data_exists = Path(DATA_PATH).exists()
model_exists = Path(MODEL_PATH).exists()

if not data_exists or not model_exists:
    st.error("Data or model artifacts are missing.")
    st.markdown(
        "Run the setup commands first:\n\n"
        "```bash\n"
        "python scripts/generate_sample_data.py\n"
        "python scripts/run_pipeline.py\n"
        "```"
    )
    st.stop()

df = load_data()
bundle = load_model_bundle()
explainer = get_shap_explainer(bundle)

# Sidebar — model info
with st.sidebar:
    st.markdown("### Model Performance")
    metrics = bundle["metrics"]
    for name, m in metrics.items():
        st.markdown(f"**{name}**")
        st.markdown(
            f"- ROC-AUC: `{m['roc_auc']:.3f}`\n"
            f"- PR-AUC: `{m['pr_auc']:.3f}`\n"
            f"- KS: `{m['ks_statistic']:.3f}`"
        )
    st.markdown(f"**Best:** `{bundle['best_name']}`")
    st.markdown("---")
    st.markdown("### Gemini API")
    if os.getenv("GEMINI_API_KEY"):
        st.success("✓ Gemini key detected")
    else:
        st.warning(
            "No `GEMINI_API_KEY` in env.\n\n"
            "LLM memos and embedding-based similarity will use fallbacks. "
            "See README to set up a free key."
        )


st.title("💰 Credit Risk Analytics")
st.caption(
    "End-to-end ML pipeline with SHAP explainability, LLM-generated decision "
    "memos, and embedding-based borrower similarity search."
)


tab1, tab2, tab3 = st.tabs([
    "Portfolio Overview",
    "Risk Scoring",
    "Threshold & Profit Tuning",
])


# --- TAB 1: Portfolio Overview ----------------------------------------------

with tab1:
    st.subheader("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Loans", f"{len(df):,}")
    col2.metric("Default rate", f"{df['is_default'].mean():.2%}")
    col3.metric("Avg loan amount", f"${df['loan_amnt'].mean():,.0f}")
    col4.metric("Avg interest rate", f"{df['int_rate'].mean():.2f}%")

    st.markdown("---")

    # Filters
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        grades = st.multiselect("Grade", sorted(df["grade"].unique()),
                                default=sorted(df["grade"].unique()))
    with fcol2:
        purposes = st.multiselect("Purpose", sorted(df["purpose"].unique()),
                                  default=sorted(df["purpose"].unique()))

    filtered = df[df["grade"].isin(grades) & df["purpose"].isin(purposes)]
    st.caption(f"Filtered to {len(filtered):,} loans.")

    # SQL-driven views — these run real DuckDB queries against the dataframe
    st.markdown("#### Default rate by grade")
    grade_table = sql_eda.query(sql_eda.DEFAULT_RATE_BY_GRADE, filtered)
    fig = px.bar(grade_table, x="grade", y="default_rate_pct",
                 hover_data=["n_loans", "avg_int_rate"],
                 labels={"default_rate_pct": "Default rate (%)"})
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Default rate by purpose")
        purpose_table = sql_eda.query(sql_eda.DEFAULT_RATE_BY_PURPOSE, filtered)
        st.dataframe(purpose_table, hide_index=True, width="stretch")

    with col_b:
        st.markdown("#### Default rate by DTI bucket")
        dti_table = sql_eda.query(sql_eda.DTI_BUCKETS, filtered)
        fig2 = px.bar(dti_table, x="dti_bucket", y="default_rate_pct",
                      labels={"default_rate_pct": "Default rate (%)",
                              "dti_bucket": "DTI bucket"})
        st.plotly_chart(fig2, width="stretch")


# --- TAB 2: Risk Scoring -----------------------------------------------------

with tab2:
    st.subheader("Score a hypothetical applicant")
    st.caption("Enter applicant details. The model returns P(default), SHAP "
               "contributions, an LLM-generated decision memo, and the most "
               "similar historical borrowers.")

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 15000, step=500)
            term = st.selectbox("Term (months)", [36, 60])
            int_rate = st.number_input("Interest rate (%)", 5.0, 30.0, 12.0, step=0.1)
            annual_inc = st.number_input("Annual income ($)", 15000, 500000, 60000, step=1000)
            grade = st.selectbox("Grade", list("ABCDEFG"))
            emp_length = st.slider("Employment length (years)", 0, 10, 5)
        with c2:
            dti = st.slider("Debt-to-income ratio (%)", 0.0, 50.0, 18.0)
            delinq_2yrs = st.number_input("Delinquencies (past 2yrs)", 0, 10, 0)
            open_acc = st.number_input("Open credit lines", 1, 50, 10)
            revol_util = st.slider("Revolving utilization (%)", 0.0, 100.0, 45.0)
            total_acc = st.number_input("Total credit accounts", 1, 80, 25)
        with c3:
            home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
            purpose = st.selectbox(
                "Loan purpose",
                ["debt_consolidation", "credit_card", "home_improvement",
                 "major_purchase", "small_business", "medical", "other"],
            )
            verification_status = st.selectbox(
                "Income verification",
                ["Verified", "Source Verified", "Not Verified"],
            )
            addr_state = st.selectbox("State", sorted(df["addr_state"].unique()))
            k_similar = st.slider("Show K similar borrowers", 5, 20, 10)

        submitted = st.form_submit_button("Score applicant", type="primary")

    if submitted:
        applicant_df = pd.DataFrame([{
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
            "grade": grade, "emp_length": emp_length,
            "home_ownership": home_ownership, "annual_inc": annual_inc,
            "verification_status": verification_status, "purpose": purpose,
            "dti": dti, "delinq_2yrs": delinq_2yrs, "open_acc": open_acc,
            "revol_util": revol_util, "total_acc": total_acc,
            "addr_state": addr_state, "issue_d": "Jan-2024",
        }])

        # Feature engineering with the same schema as training
        X_app, _ = features_module.build_features(
            applicant_df, feature_columns=bundle["feature_columns"]
        )

        # Use XGBoost for the dashboard (TreeExplainer is fast)
        prob = float(model_module.predict_proba(bundle, X_app, model_name="xgb")[0])

        # --- Headline result
        decision = "DECLINE" if prob > 0.5 else "APPROVE"
        decision_color = "🔴" if prob > 0.5 else "🟢"

        st.markdown("### Result")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("P(default)", f"{prob:.1%}")
        rc2.metric("Decision", f"{decision_color} {decision}")
        rc3.metric("Risk band",
                   "Very high" if prob > 0.7
                   else "High" if prob > 0.5
                   else "Moderate" if prob > 0.3
                   else "Low")

        # --- SHAP contributions
        shap_values = explainer.shap_values(X_app)
        if isinstance(shap_values, list):  # multi-class compat
            shap_values = shap_values[1]
        contributions = pd.DataFrame({
            "feature": X_app.columns,
            "value": X_app.iloc[0].values,
            "shap": shap_values[0],
        })
        contributions["abs_shap"] = contributions["shap"].abs()
        contributions = contributions.sort_values("abs_shap", ascending=False).head(8)

        st.markdown("### Why? (SHAP contributions)")
        fig = px.bar(
            contributions.sort_values("shap"),
            x="shap", y="feature", orientation="h",
            color="shap",
            color_continuous_scale=["green", "lightgray", "red"],
            color_continuous_midpoint=0,
            labels={"shap": "SHAP value (→ raises default risk)"},
            height=350,
        )
        st.plotly_chart(fig, width="stretch")

        # --- LLM decision memo
        st.markdown("### Risk Memo (LLM-generated)")
        with st.spinner("Generating decision memo..."):
            # Build contributor objects for the LLM
            top_contribs = []
            for _, row in contributions.iterrows():
                # Pull human-readable name; fall back to raw column for dummies
                base_name = row["feature"].split("_")[0]
                desc = FEATURE_DESCRIPTIONS.get(
                    row["feature"],
                    FEATURE_DESCRIPTIONS.get(base_name, row["feature"]),
                )
                top_contribs.append(FeatureContribution(
                    name=row["feature"],
                    value=row["value"],
                    shap_value=row["shap"],
                    description=desc,
                ))

            applicant_summary = {
                "loan_amnt": loan_amnt, "annual_inc": annual_inc,
                "dti": dti, "int_rate": int_rate, "purpose": purpose,
                "home_ownership": home_ownership,
            }
            memo = explain_decision(prob, top_contribs, applicant_summary)
        st.info(memo)

        # --- Embedding-based similar borrowers
        st.markdown(f"### {k_similar} most similar historical borrowers")
        with st.spinner("Finding similar borrowers..."):
            sim_engine = get_similarity_engine(df)
            result = sim_engine.similar_default_rate(applicant_df.iloc[0], k=k_similar)

        method_label = (
            "Gemini text embeddings"
            if result["method"] == "gemini_embeddings"
            else "Scaled numeric features (fallback)"
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("Default rate among similar",
                  f"{result['similar_default_rate']:.1%}")
        s2.metric("Avg similarity", f"{result['avg_similarity']:.3f}")
        s3.metric("Method", method_label)

        display_cols = [
            "similarity_score", "loan_amnt", "int_rate", "annual_inc",
            "dti", "purpose", "home_ownership", "is_default",
        ]
        st.dataframe(
            result["similar_loans"][display_cols],
            hide_index=True,
            width="stretch",
        )


# --- TAB 3: Threshold & Profit Tuning ----------------------------------------

with tab3:
    st.subheader("Threshold & profit tuning")
    st.caption(
        "Move the threshold to see how the approve/decline decision tradeoff "
        "changes — and where expected profit is maximized."
    )

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        profit_per_good = st.number_input(
            "Profit per approved good loan ($)", 100, 10000, 1000, step=100,
        )
    with pcol2:
        loss_per_bad = st.number_input(
            "Loss per approved bad loan ($)", 500, 50000, 4000, step=500,
        )

    # Score the full dataset and compute profit curve
    @st.cache_data
    def get_test_scores():
        X_train, X_test, y_train, y_test = data_module.split(df)
        X_test_f, _ = features_module.build_features(
            X_test, feature_columns=bundle["feature_columns"]
        )
        scores = model_module.predict_proba(bundle, X_test_f, model_name="xgb")
        return y_test.values, scores

    y_test, scores = get_test_scores()
    pc = model_module.profit_curve(y_test, scores, profit_per_good, loss_per_bad)

    optimal_row = pc.loc[pc["expected_profit"].idxmax()]
    st.markdown(
        f"**Optimal threshold:** `{optimal_row['threshold']:.2f}` — "
        f"profit `${optimal_row['expected_profit']:,.0f}`, "
        f"approval rate `{optimal_row['approval_rate']:.1%}`"
    )

    fig = px.line(pc, x="threshold", y="expected_profit",
                  labels={"expected_profit": "Expected profit ($)",
                          "threshold": "Decision threshold"})
    fig.add_vline(x=optimal_row["threshold"], line_dash="dash", line_color="red",
                  annotation_text="Optimal")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Confusion matrix at chosen threshold")
    chosen = st.slider("Pick a threshold", 0.05, 0.95,
                       float(optimal_row["threshold"]), step=0.05)
    cm = model_module.confusion_at_thresholds(y_test, scores, [chosen])
    st.dataframe(cm, hide_index=True, width="stretch")
