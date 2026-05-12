"""
AI features built on Google Gemini's free API tier.

Two capabilities:
  1. `explain_decision(...)`: LLM-generated plain-English risk memo for a
     scored applicant, conditioned on SHAP feature contributions.
  2. `BorrowerSimilarity`: embedding-based nearest-neighbor search over
     historical borrowers — for a new applicant, returns the K most similar
     past loans along with their actual outcomes.

Both gracefully degrade if no GEMINI_API_KEY is set:
  - LLM explanations fall back to a templated rule-based memo.
  - Similarity falls back to scaled-feature cosine similarity (no
    embeddings, but still functional and useful).

Free tier: https://aistudio.google.com (no credit card required).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# New unified Gemini SDK. Defer hard failures until a call is actually made.
try:
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# --- Configuration -----------------------------------------------------------

GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

_client = None  # populated lazily


def _get_client():
    """Return a configured Gemini client, or None if unavailable."""
    global _client
    if _client is not None:
        return _client
    if not _GEMINI_AVAILABLE:
        return None
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        _client = genai.Client(api_key=key)
        return _client
    except Exception:
        return None


# --- 1. LLM-powered risk explanation -----------------------------------------

@dataclass
class FeatureContribution:
    """One feature's SHAP contribution to a single prediction."""
    name: str
    value: float | str
    shap_value: float  # positive → pushes toward default; negative → away
    description: str


def _format_feature_value(name: str, value) -> str:
    """Format a value nicely for LLM and dashboard display."""
    if name in {"annual_inc", "loan_amnt"}:
        try:
            return f"${float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
    if name in {"dti", "int_rate", "revol_util"}:
        try:
            return f"{float(value):.1f}%"
        except (TypeError, ValueError):
            return str(value)
    if name == "loan_to_income":
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _templated_explanation(
    default_probability: float,
    top_contributors: list[FeatureContribution],
) -> str:
    """Rule-based fallback when no Gemini key is available."""
    decision = "DECLINE" if default_probability > 0.5 else "APPROVE"
    risk_level = (
        "Very High" if default_probability > 0.7
        else "High" if default_probability > 0.5
        else "Moderate" if default_probability > 0.3
        else "Low"
    )

    pushing_up = [c for c in top_contributors if c.shap_value > 0][:3]
    pushing_down = [c for c in top_contributors if c.shap_value < 0][:3]

    lines = [
        f"DECISION: {decision}",
        f"Predicted default probability: {default_probability:.1%} ({risk_level} risk)",
        "",
        "Key risk drivers (pushing risk UP):",
    ]
    for c in pushing_up:
        lines.append(f"  - {c.description}: {_format_feature_value(c.name, c.value)}")

    if pushing_down:
        lines.append("")
        lines.append("Mitigating factors (pushing risk DOWN):")
        for c in pushing_down:
            lines.append(f"  - {c.description}: {_format_feature_value(c.name, c.value)}")

    return "\n".join(lines)


def explain_decision(
    default_probability: float,
    top_contributors: list[FeatureContribution],
    applicant_summary: dict | None = None,
) -> str:
    """
    Generate a plain-English risk memo. Uses Gemini if available, falls
    back to a templated explanation otherwise.
    """
    client = _get_client()
    if client is None:
        return _templated_explanation(default_probability, top_contributors)

    contributor_lines = []
    for c in top_contributors:
        direction = "raises risk" if c.shap_value > 0 else "lowers risk"
        val_str = _format_feature_value(c.name, c.value)
        contributor_lines.append(
            f"  - {c.description}: {val_str}  "
            f"(SHAP={c.shap_value:+.3f}, {direction})"
        )

    summary_block = ""
    if applicant_summary:
        summary_lines = [f"  - {k}: {_format_feature_value(k, v)}"
                         for k, v in applicant_summary.items()]
        summary_block = "Applicant summary:\n" + "\n".join(summary_lines) + "\n\n"

    decision = "DECLINE" if default_probability > 0.5 else "APPROVE"

    prompt = f"""You are a senior credit risk analyst writing a brief decision memo for an underwriting review.

A machine learning model has scored a loan applicant. Write a clear, factual memo (max 150 words) explaining the decision in plain English. Be specific — use the numbers below. Do not invent facts not present in the inputs.

Model output:
  - Predicted P(default): {default_probability:.1%}
  - Recommended decision: {decision}

Top feature contributions (from SHAP analysis):
{chr(10).join(contributor_lines)}

{summary_block}Write the memo with:
1. A one-sentence decision line.
2. A short paragraph explaining the top 2-3 risk drivers.
3. If applicable, one sentence on mitigating factors.

Do NOT use markdown headers or bullet points. Write as flowing prose suitable for a one-paragraph note to a colleague.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return (
            _templated_explanation(default_probability, top_contributors)
            + f"\n\n[Note: LLM call failed — {type(e).__name__}. Fallback memo shown.]"
        )


# --- 2. Embedding-based borrower similarity ----------------------------------

def _borrower_to_text(row: pd.Series) -> str:
    """Convert a borrower row to a natural-language description for embedding."""
    return (
        f"Loan of ${row['loan_amnt']:,.0f} for {row['purpose'].replace('_', ' ')} "
        f"over {int(row['term'])} months at {row['int_rate']:.1f}% interest. "
        f"Borrower has annual income of ${row['annual_inc']:,.0f}, "
        f"debt-to-income ratio of {row['dti']:.1f}%, "
        f"{int(row['delinq_2yrs'])} delinquencies in the past 2 years, "
        f"and {int(row['emp_length'])} years of employment. "
        f"Home ownership: {row['home_ownership']}. Grade: {row['grade']}."
    )


class BorrowerSimilarity:
    """
    Nearest-neighbor search over a corpus of historical loans.

    Uses Gemini embeddings if available; otherwise falls back to cosine
    similarity over standardized numerical features.
    """

    SIMILARITY_FEATURES = [
        "loan_amnt", "term", "int_rate", "annual_inc", "dti",
        "delinq_2yrs", "open_acc", "revol_util", "emp_length",
    ]

    def __init__(self, df: pd.DataFrame, max_corpus: int = 300):
        if len(df) > max_corpus:
            df = df.sample(max_corpus, random_state=42).reset_index(drop=True)
        self.corpus = df.reset_index(drop=True)

        client = _get_client()
        self.use_embeddings = client is not None
        if self.use_embeddings:
            try:
                texts = [_borrower_to_text(row) for _, row in self.corpus.iterrows()]
                self.corpus_vecs = self._embed(client, texts)
            except Exception:
                # If embedding call fails (rate limit, network), fall back silently
                self.use_embeddings = False
                self._fit_numeric_fallback()
        else:
            self._fit_numeric_fallback()

    def _fit_numeric_fallback(self) -> None:
        self.scaler = StandardScaler()
        X = self.corpus[self.SIMILARITY_FEATURES].fillna(0).to_numpy()
        self.corpus_vecs = self.scaler.fit_transform(X)

    @staticmethod
    def _embed(client, texts: list[str]) -> np.ndarray:
        """Embed texts via Gemini. New SDK accepts a list directly."""
        result = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=texts,
        )
        # `embeddings` is a list of Embedding objects each with a `.values` list
        vecs = [e.values for e in result.embeddings]
        return np.array(vecs, dtype=np.float32)

    def _vectorize_query(self, applicant: pd.Series) -> np.ndarray:
        if self.use_embeddings:
            client = _get_client()
            return self._embed(client, [_borrower_to_text(applicant)])
        x = applicant[self.SIMILARITY_FEATURES].fillna(0).to_numpy().reshape(1, -1)
        return self.scaler.transform(x)

    def find_similar(self, applicant: pd.Series, k: int = 10) -> pd.DataFrame:
        qvec = self._vectorize_query(applicant)
        sims = cosine_similarity(qvec, self.corpus_vecs)[0]
        top_k_idx = np.argsort(-sims)[:k]
        result = self.corpus.iloc[top_k_idx].copy()
        result["similarity_score"] = sims[top_k_idx]
        return result.reset_index(drop=True)

    def similar_default_rate(self, applicant: pd.Series, k: int = 10) -> dict:
        similar = self.find_similar(applicant, k=k)
        return {
            "k": k,
            "method": "gemini_embeddings" if self.use_embeddings else "numeric_features",
            "similar_default_rate": float(similar["is_default"].mean()),
            "n_defaults": int(similar["is_default"].sum()),
            "avg_similarity": float(similar["similarity_score"].mean()),
            "similar_loans": similar,
        }
