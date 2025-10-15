import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class UncertaintyScorer(ABC):
    """Base interface for scorers."""
    @abstractmethod
    def score_with_reason(self, merged: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          - response_id
          - uncertainty_score (float, higher = more uncertain)
          - coverage_reason (str)
        """
        ...


class ProxyTfidfDisagreementScorer(UncertaintyScorer):
    """
    Disagreement proxy via TF-IDF cosine similarity.
    For each prompt_id, compute TF-IDF over that prompt's responses and set
    each response's uncertainty = 1 - mean cosine similarity to the others.
    Requires: response_id, prompt_id, response_text
    """

    def score_with_reason(self, merged: pd.DataFrame) -> pd.DataFrame:
        needed = {"response_id", "prompt_id", "response_text"}
        missing = needed.difference(merged.columns)
        if missing:
            raise ValueError(
                f"ProxyTfidfDisagreementScorer missing columns: {sorted(missing)}. "
                "Expected at least response_id, prompt_id, response_text."
            )

        # Ensure text is string
        df = merged[["response_id", "prompt_id", "response_text"]].copy()
        df["response_text"] = df["response_text"].fillna("").astype(str)

        results = []
        for pid, grp in df.groupby("prompt_id", dropna=False):
            texts = grp["response_text"].tolist()
            rids = grp["response_id"].tolist()

            if len(texts) == 1:
                # With a single response, we cannot compute pairwise similarity meaningfully.
                results.append({
                    "response_id": rids[0],
                    "uncertainty_score": 1.0,  # treat singleton as highly uncertain
                    "coverage_reason": "singleton_prompt"
                })
                continue

            # Vectorize within the prompt
            tfidf = TfidfVectorizer(min_df=1)
            X = tfidf.fit_transform(texts)  # shape (n, vocab)
            sim = cosine_similarity(X)      # shape (n, n), diagonal = 1

            # For each response, mean similarity to others (exclude self)
            n = sim.shape[0]
            # sum of row minus self-similarity, then divide by (n-1)
            row_sum_minus_self = sim.sum(axis=1) - np.ones((n,))
            mean_sim_to_others = row_sum_minus_self / max(1, (n - 1))

            # Disagreement = 1 - mean_similarity
            uncertainty = 1.0 - mean_sim_to_others

            for rid, u in zip(rids, uncertainty):
                results.append({
                    "response_id": rid,
                    "uncertainty_score": float(u),
                    "coverage_reason": "tfidf_disagreement"
                })

        out = pd.DataFrame(results)
        return out[["response_id", "uncertainty_score", "coverage_reason"]]


class MainVarianceScorer(UncertaintyScorer):
    """
    Variance-based uncertainty from human annotations (Sprint 4).
    Requires: response_id, score  (one row per (response_id, annotator_id))
    """
    def score_with_reason(self, merged: pd.DataFrame) -> pd.DataFrame:
        needed = {"response_id", "score"}
        missing = needed.difference(merged.columns)
        if missing:
            raise ValueError(
                "MainVarianceScorer requires annotation columns. "
                f"Missing: {sorted(missing)}. "
                "Tip: merge annotations (response_id, annotator_id, score) into the merged frame."
            )

        g = (
            merged.groupby("response_id", dropna=False)["score"]
                  .agg(var="var", n="count")
                  .reset_index()
        )
        # If n < 2, pandas var() yields NaN; treat as 0 uncertainty and mark coverage
        g["uncertainty_score"] = g["var"].fillna(0.0)
        g["coverage_reason"] = g["n"].apply(
            lambda n: "variance_based" if n >= 2 else "insufficient_annotations"
        )
        return g[["response_id", "uncertainty_score", "coverage_reason"]]
