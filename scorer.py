# scorer.py
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class UncertaintyScorer(ABC):
    @abstractmethod
    def score(self, merged: pd.DataFrame) -> pd.Series:
        """
        Return a Series indexed by response_id with larger values = more uncertain.
        """


class ProxyTfidfDisagreementScorer(UncertaintyScorer):
    """
    Reproduces your current logic:
      - group by prompt_id
      - compute TF-IDF(1,2) embeddings for all responses to that prompt
      - uncertainty = 1 - mean(pairwise cosine similarity)
      - broadcast prompt-level uncertainty to each response in that prompt
    """

    def _group_uncertainty(self, texts: list[str]) -> tuple[float, str]:
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if len(texts) < 2:
            return 0.35, "insufficient_responses"

        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        sims = cosine_similarity(X)
        n = len(texts)
        triu = sims[np.triu_indices(n, k=1)]
        mean_sim = float(triu.mean()) if triu.size else 1.0
        u = 1.0 - mean_sim

        if u >= 0.45:
            reason = "high_model_disagreement"
        elif u >= 0.25:
            reason = "moderate_model_disagreement"
        else:
            reason = "low_model_disagreement"
        return u, reason

    def score(self, merged: pd.DataFrame) -> pd.Series:
        # Expect merged to have: response_id, prompt_id, response_text
        grouped = (
            merged.groupby("prompt_id", dropna=False)["response_text"]
                  .apply(lambda s: self._group_uncertainty(s.tolist()))
                  .apply(pd.Series)
                  .rename(columns={0: "uncertainty_score", 1: "coverage_reason"})
                  .reset_index()
        )
        with_scores = merged.merge(grouped, on="prompt_id", how="left")
        # Return Series[response_id -> uncertainty_score]
        s = with_scores.set_index("response_id")["uncertainty_score"]
        return s.groupby(level=0).max()
