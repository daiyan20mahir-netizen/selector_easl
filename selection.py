from typing import Literal, Optional, Dict
import pandas as pd
from scorer import ProxyTfidfDisagreementScorer, MainVarianceScorer

ScorerName = Literal["proxy", "main"]


def compute_scores(merged: pd.DataFrame, scorer_type: ScorerName) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the chosen scorer and return:
      - scores_df: DataFrame[response_id, uncertainty_score, coverage_reason]
      - scores:    Series[uncertainty_score] indexed by response_id
    """
    if scorer_type == "proxy":
        scorer = ProxyTfidfDisagreementScorer()
    elif scorer_type == "main":
        scorer = MainVarianceScorer()
    else:
        raise ValueError(f"Unknown scorer '{scorer_type}'. Use 'proxy' or 'main'.")

    scores_df = scorer.score_with_reason(merged)
    scores = pd.Series(scores_df["uncertainty_score"].values, index=scores_df["response_id"])
    return scores_df, scores


def select_batch(
    merged: pd.DataFrame,
    scorer_type: ScorerName = "proxy",
    batch_size: int = 60,
    by_stream: Optional[Dict[str, int]] = None,
    max_per_prompt: int = 0,
) -> pd.DataFrame:
    """
    Main batch-selection function.
    Expects merged to contain at least: response_id, prompt_id, bias_stream, response_text (for proxy).
    """
    scores_df, _ = compute_scores(merged, scorer_type)
    merged = merged.merge(scores_df, on="response_id", how="left")

    # Sort by descending uncertainty
    merged = merged.sort_values("uncertainty_score", ascending=False)

    # Apply per-stream quotas if provided
    if by_stream:
        selected_parts = []
        for stream, quota in by_stream.items():
            sub = merged[merged["bias_stream"] == stream].head(quota)
            selected_parts.append(sub)
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else merged.head(batch_size)
    else:
        selected = merged.head(batch_size)

    # Optionally limit per prompt
    if max_per_prompt > 0 and "prompt_id" in selected.columns:
        selected = (
            selected.groupby("prompt_id", group_keys=False)
                    .head(max_per_prompt)
                    .reset_index(drop=True)
        )
    else:
        selected = selected.reset_index(drop=True)

    return selected
