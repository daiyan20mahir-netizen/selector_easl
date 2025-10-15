# selection.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List


def _rank_df(scores: pd.Series, merged: pd.DataFrame) -> pd.DataFrame:
    """
    Combine scores with minimal tie-breakers to match original behavior:
      - sort by uncertainty_score desc
      - tie-break: shorter response_text first
    """
    df = (
        merged[["response_id", "prompt_id", "bias_stream", "family", "response_text"]]
        .drop_duplicates(subset=["response_id"])
        .copy()
    )
    df["uncertainty_score"] = df["response_id"].map(scores)
    df["text_len"] = df["response_text"].astype(str).str.len()
    df = df.sort_values(["uncertainty_score", "text_len"], ascending=[False, True])
    return df


def _apply_per_prompt_cap(df_ranked: pd.DataFrame, max_per_prompt: int) -> pd.DataFrame:
    if not max_per_prompt or max_per_prompt <= 0:
        return df_ranked
    c = df_ranked.copy()
    c["_rank_within_prompt"] = c.groupby("prompt_id").cumcount()
    c = c[c["_rank_within_prompt"] < max_per_prompt].drop(columns=["_rank_within_prompt"])
    return c


def _fill_quota(df_ranked: pd.DataFrame, batch: int, quotas: Dict[str, int]) -> List[str]:
    """
    quotas: dict like {"gender": 30, "politics": 30}
    """
    selected_ids: List[str] = []
    picked = set()

    # 1) fulfill per-stream quotas in ranked order
    for stream, q in quotas.items():
        take = df_ranked[df_ranked["bias_stream"] == stream].head(q)
        ids = take["response_id"].tolist()
        selected_ids.extend(ids)
        picked.update(ids)

    # 2) fill the remainder globally
    remaining = batch - len(selected_ids)
    if remaining > 0:
        filler = df_ranked[~df_ranked["response_id"].isin(picked)].head(remaining)
        selected_ids.extend(filler["response_id"].tolist())

    # de-dup & truncate (preserve order)
    seen = set()
    ordered_unique = []
    for rid in selected_ids:
        if rid not in seen:
            seen.add(rid)
            ordered_unique.append(rid)
        if len(ordered_unique) >= batch:
            break
    return ordered_unique


def select_batch(
    scores: pd.Series,
    merged: pd.DataFrame,
    batch: int,
    quotas: Dict[str, int] | None = None,
    max_per_prompt: int = 0,
) -> List[str]:
    """
    Return a list of response_id selected for the batch, honoring:
      - global ranking by uncertainty
      - optional per-prompt cap
      - optional per-stream quotas with global remainder fill
    """
    ranked = _rank_df(scores, merged)
    ranked = _apply_per_prompt_cap(ranked, max_per_prompt)

    if quotas:
        return _fill_quota(ranked, batch, quotas)
    # no quotas: take head(batch)
    return ranked["response_id"].head(batch).tolist()