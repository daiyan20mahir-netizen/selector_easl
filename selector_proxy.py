import argparse
import os
import re
import logging
from datetime import datetime, timezone
from itertools import combinations
import pandas as pd


# ----------------------------- Logging ----------------------------

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("easl_proxy_selector")

# ----------------------------- Utils ------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def tokenize(text: str):
    """Lowercase alphanumeric tokenizer → set of tokens."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

# --- NEW: cosine-based uncertainty over TF-IDF embeddings ---
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def group_uncertainty(texts):
    """
    Compute per-prompt uncertainty from all model responses to that prompt.
    Proxy = 1 - mean cosine similarity across TF-IDF embeddings (pairwise).
    Returns (uncertainty_score, coverage_reason)
    """
    # keep only non-empty strings
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 2:
        return 0.35, "insufficient_responses"  # fallback when fewer than 2 responses exist

    # Vectorize with TF-IDF 
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
    X = vec.fit_transform(texts)  # shape: (n_texts, vocab)

    # Cosine similarity matrix
    sims = cosine_similarity(X)
    # Upper triangle (no diagonal)
    n = len(texts)
    triu = sims[np.triu_indices(n, k=1)]
    mean_sim = float(triu.mean()) if triu.size else 1.0

    # Disagreement proxy
    u = 1.0 - mean_sim

    # Standardized coverage reasons
    if u >= 0.45:
        reason = "high_model_disagreement"
    elif u >= 0.25:
        reason = "moderate_model_disagreement"
    else:
        reason = "low_model_disagreement"

    return u, reason


# --- keep these helpers unchanged (shown here for completeness) ---
def parse_stream_quota(s: str):
    """
    Parse --by-stream like 'gender:30,politics:30' -> {'gender': 30, 'politics': 30}
    """
    if not s:
        return {}
    out = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = int(v)
            except ValueError:
                pass
    return out

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def expect_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


# ----------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="EASL Proxy Selector (model disagreement) — Lean version")
    ap.add_argument("--base", required=True, help="Folder containing CSVs (runs, responses, prompts).")
    ap.add_argument("--batch", type=int, default=60, help="How many items to select for the batch.")
    ap.add_argument("--by-stream", type=str, default="", help="Quotas per bias_stream, e.g. 'gender:30,politics:30'")
    ap.add_argument("--max-per-prompt", type=int, default=0,
                    help="Max responses per prompt (0 = no cap). Prevents a single prompt dominating the batch.")
    ap.add_argument("--out", default="selector_decisions.csv", help="Output CSV filename (written in --base).")
    ap.add_argument("--history", default="selected_history.csv",
                    help="History CSV filename (in --base) to prevent repeats.")
    ap.add_argument("--archive-batches", action="store_true",
                    help="Also write the batch into ./batches/<batch_id>/selector_decisions.csv")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible tie ordering.")
    args = ap.parse_args()

    logger = setup_logging(args.verbose)
    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed); np.random.seed(args.seed)
        logger.debug(f"Seed set to {args.seed}")

    base = args.base

    # ---- Validate file presence ----
    for fname in ["runs.csv", "responses.csv", "prompts.csv"]:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")

    # ---- Load inputs (only needed cols) ----
    responses = pd.read_csv(os.path.join(base, "responses.csv"),
                            usecols=["response_id", "run_id", "response_text"], encoding="utf-8")
    runs = pd.read_csv(os.path.join(base, "runs.csv"),
                       usecols=["run_id", "prompt_id", "model_id"], encoding="utf-8")
    prompts = pd.read_csv(os.path.join(base, "prompts.csv"),
                          usecols=["prompt_id", "bias_stream", "family"], encoding="utf-8")

    expect_cols(responses, ["response_id", "run_id", "response_text"], "responses.csv")
    expect_cols(runs, ["run_id", "prompt_id", "model_id"], "runs.csv")
    expect_cols(prompts, ["prompt_id", "bias_stream", "family"], "prompts.csv")

    # ---- Enrich responses with metadata (simple merges; fine at this scale) ----
    df = responses.merge(runs, on="run_id", how="left") \
                  .merge(prompts, on="prompt_id", how="left")

    # Cast to categoricals to reduce memory at larger sizes (optional, cheap)
    for c in ["prompt_id", "model_id", "bias_stream", "family"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    if df.empty:
        logger.warning("No data after joins — check your CSVs.")
        return

    # ---- Exclude responses already selected in previous batches ----
    history_path = os.path.join(base, args.history)
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path, encoding="utf-8")
        if "response_id" in hist.columns:
            already = set(hist["response_id"].astype(str).unique())
            before = len(df)
            df = df[~df["response_id"].astype(str).isin(already)]
            logger.info(f"Excluded {before - len(df)} previously selected responses via history")
        else:
            logger.warning(f"History file found but has no 'response_id' column: {history_path}")
            hist = pd.DataFrame(columns=["batch_id", "decision_id", "response_id", "selected_utc"])
    else:
        hist = pd.DataFrame(columns=["batch_id", "decision_id", "response_id", "selected_utc"])

    if df.empty:
        logger.warning("No new candidates — all responses already selected previously.")
        return

    # ---- Compute per-prompt uncertainty (single groupby; fast & simple) ----
    scores = (
        df.groupby("prompt_id", dropna=False)["response_text"]
          .apply(lambda s: group_uncertainty(s.tolist()))
          .apply(pd.Series)
          .rename(columns={0: "uncertainty_score", 1: "coverage_reason"})
          .reset_index()
    )

    # ---- Broadcast prompt-level scores to each response ----
    decisions = df.merge(scores, on="prompt_id", how="left")
    decisions["text_len"] = decisions["response_text"].astype(str).str.len()

    # ---- Rank: highest uncertainty first; shorter text wins ties ----
    decisions = decisions.sort_values(["uncertainty_score", "text_len"], ascending=[False, True])
    logger.info(f"Ranked {len(decisions)} candidate responses")

    # ---- (Optional) Cap per-prompt items to avoid dominance ----
    if args.max_per_prompt and args.max_per_prompt > 0:
        decisions["_rank_within_prompt"] = decisions.groupby("prompt_id").cumcount()
        decisions = decisions[decisions["_rank_within_prompt"] < args.max_per_prompt]
        decisions = decisions.drop(columns=["_rank_within_prompt"])
        logger.info(f"Applied per-prompt cap: max {args.max_per_prompt} responses per prompt")

    # ---- Apply quotas, if any ----
    quotas = parse_stream_quota(args.by_stream)
    if quotas:
        logger.info(f"Applying per-stream quotas: {quotas}")
        selected_parts = []
        remaining = args.batch

        # 1) Top items per stream
        for stream, q in quotas.items():
            pick = decisions[decisions["bias_stream"] == stream].head(q)
            selected_parts.append(pick)
            remaining -= len(pick)

        # 2) Fill remainder from global list (no repeats)
        selected = pd.concat(selected_parts) if selected_parts else pd.DataFrame(columns=decisions.columns)
        picked_ids = set(selected["response_id"].unique()) if not selected.empty else set()
        if remaining > 0:
            filler = decisions[~decisions["response_id"].isin(picked_ids)].head(remaining).copy()
            filler["coverage_reason"] = "quota_fill"  # transparency
            selected = pd.concat([selected, filler], ignore_index=True)

        selected = selected.drop_duplicates(subset=["response_id"])
    else:
        selected = decisions.head(args.batch)
        logger.info("No quotas specified; using global ranking only.")

    if len(selected) < args.batch:
        logging.warning(f"Requested batch={args.batch} but selected only {len(selected)}. "
                        "Data availability or quotas may be limiting.")

    # ---- Finalize batch metadata ----
    batch_id = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M")
    selected = selected.copy()
    selected["batch_id"] = batch_id
    selected["decided_utc"] = iso_now()
    selected = selected.reset_index(drop=True)
    selected["decision_id"] = selected.index.map(lambda i: f"sel_{i+1:03d}")

    out_cols = [
        "decision_id", "response_id", "prompt_id", "model_id",
        "bias_stream", "family", "uncertainty_score", "coverage_reason",
        "batch_id", "decided_utc", "response_text"
    ]

    # ---- Write current batch (latest) ----
    out_path = os.path.join(base, args.out)
    selected[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Wrote {len(selected)} selections → {out_path}")

    # ---- Update history (prevents future repeats) ----
    history_new = selected[["decision_id", "response_id"]].copy()
    history_new["batch_id"] = batch_id
    history_new["selected_utc"] = iso_now()
    history_new = history_new[["batch_id", "decision_id", "response_id", "selected_utc"]]

    hist_all = pd.concat([hist, history_new], ignore_index=True)
    hist_all.drop_duplicates(subset=["response_id"], keep="first", inplace=True)
    hist_all.to_csv(history_path, index=False, encoding="utf-8")
    logger.info(f"Updated history → {history_path}")

    # ---- Optional archive per batch ----
    if args.archive_batches:
        archive_dir = os.path.join(base, "batches", batch_id)
        ensure_dir(archive_dir)
        selected[out_cols].to_csv(os.path.join(archive_dir, "selector_decisions.csv"),
                                  index=False, encoding="utf-8")
        logger.info(f"Archived batch → {archive_dir}")

    # ---- Preview ----
    preview = selected[["decision_id", "bias_stream", "uncertainty_score", "coverage_reason"]].head()
    logger.info("Preview of selections:\n" + preview.to_string(index=False))

if __name__ == "__main__":
    main()
