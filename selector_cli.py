# selector_cli.py
from __future__ import annotations
import argparse, os, logging
from datetime import datetime, timezone
import pandas as pd

from scorer import UncertaintyScorer, ProxyTfidfDisagreementScorer
from selection import select_batch


# ----------------------------- Logging -----------------------------
def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("selector_cli")


# ------------------------------ Utils ------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_base(user_base: str) -> str:
    # If user gives absolute path, use it. If relative, anchor to script folder.
    return user_base if os.path.isabs(user_base) else os.path.join(SCRIPT_DIR, user_base)

def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_stream_quota(s: str):
    """
    Parse like 'gender:30,politics:30' -> {'gender': 30, 'politics': 30}
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

def expect_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


# ------------------------------ CLI -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Track-A modular selector — TF-IDF disagreement proxy")
    ap.add_argument("--base", default="data", help="Folder with runs.csv, responses.csv, prompts.csv (relative to script dir by default).")
    ap.add_argument("--batch", type=int, default=60, help="How many items to select for the batch.")
    ap.add_argument("--by-stream", type=str, default="", help="Quotas per bias_stream, e.g. 'gender:30,politics:30'")
    ap.add_argument("--max-per-prompt", type=int, default=0, help="Max responses per prompt (0 = no cap).")
    ap.add_argument("--out", default="selector_decisions.csv", help="Output CSV filename (written in --base).")
    ap.add_argument("--history", default="selected_history.csv", help="History CSV filename (in --base).")
    ap.add_argument("--archive-batches", action="store_true", help="Also write the batch into ./batches/<batch_id>/selector_decisions.csv")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible tie ordering.")
    # keeping it simple for now: only 'proxy' is supported
    ap.add_argument("--scorer", choices=["proxy"], default="proxy", help="Which scorer to use.")
    args = ap.parse_args()

    logger = setup_logging(args.verbose)
    base = resolve_base(args.base)

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

    merged = responses.merge(runs, on="run_id", how="left").merge(prompts, on="prompt_id", how="left")

    if merged.empty:
        logger.warning("No data after merges — check your CSVs.")
        return

    # ---- History exclusion ----
    history_path = os.path.join(base, args.history)
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path, encoding="utf-8")
        if "response_id" in hist_df.columns:
            already = set(hist_df["response_id"].astype(str).unique())
            before = len(merged)
            merged = merged[~merged["response_id"].astype(str).isin(already)]
            logger.info(f"Excluded {before - len(merged)} previously selected responses via history")
        else:
            logger.warning(f"History file has no 'response_id' column: {history_path}")

    if merged.empty:
        logger.warning("No new candidates — all responses already selected previously.")
        return

    # ---- Scorer (proxy) ----
    scorer: UncertaintyScorer = ProxyTfidfDisagreementScorer()

    # Get scores with reasons (DataFrame), plus Series for selection
    scores_df = scorer.score_with_reason(merged)  # response_id, uncertainty_score, coverage_reason
    scores = pd.Series(scores_df["uncertainty_score"].values, index=scores_df["response_id"])

    # ---- Selection ----
    quotas = parse_stream_quota(args.by_stream)
    selected_ids = select_batch(
        scores=scores,
        merged=merged,
        batch=args.batch,
        quotas=quotas if quotas else None,
        max_per_prompt=args.max_per_prompt,
    )

    if not selected_ids:
        logger.warning("Batch selection returned 0 items. Check quotas/caps vs data size.")
        return

    selected = (
        merged[merged["response_id"].isin(selected_ids)]
        .drop_duplicates(subset=["response_id"])
        .merge(scores_df, on="response_id", how="left")  # add uncertainty columns
        .copy()
    )

    # ---- Finalize and write ----
    batch_id = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M")
    decided_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    selected = selected.assign(batch_id=batch_id, decided_utc=decided_utc)
    selected = selected.reset_index(drop=True)
    selected["decision_id"] = selected.index.map(lambda i: f"sel_{i+1:03d}")

    out_cols = [
        "decision_id", "response_id", "prompt_id", "model_id",
        "bias_stream", "family",
        "uncertainty_score", "coverage_reason",
        "batch_id", "decided_utc", "response_text"
    ]

    out_path = os.path.join(base, args.out)
    selected[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Wrote {len(selected)} selections → {out_path}")

    # ---- Update history ----
    history_new = selected[["decision_id", "response_id"]].copy()
    history_new["batch_id"] = batch_id
    history_new["selected_utc"] = decided_utc
    history_new = history_new[["batch_id", "decision_id", "response_id", "selected_utc"]]

    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path, encoding="utf-8")
        hist_df = pd.concat([hist_df, history_new], ignore_index=True)
        hist_df.drop_duplicates(subset=["response_id"], keep="first", inplace=True)
    else:
        hist_df = history_new

    hist_df.to_csv(history_path, index=False, encoding="utf-8")
    logger.info(f"Updated history → {history_path}")

    # ---- Optional archive per batch ----
    if args.archive_batches:
        archive_dir = os.path.join(base, "batches", batch_id)
        os.makedirs(archive_dir, exist_ok=True)
        selected[out_cols].to_csv(os.path.join(archive_dir, "selector_decisions.csv"),
                                  index=False, encoding="utf-8")
        logger.info(f"Archived batch → {archive_dir}")

    # ---- Preview ----
    logger.info("Preview:\n" + selected[["decision_id", "bias_stream", "uncertainty_score"]].head().to_string(index=False))


if __name__ == "__main__":
    main()

