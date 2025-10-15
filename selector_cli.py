import argparse
import os
import pandas as pd
from selection import select_batch

def parse_by_stream(arg: str):
    if not arg:
        return None
    quotas = {}
    for part in arg.split(","):
        if not part.strip():
            continue
        key, val = part.split(":")
        quotas[key.strip()] = int(val)
    return quotas

def main():
    ap = argparse.ArgumentParser(description="EASL Selector CLI")
    ap.add_argument("--base", default="data", help="Folder containing input CSVs")
    ap.add_argument("--batch", type=int, default=60, help="Batch size")
    ap.add_argument("--by-stream", default="", help='Comma-separated quotas, e.g., "gender:30,politics:30"')
    ap.add_argument("--max-per-prompt", type=int, default=0, help="Max responses per prompt (0 = no limit)")
    ap.add_argument("--out", default="selector_decisions.csv", help="Output file name")
    ap.add_argument("--history", default="selected_history.csv", help="History file name")
    ap.add_argument("--archive-batches", action="store_true", help="Save archived batch copies")
    ap.add_argument("--scorer", choices=["proxy", "main"], default="proxy",
                    help="Scoring strategy: 'proxy' (TF-IDF disagreement) or 'main' (annotator variance).")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    base = args.base.rstrip("/")

    # Load CSVs
    prompts = pd.read_csv(os.path.join(base, "prompts.csv"))
    runs = pd.read_csv(os.path.join(base, "runs.csv"))
    responses = pd.read_csv(os.path.join(base, "responses.csv"))

    # Merge
    merged = responses.merge(runs, on="run_id").merge(prompts, on="prompt_id")

    if args.verbose:
        print(f"[selector] Loaded {len(merged)} responses from {base}/")
        print(f"[selector] Using scorer: {args.scorer}")

    # Select
    selected = select_batch(
        merged,
        scorer_type=args.scorer,
        batch_size=args.batch,
        by_stream=parse_by_stream(args.by_stream),
        max_per_prompt=args.max_per_prompt,
    )

    # Write decisions
    out_path = os.path.join(base, args.out)
    selected.to_csv(out_path, index=False)
    if args.verbose:
        print(f"[selector] Wrote {len(selected)} selections → {out_path}")

    # Update history (append unique response_ids)
    history_path = os.path.join(base, args.history)
    selected_ids = selected[["response_id"]].drop_duplicates()
    try:
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, selected_ids], ignore_index=True).drop_duplicates()
    except FileNotFoundError:
        combined = selected_ids
    combined.to_csv(history_path, index=False)
    if args.verbose:
        print(f"[selector] Updated selection history → {history_path}")

    # Optional archive
    if args.archive_batches:
        batch_id = str(args.batch)
        arch_dir = os.path.join(base, "batches", batch_id)
        os.makedirs(arch_dir, exist_ok=True)
        selected.to_csv(os.path.join(arch_dir, "selector_decisions.csv"), index=False)
        if args.verbose:
            print(f"[selector] Archived → {arch_dir}/selector_decisions.csv")

if __name__ == "__main__":
    main()
