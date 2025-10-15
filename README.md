# EASL Modular Selector

A modular command-line tool for selecting batches of AI model responses for annotation.
This tool is part of the Efficient Annotation of Scalar Labels (EASL) project, designed to intelligently prioritize data for human review and improve labeling efficiency.

It supports two complementary scoring strategies:

- Proxy (TF-IDF Disagreement): Identifies responses where models disagree, serving as a proxy for uncertainty when human annotations are not yet available.
- Main (Annotator Variance): Once annotations are collected, flags responses with high disagreement among human annotators, prioritizing them for expert review.

---

## How It Works

The selector follows a simple workflow to build annotation batches:

1. Load Data: Ingests prompts.csv, runs.csv, and responses.csv and merges them into a single dataset.
2. Exclude History: Removes responses that have already been selected in previous batches (selected_history.csv).
3. Score Candidates:
   - ProxyTfidfDisagreementScorer scores responses based on textual dissimilarity among peers for the same prompt.
   - MainVarianceScorer computes annotator-score variance for already-labeled data.
4. Select Batch: Chooses the highest-scoring candidates while respecting total batch size, per-category quotas, and prompt-level limits.
5. Save Outputs: Writes selected responses to selector_decisions.csv, updates the history log, and optionally archives outputs to timestamped directories.

---

## Getting Started

### Prerequisites
- Python 3.9–3.11
- pip and venv (Python’s built-in environment manager)

### Installation

Clone the repository
```
git clone https://github.com/YOUR_USERNAME/selector_easl.git
cd selector_easl
```

Create and activate a virtual environment
```
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies
```
pip install -r requirements.txt
```

---

## Project Structure

```
selector_easl/
│
├── scorer.py                # Scoring logic (ProxyTfidfDisagreementScorer, MainVarianceScorer)
├── selection.py             # Batch selection algorithm and constraints
├── selector_cli.py          # Command-line interface
├── requirements.txt         # Dependencies
├── README.md                # Documentation
│
├── data/                    # Local input CSVs
│   ├── prompts.csv
│   ├── runs.csv
│   └── responses.csv
│
└── batches/                 # Auto-created output archives
```

---

## Usage

Place your input CSV files in the data/ directory and run the tool from your terminal.

### Example Commands

```
# 1. Using the proxy scorer for initial selection
python selector_cli.py --base data --batch 60 --scorer proxy --archive-batches --verbose

# 2. Using the main scorer to find contentious items
python selector_cli.py --base data --batch 20 --scorer main --by-stream "politics:10"
```

---

## Command-Line Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| --base | Folder containing input CSVs (prompts.csv, runs.csv, responses.csv). | data |
| --batch | Total number of items to select for the batch. | 60 |
| --by-stream | Quotas per stream, e.g., "gender:30,politics:30". | "" (none) |
| --max-per-prompt | Max responses per prompt (0 = no limit). | 0 |
| --scorer | Scoring strategy (proxy or main). | proxy |
| --out | Output filename for selection decisions. | selector_decisions.csv |
| --history | Path to the history file for avoiding duplicates. | selected_history.csv |
| --archive-batches | Save results in timestamped subfolders. | False |
| --seed | Random seed for reproducible tie-breaking. | None |
| --verbose | Enable detailed logging. | False |

---

## Example Run

```
python selector_cli.py --base data --batch 60 --scorer proxy --archive-batches --verbose
```

Expected Output:
```
[selector] Loaded 300 responses from data/
[selector] Using scorer: proxy
[selector] Wrote 60 selections → data/selector_decisions.csv
[selector] Updated selection history → data/selected_history.csv
[selector] Archived → data/batches/60/selector_decisions.csv
```

---

## Roadmap

- Current: Proxy and main scorers integrated under modular structure
- Next: Extend main scorer with real annotator variance data
- Future: Add visualization dashboard and API for real-time annotation tracking
