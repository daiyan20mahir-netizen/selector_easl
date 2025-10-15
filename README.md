# EASL Disagreement-Proxy Selector

A modular command-line tool for selecting batches of AI model responses based on uncertainty.  
This selector uses TF-IDF disagreement as a proxy for identifying the most informative responses for human annotation.

This tool serves as a proxy selector for the Efficient Annotation of Scalar Labels (EASL) project.  
It is designed to intelligently feed responses to a future main selector, which will use annotator score variance instead of model disagreement.

---

## How It Works

The selector follows a simple yet effective workflow:

1. **Load Data** – Ingests `prompts.csv`, `runs.csv`, and `responses.csv` and merges them into a single dataset.  
2. **Exclude History** – Filters out any responses that have already been selected in previous batches by checking against a history file (`selected_history.csv`).  
3. **Score Candidates** – Calculates an `uncertainty_score` for each response.  
   - The default scorer (`ProxyTfidfDisagreementScorer`) assigns higher scores to responses that are textually dissimilar from other model responses to the same prompt.  
4. **Select Batch** – Chooses the highest-scoring candidates while respecting user-defined constraints such as total batch size, per-stream quotas, and maximum responses per prompt.  
5. **Save Results** – Writes the selected batch to `selector_decisions.csv`, updates the cumulative history log, and optionally archives the batch into a timestamped folder.  

*Planned upgrade:* The main EASL selector will replace the disagreement scorer with an annotator-variance-based scorer.  
That version will plug into `selection.py` while keeping the same CLI interface.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- `pip` and `venv` (Python’s built-in environment manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/selector_easl.git
   cd selector_easl
   ```

2. **Create and activate a virtual environment**
   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Repository Layout

```
selector_easl/
│
├── selection.py             # Selection logic (batch selection + I/O)
├── scorer.py                # Scoring classes (ProxyTfidfDisagreementScorer, etc.)
├── selector_cli.py          # Command-line interface / dispatcher
├── selector_legacy.py       # Older baseline version (reference)
├── requirements.txt         # Dependencies
├── README.md                # Documentation
│
├── data/                    # (Local only) Input CSVs
│   ├── prompts.csv
│   ├── runs.csv
│   └── responses.csv
│
└── batches/                 # (Auto-created) Archived outputs per batch
```

---

## Usage

Place your input CSV files inside the `data/` directory (or any directory you specify via `--base`), then run:

### Example Command
```bash
python selector_cli.py --base data --batch 60 --by-stream "gender:30,politics:30" --archive-batches
```

### Command-Line Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--base` | Folder containing input CSVs (`prompts.csv`, `runs.csv`, `responses.csv`). | `data` |
| `--batch` | Total number of items to select for the batch. | `60` |
| `--by-stream` | Quotas per stream, e.g., `"gender:30,politics:30"`. | `""` (none) |
| `--max-per-prompt` | Maximum number of responses to select for any single prompt (`0` = no limit). | `0` |
| `--out` | Output filename for the selection decisions. | `selector_decisions.csv` |
| `--history` | Path to the history file used to avoid re-selecting responses. | `selected_history.csv` |
| `--archive-batches` | If set, saves a copy of the output to a timestamped folder in `./batches/`. | `False` |
| `--scorer` | Scoring strategy to use (currently only `proxy`). | `proxy` |
| `--seed` | Random seed for reproducible tie-breaking. | `None` |
| `--verbose` | Enable detailed logging output. | `False` |

---

## Inputs and Outputs

### Expected Inputs

The script requires three CSV files located in the `--base` directory:

| File | Required Columns | Description |
|------|------------------|-------------|
| `prompts.csv` | `prompt_id`, `bias_stream`, `family` | Prompt metadata |
| `runs.csv` | `run_id`, `prompt_id`, `model_id` | Model run information |
| `responses.csv` | `response_id`, `run_id`, `response_text` | Generated responses |

---

### Generated Outputs

| File | Description |
|------|-------------|
| `selector_decisions.csv` | Selected responses for the current batch with scores and metadata. |
| `selected_history.csv` | Cumulative log of all selected `response_id`s to prevent duplicates. |
| `./batches/<batch_id>/` | (Optional) Archived copy of `selector_decisions.csv` if `--archive-batches` is used. |

Both output files are created even if no responses meet the selection criteria, ensuring consistent logs across runs.

---

## Example Run

```bash
# Run 60 selections with logging and archiving enabled
python selector_cli.py --base data --batch 60 --archive-batches --verbose
```

**Expected output:**
```
Wrote 60 selections → data/selector_decisions.csv
Updated selection history → data/selected_history.csv
Archived → data/batches/60/selector_decisions.csv
```

---

## Roadmap

- Current: Proxy selector using TF-IDF disagreement  
- Next: EASL main selector using annotator score variance
