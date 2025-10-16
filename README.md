# EASL Selector (Proxy + Main Scorers)

A modular command-line tool for selecting batches of AI model responses based on uncertainty.  
This selector supports two complementary scoring strategies for the **Efficient Annotation of Scalar Labels (EASL)** project:

- **Proxy (TF-IDF Disagreement):** Identifies uncertain responses based on model disagreement.  
- **Main (Annotator Variance):** Uses annotator score variance to highlight responses needing review (Sprint 4).

The tool is designed to intelligently route responses for human annotation, improving efficiency and label quality.

---

## How It Works

The selector follows a simple yet effective workflow:

1. **Load Data** – Ingests `prompts.csv`, `runs.csv`, and `responses.csv` and merges them into a single dataset.  
2. **Exclude History** – Filters out any responses already selected in previous batches using `selected_history.csv`.  
3. **Score Candidates** –  
   - `ProxyTfidfDisagreementScorer` assigns higher scores to responses that are textually dissimilar from others to the same prompt.  
   - `MainVarianceScorer` computes annotator-score variance once annotation data is available.  
4. **Select Batch** – Chooses the highest-scoring candidates while respecting batch size, per-stream quotas, and prompt-level limits.  
5. **Save Results** – Writes the selected batch to `selector_decisions.csv`, updates `selected_history.csv`, and optionally archives outputs.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- `pip` and `venv` (Python’s built-in environment manager)

---

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/selector_easl.git
cd selector_easl
```

#### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Repository Layout

```
selector_easl/
│
├── scorer.py                # Scoring logic (ProxyTfidfDisagreementScorer, MainVarianceScorer)
├── selection.py             # Selection and I/O operations
├── selector_cli.py          # Command-line interface
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

Place your input CSV files inside the `data/` directory (or specify another directory via `--base`), then run one of the following commands:

### Example Commands

```bash
# Run using proxy scorer (TF-IDF disagreement)
python selector_cli.py --base data --batch 60 --scorer proxy --archive-batches

# Run using main scorer (annotator variance)
python selector_cli.py --base data --batch 60 --scorer main --archive-batches
```

---

## Command-Line Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--base` | Folder containing input CSVs (`prompts.csv`, `runs.csv`, `responses.csv`). | `data` |
| `--batch` | Total number of items to select for the batch. | `60` |
| `--by-stream` | Quotas per stream, e.g., `"gender:30,politics:30"`. | `""` (none) |
| `--max-per-prompt` | Maximum number of responses to select per prompt (0 = no limit). | `0` |
| `--out` | Output filename for the selection decisions. | `selector_decisions.csv` |
| `--history` | Path to the history file used to avoid re-selecting responses. | `selected_history.csv` |
| `--archive-batches` | Saves a copy of the output to a timestamped folder in `./batches/`. | `False` |
| `--scorer` | Scoring strategy to use (`proxy` or `main`). | `proxy` |
| `--seed` | Random seed for reproducible tie-breaking. | `None` |
| `--verbose` | Enable detailed logging output. | `False` |

---

## Inputs and Outputs

### Expected Inputs

| File | Required Columns | Description |
|------|------------------|-------------|
| **prompts.csv** | `prompt_id`, `bias_stream`, `family` | Prompt metadata containing contextual and categorical information. |
| **runs.csv** | `run_id`, `prompt_id`, `model_id` | Model run metadata linking prompts to generated responses. |
| **responses.csv** | `response_id`, `run_id`, `response_text` | Generated responses for each prompt-run combination. |

---

### Generated Outputs

| File | Description |
|------|-------------|
| **selector_decisions.csv** | Selected responses for the current batch, including scores and metadata. |
| **selected_history.csv** | Cumulative log of all selected `response_id`s to prevent duplicates across runs. |
| **./batches/<batch_id>/** | *(Optional)* Archived copy of each batch if `--archive-batches` is used. |

> Both output files are created even if no responses meet the selection criteria, ensuring consistent logs across runs.

---

## Example Run

```bash
python selector_cli.py --base data --batch 60 --scorer proxy --archive-batches --verbose
```

### Expected Output

```bash
[selector] Loaded 300 responses from data/
[selector] Using scorer: proxy
[selector] Wrote 60 selections → data/selector_decisions.csv
[selector] Updated selection history → data/selected_history.csv
[selector] Archived → data/batches/60/selector_decisions.csv
```

---

## Roadmap

- Current: Proxy and Main scorers integrated under modular structure  
- Next: Full annotator-variance support with real feedback data integration
