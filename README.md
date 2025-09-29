# EASL Proxy Selector

This repository implements a **proxy selector** for Efficient Annotation of Scalar Labels (EASL).
The selector identifies uncertain responses (based on model disagreement) for annotation. This proxy selector feeds into the **main selector** (Sprint 4),
which will switch to using **annotator score variance** instead of model disagreement.



## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/daiyan20mahir-netizen/selector_easl.git
   cd selector_easl
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure
```
selector_easl/
│── selector_proxy.py         # main proxy selector script
│── requirements.txt          # dependencies
│── README.md                 # instructions
│── .gitignore                # ignore rules
│── data/                     # input CSVs (local only, not committed)
│── batches/                  # archived outputs (auto-created, ignored in Git)
```

---

##  Usage

Prepare your input CSVs (`runs.csv`, `responses.csv`, `prompts.csv`) in a folder(data), then run:

```bash
python selector_proxy.py --base ./data --batch 20
```

### Options
- `--batch` → number of responses to select (e.g., 20, 50).
- `--by-stream "gender:10,politics:10"` → per-stream quotas.
- `--out` → filename for the decisions file (default: `selector_decisions.csv`).
- `--history` → history file to avoid repeats (default: `selected_history.csv`).
- `--archive-batches` → also save batch outputs into `./batches/<batch_id>/`.

---

##  Outputs
- `selector_decisions.csv` → responses to annotate (current batch).
- `selected_history.csv` → cumulative log of all selected responses.
- `./batches/<batch_id>/` → archived copies of each batch (if enabled).

---

