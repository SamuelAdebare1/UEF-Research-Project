# Information Loss Test — Full-Context Baseline

## Overview

This project measures how accurately a large language model answers questions when it is given an **entire long document in one context window** — no chunking, no retrieval.

It is a baseline experiment. The same document and question set are also evaluated under RAG conditions in the separate `Information Loss Test - RAG` project. The gap in accuracy between the two represents the information loss introduced by chunking and retrieval.

---

## Data Source

The QuALITY dataset (`QuALITY.v1.0.1.htmlstripped.txt`) is sourced from:

https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev

---

## Test Document

This experiment uses story **52845** from the QuALITY dataset.

| File | Description |
|------|-------------|
| `52845.txt` | Original story text extracted from the dataset |
| `52845-final.txt` | Enriched version with three "needle in a haystack" facts injected at different points in the story. This is the document fed to each model during evaluation. |

The injected needle facts are entirely foreign to the original narrative. A model can only answer them correctly if it genuinely reads the specific passage — they cannot be guessed from general knowledge or pre-training data.

---

## Setup

### Requirements

- Python 3.10+

### Create a virtual environment

**Mac / Linux**
```bash
cd "Information Loss Test"
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
cd "Information Loss Test"
python -m venv venv
venv\Scripts\activate.bat
```

### Install dependencies

```bash
pip install json5
```

No additional packages are required — `main.py` uses only the Python standard library.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Extracts a story from the QuALITY dataset by `article_id` and saves it as a plain-text file |
| `QuALITY.v1.0.1.htmlstripped.txt` | The full QuALITY dataset (one JSON object per line) |
| `52845.txt` | Story 52845 extracted from the dataset |
| `52845-final.txt` | Story 52845 with three needle facts injected — the document used in evaluation |
| `Metrics (52845-final).csv` | Per-question scores and model answers |
| `Metrics Summary.png` | Summary table image of results |
| `Using-GPT4All-for-RAG.md` | Notes on running models locally via GPT4All |

### Running `main.py`

```bash
python main.py
```

Edit the `TARGET_ARTICLE_ID` variable at the top of the file to extract a different story from the dataset.

---

## Evaluation Methodology

Each model was tested by submitting **all 22 questions one per tab** (a fresh context window per question), so no answer benefits from context accumulated across previous questions.

**Exception — follow-up question triples:** Three questions are structurally linked, asking the model to name each of the three women pursuing Blake inside his mind. These were submitted together in a single session:

1. *"Name one of the three women pursuing Blake inside his mind."*
2. *"Name another of the three women pursuing Blake inside his mind."*
3. *"Name the third woman pursuing Blake inside his mind."*

---

## Models Evaluated

| Model | Context window |
|-------|---------------|
| DeepSeek-R1-Distill-Qwen-14B | 128k |
| Llama 3.1 8B Instruct | 128k |
| Mistral Instruct | 128k |

---

## Results

Full per-question scores are in [`Metrics (52845-final).csv`](<Metrics%20(52845-final).csv>).

| Metric | DeepSeek-R1-Distill-Qwen-14B | Llama 3.1 8B Instruct 128k | Mistral Instruct |
|--------|------------------------------|---------------------------|-----------------|
| **Overall Accuracy** | 40.9% | 61.4% | 45.5% |
| **Needle Retrieval Score** | 100.0% | 100.0% | 66.7% |
| **Section Recall — early** | 37.5% | 50.0% | 50.0% |
| **Section Recall — middle** | 33.3% | 61.1% | 33.3% |
| **Section Recall — late** | 60.0% | 80.0% | 60.0% |

---

## Question Types

Questions are categorised along two axes:

- **Section** (`early` / `middle` / `late`) — where in the story the answer appears.
- **Type** — `entity`, `fact`, `location`, `event`, `concept`, `evidence`, `reasoning`, `needle`.

The three **needle** questions (Q18–Q20) test whether a model can retrieve the injected facts that do not appear in the original `52845.txt`, making them the sharpest signal for verbatim long-context recall.
