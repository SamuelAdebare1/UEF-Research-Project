# Linguistic Analysis

NLP analysis suite applied to `50-pages.pdf` — a 50-page document containing the Book of Genesis (KJV) with five fictional "needle" sentences injected at specific pages. The goal is to detect the injected content using multiple independent linguistic methods.

---

## What is `50-pages.pdf`?

The document is a hybrid: the base text is the Book of Genesis (KJV), which has a highly consistent, archaic writing style. Five short fictional passages from different genres (corporate compliance manual, scientific paper, financial contract, space telemetry log, historical archive) were injected at pages 1, 15, 25, 35, and 50. These injections are stylistically foreign to the biblical prose, making them detectable by linguistic analysis.

---

## Setup

### Requirements

- Python 3.10+

### Create a virtual environment

**Mac / Linux**
```bash
cd "Linguistic Analysis"
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
cd "Linguistic Analysis"
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
cd "Linguistic Analysis"
python -m venv venv
venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install pdfminer.six sentence-transformers transformers scikit-learn scipy numpy torch
```

---

## Files

| File | Purpose |
|------|---------|
| `50-pages.pdf` | Source document — Genesis text with five injected sentences |
| `analysis_utils.py` | Shared PDF loader and sentence splitter used by all analysis scripts |
| `semantic_anomaly.py` | Semantic anomaly detection via neighbour similarity |
| `word_frequency.py` | Word frequency, type-token ratio, hapax legomena |
| `generate_dashboard.py` | Reads all result JSON files and writes `dashboard.html` |
| `dashboard.html` | Interactive visual dashboard — open in any browser |
| `results_anomalies.json` | Output of `semantic_anomaly.py` |
| `results_word_frequency.json` | Output of `word_frequency.py` |
| `TODO.md` | Pending analyses |

---

## Running the analyses

Run each script from inside the `Linguistic Analysis` folder with the virtual environment active.

### 1. Semantic Anomaly Detection

```bash
python semantic_anomaly.py
```

Embeds every sentence with `all-MiniLM-L6-v2` and computes each sentence's mean cosine similarity to its 5 nearest neighbours. Sentences scoring below **0.30** are flagged as anomalous.

Output → `results_anomalies.json`

**Key result:** All five injected needles scored **negative** similarity — far below any genuine biblical sentence.

### 2. Word Frequency & Vocabulary

```bash
python word_frequency.py
```

Computes type-token ratio, hapax legomena count, and a ranked frequency list of the top 100 content words.

Output → `results_word_frequency.json`

**Key result:** TTR = 0.08, 839 hapax legomena (43.7% of vocabulary). Top entities by frequency: Abraham (126), son (122), earth (113), Jacob (101).

### 3. Generate the Dashboard

After running the analyses above:

```bash
python generate_dashboard.py
```

Writes `dashboard.html`. Open it in a browser — no server needed.

---

## Pending Analyses

See [`TODO.md`](TODO.md) for planned analyses:

- **Stylometric Analysis** — sentence length and vocabulary richness to detect style shifts at injection points
- **Sentence Embedding & Clustering** — group sentences by semantic topic
- **Readability & Complexity Scoring** — Flesch-Kincaid on biblical vs. injected text
- **Cohesion & Coherence Analysis** — lexical chain analysis across the document
