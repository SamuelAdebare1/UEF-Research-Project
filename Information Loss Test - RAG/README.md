# Information Loss Test — RAG Condition

## Overview

This is the **RAG (Retrieval-Augmented Generation) condition** of the information loss experiment.

The experiment uses a _needle-in-a-haystack_ design. Five highly specific facts ("needles") are
embedded at known locations across a 50-page document. Each needle is a detail that a reader could
only answer correctly if the exact passage was in front of them — it cannot be guessed or inferred
from general knowledge. The model is then queried under two conditions:

| Condition               | What the model sees                                  |
| ----------------------- | ---------------------------------------------------- |
| Full-context (baseline) | Entire document in one context window                |
| **RAG (this folder)**   | Only the top-k chunks retrieved by cosine similarity |

The gap in accuracy between the two conditions is the measure of information loss introduced by
chunking and retrieval.

---

## Evaluation Results

### Models Evaluated

| Model                                | Status           | Context window |
| ------------------------------------ | ---------------- | -------------- |
| Meta-Llama-3.1-8B-Instruct-128k-Q4_0 | ✓ Tested         | 128k           |
| Mistral 7B Instruct v0.1 Q4_0        | ✓ Tested         | 32k            |
| DeepSeek-R1-Distill-Qwen-14B Q4_0    | ✗ Failed to load | —              |

> **DeepSeek note:** The GPT4All Python library (`gpt4all` pip package) does not support the
> `deepseek-r1-qwen` pre-tokenizer type used by this model. All 30 questions returned an empty
> answer with `LLaMA ERROR: prompt won't work with an unloaded model!`. Results for this model
> are excluded from all metrics.

---

### Metrics Summary

Full per-question scores, raw answers, and retrieved chunk IDs are in
[`Metrics.csv`](Metrics.csv).

| Metric                                    | Llama 3.1 8B 128k | Mistral 7B Instruct |
| ----------------------------------------- | ----------------- | ------------------- |
| **Overall Accuracy (30 questions)**       | **93.3%** (28/30) | **86.7%** (26/30)   |
| **Needle Score (Q1–Q5)**                  | 100.0% (5/5)      | 100.0% (5/5)        |
| **Adjacent-to-Needle (Q6–Q15)**           | 90.0% (9/10)      | 100.0% (10/10)      |
| **Fact Recall (Q16–Q30)**                 | 93.3% (14/15)     | 73.3% (11/15)       |
| **Section Recall — early (pages 1–15)**   | 93.3% (14/15)     | 80.0% (12/15)       |
| **Section Recall — middle (pages 16–33)** | 100.0% (4/4)      | 100.0% (4/4)        |
| **Section Recall — late (pages 34–50)**   | 90.9% (10/11)     | 90.9% (10/11)       |
| **Difficulty — easy**                     | 100.0% (4/4)      | 100.0% (4/4)        |
| **Difficulty — medium**                   | 94.4% (17/18)     | 83.3% (15/18)       |
| **Difficulty — hard**                     | 87.5% (7/8)       | 87.5% (7/8)         |

---

### Question Structure

Questions span three categories, chosen to address the professor's feedback that LLMs assume
rather than retrieve:

| #       | Category               | Rationale                                                                                                                                                                                                                                                                               |
| ------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1–Q5   | **Needle**             | Facts injected into the document that do not exist in any training corpus. A correct answer _requires_ retrieval — there is no other source.                                                                                                                                            |
| Q6–Q15  | **Adjacent-to-needle** | Details from the same chunk as a needle. Tests whether the retrieved context is rich enough to support follow-up specifics beyond the headline fact.                                                                                                                                    |
| Q16–Q30 | **Biblical fact**      | Facts drawn from the Genesis text (the base document). The LLM may answer these from pre-training knowledge even when retrieval fails, making it impossible to confirm RAG helped from accuracy alone — the `retrieved_chunk_ids` column reveals whether the right chunk was retrieved. |

---

### Genuine Failures (not scoring bugs)

#### Llama 3.1 8B 128k

| Q   | Question                                        | Llama answer                          | Correct answer | Analysis                                                                                                                                                                                                                                                                              |
| --- | ----------------------------------------------- | ------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q11 | Which arm of the galaxy was the probe scanning? | "I don't know based on the document." | Perseus arm    | Chunk 54 was retrieved (it appears in `retrieved_chunk_ids`) and contains the answer. Llama found Xylanthia-9 in the same chunk (Q4 = 1) but failed to extract the preceding "Perseus arm" phrase — a within-chunk attention failure.                                                 |
| Q20 | How many years did Enoch live?                  | "365 + 300 = 665 years"               | 365 years      | The chunk states "all the days of Enoch were three hundred sixty and five years." Llama correctly quoted this but then added the sub-components (65 + 300) a second time, reaching 665. The model understood the question required addition but applied it to already-summed figures. |

#### Mistral 7B Instruct

| Q   | Question                              | Mistral answer                                      | Correct answer                        | Analysis                                                                                                                                                                                                                                                                                                                                                                           |
| --- | ------------------------------------- | --------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q19 | What did Adam and Eve sew together?   | "They made aprons."                                 | Fig leaves                            | The chunk clearly states "they sewed fig leaves together, and made themselves aprons." Mistral answered with the _purpose_ (aprons) rather than the _material_ (fig leaves). The retrieved chunk was correct; the model focused on the wrong semantic element.                                                                                                                     |
| Q21 | How many years did Methuselah live?   | "187 years"                                         | 969 years                             | 187 is Methuselah's age when he _begat Lamech_ — a number that appears two sentences before the total. Mistral extracted the first relevant number rather than the total, a classic partial-context read.                                                                                                                                                                          |
| Q23 | What did God promise Abram at Sichem? | "the land from the river of Egypt to the Euphrates" | "Unto thy seed will I give this land" | **LLM assumption in action.** The retrieved chunk (chunk 23) contains the Sichem promise (Genesis 12:7). But Mistral answered with the much larger territorial covenant from Genesis 15:18 — a passage that was _not_ in the retrieved context. The model overrode retrieved evidence with memorised training knowledge, illustrating exactly the concern raised by the professor. |
| Q27 | How old was Abraham when he died?     | "one hundred and thirty-seven years"                | 175 years                             | 137 is Sarah's age at death (Genesis 23:1), not Abraham's (Genesis 25:7, 175). Chunk 54 (containing the 175-year figure) was retrieved. Mistral produced a plausible biblical number from training memory rather than reading the retrieved passage. Another LLM assumption failure.                                                                                               |

---

### Key Observations

**1. Needle retrieval is robust at top-5.**
Both models scored 100% on Q1–Q5. The `all-mpnet-base-v2` embedder with cosine similarity
and `top_k = 5` reliably surfaced every injected fact regardless of its depth in the document
(page 1 through page 50). The RAG retrieval step does not appear to be the bottleneck.

**2. The bottleneck is within-chunk extraction, not retrieval.**
Llama missed the Perseus arm (Q11) despite chunk 54 appearing in `retrieved_chunk_ids`. The
chunk was retrieved; the LLM failed to surface the right sentence within it. Similarly, Mistral's
"aprons" answer (Q19) came from a chunk that contained "fig leaves" three words earlier. This
suggests that with longer, denser chunks (500 tokens each), models can retrieve the right region
yet still miss specific details.

**3. LLMs assume — pre-training knowledge overrides retrieved context.**
Mistral's Q23 answer (Genesis 15:18 covenant boundaries) and Q27 answer (Sarah's age confused
with Abraham's) both show the model producing confident, plausible-sounding answers from
training memory that directly contradict or ignore the retrieved passage. This validates Dr. Pala's concern: correct retrieval does not guarantee the model will _use_ what it retrieved.
Llama's verbose self-correction pattern (answer → "I don't know based on the document" →
re-answer) reflects the same tension, though Llama typically settled on the retrieved answer.

**4. Mistral is cleaner but less grounded for complex facts.**
Mistral's answers are concise and precise (no repetition or self-contradiction), which made it
perfect on adjacent-to-needle questions (100% vs Llama's 90%). However, for the longer
biblical narrative questions that require reading multiple sentences in the retrieved chunk,
Mistral's short-form reading strategy caused it to extract the wrong number (Q21: 187 vs 969)
or the wrong action (Q19: aprons vs fig leaves).

**5. Section depth had no clear effect on retrieval.**
Both models performed similarly across early, middle, and late sections of the 50-page document.
The dense embedding model appears to retrieve equally well at any document depth within this
corpus size (79 chunks).

---

## Needles

All five needles are documented in [`Needle in haystack questions.txt`](Needle%20in%20haystack%20questions.txt).
They span five fictional document genres placed at different depths in the 50-page document:

| Test | Genre                        | Needle location | Target answer                     |
| ---- | ---------------------------- | --------------- | --------------------------------- |
| 1    | Corporate compliance manual  | Page 1          | Neon-orange anti-static wristband |
| 2    | Scientific research paper    | Page 15         | A distinct metallic taste         |
| 3    | Financial contract           | Page 25         | Helsinki                          |
| 4    | Deep space telemetry log     | Page 35         | Xylanthia-9                       |
| 5    | Municipal historical archive | Page 50         | Its secondary rudder              |

The needles are intentionally scattered across pages 1, 15, 25, 35, and 50 to test whether
retrieval performance degrades with document depth.

---

## Pipeline

The pipeline runs in three sequential steps:

```
50-pages.pdf
     │
     ▼  chunker.py
text extraction → sliding window (500 tokens, 50 overlap) → chunks.json
     │
     ▼  embedder.py
chunks.json → dense vectors (all-mpnet-base-v2, 768 dim) → embeddings.json
     │
     ▼  api.py  (uvicorn)
embeddings.json → FastAPI server → /query  (batch)
                                 → /query/stream  (token-by-token SSE)
```

At query time, each needle question is embedded with the same model, and cosine similarity
is computed against all chunk vectors to retrieve the top-k candidates.

---

## Step 1 — Chunking (`chunker.py`)

### What it does

Extracts all text from the PDF and splits it into overlapping token windows using a sliding
stride. Each chunk is saved with its ID, token count, and raw text.

### Parameters

| Parameter    | Value         | Rationale                                                                                                                                   |
| ------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `CHUNK_SIZE` | 500 tokens    | Keeps retrieved passages focused; reduces the amount of irrelevant context fed to the LLM per chunk                                         |
| `OVERLAP`    | 50 tokens     | Prevents a needle from being split cleanly across two chunk boundaries                                                                      |
| Tokeniser    | `cl100k_base` | The tiktoken tokeniser used by GPT-4 and OpenAI embedding models — ensures token counts are meaningful if the downstream model is API-based |

> The tiktoken tokeniser here is **only used for window sizing**. It does not produce input
> for the embedding model. The embedding model runs its own internal tokenizer (see Step 2).

### Output

`chunks.json` — a JSON array where each element is:

```json
{
  "chunk_id": 0,
  "token_count": 500,
  "text": "..."
}
```

Current stats: **79 chunks**, avg 497 tokens (~1820 chars) each.

### Run

```bash
python chunker.py                      # reads 50-pages.pdf from this folder
python chunker.py path/to/other.pdf    # or pass a different PDF
```

---

## Step 2 — Embedding (`embedder.py`)

### What it does

Passes each chunk's raw text through a sentence-transformer model and appends the resulting
vector to the chunk object. The output file is a drop-in replacement for `chunks.json` —
same structure, with an `"embedding"` field added to each entry.

### Model choice

**`all-mpnet-base-v2`** (sentence-transformers)

Selected from the SBERT pretrained model catalogue:
[https://www.sbert.net/docs/sentence_transformer/pretrained_models.html](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

The documentation identifies `all-mpnet-base-v2` as the model that _"provides the best quality"_
among the general-purpose pretrained models. It was trained on more than 1 billion sentence pairs
(the full "all-\*" dataset), making it the strongest baseline for semantic retrieval quality.

`all-MiniLM-L6-v2` is 5× faster, but speed is irrelevant here — embeddings are computed once at
index time, not during inference. Quality is the only meaningful criterion, since a retrieval miss
means the needle is never shown to the model.

| Property            | Value                     |
| ------------------- | ------------------------- |
| Embedding dimension | 768                       |
| Max sequence length | 384 tokens                |
| Training data       | 1 billion+ pairs          |
| Runs locally        | Yes — no API key required |

### Output

`embeddings.json` — same array as `chunks.json`, each element extended with:

```json
{
  "chunk_id": 0,
  "token_count": 500,
  "text": "...",
  "embedding": [0.0428, 0.0122, ...]   // 768 floats
}
```

### Run

```bash
python embedder.py                       # reads chunks.json from this folder
python embedder.py path/to/chunks.json   # or pass a different chunks file
```

---

## Step 3 — API server (`api.py`)

### What it does

Loads `embeddings.json` once at startup, then for each query:

1. Embeds the question with `all-mpnet-base-v2`
2. Runs cosine-similarity search → top-k chunks
3. Builds a prompt (context + question) and calls the local LLM
4. Returns the answer and source chunks

The LLM is **Meta-Llama-3.1-8B-Instruct-128k-Q4_0** running locally via GPT4All — no internet
or API key required after the model file is downloaded.

### Endpoints

| Method | Path            | Description                                                                            |
| ------ | --------------- | -------------------------------------------------------------------------------------- |
| `GET`  | `/health`       | Returns `{"status": "ok"}`                                                             |
| `POST` | `/query`        | Full response — waits for complete generation then returns JSON                        |
| `POST` | `/query/stream` | Streaming response — sends sources first, then streams tokens one-by-one as SSE events |

### Request body (`/query` and `/query/stream`)

```json
{
  "question": "What is the age of Abraham?",
  "top_k": 5,
  "max_tokens": 512,
  "temperature": 0.7
}
```

### Streaming event format (`/query/stream`)

```
data: {"type": "sources", "sources": [...]}

data: {"type": "token", "token": " 175"}

data: {"type": "done"}
```

### Run

```bash
source ../venv/bin/activate
uvicorn api:app --reload --port 8000
```

The server is available at `http://localhost:8000`. CORS is pre-configured for
`localhost:5173`, `localhost:5174`, and `localhost:3000`.

---

## Chat UI

A React chat interface for the API lives in [`../rag-ui/`](../rag-ui/). It uses the
`/query/stream` endpoint so responses appear token-by-token. See the UI README for setup.

---

## Reproducibility

### Environment

This project uses a shared virtual environment at the repository root:

```bash
source ../venv/bin/activate
```

### Dependencies

```bash
pip install -r ../requirements.txt
```

### Full pipeline (first time or after changing the PDF)

```bash
source ../venv/bin/activate
python chunker.py
python embedder.py
uvicorn api:app --port 8000
```

No external accounts, API keys, or internet access are required after model weights are
downloaded on the first run (weights are cached locally by `sentence-transformers` and GPT4All).

---

## File reference

| File                               | Role                                                          |
| ---------------------------------- | ------------------------------------------------------------- |
| `50-pages.pdf`                     | Source document containing the five needles                   |
| `Needle in haystack questions.txt` | Needle locations, questions, and target answers               |
| `chunker.py`                       | Step 1 — extracts and chunks the PDF                          |
| `embedder.py`                      | Step 2 — encodes chunks into embedding vectors                |
| `api.py`                           | Step 3 — FastAPI server exposing `/query` and `/query/stream` |
| `chunks.json`                      | Output of Step 1                                              |
| `embeddings.json`                  | Output of Step 2                                              |
