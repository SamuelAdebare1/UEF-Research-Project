"""
test_runner.py — automated information loss test for the RAG pipeline.

Loads embeddings once, then for each of the three local models:
  1. Loads the GPT4All model
  2. Runs all 30 questions through the RAG pipeline
  3. Auto-scores answers via keyword matching
  4. Frees the model from memory before loading the next

Output: Metrics.csv (raw answers + scores) and a printed summary table.

Run:
    source ../venv/bin/activate
    python test_runner.py

NOTE: Stop the API server (uvicorn) before running — it holds the Llama model
      in memory and the 14B model especially needs room to load cleanly.
"""

import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
EMBEDDINGS_FILE = BASE_DIR / "embeddings.json"
MODEL_DIR = Path.home() / "Library/Application Support/nomic.ai/GPT4All"
EMBED_MODEL_NAME = "all-mpnet-base-v2"
TOP_K = 5
N_CTX = 4096
MAX_NEW_TOKENS = 256  # shorter to keep test fast; needles need only ~10 tokens
TEMPERATURE = 0.1     # low temp for deterministic recall answers

MODELS = [
    "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf",
    "DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf",
    "mistral-7b-instruct-v0.1.Q4_0.gguf",
]

# ── questions ─────────────────────────────────────────────────────────────────
# keywords: ALL listed terms must appear in the answer (case-insensitive) for
# score = 1. Design them to match the unique, specific part of the answer only.
QUESTIONS = [
    # ── NEEDLE QUESTIONS (5) ──────────────────────────────────────────────────
    # These are the injected facts. The model has zero chance of knowing them
    # from pre-training; the correct chunk MUST be retrieved.
    {
        "id": 1,
        "question": "Which specific type of wristband must employees wear when operating the TS-500 machinery in Sector 4?",
        "target_answer": "Neon-orange anti-static wristband",
        "keywords": ["neon", "wristband"],
        "section": "early",
        "difficulty": "hard",
        "type": "needle",
    },
    {
        "id": 2,
        "question": "What specific sensory anomaly did subject 402-B report after receiving the placebo?",
        "target_answer": "A distinct metallic taste",
        "keywords": ["metallic", "taste"],
        "section": "early",
        "difficulty": "hard",
        "type": "needle",
    },
    {
        "id": 3,
        "question": "To which city must the Request for Dissolution form be submitted when a force majeure lasts longer than thirty consecutive days?",
        "target_answer": "Helsinki",
        "keywords": ["helsinki"],
        "section": "middle",
        "difficulty": "hard",
        "type": "needle",
    },
    {
        "id": 4,
        "question": "What is the name of the asteroid near which the secondary probe telemetry indicated a sudden temperature drop?",
        "target_answer": "Xylanthia-9",
        "keywords": ["xylanthia"],
        "section": "late",
        "difficulty": "hard",
        "type": "needle",
    },
    {
        "id": 5,
        "question": "What specific part of the Windsor Merchant was lost during the squall off the coast of Yarmouth?",
        "target_answer": "Its secondary rudder",
        "keywords": ["secondary", "rudder"],
        "section": "late",
        "difficulty": "hard",
        "type": "needle",
    },

    # ── ADJACENT-TO-NEEDLE QUESTIONS (10) ────────────────────────────────────
    # Details found in the SAME chunk as a needle.
    # If the model retrieved the needle chunk it should also answer these —
    # failures here reveal partial context capture or LLM hallucination.
    {
        "id": 6,
        "question": "On which specific arm of the body must the anti-static wristband be worn when operating TS-500 machinery?",
        "target_answer": "Left arm",
        "keywords": ["left"],
        "section": "early",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 7,
        "question": "In the compliance manual, which sector of the warehouse uses the TS-500 machinery?",
        "target_answer": "Sector 4",
        "keywords": ["4"],
        "section": "early",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 8,
        "question": "Exactly how many minutes after receiving the placebo did subject 402-B report the sensory anomaly?",
        "target_answer": "Forty-five minutes",
        "keywords": ["forty-five", "45"],
        "section": "early",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 9,
        "question": "What symptom did the majority of the control group exhibit after the experiment, before subject 402-B reported their anomaly?",
        "target_answer": "Mild lethargy",
        "keywords": ["lethargy"],
        "section": "early",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 10,
        "question": "How many consecutive days of force majeure must pass before the subcontractor may claim a termination fee?",
        "target_answer": "Thirty consecutive days",
        "keywords": ["thirty", "30"],
        "section": "middle",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 11,
        "question": "Which arm of the galaxy was the probe scanning when it detected the temperature anomaly near Xylanthia-9?",
        "target_answer": "The Perseus arm",
        "keywords": ["perseus"],
        "section": "late",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 12,
        "question": "By how many Kelvin did the temperature drop near the asteroid Xylanthia-9?",
        "target_answer": "14 Kelvin",
        "keywords": ["14"],
        "section": "late",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 13,
        "question": "Which year's winter rations were delayed by the Windsor Merchant incident?",
        "target_answer": "1842",
        "keywords": ["1842"],
        "section": "late",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 14,
        "question": "What was the name of the primary transport vessel involved in the grain-delay incident?",
        "target_answer": "The Windsor Merchant",
        "keywords": ["windsor"],
        "section": "late",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },
    {
        "id": 15,
        "question": "How many sons did Jacob take with him when he crossed the ford Jabbok?",
        "target_answer": "Eleven sons",
        "keywords": ["eleven", "11"],
        "section": "late",
        "difficulty": "medium",
        "type": "needle_adjacent",
    },

    # ── BIBLICAL FACT QUESTIONS (15) ─────────────────────────────────────────
    # Facts from the Genesis text itself.
    # The LLM is likely pre-trained on the Bible, so it may answer these
    # CORRECTLY from memorised knowledge even if RAG fails to retrieve the chunk.
    # Low scores here therefore implicate retrieval AND LLM recall together.
    # High scores do NOT confirm RAG worked — see retrieved_chunks column.
    {
        "id": 16,
        "question": "What name did God give to the light, and what name did he give to the darkness?",
        "target_answer": "Light = Day, Darkness = Night",
        "keywords": ["day", "night"],
        "section": "early",
        "difficulty": "easy",
        "type": "fact",
    },
    {
        "id": 17,
        "question": "What was the name of the first of the four rivers that flowed out of the garden of Eden?",
        "target_answer": "Pison",
        "keywords": ["pison"],
        "section": "early",
        "difficulty": "medium",
        "type": "entity",
    },
    {
        "id": 18,
        "question": "What specific physical part of Adam did God use to create Eve?",
        "target_answer": "One of his ribs",
        "keywords": ["rib"],
        "section": "early",
        "difficulty": "easy",
        "type": "fact",
    },
    {
        "id": 19,
        "question": "What did Adam and Eve sew together to cover themselves after eating the forbidden fruit?",
        "target_answer": "Fig leaves",
        "keywords": ["fig"],
        "section": "early",
        "difficulty": "medium",
        "type": "fact",
    },
    {
        "id": 20,
        "question": "How many total years did Enoch live, according to the genealogy in Genesis?",
        "target_answer": "365 years (three hundred sixty and five years)",
        "keywords": ["365", "sixty and five"],
        "section": "early",
        "difficulty": "hard",
        "type": "fact",
    },
    {
        "id": 21,
        "question": "How many total years did Methuselah live?",
        "target_answer": "969 years (nine hundred sixty and nine)",
        "keywords": ["969", "sixty and nine"],
        "section": "early",
        "difficulty": "medium",
        "type": "fact",
    },
    {
        "id": 22,
        "question": "According to the text, how did Enoch's life end — did he die?",
        "target_answer": "He did not die; God took him (he was not, for God took him)",
        "keywords": ["took him"],
        "section": "early",
        "difficulty": "medium",
        "type": "event",
    },
    {
        "id": 23,
        "question": "What did God promise Abram when he appeared to him at the plain of Moreh in Sichem?",
        "target_answer": "God promised to give that land to Abram's seed (descendants)",
        "keywords": ["seed", "land"],
        "section": "early",
        "difficulty": "medium",
        "type": "event",
    },
    {
        "id": 24,
        "question": "What structure did Abram build after God appeared to him at Sichem?",
        "target_answer": "An altar unto the LORD",
        "keywords": ["altar"],
        "section": "early",
        "difficulty": "medium",
        "type": "fact",
    },
    {
        "id": 25,
        "question": "What was the name of the small city that Lot escaped to before God destroyed Sodom?",
        "target_answer": "Zoar",
        "keywords": ["zoar"],
        "section": "middle",
        "difficulty": "hard",
        "type": "entity",
    },
    {
        "id": 26,
        "question": "What did God rain down from heaven upon Sodom and Gomorrah?",
        "target_answer": "Brimstone and fire",
        "keywords": ["brimstone", "fire"],
        "section": "middle",
        "difficulty": "easy",
        "type": "fact",
    },
    {
        "id": 27,
        "question": "How old was Abraham when he died, according to the text?",
        "target_answer": "175 years (an hundred threescore and fifteen years)",
        "keywords": ["175", "threescore and fifteen"],
        "section": "late",
        "difficulty": "hard",
        "type": "fact",
    },
    {
        "id": 28,
        "question": "In which specific place was Abraham buried?",
        "target_answer": "The cave of Machpelah",
        "keywords": ["machpelah"],
        "section": "late",
        "difficulty": "medium",
        "type": "location",
    },
    {
        "id": 29,
        "question": "How did Jacob describe his brother Esau's physical appearance in contrast to his own?",
        "target_answer": "Esau is a hairy man; Jacob described himself as a smooth man",
        "keywords": ["hairy"],
        "section": "late",
        "difficulty": "easy",
        "type": "fact",
    },
    {
        "id": 30,
        "question": "What did Rebekah instruct Jacob to fetch from the flock so she could prepare food for Isaac?",
        "target_answer": "Two good kids of the goats",
        "keywords": ["goat"],
        "section": "late",
        "difficulty": "medium",
        "type": "fact",
    },
]

# ── helpers ────────────────────────────────────────────────────────────────────

def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    sims = (matrix / norms) @ q
    return np.argsort(sims)[::-1][:k].tolist()


def build_prompt(question: str, contexts: list[dict]) -> str:
    max_ctx_chars = (N_CTX - 600 - MAX_NEW_TOKENS) * 4
    budget = max_ctx_chars // max(len(contexts), 1)
    ctx_block = "\n\n---\n\n".join(
        f"[Chunk {c['chunk_id']}]\n{c['text'][:budget]}" for c in contexts
    )
    return (
        "You are a helpful assistant. Use ONLY the context below to answer the question. "
        "If the answer is not in the context, say exactly: I don't know based on the document.\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (be brief and specific):"
    )


def strip_think(text: str) -> str:
    """Remove DeepSeek <think>…</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def auto_score(answer: str, keywords: list[str]) -> int:
    """Return 1 if ALL keyword alternatives match, else 0.

    Each entry in keywords may be a pipe-separated string of alternatives,
    e.g. 'forty-five|45' means either form is acceptable.
    """
    a = answer.lower()
    for kw in keywords:
        alts = [a_.strip() for a_ in kw.split("|")]
        if not any(alt in a for alt in alts):
            return 0
    return 1


def short_model_name(filename: str) -> str:
    name = filename.replace(".gguf", "")
    mapping = {
        "Meta-Llama-3.1-8B-Instruct-128k-Q4_0": "Llama 3.1 8B 128k",
        "DeepSeek-R1-Distill-Qwen-14B-Q4_0": "DeepSeek-R1-14B",
        "mistral-7b-instruct-v0.1.Q4_0": "Mistral 7B Instruct",
    }
    return mapping.get(name, name)


# ── load shared resources ──────────────────────────────────────────────────────

print("Loading embeddings …")
_raw = json.loads(EMBEDDINGS_FILE.read_text())
_matrix = np.array([c["embedding"] for c in _raw], dtype=np.float32)
_chunks = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in _raw]
print(f"  {len(_chunks)} chunks loaded.")

print(f"Loading embed model '{EMBED_MODEL_NAME}' …")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("  Embed model ready.")

# ── main test loop ─────────────────────────────────────────────────────────────

all_rows: list[dict] = []

for model_file in MODELS:
    label = short_model_name(model_file)
    print(f"\n{'='*64}")
    print(f"  MODEL: {label}")
    print(f"{'='*64}")

    llm = GPT4All(
        model_name=model_file,
        model_path=str(MODEL_DIR),
        allow_download=False,
        verbose=False,
        n_ctx=N_CTX,
    )

    for q in QUESTIONS:
        qid = q["id"]
        print(f"  Q{qid:02d} [{q['type']:15s}] {q['question'][:55]}…", end="", flush=True)

        query_vec = _embed_model.encode(q["question"], convert_to_numpy=True)
        top_idx = cosine_top_k(query_vec, _matrix, TOP_K)
        top_chunks = [_chunks[i] for i in top_idx]

        prompt = build_prompt(q["question"], top_chunks)
        t0 = time.time()
        raw = llm.generate(prompt, max_tokens=MAX_NEW_TOKENS, temp=TEMPERATURE)
        elapsed = time.time() - t0

        answer = strip_think(raw).strip()
        score = auto_score(answer, q["keywords"])
        retrieved = [_chunks[i]["chunk_id"] for i in top_idx]

        print(f"  score={score}  ({elapsed:.1f}s)")
        print(f"         ↳ {answer[:100]}")

        all_rows.append({
            "id": qid,
            "question": q["question"],
            "target_answer": q["target_answer"],
            "section": q["section"],
            "difficulty": q["difficulty"],
            "type": q["type"],
            "model": label,
            "answer": answer,
            "score": score,
            "retrieved_chunk_ids": str(retrieved),
            "latency_s": round(elapsed, 1),
        })

    del llm
    print(f"\n  Done with {label}. Model unloaded.")

# ── write CSV ──────────────────────────────────────────────────────────────────

csv_path = BASE_DIR / "Metrics.csv"
fieldnames = [
    "id", "question", "target_answer", "section", "difficulty", "type",
    "model", "answer", "score", "retrieved_chunk_ids", "latency_s",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_rows)

print(f"\nResults written to {csv_path}")

# ── summary table ──────────────────────────────────────────────────────────────

model_names = [short_model_name(m) for m in MODELS]


def pct(rows, pred):
    subset = [r for r in rows if pred(r)]
    if not subset:
        return "—"
    return f"{100 * sum(r['score'] for r in subset) / len(subset):.1f}%"


print("\n" + "="*64)
print("SUMMARY")
print("="*64)
header = f"{'Metric':<40}" + "".join(f"{m:<22}" for m in model_names)
print(header)
print("-" * (40 + 22 * len(model_names)))

metrics = [
    ("Overall Accuracy (all 30)",        lambda r: True),
    ("Needle Score (Q1-Q5)",             lambda r: r["type"] == "needle"),
    ("Adjacent-to-Needle (Q6-Q15)",      lambda r: r["type"] == "needle_adjacent"),
    ("Biblical Facts (Q16-Q30)",         lambda r: r["type"] in ("fact", "entity", "location", "event")),
    ("Section Recall — early",           lambda r: r["section"] == "early"),
    ("Section Recall — middle",          lambda r: r["section"] == "middle"),
    ("Section Recall — late",            lambda r: r["section"] == "late"),
    ("Difficulty — easy",                lambda r: r["difficulty"] == "easy"),
    ("Difficulty — medium",              lambda r: r["difficulty"] == "medium"),
    ("Difficulty — hard",                lambda r: r["difficulty"] == "hard"),
]

for name, pred in metrics:
    row_str = f"{name:<40}"
    for m in model_names:
        model_rows = [r for r in all_rows if r["model"] == m]
        row_str += f"{pct(model_rows, pred):<22}"
    print(row_str)

print("="*64)
print("\nDone. Review Metrics.csv for per-question raw answers.")
