"""
embedder.py — generates dense vector embeddings for each chunk produced by chunker.py.

Model: all-mpnet-base-v2  (sentence-transformers)
  Source : https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
  Reason : The SBERT documentation states that all-mpnet-base-v2 "provides the best
           quality" among the general-purpose pretrained models. It was trained on
           more than 1 billion training pairs (the full "all-*" dataset), making it
           the recommended default when retrieval quality matters more than speed.
           (all-MiniLM-L6-v2 is 5× faster but lower quality — a reasonable swap if
           latency becomes a constraint.)

Input  : chunks.json   — produced by chunker.py
Output : embeddings.json — same list of chunk dicts, each with an "embedding" field

Dependencies (install once):
    pip install sentence-transformers

Usage:
    python embedder.py                        # uses default paths below
    python embedder.py path/to/chunks.json    # explicit input path
"""

import json
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ── configuration ──────────────────────────────────────────────────────────────
MODEL_NAME    = "all-mpnet-base-v2"
BATCH_SIZE    = 32          # lower if you run out of RAM
DEFAULT_INPUT = Path(__file__).parent / "chunks.json"
OUTPUT_FILE   = Path(__file__).parent / "embeddings.json"
# ───────────────────────────────────────────────────────────────────────────────


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks with '{MODEL_NAME}' (batch={BATCH_SIZE})…")
    t0 = time.perf_counter()

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed / len(texts):.2f}s per chunk)")

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec.tolist()   # JSON-serialisable float list

    return chunks


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT

    if not input_path.exists():
        sys.exit(
            f"[ERROR] Chunks file not found: {input_path}\n"
            "        Run chunker.py first to generate it."
        )

    chunks = json.loads(input_path.read_text())
    print(f"Loaded   : {len(chunks)} chunks from {input_path}")

    print(f"Model    : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim   = model.get_embedding_dimension()
    print(f"Dim      : {dim}")

    chunks = embed_chunks(chunks, model)

    OUTPUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"Saved to : {OUTPUT_FILE}")

    # sanity check
    sample = chunks[0]
    print(f"\n── chunk 0 sanity check ───────────────────────────────────────────")
    print(f"  chunk_id    : {sample['chunk_id']}")
    print(f"  token_count : {sample['token_count']}")
    print(f"  embedding   : [{sample['embedding'][0]:.6f}, {sample['embedding'][1]:.6f}, … ] (len={len(sample['embedding'])})")


if __name__ == "__main__":
    main()
