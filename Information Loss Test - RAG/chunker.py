"""
chunker.py — splits a PDF into overlapping text chunks for RAG pipelines.

Chunk size : 500 tokens  (approximated as words)
Overlap    : 50 tokens

Dependencies (install once):
    pip install pymupdf tiktoken

Usage:
    python chunker.py                        # uses default PDF path below
    python chunker.py path/to/document.pdf   # explicit path
"""

import json
import sys
from pathlib import Path

import fitz          # pip install pymupdf
import tiktoken      # pip install tiktoken


# ── configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE   = 500   # tokens per chunk
OVERLAP_SIZE = 50    # tokens shared between consecutive chunks
ENCODING     = "cl100k_base"   # same tokeniser used by GPT-4 / text-embedding-3
DEFAULT_PDF  = Path(__file__).parent / "50-pages.pdf"
OUTPUT_FILE  = Path(__file__).parent / "chunks.json"
# ───────────────────────────────────────────────────────────────────────────────


def extract_text(pdf_path: Path) -> str:
    """Return all text from a PDF as a single string."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int, enc) -> list[dict]:
    """
    Tokenise `text`, then slide a window of `chunk_size` tokens
    with `overlap` tokens carried forward into the next chunk.

    Returns a list of dicts:
        {
            "chunk_id"    : int,
            "token_count" : int,
            "text"        : str,
        }
    """
    token_ids = enc.encode(text)
    chunks    = []
    step      = chunk_size - overlap
    start     = 0

    while start < len(token_ids):
        end        = min(start + chunk_size, len(token_ids))
        window_ids = token_ids[start:end]
        chunk_text = enc.decode(window_ids)

        chunks.append({
            "chunk_id"    : len(chunks),
            "token_count" : len(window_ids),
            "text"        : chunk_text,
        })

        if end == len(token_ids):
            break
        start += step

    return chunks


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF

    if not pdf_path.exists():
        sys.exit(f"[ERROR] PDF not found: {pdf_path}")

    print(f"PDF      : {pdf_path}")
    print(f"Chunk    : {CHUNK_SIZE} tokens  |  Overlap: {OVERLAP_SIZE} tokens")

    enc   = tiktoken.get_encoding(ENCODING)
    text  = extract_text(pdf_path)
    total = len(enc.encode(text))
    print(f"Total    : {total:,} tokens")

    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP_SIZE, enc)
    print(f"Chunks   : {len(chunks)}")

    OUTPUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"Saved to : {OUTPUT_FILE}")

    # preview first chunk
    print("\n── chunk 0 preview ────────────────────────────────────────────────")
    print(chunks[0]["text"][:300], "…")


if __name__ == "__main__":
    main()
