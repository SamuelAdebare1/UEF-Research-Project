"""
api.py — FastAPI backend for the RAG pipeline.

Loads embeddings.json once at startup, then for each query:
  1. Embeds the question with all-mpnet-base-v2
  2. Cosine-similarity search → top-k chunks
  3. Feeds context + question to a local GPT4All model
  4. Returns answer + source chunks

Run:
    uvicorn api:app --reload --port 8000
"""

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gpt4all import GPT4All
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── config ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
EMBEDDINGS_FILE = BASE_DIR / "embeddings.json"
EMBED_MODEL     = "all-mpnet-base-v2"
GPT4ALL_MODEL   = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
MODEL_DIR       = Path.home() / "Library/Application Support/nomic.ai/GPT4All"
TOP_K           = 5
MAX_NEW_TOKENS  = 512
# 4096 balances context depth vs. KV-cache memory on Apple Silicon.
# 8192 triggers a backend allocation crash (GGML_ASSERT) on this hardware.
N_CTX           = 4096
# ~4 chars per token; reserve 600 tokens for system prompt + question + answer
MAX_CONTEXT_CHARS = (N_CTX - 600 - MAX_NEW_TOKENS) * 4
# ───────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── startup: load models & embeddings ──────────────────────────────────────────
print("Loading embeddings …")
_raw = json.loads(EMBEDDINGS_FILE.read_text())
_embeddings_matrix = np.array([c["embedding"] for c in _raw], dtype=np.float32)
_chunks = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in _raw]
print(f"  {len(_chunks)} chunks loaded.")

print(f"Loading embed model '{EMBED_MODEL}' …")
_embed_model = SentenceTransformer(EMBED_MODEL)

print(f"Loading LLM '{GPT4ALL_MODEL}' …")
_llm = GPT4All(
    model_name=GPT4ALL_MODEL,
    model_path=str(MODEL_DIR),
    allow_download=False,
    verbose=False,
    n_ctx=N_CTX,
)
print("Ready.")


# ── helpers ────────────────────────────────────────────────────────────────────
def _cosine_top_k(query_vec: np.ndarray, k: int) -> list[dict]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(_embeddings_matrix, axis=1, keepdims=True) + 1e-10
    sims = (_embeddings_matrix / norms) @ q
    top_idx = np.argsort(sims)[::-1][:k]
    return [
        {**_chunks[i], "score": float(sims[i])}
        for i in top_idx
    ]


def _build_prompt(question: str, contexts: list[dict], max_context_chars: int = MAX_CONTEXT_CHARS) -> str:
    # Distribute the char budget evenly across chunks; truncate each if needed.
    budget_per_chunk = max_context_chars // max(len(contexts), 1)
    ctx_block = "\n\n---\n\n".join(
        f"[Chunk {c['chunk_id']}]\n{c['text'][:budget_per_chunk]}" for c in contexts
    )
    return (
        "You are a helpful assistant. Use ONLY the context below to answer the question. "
        "If the answer is not in the context, say 'I don't know based on the document.'\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


# ── request / response models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.7


class SourceChunk(BaseModel):
    chunk_id: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_vec = _embed_model.encode(req.question, convert_to_numpy=True)
    top_chunks = _cosine_top_k(query_vec, req.top_k)

    max_ctx_chars = (N_CTX - 600 - req.max_tokens) * 4
    prompt = _build_prompt(req.question, top_chunks, max_context_chars=max_ctx_chars)

    # Don't use chat_session() — it accumulates context across calls and overflows.
    answer = _llm.generate(prompt, max_tokens=req.max_tokens, temp=req.temperature)

    return QueryResponse(
        answer=answer.strip(),
        sources=[SourceChunk(**c) for c in top_chunks],
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_vec = _embed_model.encode(req.question, convert_to_numpy=True)
    top_chunks = _cosine_top_k(query_vec, req.top_k)

    max_ctx_chars = (N_CTX - 600 - req.max_tokens) * 4
    prompt = _build_prompt(req.question, top_chunks, max_context_chars=max_ctx_chars)

    def token_stream():
        sources = [{"chunk_id": c["chunk_id"], "text": c["text"], "score": c["score"]} for c in top_chunks]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        for token in _llm.generate(prompt, max_tokens=req.max_tokens, temp=req.temperature, streaming=True):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
