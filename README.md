# UEF Research Project — Information Loss in RAG Systems

This repository contains two experiments measuring information loss when a large language model answers questions from a document under different retrieval conditions.

| Folder | Description |
|--------|-------------|
| `Information Loss Test/` | Full-context baseline — entire document in one context window |
| `Information Loss Test - RAG/` | RAG condition — top-k chunks retrieved by cosine similarity |
| `rag-ui/` | React chat interface for the RAG API |

The gap in accuracy between the two conditions is the measure of information loss introduced by chunking and retrieval.

For setup and reproduction instructions, see [`Information Loss Test - RAG/SETUP.md`](Information%20Loss%20Test%20-%20RAG/SETUP.md).
