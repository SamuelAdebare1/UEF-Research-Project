# Research Projects — Overview

Three independent projects, each in its own self-contained folder.

| Folder | What it does |
|--------|-------------|
| [`Information Loss Test/`](Information%20Loss%20Test/) | Tests how well LLMs answer questions when given an entire long document in one context window (full-context baseline). |
| [`Information Loss Test - RAG/`](Information%20Loss%20Test%20-%20RAG/) | Tests the same questions using a RAG pipeline — only the top retrieved chunks are shown to the model. Includes an automated test runner and a React chat UI. |
| [`Linguistic Analysis/`](Linguistic%20Analysis/) | NLP analysis suite applied to the 50-page test document — named entity recognition, semantic anomaly detection, and vocabulary profiling. |

Each folder has its own `README.md` with setup instructions and can be run independently.
