# RAGChat UI

A local chat interface for the Information Loss Test RAG pipeline. Sends questions to
the FastAPI backend, streams tokens as they are generated, and displays the retrieved
source chunks alongside each answer.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 + Vite |
| Styling | Plain CSS (custom design system, light/dark themes) |
| Backend | FastAPI (`../Information Loss Test - RAG/api.py`) |
| LLM | Meta-Llama-3.1-8B-Instruct-128k-Q4_0 via GPT4All (local) |
| Retrieval | `all-mpnet-base-v2` embeddings + cosine similarity |

---

## Features

- **Token streaming** — responses appear word-by-word via the `/query/stream` SSE endpoint; no waiting for full generation
- **Stop button** — cancels an in-flight request mid-stream
- **Source cards** — each answer shows the retrieved document chunks with similarity scores; click to expand
- **Settings panel** — adjust top-k, max tokens, and temperature without restarting the server
- **Dark / light theme** — toggle in the settings panel

---

## Prerequisites

1. The backend server must be running on `http://localhost:8000`:
   ```bash
   cd "../Information Loss Test - RAG"
   source ../venv/bin/activate
   uvicorn api:app --port 8000
   ```

2. Node 18+ installed.

---

## Setup

```bash
cd rag-ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Build

```bash
npm run build       # output goes to dist/
npm run preview     # serve the production build locally
```

---

## Configuration

The backend URL is set at the top of [src/App.jsx](src/App.jsx):

```js
const API = "http://localhost:8000";
```

Default query settings (editable at runtime via the settings panel):

| Setting | Default | Description |
|---------|---------|-------------|
| Top K | 5 | Number of chunks retrieved per query |
| Max tokens | 512 | Maximum tokens the LLM generates |
| Temperature | 0.70 | Generation randomness (0 = deterministic, 1.5 = creative) |

---

## Project structure

```
rag-ui/
├── src/
│   ├── App.jsx      # All components and application logic
│   ├── App.css      # Styles (CSS variables, dark/light themes)
│   ├── main.jsx     # React entry point
│   └── index.css    # Base reset
├── index.html
├── vite.config.js
└── package.json
```
