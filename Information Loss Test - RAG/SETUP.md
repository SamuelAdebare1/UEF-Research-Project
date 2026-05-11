# Setup Guide — Reproducing the RAG Experiment

## Prerequisites

| Tool            | Minimum version   | Check (Mac/Linux)   | Check (Windows)    |
| --------------- | ----------------- | ------------------- | ------------------ |
| Python          | 3.10+             | `python3 --version` | `python --version` |
| Node.js         | 18+               | `node --version`    | `node --version`   |
| npm             | 9+                | `npm --version`     | `npm --version`    |
| ~6 GB free disk | for model weights | —                   | —                  |

Download links if needed: [python.org](https://www.python.org/downloads/) · [nodejs.org](https://nodejs.org) · [pip](https://pip.pypa.io/en/stable/installation/)

> **Note:** pip comes bundled with Python 3.4+. If you installed Python from python.org, you already have pip. Also, npm comes bundled with Node.js. If you installed Node.js from nodejs.org, you already have npm.

---

## Step 1 — Download and unzip

1. Go to the GitHub repository page.
2. Click **Code → Download ZIP**.
3. Unzip the file — you will get a folder named `Information-Loss-Test-RAG`.
4. Open a terminal and navigate into the `Information Loss Test - RAG` folder:

**Mac / Linux**

```bash
cd path/to/Information-Loss-Test-RAG
```

**Windows (Command Prompt)**

```cmd
cd path\to\Information-Loss-Test-RAG
```

**Windows (PowerShell)**

```powershell
cd path\to\Information-Loss-Test-RAG
```

> PowerShell also accepts forward slashes: `cd path/to/Information-Loss-Test-RAG`

---

## Step 2 — Create a Python virtual environment

**Mac / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script with an "execution policy" error, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> then retry `venv\Scripts\Activate.ps1`.

Your terminal prompt should show `(venv)` to confirm the environment is active.

---

## Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package                 | Purpose                               |
| ----------------------- | ------------------------------------- |
| `pymupdf`               | Extracts text from the PDF            |
| `tiktoken`              | Tokeniser used for chunk sizing       |
| `sentence-transformers` | Embedding model (`all-mpnet-base-v2`) |
| `fastapi` + `uvicorn`   | RAG API server                        |
| `gpt4all`               | Runs the local LLM                    |
| `scikit-learn`          | Cosine similarity search              |

Installation takes 2–5 minutes. PyTorch and transformers are pulled in automatically as dependencies.

---

## Step 4 — Run the RAG pipeline

### 4a — Chunk the PDF

```bash
python chunker.py
```

Reads `50-pages.pdf` and writes `chunks.json` (113 chunks, ~350 tokens each). Completes in seconds.

### 4b — Embed the chunks

```bash
python embedder.py
```

Downloads `all-mpnet-base-v2` (~420 MB) on the first run — automatic, no account needed. Writes `embeddings.json`. Takes 1–3 minutes.

### 4c — Start the API server (Terminal 1)

```bash
uvicorn api:app --port 8000
```

On the **first query**, GPT4All downloads the LLM weights:

| Model                                | Size    |
| ------------------------------------ | ------- |
| Meta-Llama-3.1-8B-Instruct-128k-Q4_0 | ~4.7 GB |

This download happens once and is cached locally. After that, startup is instant.

The server runs at `http://localhost:8000`.

> **Important:** The API server must stay running in this terminal the entire time you use the app. Do not close it.

---

## Step 5 — Start the chat UI (Terminal 2)

The backend (Terminal 1) and the frontend (Terminal 2) must run **at the same time** in two separate terminal windows.

Open a **new, second terminal window** — do not use the same terminal as the API server.

**Mac / Linux**

```bash
cd path/to/Information-Loss-Test-RAG/rag-ui
npm install
npm run dev
```

**Windows (Command Prompt)**

```cmd
cd path\to\Information-Loss-Test-RAG\rag-ui
npm install
npm run dev
```

**Windows (PowerShell)**

```powershell
cd path\to\Information-Loss-Test-RAG\rag-ui
npm install
npm run dev
```

Open `http://localhost:5173` in a browser.

---

## Step 6 — Run the automated test suite (optional)

With the API server running (Step 4c), open another terminal:

**Mac / Linux**

```bash
cd "Information Loss Test - RAG"
source venv/bin/activate
python test_runner.py
```

**Windows (Command Prompt)**

```cmd
cd "Information Loss Test - RAG"
..\venv\Scripts\activate.bat
python test_runner.py
```

**Windows (PowerShell)**

```powershell
cd "Information Loss Test - RAG"
..\venv\Scripts\Activate.ps1
python test_runner.py
```

Results are written to `Metrics.csv`.

---

## Quick-start cheatsheet (after first-time setup)

You need **two terminal windows open at the same time**.

**Mac / Linux**

Terminal 1 (backend):

```bash
cd path/to/Information-Loss-Test-RAG
uvicorn api:app --port 8000
```

Terminal 2 (frontend):

```bash
cd path/to/Information-Loss-Test-RAG/rag-ui
npm run dev
```

**Windows (Command Prompt)**

Terminal 1 (backend):

```cmd
cd path\to\Information-Loss-Test-RAG
uvicorn api:app --port 8000
```

Terminal 2 (frontend):

```cmd
cd path\to\Information-Loss-Test-RAG\rag-ui
npm run dev
```

**Windows (PowerShell)**

Terminal 1 (backend):

```powershell
cd path\to\Information-Loss-Test-RAG
uvicorn api:app --port 8000
```

Terminal 2 (frontend):

```powershell
cd path\to\Information-Loss-Test-RAG\rag-ui
npm run dev
```

Then open `http://localhost:5173` in a browser.

---

## Troubleshooting

| Problem                                                   | Fix                                                                                                                   |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `source: no such file or directory: venv/bin/activate` | The venv hasn't been created yet. Run Step 2 from inside this folder.                                                   |
| `venv\Scripts\activate.bat is not recognized`             | Make sure you are inside the `Information Loss Test - RAG` folder before running this command.                   |
| PowerShell says "running scripts is disabled"             | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry `.\venv\Scripts\Activate.ps1`. |
| `python3` not found on Windows                            | Use `python` instead of `python3` — Windows installers register the command as `python`.                              |
| `ModuleNotFoundError`                                     | Confirm `(venv)` is shown in your prompt before running any Python script.                                            |
| Chat UI shows "failed to fetch" or no response            | The API server (Terminal 1) is not running. Go back to Step 4c and start it before using the UI.                      |
| Port 8000 already in use                                  | Use `uvicorn api:app --port 8001` and reload the UI page.                                                             |
| GPT4All model download hangs                              | The model is ~4.7 GB — check your internet connection and disk space.                                                 |
| `chunks.json` or `embeddings.json` not found              | Run `chunker.py` before `embedder.py`, and both before starting the API server.                                       |

---

## No internet or API keys required

After the one-time model downloads, the entire pipeline runs fully offline. No OpenAI, Anthropic, or any cloud service account is needed.
