# Using GPT4All for Retrieval-Augmented Generation (RAG) on Local Files

## 1. Download GPT4All

Download GPT4All from [https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all)

## 2. Add a Model

1. Open the GPT4All app and navigate to the **Models** menu.
2. Click **"+Add Model"**.
3. Download a model of your choice. Suggestions:
   - DeepSeek-R1-Distill-Qwen-14B *(requires 16GB of RAM)*
   - Llama 3.1 8B Instruct 128k
   - Mistral Instruct

## 3. Add a Local Document Collection

1. Navigate to the **"LocalDocs"** tab.
2. Click **"+Add Collection"** to select a folder (only folders can be selected) and give it a name.
3. Wait for the embedding step to complete — it will show **"Ready"** when done.

> **Note:** The folder should contain text files that the LLM can read to answer user queries.

## 4. Use the Chat

1. Navigate to the **"Chats"** tab and choose a model.
2. Click on **LocalDocs** in the top-right corner and tick the checkbox of the uploaded folder needed for retrieval.
3. Ask your questions.
