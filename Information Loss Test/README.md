# UEF Research Project — Information Loss Test

## Data Source

The QuALITY dataset (`QuALITY.v1.0.1.htmlstripped.txt`) used in this project is sourced from:

https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev

---

## Test Document

This experiment uses story **52845** from the QuALITY dataset.

- **`52845.txt`** — the original story text as extracted from the dataset.
- **`52845-final.txt`** — an enriched version of the original. Three "needle in a haystack" facts were injected into the text at different points in the story. These facts are entirely foreign to the original narrative and serve as a controlled probe for verbatim retrieval accuracy. This is the document that was fed to each model during evaluation.

---

## Evaluation Methodology

Each model was tested by submitting **all 22 questions one per tab** (i.e. in a fresh context window for each question), so that no answer could benefit from context accumulated across previous questions.

**Exception — follow-up question triples:** Three questions are structurally linked and ask the model to name each of the three women pursuing Blake inside his mind. Because they reference one another, these were submitted together in a single session rather than in isolated tabs:

1. _"Name one of the three women pursuing Blake inside his mind."_
2. _"Name another of the three women pursuing Blake inside his mind."_
3. _"Name the third woman pursuing Blake inside his mind."_

---

## Models Evaluated

| Model                        | Context window |
| ---------------------------- | -------------- |
| DeepSeek-R1-Distill-Qwen-14B | 128k           |
| Llama 3.1 8B Instruct        | 128k           |
| Mistral Instruct             | 128k           |

---

## Metrics Summary

![Metrics Summary](Metrics%20Summary.png)

Full per-question scores are in [Metrics (52845-final).csv](<Metrics%20(52845-final).csv>).

| Metric                      | DeepSeek-R1-Distill-Qwen-14B | Llama 3.1 8B Instruct 128k | Mistral Instruct |
| --------------------------- | ---------------------------- | -------------------------- | ---------------- |
| **Overall Accuracy**        | 40.9%                        | 61.4%                      | 45.5%            |
| **Needle Retrieval Score**  | 100.0%                       | 100.0%                     | 66.7%            |
| **Section Recall — early**  | 37.5%                        | 50.0%                      | 50.0%            |
| **Section Recall — middle** | 33.3%                        | 61.1%                      | 33.3%            |
| **Section Recall — late**   | 60.0%                        | 80.0%                      | 60.0%            |

---

## Question Types

Questions are categorised along two axes:

- **Section** (`early` / `middle` / `late`) — where in the story the answer appears.
- **Type** — `entity`, `fact`, `location`, `event`, `concept`, `evidence`, `reasoning`, `needle`.

The three **needle** questions (Q18–Q20) test whether a model can retrieve the injected facts that do not appear in the original `52845.txt`, making them the sharpest signal for verbatim long-context recall.
