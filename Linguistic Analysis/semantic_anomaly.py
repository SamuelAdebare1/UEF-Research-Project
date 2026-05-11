"""
Semantic Anomaly Detection
Embeds every sentence, computes each sentence's mean cosine similarity
to its neighbours, and flags low-similarity outliers as potential injections.
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from analysis_utils import load_text, split_sentences

PDF_PATH = "50-pages.pdf"
WINDOW = 5          # neighbours on each side
THRESHOLD = 0.30    # sentences below this mean similarity are flagged


def mean_neighbour_similarity(embeddings: np.ndarray, idx: int, window: int) -> float:
    start = max(0, idx - window)
    end = min(len(embeddings), idx + window + 1)
    neighbours = [embeddings[j] for j in range(start, end) if j != idx]
    if not neighbours:
        return 1.0
    sims = [1 - cosine(embeddings[idx], n) for n in neighbours]
    return float(np.mean(sims))


def run():
    print("=== Semantic Anomaly Detection ===\n")
    sentences = split_sentences(load_text(PDF_PATH))
    print(f"Total sentences: {len(sentences)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=True)

    scores = [mean_neighbour_similarity(embeddings, i, WINDOW) for i in range(len(sentences))]

    anomalies = [
        {"index": i, "similarity_score": round(scores[i], 4), "sentence": sentences[i]}
        for i in range(len(sentences))
        if scores[i] < THRESHOLD
    ]
    anomalies.sort(key=lambda x: x["similarity_score"])

    print(f"\nAnomalies detected (threshold < {THRESHOLD}): {len(anomalies)}\n")
    for a in anomalies:
        print(f"  [{a['index']}] score={a['similarity_score']:.4f}")
        print(f"      {a['sentence'][:120]}\n")

    with open("results_anomalies.json", "w") as f:
        json.dump({"threshold": THRESHOLD, "anomalies": anomalies}, f, indent=2)
    print("Results saved to results_anomalies.json")


if __name__ == "__main__":
    run()
