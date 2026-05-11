"""
Named Entity Recognition (NER)
Uses a HuggingFace token-classification pipeline to extract persons,
locations, organisations, and misc entities from the document.
"""
import json
from collections import defaultdict
from transformers import pipeline
from analysis_utils import load_text, split_sentences

PDF_PATH = "50-pages.pdf"


def run():
    print("=== Named Entity Recognition ===\n")
    sentences = split_sentences(load_text(PDF_PATH))
    print(f"Total sentences: {len(sentences)}")

    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )

    entity_map: dict[str, list[dict]] = defaultdict(list)
    for i, sent in enumerate(sentences):
        for ent in ner(sent[:512]):
            word = ent["word"]
            if word.startswith("##") or len(word) < 2:
                continue
            entity_map[ent["entity_group"]].append({
                "text": word,
                "score": round(float(ent["score"]), 4),
                "sentence_index": i,
            })

    print("\nEntity summary:")
    for label, ents in sorted(entity_map.items()):
        unique = sorted({e["text"] for e in ents})
        print(f"  {label} ({len(ents)} mentions, {len(unique)} unique): {unique[:15]}")

    with open("results_ner.json", "w") as f:
        json.dump({k: v for k, v in entity_map.items()}, f, indent=2)
    print("\nResults saved to results_ner.json")


if __name__ == "__main__":
    run()
