"""
Word Frequency & Vocabulary Analysis
Type-token ratio, most frequent content words, hapax legomena,
and a frequency distribution saved to JSON.
"""
import json
import re
from collections import Counter
from analysis_utils import load_text

PDF_PATH = "50-pages.pdf"

STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "that", "he", "his", "was",
    "it", "for", "they", "were", "with", "all", "as", "unto", "him",
    "upon", "said", "is", "be", "not", "from", "which", "had", "an",
    "by", "their", "but", "so", "them", "then", "out", "up", "god",
    "lord", "shall", "thou", "thy", "thee", "hath", "have", "this",
    "her", "she", "we", "at", "my", "i", "me", "are", "or", "no",
    "ye", "our", "there", "made", "after", "into", "when", "what",
    "also", "let",
}


def tokenise(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]


def run():
    print("=== Word Frequency & Vocabulary Analysis ===\n")
    text = load_text(PDF_PATH)
    tokens = tokenise(text)
    total = len(tokens)
    vocab = set(tokens)

    ttr = len(vocab) / total
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    freq = Counter(content)
    hapax = [w for w, c in freq.items() if c == 1]

    print(f"Total tokens     : {total:,}")
    print(f"Unique types     : {len(vocab):,}")
    print(f"Type-Token Ratio : {ttr:.4f}")
    print(f"Hapax legomena   : {len(hapax):,}  ({100*len(hapax)/len(vocab):.1f}% of vocab)")
    print(f"\nTop 30 content words:")
    for word, count in freq.most_common(30):
        print(f"  {word:<20} {count}")

    results = {
        "total_tokens": total,
        "unique_types": len(vocab),
        "type_token_ratio": round(ttr, 4),
        "hapax_legomena_count": len(hapax),
        "hapax_legomena": sorted(hapax),
        "top_100_content_words": dict(freq.most_common(100)),
    }
    with open("results_word_frequency.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_word_frequency.json")


if __name__ == "__main__":
    run()
