# Linguistic Analysis – TODO

## Pending Tasks

### Task 2 – Stylometric Analysis
Measure writing style features (sentence length, vocabulary richness, syntactic patterns) to detect style shifts where injections occur. The biblical KJV style is highly consistent, making anomalies stand out.

### Task 6 – Readability & Complexity Scoring
Flesch-Kincaid, sentence length distribution — the injected sentences likely have different complexity profiles than the biblical prose.

### Task 5 – Sentence Embedding & Clustering
Encodes all sentences with a sentence-transformer, clusters them with K-Means, then identifies sentences whose distance to their cluster centroid is an outlier (> mean + 2σ) — a strong signal for injected content.

### Task 7 – Cohesion & Coherence Analysis
Builds lexical chains (shared content words between consecutive sentence windows) and computes a continuity score per sentence. Sentences with very low continuity break the lexical flow and are flagged.
