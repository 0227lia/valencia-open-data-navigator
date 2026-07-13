# Methodology

## Corpus preparation

The catalog text for one dataset joins title, slug terms, description, tags, groups and declared formats. Spanish and Valencian accents are folded for matching, text is lowercased and a small function-word list is removed for BM25 tokenization.

## Retrieval candidates

1. **BM25** applies `k1=1.5` and `b=0.75` over the normalized catalog text.
2. **LSA** combines word 1-2 grams and character 3-5 grams in TF-IDF, then uses truncated SVD and cosine similarity.
3. **Hybrid** fuses BM25 and LSA rank positions with reciprocal-rank fusion, `k=60`.
4. **Hybrid MMR** selects from the top 40 fused candidates with a 0.24 diversity penalty based on latent-vector cosine similarity.

The model is deliberately small and deterministic. It is not an embedding model or a generative system.

## Evaluation

`data/evaluation_queries.json` contains manually curated information needs and relevant official dataset slugs. For each strategy the pipeline reports mean Recall@5, MRR@10 and nDCG@10. Labels are validated against the snapshot before scoring.

## Error interpretation

The benchmark is useful for checking regressions after preprocessing or ranking changes. It is not a representative user study, does not test every language variation and cannot establish search quality for all portal users.
