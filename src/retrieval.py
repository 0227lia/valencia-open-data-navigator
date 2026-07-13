"""Auditable BM25, latent semantic and hybrid retrieval for CKAN metadata."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .catalog import CatalogRecord, normalize_text, tokenize

SUPPORTED_STRATEGIES = ("bm25", "lsa", "hybrid", "hybrid_mmr")


@dataclass(frozen=True)
class SearchHit:
    rank: int
    record: CatalogRecord
    score: float
    bm25_score: float
    lsa_score: float
    rrf_score: float
    matched_terms: tuple[str, ...]


class BM25Index:
    """A small deterministic BM25 implementation that exposes its assumptions."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25Index requires at least one document.")
        self.k1 = k1
        self.b = b
        self.document_tokens = [tokenize(document) for document in documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.document_tokens]
        self.document_lengths = np.array([len(tokens) for tokens in self.document_tokens], dtype=float)
        self.average_length = float(self.document_lengths.mean()) or 1.0
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.document_tokens:
            self.document_frequency.update(set(tokens))
        self.document_count = len(documents)

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.document_count, dtype=float)
        query_terms = tokenize(query)
        if not query_terms:
            return scores

        for term in query_terms:
            frequency = self.document_frequency.get(term, 0)
            if frequency == 0:
                continue
            inverse_frequency = math.log(1 + (self.document_count - frequency + 0.5) / (frequency + 0.5))
            for index, term_frequency in enumerate(self.term_frequencies):
                occurrences = term_frequency.get(term, 0)
                if occurrences == 0:
                    continue
                denominator = occurrences + self.k1 * (
                    1 - self.b + self.b * self.document_lengths[index] / self.average_length
                )
                scores[index] += inverse_frequency * occurrences * (self.k1 + 1) / denominator
        return scores


class HybridRetriever:
    """Combine lexical and latent-semantic retrieval with transparent rank fusion."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.records: list[CatalogRecord] = []
        self.documents: list[str] = []
        self.normalized_documents: list[str] = []
        self.bm25: BM25Index | None = None
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.document_vectors: np.ndarray | None = None

    def fit(self, records: list[CatalogRecord]) -> HybridRetriever:
        if not records:
            raise ValueError("HybridRetriever requires at least one catalog record.")

        self.records = records
        self.documents = [record.document for record in records]
        self.normalized_documents = [normalize_text(document) for document in self.documents]
        self.bm25 = BM25Index(self.documents)

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        word_matrix = self.word_vectorizer.fit_transform(self.normalized_documents)
        char_matrix = self.char_vectorizer.fit_transform(self.normalized_documents)
        feature_matrix = hstack([word_matrix, char_matrix], format="csr")

        components = min(48, feature_matrix.shape[0] - 1, feature_matrix.shape[1] - 1)
        if components >= 2:
            self.svd = TruncatedSVD(n_components=components, random_state=self.random_state)
            self.document_vectors = normalize(self.svd.fit_transform(feature_matrix))
        else:
            self.svd = None
            self.document_vectors = normalize(feature_matrix).toarray()
        return self

    def _assert_fitted(self) -> None:
        if (
            self.bm25 is None
            or self.word_vectorizer is None
            or self.char_vectorizer is None
            or self.document_vectors is None
        ):
            raise RuntimeError("Fit the retriever before calling search.")

    def _lsa_scores(self, query: str) -> np.ndarray:
        self._assert_fitted()
        assert self.word_vectorizer is not None
        assert self.char_vectorizer is not None
        assert self.document_vectors is not None
        normalized_query = normalize_text(query)
        if not normalized_query:
            return np.zeros(len(self.records), dtype=float)
        query_features = hstack(
            [
                self.word_vectorizer.transform([normalized_query]),
                self.char_vectorizer.transform([normalized_query]),
            ],
            format="csr",
        )
        if self.svd is not None:
            query_vector = normalize(self.svd.transform(query_features))
        else:
            query_vector = normalize(query_features).toarray()
        return np.asarray(query_vector @ self.document_vectors.T).ravel()

    @staticmethod
    def _rrf_scores(*score_arrays: np.ndarray, constant: int = 60) -> np.ndarray:
        if not score_arrays:
            raise ValueError("RRF needs at least one score array.")
        fused = np.zeros_like(score_arrays[0], dtype=float)
        for scores in score_arrays:
            ranks = np.empty(len(scores), dtype=int)
            ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
            fused += 1 / (constant + ranks)
        return fused

    def _candidate_indices(self, tag: str | None) -> np.ndarray:
        if not tag:
            return np.arange(len(self.records))
        expected = normalize_text(tag)
        return np.array(
            [
                index
                for index, record in enumerate(self.records)
                if expected in {normalize_text(item) for item in record.tags}
            ],
            dtype=int,
        )

    def _mmr(self, base_scores: np.ndarray, candidates: np.ndarray, limit: int) -> list[int]:
        self._assert_fitted()
        assert self.document_vectors is not None
        if len(candidates) == 0:
            return []

        ordered = list(candidates[np.argsort(-base_scores[candidates], kind="stable")])
        pool = ordered[: min(40, len(ordered))]
        maximum = float(base_scores[pool].max())
        minimum = float(base_scores[pool].min())
        scale = maximum - minimum or 1.0
        selected: list[int] = []
        diversity_weight = 0.24

        while pool and len(selected) < limit:
            best_index = pool[0]
            best_value = -float("inf")
            for candidate in pool:
                relevance = (float(base_scores[candidate]) - minimum) / scale
                diversity = (
                    max(float(np.dot(self.document_vectors[candidate], self.document_vectors[chosen])) for chosen in selected)
                    if selected
                    else 0.0
                )
                value = (1 - diversity_weight) * relevance - diversity_weight * diversity
                if value > best_value:
                    best_index = candidate
                    best_value = value
            selected.append(best_index)
            pool.remove(best_index)
        return selected

    def search(
        self,
        query: str,
        strategy: str = "hybrid_mmr",
        limit: int = 5,
        tag: str | None = None,
    ) -> list[SearchHit]:
        self._assert_fitted()
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if not tokenize(query):
            return []

        assert self.bm25 is not None
        bm25_scores = self.bm25.score(query)
        lsa_scores = self._lsa_scores(query)
        rrf_scores = self._rrf_scores(bm25_scores, lsa_scores)
        scores_by_strategy = {
            "bm25": bm25_scores,
            "lsa": lsa_scores,
            "hybrid": rrf_scores,
            "hybrid_mmr": rrf_scores,
        }
        selected_scores = scores_by_strategy[strategy]
        candidates = self._candidate_indices(tag)
        if strategy == "hybrid_mmr":
            indices = self._mmr(selected_scores, candidates, limit)
        else:
            indices = list(candidates[np.argsort(-selected_scores[candidates], kind="stable")][:limit])

        query_terms = set(tokenize(query))
        hits: list[SearchHit] = []
        for rank, index in enumerate(indices, start=1):
            document_terms = set(tokenize(self.documents[index]))
            hits.append(
                SearchHit(
                    rank=rank,
                    record=self.records[index],
                    score=float(selected_scores[index]),
                    bm25_score=float(bm25_scores[index]),
                    lsa_score=float(lsa_scores[index]),
                    rrf_score=float(rrf_scores[index]),
                    matched_terms=tuple(sorted(query_terms.intersection(document_terms))),
                )
            )
        return hits

    def get_record(self, name: str) -> CatalogRecord | None:
        return next((record for record in self.records if record.name == name), None)
