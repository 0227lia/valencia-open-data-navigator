"""Offline ranking evaluation against versioned, manual relevance labels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .retrieval import SUPPORTED_STRATEGIES, HybridRetriever


@dataclass(frozen=True)
class RelevanceQuery:
    identifier: str
    query: str
    domain: str
    relevant: tuple[str, ...]


def load_evaluation_queries(path: Path) -> list[RelevanceQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = [
        RelevanceQuery(
            identifier=item["id"],
            query=item["query"],
            domain=item["domain"],
            relevant=tuple(item["relevant"]),
        )
        for item in payload
    ]
    if not queries:
        raise ValueError("Evaluation query file is empty.")
    return queries


def validate_relevance_labels(queries: list[RelevanceQuery], available_names: set[str]) -> None:
    missing = sorted({name for query in queries for name in query.relevant if name not in available_names})
    if missing:
        raise ValueError(f"Evaluation labels reference missing dataset slugs: {missing}")


def _reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for index, name in enumerate(ranked, start=1):
        if name in relevant:
            return 1 / index
    return 0.0


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]).intersection(relevant)) / len(relevant)


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(index + 2) for index, name in enumerate(ranked[:k]) if name in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def evaluate_retriever(
    retriever: HybridRetriever,
    queries: list[RelevanceQuery],
    strategies: tuple[str, ...] = SUPPORTED_STRATEGIES,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        for query in queries:
            ranked = [hit.record.name for hit in retriever.search(query.query, strategy=strategy, limit=10)]
            relevant = set(query.relevant)
            rows.append(
                {
                    "strategy": strategy,
                    "query_id": query.identifier,
                    "query": query.query,
                    "domain": query.domain,
                    "reciprocal_rank": _reciprocal_rank(ranked, relevant),
                    "recall_at_5": _recall_at_k(ranked, relevant, 5),
                    "ndcg_at_10": _ndcg_at_k(ranked, relevant, 10),
                    "top_result": ranked[0] if ranked else None,
                    "ranked_results": "|".join(ranked),
                }
            )
    frame = pd.DataFrame(rows)
    metrics = {
        strategy: {
            "queries": int(group.shape[0]),
            "mrr_at_10": float(group["reciprocal_rank"].mean()),
            "recall_at_5": float(group["recall_at_5"].mean()),
            "ndcg_at_10": float(group["ndcg_at_10"].mean()),
        }
        for strategy, group in frame.groupby("strategy", sort=False)
    }
    return metrics, frame
