"""FastAPI surface for reproducible metadata retrieval."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .catalog import CatalogRecord, load_catalog
from .retrieval import SUPPORTED_STRATEGIES, HybridRetriever, SearchHit

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "raw" / "ckan_catalog_snapshot.json"
METRICS = ROOT / "reports" / "evaluation_metrics.json"

app = FastAPI(
    title="Valencia Open Data Navigator API",
    version="1.0.0",
    description="Search a versioned snapshot of public Valencia CKAN metadata. Not an official portal API.",
)


class DatasetResponse(BaseModel):
    name: str
    title: str
    notes: str
    tags: list[str]
    groups: list[str]
    resource_count: int
    formats: list[str]
    metadata_created: str | None
    metadata_modified: str | None
    license_title: str | None
    portal_url: str


class SearchHitResponse(DatasetResponse):
    rank: int
    score: float
    bm25_score: float
    lsa_score: float
    rrf_score: float
    matched_terms: list[str]


class SearchResponse(BaseModel):
    query: str
    strategy: str
    tag: str | None
    count: int
    hits: list[SearchHitResponse]


class HealthResponse(BaseModel):
    status: str
    records: int
    snapshot_path: str


def _dataset_response(record: CatalogRecord) -> DatasetResponse:
    return DatasetResponse(
        name=record.name,
        title=record.title,
        notes=record.notes,
        tags=list(record.tags),
        groups=list(record.groups),
        resource_count=record.resource_count,
        formats=list(record.formats),
        metadata_created=record.metadata_created,
        metadata_modified=record.metadata_modified,
        license_title=record.license_title,
        portal_url=record.portal_url,
    )


def _hit_response(hit: SearchHit) -> SearchHitResponse:
    dataset = _dataset_response(hit.record).model_dump()
    return SearchHitResponse(
        **dataset,
        rank=hit.rank,
        score=hit.score,
        bm25_score=hit.bm25_score,
        lsa_score=hit.lsa_score,
        rrf_score=hit.rrf_score,
        matched_terms=list(hit.matched_terms),
    )


@lru_cache
def get_retriever() -> HybridRetriever:
    records, _ = load_catalog(SNAPSHOT)
    return HybridRetriever().fit(records)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    retriever = get_retriever()
    return HealthResponse(status="ok", records=len(retriever.records), snapshot_path=str(SNAPSHOT.relative_to(ROOT)))


@app.get("/search", response_model=SearchResponse, tags=["retrieval"])
def search(
    query: str = Query(min_length=2, max_length=240),
    strategy: Literal["bm25", "lsa", "hybrid", "hybrid_mmr"] = "hybrid_mmr",
    limit: int = Query(default=5, ge=1, le=20),
    tag: str | None = Query(default=None, min_length=2, max_length=120),
) -> SearchResponse:
    hits = get_retriever().search(query=query, strategy=strategy, limit=limit, tag=tag)
    return SearchResponse(
        query=query,
        strategy=strategy,
        tag=tag,
        count=len(hits),
        hits=[_hit_response(hit) for hit in hits],
    )


@app.get("/datasets/{name}", response_model=DatasetResponse, tags=["catalog"])
def dataset(name: str) -> DatasetResponse:
    record = get_retriever().get_record(name)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset slug not found in the versioned snapshot.")
    return _dataset_response(record)


@app.get("/evaluation", tags=["evaluation"])
def evaluation() -> dict[str, object]:
    if not METRICS.exists():
        raise HTTPException(status_code=404, detail="Run python -m src.pipeline before requesting evaluation metrics.")
    return json.loads(METRICS.read_text(encoding="utf-8"))


@app.get("/strategies", tags=["retrieval"])
def strategies() -> dict[str, list[str]]:
    return {"strategies": list(SUPPORTED_STRATEGIES)}
