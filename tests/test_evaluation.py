from src.catalog import CatalogRecord
from src.evaluation import RelevanceQuery, evaluate_retriever, validate_relevance_labels
from src.retrieval import HybridRetriever


def make_record(name: str, title: str) -> CatalogRecord:
    return CatalogRecord(
        name=name,
        title=title,
        notes=title,
        tags=(),
        groups=(),
        resource_count=1,
        formats=("CSV",),
        metadata_created=None,
        metadata_modified=None,
        license_title=None,
        portal_url=f"https://example.test/{name}",
    )


def test_evaluation_reports_perfect_rank_for_exact_query() -> None:
    records = [make_record("bike", "Aparcamientos bicicletas"), make_record("air", "Calidad aire")]
    retriever = HybridRetriever().fit(records)
    queries = [RelevanceQuery("bike", "aparcamientos bicicletas", "mobility", ("bike",))]
    validate_relevance_labels(queries, {record.name for record in records})
    metrics, rows = evaluate_retriever(retriever, queries, strategies=("bm25",))
    assert metrics["bm25"]["mrr_at_10"] == 1.0
    assert rows.loc[0, "top_result"] == "bike"
