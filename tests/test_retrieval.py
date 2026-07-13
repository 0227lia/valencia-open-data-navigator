from src.catalog import CatalogRecord
from src.retrieval import HybridRetriever


def record(name: str, title: str, notes: str, tags: tuple[str, ...]) -> CatalogRecord:
    return CatalogRecord(
        name=name,
        title=title,
        notes=notes,
        tags=tags,
        groups=(),
        resource_count=1,
        formats=("CSV",),
        metadata_created=None,
        metadata_modified=None,
        license_title=None,
        portal_url=f"https://example.test/dataset/{name}",
    )


def test_hybrid_retrieval_returns_the_expected_catalog_record() -> None:
    retriever = HybridRetriever().fit(
        [
            record("bike-parking", "Aparcamientos bicicletas", "Plazas para bicicletas", ("Movilidad",)),
            record("air", "Calidad del aire", "Contaminacion atmosferica horaria", ("Ambiente",)),
            record("tourism", "Oficinas turismo", "Recursos turisticos", ("Turismo",)),
        ]
    )
    hits = retriever.search("parking de bicicletas", strategy="hybrid_mmr", limit=2)
    assert hits[0].record.name == "bike-parking"
    assert "bicicletas" in hits[0].matched_terms


def test_tag_filter_restricts_the_candidate_pool() -> None:
    retriever = HybridRetriever().fit(
        [
            record("bike", "Bicicletas", "Movilidad urbana", ("Movilidad",)),
            record("air", "Aire", "Ambiente urbano", ("Ambiente",)),
            record("bus", "Autobus", "Movilidad publica", ("Movilidad",)),
        ]
    )
    hits = retriever.search("urbana", strategy="hybrid", limit=5, tag="Ambiente")
    assert [hit.record.name for hit in hits] == ["air"]
