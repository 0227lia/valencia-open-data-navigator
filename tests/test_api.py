from fastapi.testclient import TestClient

from src.api import app, get_retriever


def test_health_and_search_endpoints_use_the_versioned_snapshot() -> None:
    get_retriever.cache_clear()
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["records"] > 0

    response = client.get("/search", params={"query": "aparcamientos bicicletas", "strategy": "hybrid_mmr"})
    assert response.status_code == 200
    assert response.json()["count"] > 0
    assert response.json()["hits"][0]["name"] == "aparcaments-bicicletes-aparcamientos-bicicletas"


def test_unknown_dataset_returns_404() -> None:
    client = TestClient(app)
    response = client.get("/datasets/not-a-real-dataset")
    assert response.status_code == 404
