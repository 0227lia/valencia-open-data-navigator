"""End-to-end reproducible build for the Valencia metadata retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import fetch_catalog_snapshot, load_catalog
from .evaluation import evaluate_retriever, load_evaluation_queries, validate_relevance_labels
from .reporting import write_artifacts
from .retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
RAW_SNAPSHOT = ROOT / "data" / "raw" / "ckan_catalog_snapshot.json"
MANIFEST = ROOT / "data" / "raw" / "source_manifest.json"
EVALUATION_QUERIES = ROOT / "data" / "evaluation_queries.json"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def run(refresh: bool = False, offline: bool = False) -> dict[str, object]:
    if refresh or not RAW_SNAPSHOT.exists():
        if offline:
            raise FileNotFoundError("Offline mode requires data/raw/ckan_catalog_snapshot.json.")
        fetch_catalog_snapshot(RAW_SNAPSHOT, MANIFEST)

    records, validation = load_catalog(RAW_SNAPSHOT)
    queries = load_evaluation_queries(EVALUATION_QUERIES)
    validate_relevance_labels(queries, {record.name for record in records})
    retriever = HybridRetriever().fit(records)
    metrics, rows = evaluate_retriever(retriever, queries)
    summary = write_artifacts(
        records=records,
        metrics=metrics,
        evaluation_rows=rows,
        validation_summary={
            "records": validation.records,
            "duplicate_names": list(validation.duplicate_names),
            "missing_titles": list(validation.missing_titles),
            "records_without_resources": validation.records_without_resources,
        },
        processed_dir=PROCESSED_DIR,
        reports_dir=REPORTS_DIR,
    )
    return {"catalog": summary, "metrics": metrics, "queries": len(queries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh the official CKAN metadata snapshot.")
    parser.add_argument("--offline", action="store_true", help="Fail instead of using the network.")
    args = parser.parse_args()
    result = run(refresh=args.refresh, offline=args.offline)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
