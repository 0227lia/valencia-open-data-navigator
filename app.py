"""Streamlit interface for public-data discovery and retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.catalog import load_catalog
from src.retrieval import SUPPORTED_STRATEGIES, HybridRetriever

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "data" / "raw" / "ckan_catalog_snapshot.json"
METRICS = ROOT / "reports" / "evaluation_metrics.json"
SUMMARY = ROOT / "reports" / "catalog_summary.json"
BY_QUERY = ROOT / "reports" / "evaluation_by_query.csv"

st.set_page_config(page_title="Valencia Open Data Navigator", page_icon="V", layout="wide")


@st.cache_resource
def load_index() -> tuple[HybridRetriever, list[str]]:
    records, _ = load_catalog(SNAPSHOT)
    retriever = HybridRetriever().fit(records)
    tags = sorted({tag for record in records for tag in record.tags})
    return retriever, tags


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


retriever, tags = load_index()

st.title("Valencia Open Data Navigator")
st.caption(
    "Hybrid NLP retrieval over a versioned snapshot of official CKAN metadata. "
    "It is a portfolio prototype, not an official Valencia search service."
)

discover_tab, benchmark_tab, quality_tab, use_tab = st.tabs(
    ["Discover", "Benchmark", "Catalog quality", "Use and limits"]
)

with discover_tab:
    controls_column, query_column = st.columns([1, 2])
    with controls_column:
        strategy = st.selectbox("Retrieval strategy", SUPPORTED_STRATEGIES, index=3)
        selected_tag = st.selectbox("Optional tag filter", ["All tags", *tags])
        limit = st.slider("Results", min_value=3, max_value=12, value=6)
    with query_column:
        query = st.text_input("Describe the dataset you need", value="aparcamientos para bicicletas")

    active_tag = None if selected_tag == "All tags" else selected_tag
    hits = retriever.search(query, strategy=strategy, limit=limit, tag=active_tag)
    st.subheader(f"{len(hits)} results for: {query}")
    for hit in hits:
        with st.container(border=True):
            header, score = st.columns([5, 1])
            with header:
                st.markdown(f"### {hit.rank}. [{hit.record.title}]({hit.record.portal_url})")
                st.write(hit.record.notes or "No public description is available in this snapshot.")
            with score:
                st.metric("rank score", f"{hit.score:.4f}")
            st.caption(
                " | ".join(
                    [
                        f"resources: {hit.record.resource_count}",
                        f"formats: {', '.join(hit.record.formats) or 'not declared'}",
                        f"matched terms: {', '.join(hit.matched_terms) or 'none'}",
                    ]
                )
            )
            st.json(
                {
                    "bm25": round(hit.bm25_score, 5),
                    "lsa": round(hit.lsa_score, 5),
                    "rrf": round(hit.rrf_score, 5),
                },
                expanded=False,
            )

with benchmark_tab:
    metrics = load_json(METRICS)
    metric_columns = st.columns(len(metrics))
    for column, (strategy, values) in zip(metric_columns, metrics.items(), strict=True):
        with column:
            st.markdown(f"**{strategy}**")
            st.metric("MRR@10", f"{values['mrr_at_10']:.3f}")
            st.metric("Recall@5", f"{values['recall_at_5']:.3f}")
            st.metric("nDCG@10", f"{values['ndcg_at_10']:.3f}")
    st.image(str(ROOT / "reports" / "figures" / "retrieval_scorecard.png"), use_container_width=True)
    st.dataframe(pd.read_csv(BY_QUERY), use_container_width=True, hide_index=True)

with quality_tab:
    summary = load_json(SUMMARY)
    columns = st.columns(4)
    columns[0].metric("datasets", summary["records"])
    columns[1].metric("declared resources", summary["resources"])
    columns[2].metric("description coverage", f"{summary['description_coverage']:.0%}")
    columns[3].metric("tag coverage", f"{summary['tag_coverage']:.0%}")
    st.image(str(ROOT / "reports" / "figures" / "catalog_quality_dashboard.png"), use_container_width=True)
    st.caption("Catalog metadata completeness is not the same as source data quality or freshness.")

with use_tab:
    st.subheader("Appropriate use")
    st.markdown(
        """
        - Discover versioned public-dataset metadata and inspect the official portal record before reuse.
        - Compare retrieval behavior through the included manual relevance benchmark.
        - Treat scores as ranking signals, not guarantees that a resource is current, available or fit for purpose.
        """
    )
    st.subheader("Boundaries")
    st.markdown(
        """
        - The corpus is a snapshot and does not update itself while the app is running.
        - Relevance labels are small, manually curated and intended for regression testing, not a general benchmark.
        - The app does not download or validate the contents of every linked resource.
        """
    )
