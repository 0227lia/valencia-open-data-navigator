"""Generate reproducible catalog and retrieval evaluation artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .catalog import CatalogRecord, record_to_dict


def _catalog_frame(records: list[CatalogRecord]) -> pd.DataFrame:
    frame = pd.DataFrame([record_to_dict(record) for record in records])
    frame["has_description"] = frame["notes"].str.len().fillna(0).gt(0)
    frame["has_tags"] = frame["tags"].map(bool)
    frame["has_license"] = frame["license_title"].notna()
    return frame


def _catalog_summary(frame: pd.DataFrame) -> dict[str, object]:
    tags = Counter(tag for values in frame["tags"] for tag in values)
    formats = Counter(item for values in frame["formats"] for item in values)
    return {
        "records": int(frame.shape[0]),
        "resources": int(frame["resource_count"].sum()),
        "records_without_resources": int(frame["resource_count"].eq(0).sum()),
        "description_coverage": float(frame["has_description"].mean()),
        "tag_coverage": float(frame["has_tags"].mean()),
        "license_coverage": float(frame["has_license"].mean()),
        "top_tags": tags.most_common(12),
        "top_formats": formats.most_common(12),
    }


def _short_label(value: str, limit: int = 34) -> str:
    return value if len(value) <= limit else f"{value[: limit - 3]}..."


def _plot_scorecard(metrics: dict[str, dict[str, float]], destination: Path) -> None:
    frame = pd.DataFrame(metrics).T.reset_index(names="strategy")
    label_map = {
        "mrr_at_10": "MRR@10",
        "recall_at_5": "Recall@5",
        "ndcg_at_10": "nDCG@10",
    }
    colors = ["#00795a", "#0c8a79", "#ed6a4a", "#496a59"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=170)
    for axis, (column, title) in zip(axes, label_map.items(), strict=True):
        bars = axis.bar(frame["strategy"], frame[column], color=colors)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("mean score")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, frame[column], strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.3f}", ha="center", fontsize=9)
    figure.suptitle("Versioned relevance benchmark: retrieval strategy comparison", x=0.05, ha="left", fontsize=14)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _plot_catalog_quality(frame: pd.DataFrame, summary: dict[str, object], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=170)
    palette = {"green": "#00795a", "coral": "#ed6a4a", "ink": "#16342b", "soft": "#8ab7a3"}

    axes[0, 0].hist(frame["resource_count"], bins=min(16, max(4, frame["resource_count"].nunique())), color=palette["green"], edgecolor="white")
    axes[0, 0].set_title("Resources per dataset", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("declared resources")
    axes[0, 0].set_ylabel("datasets")

    coverage = pd.Series(
        {
            "description": summary["description_coverage"],
            "tags": summary["tag_coverage"],
            "license": summary["license_coverage"],
        }
    )
    axes[0, 1].bar(coverage.index, coverage.values, color=[palette["green"], palette["soft"], palette["coral"]])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Metadata coverage", loc="left", fontweight="bold")
    for index, value in enumerate(coverage.values):
        axes[0, 1].text(index, value + 0.03, f"{value:.0%}", ha="center")

    top_tags = summary["top_tags"][:8]
    tag_names = [_short_label(item[0]) for item in top_tags][::-1]
    tag_counts = [item[1] for item in top_tags][::-1]
    axes[1, 0].barh(tag_names, tag_counts, color=palette["ink"])
    axes[1, 0].set_title("Most frequent tags", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("datasets")

    axes[1, 1].axis("off")
    text = "\n".join(
        [
            "CATALOG SNAPSHOT",
            f"{summary['records']} datasets",
            f"{summary['resources']} declared resources",
            f"{summary['records_without_resources']} records without a resource",
            "",
            "This is metadata quality, not data-quality validation",
        ]
    )
    axes[1, 1].text(
        0.04,
        0.92,
        text,
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=13,
        linespacing=1.7,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#e8f0e9", "edgecolor": "#8ab7a3"},
    )
    figure.suptitle("Valencia Open Data Navigator: catalog quality snapshot", x=0.05, ha="left", fontsize=14)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def write_artifacts(
    records: list[CatalogRecord],
    metrics: dict[str, dict[str, float]],
    evaluation_rows: pd.DataFrame,
    validation_summary: dict[str, object],
    processed_dir: Path,
    reports_dir: Path,
) -> dict[str, object]:
    """Write structured outputs and report figures from an already-evaluated index."""

    figures_dir = reports_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame = _catalog_frame(records)
    summary = _catalog_summary(frame)
    summary["validation"] = validation_summary
    summary["generated_at"] = datetime.now(UTC).isoformat()

    frame.to_csv(processed_dir / "catalog_records.csv", index=False)
    evaluation_rows.to_csv(reports_dir / "evaluation_by_query.csv", index=False)
    (reports_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot_scorecard(metrics, figures_dir / "retrieval_scorecard.png")
    _plot_catalog_quality(frame, summary, figures_dir / "catalog_quality_dashboard.png")
    return summary
