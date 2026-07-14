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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .catalog import CatalogRecord, record_to_dict

INK = "#0B2130"
TEAL = "#0F766E"
GREEN = "#247A68"
BLUE = "#2F6BFF"
CORAL = "#E85D45"
MUTED = "#597181"
GRID = "#D6E0E5"
PAPER = "#F4F7F6"
WHITE = "#FFFFFF"


def _apply_report_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": PAPER,
            "axes.facecolor": WHITE,
        }
    )


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
    _apply_report_style()
    frame = pd.DataFrame(metrics).T.reset_index(names="strategy")
    metric_labels = {
        "mrr_at_10": "MRR@10",
        "recall_at_5": "Recall@5",
        "ndcg_at_10": "nDCG@10",
    }
    strategy_labels = {
        "bm25": "BM25 lexical",
        "lsa": "LSA semantic",
        "hybrid": "Hybrid RRF",
        "hybrid_mmr": "Hybrid RRF + MMR",
    }
    frame["label"] = frame["strategy"].map(strategy_labels).fillna(frame["strategy"])
    metric_columns = list(metric_labels)
    matrix = frame[metric_columns].to_numpy()

    figure = plt.figure(figsize=(15, 6.4), dpi=170, facecolor=PAPER)
    grid = figure.add_gridspec(1, 2, left=0.07, right=0.965, top=0.72, bottom=0.16, wspace=0.28)
    matrix_axis = figure.add_subplot(grid[0, 0])
    tradeoff_axis = figure.add_subplot(grid[0, 1])

    cmap = LinearSegmentedColormap.from_list("retrieval", ["#E7EEF1", "#8CC8C0", TEAL])
    matrix_axis.imshow(matrix, cmap=cmap, vmin=max(0.0, matrix.min() - 0.05), vmax=1.0, aspect="auto")
    matrix_axis.set_xticks(range(len(metric_columns)), [metric_labels[column] for column in metric_columns])
    matrix_axis.set_yticks(range(len(frame)), frame["label"])
    matrix_axis.tick_params(length=0, pad=8)
    matrix_axis.set_title("Offline relevance matrix", loc="left", color=INK, pad=14)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = WHITE if value >= 0.86 else INK
            matrix_axis.text(column, row, f"{value:.3f}", ha="center", va="center", color=text_color, weight="bold", fontsize=11)
    for column in range(matrix.shape[1]):
        best_row = int(matrix[:, column].argmax())
        matrix_axis.add_patch(Rectangle((column - 0.48, best_row - 0.48), 0.96, 0.96, fill=False, edgecolor=CORAL, linewidth=2.4))
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)

    colors = [INK, BLUE, CORAL, GREEN]
    label_offsets = {
        "bm25": (8, 7),
        "lsa": (8, 7),
        "hybrid": (8, 12),
        "hybrid_mmr": (8, -20),
    }
    for color, (_, row) in zip(colors, frame.iterrows(), strict=True):
        tradeoff_axis.scatter(
            row["mrr_at_10"],
            row["recall_at_5"],
            s=420 * row["ndcg_at_10"],
            color=color,
            edgecolor=WHITE,
            linewidth=1.6,
            zorder=3,
        )
        tradeoff_axis.annotate(
            row["label"],
            (row["mrr_at_10"], row["recall_at_5"]),
            xytext=label_offsets.get(row["strategy"], (8, 7)),
            textcoords="offset points",
            color=INK,
            fontsize=9.5,
            weight="bold" if row["strategy"] == "bm25" else "normal",
        )
    tradeoff_axis.axvline(frame["mrr_at_10"].mean(), color=GRID, linestyle="--", linewidth=1.2)
    tradeoff_axis.axhline(frame["recall_at_5"].mean(), color=GRID, linestyle="--", linewidth=1.2)
    tradeoff_axis.set_xlim(frame["mrr_at_10"].min() - 0.04, frame["mrr_at_10"].max() + 0.06)
    tradeoff_axis.set_ylim(frame["recall_at_5"].min() - 0.04, 1.015)
    tradeoff_axis.set_xlabel("MRR@10  |  first relevant result")
    tradeoff_axis.set_ylabel("Recall@5  |  early retrieval coverage")
    tradeoff_axis.set_title("Precision-recall operating space", loc="left", color=INK, pad=14)
    tradeoff_axis.grid(color=GRID, linewidth=0.8)
    tradeoff_axis.set_axisbelow(True)
    tradeoff_axis.spines[["top", "right"]].set_visible(False)

    figure.text(0.07, 0.92, "VALENCIA OPEN DATA NAVIGATOR", color=TEAL, fontsize=10, weight="bold")
    figure.text(0.07, 0.855, "Retrieval quality, without hiding the trade-off", color=INK, fontsize=24, weight="bold")
    figure.text(
        0.07,
        0.805,
        "Versioned offline benchmark across lexical, semantic and hybrid ranking strategies. Higher is better.",
        color=MUTED,
        fontsize=11,
    )
    figure.text(
        0.07,
        0.06,
        "Coral outlines mark each metric leader. Bubble area encodes nDCG@10. Results are specific to the committed relevance set; they do not measure source-data quality.",
        color=MUTED,
        fontsize=9,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(figure)


def _plot_catalog_quality(frame: pd.DataFrame, summary: dict[str, object], destination: Path) -> None:
    _apply_report_style()
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=170)
    figure.patch.set_facecolor(PAPER)
    palette = {"green": TEAL, "coral": CORAL, "ink": INK, "soft": "#8CC8C0"}

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
    figure.suptitle("Valencia Open Data Navigator | Catalog quality snapshot", x=0.05, ha="left", fontsize=17, fontweight="bold", color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=PAPER)
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
