"""CKAN extraction, catalog validation and multilingual text normalization."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CKAN_ACTION_URL = "https://opendata.vlci.valencia.es/api/3/action/package_search"
PORTAL_DATASET_URL = "https://opendata.vlci.valencia.es/dataset/"

STOPWORDS = {
    "a",
    "al",
    "and",
    "con",
    "da",
    "de",
    "del",
    "des",
    "el",
    "en",
    "es",
    "i",
    "la",
    "las",
    "los",
    "per",
    "para",
    "por",
    "the",
    "un",
    "una",
    "y",
}


class CatalogValidationError(ValueError):
    """Raised when a catalog snapshot violates required metadata controls."""


@dataclass(frozen=True)
class CatalogRecord:
    """A privacy-conscious view of one public CKAN dataset record."""

    name: str
    title: str
    notes: str
    tags: tuple[str, ...]
    groups: tuple[str, ...]
    resource_count: int
    formats: tuple[str, ...]
    metadata_created: str | None
    metadata_modified: str | None
    license_title: str | None
    portal_url: str

    @property
    def document(self) -> str:
        return " ".join(
            part
            for part in (
                self.title,
                self.name.replace("-", " "),
                self.notes,
                " ".join(self.tags),
                " ".join(self.groups),
                " ".join(self.formats),
            )
            if part
        )


@dataclass(frozen=True)
class CatalogValidation:
    records: int
    duplicate_names: tuple[str, ...]
    missing_titles: tuple[str, ...]
    records_without_resources: int

    @property
    def is_valid(self) -> bool:
        return not self.duplicate_names and not self.missing_titles and self.records > 0

    def require_valid(self) -> None:
        if not self.is_valid:
            raise CatalogValidationError(
                "Catalog validation failed: "
                f"records={self.records}, duplicate_names={list(self.duplicate_names)}, "
                f"missing_titles={list(self.missing_titles)}"
            )


def normalize_text(text: str) -> str:
    """Fold Spanish and Valencian accents, lowercase and retain searchable word tokens."""

    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", ascii_text.lower())).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token not in STOPWORDS and len(token) > 1]


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return "\n".join(line.rstrip() for line in str(value).splitlines()).strip()


def _compact_package(package: dict[str, Any]) -> dict[str, Any]:
    resources = package.get("resources") or []
    return {
        "name": _safe_string(package.get("name")),
        "title": _safe_string(package.get("title")),
        "notes": _safe_string(package.get("notes")),
        "tags": sorted(
            {_safe_string(tag.get("name")) for tag in package.get("tags", []) if _safe_string(tag.get("name"))}
        ),
        "groups": sorted(
            {
                _safe_string(group.get("display_name") or group.get("name"))
                for group in package.get("groups", [])
                if _safe_string(group.get("display_name") or group.get("name"))
            }
        ),
        "resources": [
            {
                "name": _safe_string(resource.get("name")),
                "format": _safe_string(resource.get("format")),
                "url": _safe_string(resource.get("url")),
            }
            for resource in resources
        ],
        "metadata_created": _safe_string(package.get("metadata_created")) or None,
        "metadata_modified": _safe_string(package.get("metadata_modified")) or None,
        "license_title": _safe_string(package.get("license_title")) or None,
    }


def fetch_catalog_snapshot(
    snapshot_path: Path,
    manifest_path: Path,
    page_size: int = 100,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Fetch all public package metadata via CKAN pagination and write compact artifacts."""

    packages: list[dict[str, Any]] = []
    start = 0
    declared_count: int | None = None

    while declared_count is None or start < declared_count:
        query = urlencode({"rows": page_size, "start": start})
        request = Request(
            f"{CKAN_ACTION_URL}?{query}",
            headers={"User-Agent": "valencia-open-data-navigator/1.0 (portfolio reproducibility)"},
        )
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - official HTTPS endpoint
            payload = json.loads(response.read().decode("utf-8"))

        if not payload.get("success"):
            raise RuntimeError("CKAN API returned success=false while fetching package metadata.")

        result = payload["result"]
        declared_count = int(result["count"])
        page = result.get("results") or []
        if not page:
            break
        packages.extend(_compact_package(package) for package in page)
        start += len(page)

    if declared_count is not None and len(packages) != declared_count:
        raise RuntimeError(
            f"Expected {declared_count} catalog packages from CKAN but received {len(packages)}."
        )

    snapshot = {
        "source": CKAN_ACTION_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "declared_count": declared_count,
        "packages": packages,
    }
    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(serialized + "\n", encoding="utf-8")

    manifest = {
        "source": CKAN_ACTION_URL,
        "retrieved_at": snapshot["retrieved_at"],
        "declared_count": declared_count,
        "received_count": len(packages),
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "snapshot_file": snapshot_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


def validate_packages(packages: list[dict[str, Any]]) -> CatalogValidation:
    names = [_safe_string(package.get("name")) for package in packages]
    duplicate_names = tuple(sorted(name for name, count in Counter(names).items() if name and count > 1))
    missing_titles = tuple(
        package_name or "<unnamed>"
        for package_name, package in zip(names, packages, strict=True)
        if not _safe_string(package.get("title"))
    )
    records_without_resources = sum(not (package.get("resources") or []) for package in packages)
    return CatalogValidation(
        records=len(packages),
        duplicate_names=duplicate_names,
        missing_titles=missing_titles,
        records_without_resources=records_without_resources,
    )


def record_from_package(package: dict[str, Any]) -> CatalogRecord:
    formats = tuple(
        sorted(
            {
                _safe_string(resource.get("format")).upper()
                for resource in package.get("resources") or []
                if _safe_string(resource.get("format"))
            }
        )
    )
    name = _safe_string(package.get("name"))
    return CatalogRecord(
        name=name,
        title=_safe_string(package.get("title")),
        notes=_safe_string(package.get("notes")),
        tags=tuple(_safe_string(tag) for tag in package.get("tags") or [] if _safe_string(tag)),
        groups=tuple(_safe_string(group) for group in package.get("groups") or [] if _safe_string(group)),
        resource_count=len(package.get("resources") or []),
        formats=formats,
        metadata_created=package.get("metadata_created"),
        metadata_modified=package.get("metadata_modified"),
        license_title=package.get("license_title"),
        portal_url=f"{PORTAL_DATASET_URL}{name}",
    )


def load_catalog(snapshot_path: Path) -> tuple[list[CatalogRecord], CatalogValidation]:
    """Load an already-versioned snapshot and validate its required fields."""

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    packages = payload.get("packages")
    if not isinstance(packages, list):
        raise CatalogValidationError("Snapshot must contain a list under 'packages'.")
    validation = validate_packages(packages)
    validation.require_valid()
    return [record_from_package(package) for package in packages], validation


def record_to_dict(record: CatalogRecord) -> dict[str, Any]:
    return asdict(record)
