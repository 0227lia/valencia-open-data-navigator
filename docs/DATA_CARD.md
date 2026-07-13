# Data Card

## Source

The corpus is metadata returned by the public CKAN `package_search` endpoint of the Valencia Open Data portal. The extractor stores only fields needed for discovery: dataset identifiers, titles, descriptions, tags, groups, formats, counts, timestamps, license title and generated portal links.

## Privacy

The extractor intentionally omits catalog contact fields and does not download the contents of linked resources. The project should not be used to infer information about people.

## Snapshot policy

`data/raw/ckan_catalog_snapshot.json` is the reproducible input. `source_manifest.json` records the source URL, retrieval timestamp, record count and SHA-256. Refreshes are explicit and must be reviewed before committing.

## Reuse

The portal states that public datasets are reusable subject to the conditions published by the City of Valencia. Each linked dataset may have additional terms; inspect its portal record before reuse.
