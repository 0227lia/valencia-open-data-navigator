# Retrieval Card

## Intended use

Help a user inspect a versioned municipal-data catalog and navigate to the official dataset record.

## Not intended for

- Replacing the official portal search.
- Determining whether an underlying resource is current, complete, licensed for a particular use or suitable for policy action.
- Generating factual answers from source data that has not been retrieved and validated.

## Transparency

Each result returns rank, overall score, BM25 score, LSA score, fused RRF score and matched normalized query terms. The dashboard shows the strategy selected by the user and the benchmark results for each strategy.

## Maintenance

Refresh the metadata snapshot only with `python -m src.pipeline --refresh`, review the manifest and rerun the benchmark before publishing a change.
