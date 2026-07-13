# API

Run locally with:

```powershell
python -m uvicorn src.api:app --reload
```

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Snapshot availability and record count. |
| `GET /search` | Retrieve metadata with `query`, `strategy`, `limit` and optional `tag`. |
| `GET /datasets/{name}` | Inspect a single dataset from the versioned snapshot. |
| `GET /evaluation` | Read generated evaluation metrics. |
| `GET /strategies` | List supported retrieval strategies. |

The API serves local, versioned metadata. It does not proxy the official portal or expose private credentials.
