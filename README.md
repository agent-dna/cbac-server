# cbac-server

Context Based Access Control for Apps and Tools.

A standalone FastAPI decision service that answers one question per agent action:
**should this agent be allowed to do this?** A guard calls it over HTTP, gets back
`allow` / `deny` / `advise`, and the service folds the outcome into a trust score.

## Layout

| Path | What it is |
|---|---|
| `cbac_service/` | The decision service — FastAPI app, decision engine, DB layer. All ML dependencies live here. |
| `cbac/` | The framework-agnostic guard + optional MCP glue. Imports none of the ML stack. |

`pyproject.toml`, `uv.lock`, and `.venv` live at the **repo root**. Run everything
from there — except Alembic and Docker Compose, which read config relative to
`cbac_service/`.

## Requirements

- Python `>=3.10,<3.13`
- [uv](https://docs.astral.sh/uv/)
- Docker (for local Postgres), or an existing Postgres 17+ with `pgvector` and
  `pg_textsearch`

## Setup

```bash
# 1. Install dependencies (creates .venv from uv.lock)
uv sync --locked

# 2. Start Postgres with the required extensions
cd cbac_service
docker compose up -d

# 3. Create the schema
alembic upgrade head
cd ..
```

Compose brings up Postgres on `localhost:5432` with database `cbac`, user
`cbac_user`, password `cbac_pass`, and runs `init-extensions.sql` to enable
`vector` and `pg_textsearch` on first boot.

## Running the server

From the repo root:

```bash
uv run python -m cbac_service.main
```

Serves on `http://127.0.0.1:8767` by default. With `uv run` you can invoke this
from any subdirectory — uv walks up to find `pyproject.toml`.

For development with auto-reload:

```bash
uv run uvicorn cbac_service.main:app --reload --port 8767
```

`cbac_service.main:app` is also the deployment entrypoint — point gunicorn or
uvicorn at that dotted path.


## Configuration

All settings are environment variables; defaults are in `cbac_service/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | see below | Async Postgres connection string (asyncpg driver) |
| `AGENTDNA_API_KEY` | `""` | Provenance Layer access |
| `CBAC_SERVICE_HOST` | `127.0.0.1` | Bind address |
| `CBAC_SERVICE_PORT` | `8767` | Bind port |
| `HYBRID_SEARCH_ENABLED` | `true` | Toggle BM25 fusion alongside vector search |
| `VECTOR_INDEX_TYPE` | `hnsw` | `hnsw` (low latency) or `ivfflat` (large scale) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant; higher = less aggressive re-ranking |

**The built-in `DATABASE_URL` default does not match Docker Compose.** Set it
explicitly to match the compose credentials:

```bash
export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/authorize-cbac` | The decision gate — returns `allow` / `deny` / `advise` |
| `POST` | `/compute-lhi` | Fold interaction scores into the caller→callee trust score |
| `POST` | `/precompute-policy` | Explicitly trigger policy embedding precomputation |
| `GET` | `/health` | Database connectivity check |

## How a decision is made

Each request walks a tiered pipeline, escalating only when the cheaper tier is
inconclusive:

1. **NLI drift** — if a `user_intent` was supplied, check whether the agent's
   intended action contradicts it. Contradiction ≥ 0.60 → immediate deny.
2. **Policy fetch + cache** — pull the agent's policy, compare its hash against
   `policy_meta`. Stale or missing → chunk, classify, encode, store.
3. **Tier 1 — cosine gap** (pgvector) — compare the intent against allowed and
   forbidden chunks. Clear margin either way decides; otherwise escalate.
4. **Tier 2 — NLI entailment** — hybrid search (pgvector + BM25, RRF-fused) picks
   the best allowed chunk, then a cross-encoder judges entailment.
5. **Tier 3 — LLM** (optional) — if configured, sends intent + policy to an LLM.
   Otherwise returns `advise`.

A hallucination score (HHEM) is attached to the result but never gates it. The
pipeline is **fail-closed**: any error becomes a `deny`.

## Development

```bash
uv run pytest                                   # tests
uv run ruff format cbac cbac_service            # format
uv run ruff check cbac cbac_service             # lint
uv run pyright                                  # type check
```

CI runs exactly these (see `.github/workflows/`). `ruff` and `pyright` are pinned
to exact versions in `pyproject.toml` so CI and local checks never diverge —
bump them there.

### Database schema

Managed with Alembic (async). Models are in `cbac_service/db/models.py`,
migrations in `cbac_service/db/migrations/versions/`.

```bash
cd cbac_service
alembic upgrade head                            # apply
alembic revision --autogenerate -m "message"    # create
```
