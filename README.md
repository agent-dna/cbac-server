# cbac-server

Context Based Access Control for Apps and Tools.

A standalone FastAPI decision service that answers one question per agent action:
**should this agent be allowed to do this?** A guard calls it over HTTP and gets
back `allow` / `deny` / `advise`. Reaching a decision also folds the component
scores into the caller→callee trust score, so a guard makes exactly one call per
action.

## Layout

| Path | What it is |
|---|---|
| `cbac_service/` | The decision service — FastAPI app, decision engine, DB layer. All ML dependencies live here. |
| `cbac/` | The framework-agnostic guard + optional MCP glue. Imports none of the ML stack. |
| `scripts/` | `test_lifecycle.py` integration script and `cbac_benchmark/`. |

`pyproject.toml`, `uv.lock`, and `.venv` live at the **repo root**. Run everything
from there — except Alembic and Docker Compose, which read config relative to
`cbac_service/`.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Postgres)
- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- Python `>=3.10,<3.13`

## Quick Start

### 1. Create the Python virtual environment

```bash
cd cbac-server
uv venv --python 3.11
uv sync --locked
```

Installs every dependency (`agent-dna`, FastAPI, SQLAlchemy, the ML stack) into
`.venv/`. `--locked` fails if `uv.lock` has drifted from `pyproject.toml`
instead of silently re-resolving — the same thing CI does.

### 2. Build the Docker image and start Postgres

```bash
cd cbac_service
docker compose build
docker compose up -d
```

Compose **builds** rather than pulls: `Dockerfile.postgres` compiles
`pg_textsearch` from source on top of `pgvector/pgvector:pg18` (~2-3 min on first
run), because no published image carries PG18 + pgvector + pg_textsearch
together. Force a clean rebuild after Dockerfile changes:

```bash
docker compose build --no-cache
```

Both extensions are created automatically on first startup via
`init-extensions.sql` — no manual SQL. The image also appends
`shared_preload_libraries = 'pg_textsearch'` to `postgresql.conf.sample`, which
that extension requires at server start.

### 3. Run database migrations

Still inside `cbac_service/`:

```bash
export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
uv run alembic upgrade head
```

### 4. Start the service

From the **project root**:

```bash
cd cbac-server
export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
uv run uvicorn cbac_service.main:app
```

Serves on `http://localhost:8000`, with API docs at
[`/docs`](http://localhost:8000/docs). Add `--reload` for development.

Alternatively, run the module's own entrypoint, which reads `CBAC_SERVICE_HOST`
and `CBAC_SERVICE_PORT` and so defaults to port **8767**:

```bash
uv run python -m cbac_service.main
```

> Run it as `python -m cbac_service.main`, not `python -m main` from inside
> `cbac_service/`. The package imports itself absolutely (`from
> cbac_service.config import ...`), so it must be importable by its full package
> name. `cbac_service.main:app` is also the deployment entrypoint.

### 5. Run the lifecycle integration test

Exercises the full pipeline — chunking, NLI classification, embedding,
vector/BM25/hybrid search, tiered decisions — against the live Docker Postgres:

```bash
cd cbac-server
export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
PYTHONPATH=. uv run python scripts/test_lifecycle.py
```

## Useful Commands

| Task | Command |
|------|---------|
| Stop Postgres | `cd cbac_service && docker compose down` |
| Stop + destroy data | `cd cbac_service && docker compose down -v` |
| Rebuild Postgres image | `cd cbac_service && docker compose build --no-cache` |
| Connect via psql | `psql postgresql://cbac_user:cbac_pass@localhost:5432/cbac` |
| Run tests | `uv run pytest` |
| Format | `uv run ruff format cbac cbac_service` |
| Lint | `uv run ruff check cbac cbac_service` |
| Type check | `uv run pyright` |

CI runs exactly those last four (see `.github/workflows/`). `ruff` and `pyright`
are pinned to exact versions in `pyproject.toml` so CI and local checks never
diverge — bump them there.

## Configuration

All settings are environment variables; defaults live in `cbac_service/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | see below | Async Postgres connection string (asyncpg driver) |
| `AGENTDNA_API_KEY` | `""` | Provenance Layer access |
| `CBAC_SERVICE_HOST` | `127.0.0.1` | Bind address (`python -m` entrypoint only) |
| `CBAC_SERVICE_PORT` | `8767` | Bind port (`python -m` entrypoint only) |
| `HYBRID_SEARCH_ENABLED` | `true` | Toggle BM25 fusion alongside vector search |
| `VECTOR_INDEX_TYPE` | `hnsw` | `hnsw` (low latency) or `ivfflat` (large scale) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant; higher = less aggressive re-ranking |

**The built-in `DATABASE_URL` default is a local macOS socket connection and does
not match Docker Compose.** Export it as shown in the Quick Start whenever you
run against the container.

## Endpoints

Every response is the same envelope:

```json
{"success": true, "message": "human-readable reason", "data": {}}
```

`success` says the request was **processed**, never what the answer was — an
authorization deny is a successful call. The verdict is `data.decision`; reading
`success` as "allowed" would invert the gate. Malformed input answers in the
same envelope with HTTP 422 and the field errors under `data.errors`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/cbac/v1/authorize` | The decision gate. `data.decision` is `allow`/`deny`/`advise`/`error` (also mirrored in the `X-CBAC-Decision` header for proxies) and `message` the reason. Folds the component scores into the caller→callee trust score |
| `GET` | `/cbac/v1/decisions?agent_id=&limit=&offset=` | An agent's decision history, newest first |
| `GET` | `/cbac/v1/decisions/{id}` | One decision by id |
| `GET` | `/cbac/v1/decisions/by-hash/{interaction_hash}` | One decision by its interaction hash |
| `POST` | `/cbac/v1/policies/precompute` | Explicitly trigger policy embedding precomputation |
| `GET` | `/health` | Database connectivity check. Unversioned on purpose — probes are wired once and must not track API versions |

## How a decision is made

Each request walks a tiered pipeline, escalating only when the cheaper tier is
inconclusive:

1. **NLI drift** — if a `user_intent` was supplied, check whether the agent's
   intended action contradicts it. Contradiction ≥ 0.60 → immediate deny.
2. **Policy fetch + cache** — pull the agent's policy, compare its hash against
   `policy_meta`. Stale or missing → chunk, classify, encode, store.
3. **Tier 1 — cosine gap** (pgvector) — compare the intent against allowed and
   forbidden chunks. A clear margin either way decides; otherwise escalate.
4. **Tier 2 — NLI entailment** — hybrid search (pgvector + BM25, RRF-fused) picks
   the best allowed chunk, then a cross-encoder judges entailment.
5. **Tier 3 — LLM** (optional) — if configured, sends intent + policy to an LLM.
   Otherwise returns `advise`.

A hallucination score (HHEM) is attached to the result but never gates it. Once a
decision is reached, its component scores fold into the stored trust value for
that caller→callee edge. The pipeline is **fail-closed**: any error becomes a
`deny`.

## Architecture Notes

- **Database:** PostgreSQL 18 with `pgvector` (cosine similarity) and
  `pg_textsearch` (BM25 keyword search), combined for hybrid retrieval via
  Reciprocal Rank Fusion.
- **Migrations** are managed with Alembic (async, `asyncpg` driver). Models live
  in `cbac_service/db/models.py`, migrations in
  `cbac_service/db/migrations/versions/`.

```bash
cd cbac_service
uv run alembic upgrade head                            # apply
uv run alembic revision --autogenerate -m "message"    # create
```

- **`requirements.txt`** is generated, not hand-edited — it exists for deploy
  targets that don't use uv. Regenerate with:

```bash
uv export --no-hashes --no-dev -o requirements.txt
```
