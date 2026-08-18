# cbac-server


## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Postgres)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
  ```bash
  brew install uv
  ```

## Quick Start

### 1. Create the Python virtual environment

```bash
cd cbac-server
uv venv --python 3.11
uv sync
```

This installs all dependencies (including `agent-dna`, FastAPI, SQLAlchemy, etc.) into `.venv/`.

### 2. Build the Docker image and start Postgres (PG18 + pgvector + pg_textsearch)

```bash
cd cbac_service
docker compose build
docker compose up -d
```

The `build` step compiles `pg_textsearch` from source on top of the `pgvector/pgvector:pg18` base image (~2-3 min on first run). To force a clean rebuild (e.g. after Dockerfile changes):

```bash
docker compose build --no-cache
```

Once built, `docker compose up -d` starts the container. Both extensions (`vector`, `pg_textsearch`) are created automatically on first startup via `init-extensions.sql` — no manual SQL commands needed.

### 3. Run database migrations

Still inside `cbac_service/`:

```bash
export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
uv run alembic upgrade head
```

### 4. Start the service

From the **project root** (not `cbac_service/`):

```bash
cd cbac-server
export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
uv run uvicorn cbac_service.main:app
```

API docs available at http://localhost:8000/docs

### 5. Run the lifecycle integration test

This script exercises the full CBAC pipeline (chunking, NLI classification, embedding, vector/BM25/hybrid search, tiered decisions) against the live Docker Postgres:

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
| Run linter | `uv run ruff check .` |

## Architecture Notes

- **Database:** PostgreSQL 18 with `pgvector` (cosine similarity search) and `pg_textsearch` (BM25 keyword search). Both are used for hybrid retrieval via Reciprocal Rank Fusion.
- **pg_textsearch** requires `shared_preload_libraries` — this is configured automatically in the Docker image via `postgresql.conf.sample`.
- **Migrations** are managed with Alembic (async, using `asyncpg` driver).
- The `DATABASE_URL` env var defaults to a local macOS socket connection. Override it to point at the Docker container as shown above.
