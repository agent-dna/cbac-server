# cbac-server

Context Based Access Control for Apps and Tools.

A standalone FastAPI decision service that answers one question per agent action:
**should this agent be allowed to do this?** A guard calls it over HTTP and gets
back `allow` / `deny`. Reaching a decision also folds the component scores into
the caller→callee trust score, so a guard makes exactly one call per action.

It runs two gates:

- **v2** (`/cbac/v2/*`) — permissions come from trusted inputs only. An LLM
  parses the admin's policy into deny rules and the user's query into exact
  grants, both stored as OpenFGA tuples. Each tool call is mapped to tuples by
  code and checked; only its free-text arguments get semantic (NLI) checks, and
  those can only deny.
- **v1** (`/cbac/v1/*`) — the whole call is scored semantically against the
  policy text.

## Layout

| Path | What it is |
|---|---|
| `cbac_service/` | The decision service — FastAPI app, decision engine, DB layer. All ML dependencies live here. |
| `cbac/` | The framework-agnostic guard + optional MCP glue. Imports none of the ML stack. |
| `agentdna/` | The `agent-dna` library (0.6.0), vendored. The service and the examples install it from here. |
| `examples/` | Agents behind the v2 gate, run entirely on one machine. `examples/agentdna_local/` replaces agentdna's hosted services at runtime and holds the shared CBAC service runner. See [`examples/agentdna_local/README.md`](examples/agentdna_local/README.md). |
| `scripts/` | Developer tools — pipeline inspectors, the lifecycle smoke test, the dashboard rebuilder. See [`scripts/README.md`](scripts/README.md). |

`pyproject.toml`, `uv.lock`, and `.venv` live at the **repo root**. Run everything
from there — except Alembic and Docker Compose, which read config relative to
`cbac_service/`.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Postgres
  and OpenFGA)
- For the v2 parsers: [Ollama](https://ollama.com) with the configured model
  pulled (`ollama pull qwen3:14b`), or a Claude API key
- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- Python `>=3.10,<3.13`

## Quick Start

### 1. Create the Python virtual environment

```bash
cd cbac-server
uv venv --python 3.11
uv sync --locked
```

Installs every dependency (FastAPI, SQLAlchemy, the ML stack) into `.venv/`,
and `agent-dna` and `cbac` editable from `agentdna/` and `cbac/` in this
checkout, so edits to either are live. `--locked` fails if `uv.lock` has
drifted from `pyproject.toml` instead of silently re-resolving — the same thing
CI does.

### 2. Build the Docker image and start Postgres and OpenFGA

```bash
cd cbac_service
docker compose --env-file .env.dev build
docker compose --env-file .env.dev up -d
```

Set `OPENFGA_API_TOKEN` in the env file first: compose refuses to start OpenFGA
without its pre-shared key, because whoever can write to the store can mint
grants. OpenFGA's HTTP API is published on loopback only, at
`OPENFGA_HTTP_PORT`, and keeps its tuples in an `openfga` database inside the
same Postgres (the one-shot `openfga-migrate` service creates its tables).
`init-extensions.sql` creates that database on a fresh volume; on a volume
that predates it, create it once:

```bash
docker compose --env-file .env.dev exec postgres \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE DATABASE openfga"'
docker compose --env-file .env.dev up -d
```

The env file picks the environment — `.env.dev` here, `.env.test-prod` for the
other stack. See [Environments](#environments).

Compose **builds** rather than pulls: `Dockerfile.postgres` compiles
`pg_textsearch` from source on top of `pgvector/pgvector:pg18` (~2-3 min on first
run), because no published image carries PG18 + pgvector + pg_textsearch
together. Force a clean rebuild after Dockerfile changes:

```bash
docker compose --env-file .env.dev build --no-cache
```

Both extensions are created automatically on first startup via
`init-extensions.sql` — no manual SQL. The image also appends
`shared_preload_libraries = 'pg_textsearch'` to `postgresql.conf.sample`, which
that extension requires at server start.

The data lands wherever `PG_DATA_PATH` in the env file points — a host directory
you can back up, on a disk with room. Either the variable or a Docker named
volume, never both: an unset (or empty) `PG_DATA_PATH` falls back to the named
volume `cbac-$CBAC_ENV_pgdata`, and a set one replaces it. The two are separate
databases, so a path used for the first time starts empty and needs its own
`alembic upgrade head` — switching back to the named volume finds everything as
it was. On Linux the directory must exist and be writable by the container's
postgres user (uid 999):

```bash
mkdir -p "$PG_DATA_PATH" && chown -R 999:999 "$PG_DATA_PATH"
```

Docker Desktop handles the ownership itself, but only for paths it is allowed to
share.

### 3. Run database migrations

Still inside `cbac_service/` — `alembic.ini` resolves `db/migrations` relative to
it:

```bash
uv run --env-file .env.dev alembic upgrade head
```

`--env-file` rather than exporting the variables by hand, because it fails on a
missing or unreadable file. A half-loaded environment is worse than no
environment: `DATABASE_URL` silently falls back to the `config.py` default, which
is the dev database, and the migration lands on the wrong environment.

### 4. Start the service

From the **project root**:

```bash
cd cbac-server
uv run --env-file cbac_service/.env.dev uvicorn cbac_service.main:app
```

Serves on `http://localhost:8000`, with API docs at
[`/docs`](http://localhost:8000/docs). Add `--reload` for development.

Alternatively, run the module's own entrypoint, which binds to
`CBAC_SERVICE_HOST` and `CBAC_SERVICE_PORT` from that same env file:

```bash
uv run --env-file cbac_service/.env.dev python -m cbac_service.main
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
PYTHONPATH=. uv run --env-file cbac_service/.env.dev python scripts/test_lifecycle.py
```

## Environments

One image, one compose file, one variable: `CBAC_ENV`. It names the compose
project, so the container, the network and the named volume are namespaced by it
and two environments on the same host cannot reach each other's data. Nothing
environment-specific is baked into `Dockerfile.postgres` — credentials, host port
and data directory all arrive at run time — so `test-prod` reuses the `dev` build.

| | `.env.dev` | `.env.test-prod` |
|---|---|---|
| compose project | `cbac-dev` | `cbac-test-prod` |
| container | `dev-cbac-postgres` | `test-prod-cbac-postgres` |
| host port → 5432 | `5432` | `5433` |
| data (`PG_DATA_PATH`) | `…/dev/pgdump/cbac_service` | `…/test_prod/pgdump/cbac_service` |
| service port | `8000` | `8768` |

Both files are gitignored — they hold real credentials. `.env.sample` is the
checked-in reference for the full variable list; anything identical across
environments belongs in `config.py`, not in these files.

Every command takes the environment the same way — one `--env-file` flag:

```bash
cd cbac_service
docker compose --env-file .env.test-prod up -d           # Postgres
uv run --env-file .env.test-prod alembic upgrade head    # schema
```

Each environment is a separate, empty database on first start, so each needs its
own `alembic upgrade head`. Never let the flag fall off: without it `DATABASE_URL`
reverts to the `config.py` default and the command silently hits **dev**.

## Useful Commands

Shown for `dev` — swap the env file for the other environment.

| Task | Command |
|------|---------|
| Stop Postgres | `cd cbac_service && docker compose --env-file .env.dev down` |
| Stop + destroy data | `cd cbac_service && docker compose --env-file .env.dev down -v` (named volume only — a `PG_DATA_PATH` directory survives) |
| Rebuild Postgres image | `cd cbac_service && docker compose --env-file .env.dev build --no-cache` |
| Connect via psql | `psql postgresql://cbac_user:cbac_pass@localhost:5432/cbac` |
| Run tests | `uv run pytest` |
| Format | `uv run ruff format cbac cbac_service` |
| Lint | `uv run ruff check cbac cbac_service` |
| Type check | `uv run pyright` |
| Evaluate the pipeline | `uv run pytest cbac_service/tests/eval --run-eval -q --eval-html eval.html` |

CI runs `pytest`, `ruff format`, `ruff check` and `pyright` (see
`.github/workflows/`). `ruff` and `pyright` are pinned to exact versions in
`pyproject.toml` so CI and local checks never diverge — bump them there.

The evaluation suite is separate from the unit tests and opt-in behind
`--run-eval`, because it loads the encoder, NLI and HHEM models. It measures
each stage of the decision pipeline against a corpus written from the policy
author's point of view, so its failures are findings about the pipeline rather
than broken tests — its exit code is not a gate. `--eval-html` writes a
self-contained dashboard. See
[`cbac_service/tests/eval/README.md`](cbac_service/tests/eval/README.md).

## Configuration

All settings are environment variables; defaults live in `cbac_service/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | see below | Async Postgres connection string (asyncpg driver) |
| `AGENTDNA_API_KEY` | `""` | Provenance Layer access |
| `CBAC_SERVICE_HOST` | `127.0.0.1` | Bind address (`python -m` entrypoint only) |
| `CBAC_SERVICE_PORT` | `8767` | Bind port (`python -m` entrypoint only) |
| `HYBRID_SEARCH_ENABLED` | `false` | Toggle BM25 fusion alongside vector search |
| `VECTOR_INDEX_TYPE` | `hnsw` | `hnsw` (low latency) or `ivfflat` (large scale) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant; higher = less aggressive re-ranking |
| `OPENFGA_API_URL` | `http://localhost:8080` | The OpenFGA HTTP API |
| `OPENFGA_STORE_NAME` | *(required)* | Store found or created at startup — one per environment. Unset = every v2 call denies (`4104`) |
| `OPENFGA_API_TOKEN` | `""` | OpenFGA's pre-shared key (compose requires it) |
| `INTENT_TTL_SECONDS` | `3600` | How long an intent's grants stay live |
| `POLICY_PARSER_MODEL` | `ollama:qwen3:14b` | `<backend>:<model>`, backend `ollama` or `anthropic` |
| `QUERY_PARSER_MODEL` | `ollama:qwen3:14b` | Same format; runs on the first call of each intent |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `ANTHROPIC_API_KEY` | `""` | For an `anthropic:` model; empty lets the SDK find credentials itself |
| `CBAC_POLICY_KEY` | `""` | Shared secret for `/cbac/v2/policies*` (`X-CBAC-Key`); unset = refuse all |

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
| `POST` | `/cbac/v2/policies` | Upload `{agent_id, policy, trusted_data?, review_required?}` — the policy document (instructions and tool schemas in one Markdown file, see below; required) and optionally the agent's trusted data. The new version activates at once, unless `review_required` (default `false`) holds it pending for approval. A document that does not split is a 422. Returns the parsed version: catalog, deny rules, and what was rejected or cannot be enforced. Needs `X-CBAC-Key` |
| `GET` | `/cbac/v2/policies/{agent_id}` | The agent's active and pending versions. Needs `X-CBAC-Key` |
| `POST` | `/cbac/v2/policies/{agent_id}/approve` | `{version_id}` — activate a pending version. Needs `X-CBAC-Key` |
| `POST` | `/cbac/v2/authorize` | The v2 gate: `{agent_id, intent_id, user_query, tool, arguments, callee_type, mcp_did}` → `{decision, status_code, hash}`. Allow codes are `4301`/`4302` |
| `POST` | `/cbac/v1/authorize` | The v1 gate. `data.decision` is `allow`/`deny`/`error` (also mirrored in the `X-CBAC-Decision` header for proxies) and `message` the reason. Folds the component scores into the caller→callee trust score |
| `GET` | `/cbac/v1/decisions?agent_id=&limit=&offset=` | An agent's decision history, newest first |
| `GET` | `/cbac/v1/decisions/{id}` | One decision by id |
| `GET` | `/cbac/v1/decisions/by-hash/{interaction_hash}` | One decision by its interaction hash |
| `POST` | `/cbac/v1/policies/precompute` | Explicitly trigger policy embedding precomputation |
| `POST` | `/cbac/v1/lhi-scores` | Current trust for a batch of agents (`{"agent_ids": [...]}`), one entry per caller→callee edge |
| `GET` | `/health` | Database connectivity check. Unversioned on purpose — probes are wired once and must not track API versions |

## How a decision is made (v2)

**The policy is one Markdown document** holding both the instructions and the
schemas of the tools they govern, in sections separated by lines that are
exactly `---`:

```
---
Policy Instructions
<the policy, in prose>
---
Tool Schema
<JSON: a list of MCP tools, or a tools/list result {"tools": [...]}>
---
```

The service splits it in code — both sections required, the JSON optionally in
a fenced code block. **Once per policy version**, `POST /cbac/v2/policies` then
has an LLM read the instructions and the tool schemas and return a *catalog* —
for each tool, which arguments are resources (values naming what the call acts
on) and which are free text — plus deny rules over each tool's resource values
and the statements no rule can express. Code then checks it: every argument of
every schema must be classified, every deny rule must quote the policy
verbatim. Exact deny rules become OpenFGA deny tuples.

**On the first call of each user request** (a new `intent_id`), an LLM parses
the user's query — with the agent's trusted data, if any — into grants of one
tool, one of its resource arguments, and one value each, keeps only the ones
whose argument belongs to the tool and whose value appears in
the query or the trusted data, and writes them as allow tuples that expire
after `INTENT_TTL_SECONDS`. Later calls with that `intent_id` reuse them, and
must carry the same query.

Nothing here is domain-specific. A resource value is compared exactly, in one
canonical form (strings Unicode-normalized and trimmed; numbers and booleans as
JSON, so `3` and `"3"` agree). Any other spelling is a different resource and
is denied — a false deny, never a false allow.

**Every call** then goes through, in order:

1. **Mapping** — the catalog maps the call's arguments to resource values. An
   unknown tool, an unknown argument, or a value with no canonical form (an
   object, an empty string) is a deny.
2. **Glob deny rules** — patterns over a tool's resource values, matched in the
   service.
3. **OpenFGA** — every resource needs an unexpired allow tuple and no deny
   tuple.
4. **Semantic checks** on free-text arguments only, and only able to deny:
   intent drift (does the text contradict the user's query?) and policy drift
   (Tier 1 cosine gap, Tier 2 NLI against the policy chunks).

`status_code` names the check that decided (`cbac_service/error_codes.py`,
the 4000s). Any error is a deny.

## How a decision is made (v1)

Each request walks a tiered pipeline, escalating only when the cheaper tier is
inconclusive:

1. **NLI drift** — if a `user_intent` was supplied, check whether the agent's
   intended action contradicts it. Contradiction ≥ 0.60 → immediate deny.
2. **Policy** — read the agent's indexed chunks from Postgres. Only when there
   are none does the service call the Provenance Layer, chunk, classify, encode
   and store. A policy already indexed is decided against as-is: republishing a
   card on chain changes nothing until `POST /cbac/v1/policies/precompute`
   re-indexes it.
3. **Tier 1 — cosine gap** (pgvector) — compare the intent against allowed and
   forbidden chunks. A clear margin either way decides; otherwise escalate.
4. **Tier 2 — NLI entailment** — hybrid search (pgvector + BM25, RRF-fused) picks
   the best allowed chunk, then a cross-encoder judges entailment.
5. **Tier 3 — LLM** (optional) — if configured, sends intent + policy to an LLM.
   Otherwise returns `deny`. A configured backend that itself returns `advise`
   is also folded to `deny`.

A hallucination score (HHEM) is attached to the result but never gates it. Once a
decision is reached, its component scores fold into the stored trust value for
that caller→callee edge. The pipeline is **fail-closed**: any error, or an
inconclusive/misbehaving Tier 3, becomes a `deny`.

## Architecture Notes

- **Database:** PostgreSQL 18 with `pgvector` (cosine similarity) and
  `pg_textsearch` (BM25 keyword search), combined for hybrid retrieval via
  Reciprocal Rank Fusion.
- **Tuples:** OpenFGA v1.22.0 (`cbac_service/fga_model.json`) holds v2's allow
  and deny tuples; its datastore is the `openfga` database in the same
  Postgres. `policy_versions` and `cbac_intents` keep the readable record of
  what was written there and why.
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
