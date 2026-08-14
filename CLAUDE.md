# CLAUDE.md — cbac-server

Guidance for this repo. It holds two things:

- `cbac_service/` — the decision service (this file is mostly about it).
- `cbac/` — the framework-agnostic guard + optional MCP glue. `import cbac` gets
  it; this repo owns it outright.

## What this is

`cbac_service` is the **reference CBAC decision service** — a standalone FastAPI
app that the guard calls over HTTP. All the ML deps live here; `cbac/` imports
**none** of them.

- `main.py` — the HTTP boundary: `app = FastAPI()`, the lazy `CBAC` singleton,
  DB lifecycle via lifespan, and the `__main__` uvicorn runner.
- `cbac.py` — the decision engine (class `CBAC`), no HTTP.
- `config.py` — pipeline tunables as module-level constants (`ALLOW_GAP`,
  `ENCODER_MODEL`, `LHI_WEIGHTS`, `DATABASE_URL`, …). Change a value here and
  redeploy.
- `chunking.py` — structure-aware policy-text chunking (`chunk_body_text`).
- `skills.py` — `skill.md` parsing + the CBAC result dataclasses.
- **Not published as a wheel** (`[tool.uv] package = false`). Deployed from a
  checkout: `uvicorn cbac_service.main:app`.
- Endpoints:
  - **`POST /authorize-cbac`** — main decision gate. Also folds the decision
    into the caller→callee trust score, so it is the *only* call a guard makes.
  - **`POST /precompute-policy`** — explicitly trigger embedding precomputation.
  - **`GET /health`** — DB connectivity check.
- Depends on `agent-dna` (for `Provenance`, `AgentCard`, `IntentWorkflow`, `id`).

## Architecture

```
Guard (cbac/) --HTTP--> cbac_service (FastAPI)
                                     |
                                PostgreSQL 18
                                ├── pgvector 0.8.6 (semantic search)
                                ├── pg_textsearch 1.4.0 (BM25 keyword search)
                                ├── policy_chunks table (embeddings + text)
                                └── policy_meta table (cache invalidation)
```

## Database

The service uses **PostgreSQL 18** with two extensions:
- **pgvector** — stores 384-dim embeddings, HNSW index for cosine similarity.
- **pg_textsearch** — BM25 ranked keyword search on chunk text.

### Tables

**`policy_chunks`** — one row per chunk per agent:
- `agent_id` — who this belongs to
- `chunk_text` — the original text (needed for Tier 2 NLI)
- `chunk_type` — `allowed` or `forbidden`
- `embedding` — vector(384), the searchable embedding
- `policy_hash` — for cache invalidation
- `section` / `chunk_index` — ordering and provenance

**`policy_meta`** — one row per agent, lightweight cache check:
- `policy_hash` — compared against on-chain hash at runtime
- `encoder_model` / `nli_model` — detect if models changed
- `chunk_count` / `cached_at` — operational metadata

### Indexes
- `policy_chunks_embedding_idx` — HNSW (vector_cosine_ops)
- `policy_chunks_bm25_idx` — BM25 (text_config='english')
- `ix_policy_chunks_agent_hash` — B-tree (agent_id, policy_hash)
- `ix_policy_chunks_agent_id` — B-tree (agent_id)

### Schema management
Schema is managed via **Alembic** (async). Models are defined in
`db/models.py`; migrations live in `db/migrations/versions/`. Run:
```bash
cd cbac_service
alembic upgrade head
```

## Environment

`pyproject.toml`, `uv.lock`, and `.venv` all live at the **repo root** — run
everything from there, not from inside `cbac_service/`:

- `uv sync --locked` — install from `uv.lock`. `agent-dna` resolves from PyPI
  like any other dependency; there is no path source to the sibling checkout, so
  library edits are **not** picked up live — publish, then bump the pin here.
  The `dev` group declares `mcp` directly, which `cbac/mcp.py` needs (the
  published `agent-dna` wheel ships no `mcp` extra).
- `uv run pytest` — runs `cbac_service/tests/` (`[tool.pytest.ini_options]`,
  `pythonpath = ["."]`).
- `ruff` and `pyright` are pinned **exactly**, not floored: CI and the local
  `PostToolUse` hook must run the same linter. Bump them in `pyproject.toml`.
- `transformers` is pinned `<5` on purpose — HHEM-2.1's remote code
  (`hallucination_score`) breaks on transformers 5.x. Don't loosen it.

Alembic and docker-compose are the exception: both read config relative to
`cbac_service/`, so `cd cbac_service` first for those two.

### Required environment variables

See `.env.sample` for the full list. Key ones:
- `DATABASE_URL` — async Postgres connection string
- `AGENTDNA_API_KEY` — Provenance Layer access
- `CBAC_SERVICE_HOST` / `CBAC_SERVICE_PORT` — binding
- `HYBRID_SEARCH_ENABLED` — toggle BM25 fusion (default: true)

## The decision pipeline (`cbac.py`)

Class `CBAC`. On each request:

1. **Check 1 — NLI drift** (when `user_intent` supplied): NLI cross-encoder
   checks if the agent's intended action contradicts the user intent.
   Contradiction ≥ 0.60 → immediate deny.

2. **Policy fetch + cache check**: Fetches the agent's latest policy from the
   Provenance Layer. Compares the policy hash against `policy_meta` in Postgres.
   If stale or missing → `index_policy` (chunk, classify, encode, store).

3. **Tier 1 — Cosine gap** (via pgvector): Encodes the intent, runs
   `vector_search(allowed)` and `vector_search(forbidden)`. If
   `max_allowed − max_forbidden > ALLOW_GAP` → allow. If gap < −`DENY_GAP` → deny.
   Otherwise escalate.

4. **Tier 2 — NLI entailment**: Uses `hybrid_search` (pgvector + pg_textsearch
   RRF fusion) to find the best allowed chunk candidate, then runs the NLI
   cross-encoder. Entailment ≥ 0.55 → allow, contradiction ≥ 0.60 → deny.

5. **Tier 3 — LLM backend** (optional): If configured, sends intent + full
   policy text to an LLM for judgment. Otherwise returns `"advise"`.

6. **Hallucination score** (HHEM): Attached after the decision, never gates it.

7. **LHI trust fold** (`_fold_trust` → `compute_lhi`): once a decision is
   *reached*, its component scores are folded into the (agent → callee) trust
   score and the new value is attached to the result. Also never gates the
   decision — a failed trust update is logged, not raised. Skipped when no
   `callee_name` was supplied, or when no component was measured. The early
   returns *above* the decision (policy lookup down, no policy, no chunks) are
   infrastructure failures, not evidence about the agent, and record nothing.

Decisions are `"allow" | "deny" | "advise"` and the pipeline is **fail-closed**
(any error → `deny`).

## Scoring attached to a decision

- **Hallucination score (HHEM):** when a decision is reached with a user intent
  present, `CBAC.hallucination_score` (vectara HHEM model, 1 = grounded,
  0 = hallucinated) is attached to the result.
- **LHI trust:** `compute_lhi` combines three component scores
  (intent, policy, hallucination) as a **weighted arithmetic mean**
  (`LHI_WEIGHTS`) — deliberately compensatory, since the allow/deny gates
  already enforce the hard constraints — then folds it into a stored trust value
  via an **asymmetric EMA — slow to build (`LHI_LAMBDA_UP`), fast to lose
  (`LHI_LAMBDA_DOWN`)**. A zero component costs only its weight, not the whole
  score.

  It is called **from `verify_cbac` itself**, at decision time. There is no
  post-execution step and no `/compute-lhi` endpoint: every component is known
  the moment a decision is reached, so a guard makes exactly **one** HTTP call
  per action and no score ever round-trips through the client. **Every decision
  records** — allow, deny and advise alike — so an agent probing forbidden
  actions loses trust instead of keeping a pristine record.

  The mean **renormalizes over the observed components** (`s = Σ wᵢxᵢ / Σ wᵢ`)
  rather than skipping records with a missing one: the components are not
  missing at random (`policy_score` is `None` exactly on Tier-3 gray-zone
  decisions; intent/hallucination are `None` without a `user_intent`), so
  complete-case analysis would silently exclude the very interactions trust
  exists to arbitrate. An unmeasured component is stored **NULL**, never
  substituted. Nothing observed → no record. Because of the renormalizing
  denominator, `LHI_WEIGHTS` is **scale-invariant** — only the ratios matter, so
  adding or removing a component needs no retuning.

  Trust is tracked **per caller→callee edge** — the edge key is (`agent_id`,
  `callee_name`, `callee_type`) — and stored in the **`lhi_records` table, one
  row per decision**. The table *is* the trust history: rows are append-only, an
  edge's current trust is its latest row (`repository.get_latest_trust`), and
  `get_trust_history` reads the series. `compute_lhi` is **async and takes an
  `AsyncSession`** like the rest of the pipeline.

  The on-chain mirror (appending each record to the agent's `{agent_id}:cbac`
  provenance card) is **currently commented out** in `compute_lhi` — the DB is
  the working copy and nothing reads the card yet. The block names the imports
  to restore alongside it.

  The LHI math is covered by `cbac_service/tests/test_cbac_lhi.py` (repository
  functions faked in-memory via the shared `rows` fixture in
  `tests/conftest.py`, no DB needed); the fold into `verify_cbac` by
  `cbac_service/tests/test_cbac_verify.py` — keep both green when touching it.
