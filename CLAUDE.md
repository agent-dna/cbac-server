# CLAUDE.md — cbac-server

Guidance for this repo. It holds two things:

- `cbac_service/` — the decision service (this file is mostly about it).
- `cbac/` — the framework-agnostic guard + optional MCP glue, its own
  distribution (`cbac` on PyPI) with its own `pyproject.toml`. The import
  package is `cbac/src/cbac/`; the service depends on it as a **uv workspace
  member**, so `import cbac` in this checkout always resolves to that source,
  never to a published wheel.

## Writing docs and comments

`README.md`, every doc here, and every **code comment and docstring** describe the
code **as it currently stands**, never how it got there. Verify each claim against
the actual files before writing it — endpoints against `main.py`, env vars against
`config.py`, commands against `pyproject.toml` and `.github/workflows/` — rather
than reconstructing from a conversation or from a previous version of the text. No
"we used to", no "removed X because it's no longer needed", no "this previously
returned Y", no rationale that only makes sense as a diff.

Explaining *why the current code is the way it is* stays — that is what makes a
comment worth reading ("fail-closed by design", "asarray narrows the declared
union"). Restating it as a change does not. When resolving a doc merge conflict,
re-check *both* sides against current code; neither side is presumed right.

## What this is

`cbac_service` is the **reference CBAC decision service** — a standalone FastAPI
app that the guard calls over HTTP. All the ML deps live here; `cbac/` imports
**none** of them.

It runs **two gates** side by side:

- **v2** (`/cbac/v2/*`, `CBAC.authorize_call`) — the IBAC gate: permissions
  parsed from trusted inputs into exact OpenFGA tuples, each tool call mapped
  to tuples deterministically, and deny-only semantic checks on free-text
  arguments. See "The v2 pipeline" below.
- **v1** (`/cbac/v1/authorize`, `CBAC.verify_cbac`) — the semantic-only gate:
  the whole call scored against the policy text. See "The v1 pipeline".

Modules:

- `main.py` — the HTTP boundary: `app = FastAPI()`, the lazy `CBAC` singleton,
  DB and OpenFGA lifecycle via lifespan, and the `__main__` uvicorn runner.
- `cbac.py` — the decision engine (class `CBAC`), no HTTP. Both gates.
- `ibac.py` — v2's deterministic half, pure functions: the canonical form of a
  resource value, OpenFGA object/user ids, catalog validation. Domain-neutral:
  nothing in it knows what an email, a path or a URL is.
- `parsers.py` — policy text in, structures out. Deterministic: structure-aware
  chunking for the embedding index (`flatten_policy_chunks`, used by both
  gates), the hard word cap that keeps every chunk and every free-text window
  inside the cross-encoders' 512 tokens (`split_by_word_budget`), and the v2
  document split (`split_policy_document`). LLM: v2's two parsers
  (`parse_policy`, `parse_query`), the `chat_json` backend dispatch (Ollama
  over stdlib HTTP, or the Claude API), and the validation of everything they
  return.
- `fga.py` + `fga_model.json` — OpenFGA access: store binding, tuple writes,
  checks.
- `config.py` — pipeline tunables as module-level constants (`ALLOW_GAP`,
  `ENCODER_MODEL`, `LHI_WEIGHTS`, `DATABASE_URL`, …). Change a value here and
  redeploy.
- `skills.py` — `skill.md` parsing, `render_intent`, the CBAC result dataclasses.
- `entity.py` — pydantic request models for the HTTP API.
- **Not published as a wheel** (`[tool.uv] package = false`). Deployed from a
  checkout: `uvicorn cbac_service.main:app`.
- Every response is `{success, message, data}` (`main.api_response`). `success`
  means the request was *processed*, **not** that the action was allowed — a
  deny is a successful call, and the verdict is `data.decision`. Request bodies
  are pydantic models in `entity.py`; a schema violation answers in the same
  envelope with 422 via the `RequestValidationError` handler.
- Endpoints:
  - **`POST /cbac/v2/policies`** — `{agent_id, policy, trusted_data?,
    review_required?}`: upload a policy document (instructions + tool schemas
    in one Markdown file, see "The v2 pipeline"; required — v2 never reads the
    Provenance Layer) and optional trusted data; the policy parser
    turns them into a new `policy_versions` row, activated at once unless
    `review_required` (default `false`) holds it `pending` for approval. A
    document that does not split is a 422. Requires `X-CBAC-Key:
    $CBAC_POLICY_KEY`; an unset key refuses every request (503).
  - **`GET /cbac/v2/policies/{agent_id}`** — the agent's active and pending
    versions, for a reviewer. Same key.
  - **`POST /cbac/v2/policies/{agent_id}/approve`** — `{"version_id": n}`:
    activate a pending version (the optional human-review step). Same key.
  - **`POST /cbac/v2/authorize`** — the v2 decision gate:
    `{agent_id, intent_id, user_query, tool, arguments, callee_type, mcp_did}`
    → `data` `{decision, status_code, hash}`. Always HTTP 200, like v1.
  - **`POST /cbac/v1/authorize`** — the v1 decision gate. Also folds the decision
    into the caller→callee trust score, so it is the *only* call a guard makes.
    Always HTTP 200, including the fail-closed `"error"` verdict: a 5xx would
    invite a retry, and the trust fold is not idempotent.
  - **`POST /cbac/v1/policies/precompute`** — explicitly trigger embedding precomputation.
  - **`GET /cbac/v1/decisions?agent_id=&limit=&offset=`** — an agent's decision
    history, newest first.
  - **`GET /cbac/v1/decisions/{id}`** — one decision by id.
  - **`GET /cbac/v1/decisions/by-hash/{interaction_hash}`** — one decision by
    its interaction hash.
  - **`POST /cbac/v1/lhi-scores`** — current trust for a batch of agents
    (`{"agent_ids": [...]}`), one entry per caller→callee edge, each carrying
    the `mcp_did` its latest record was written with. A POST for a read because
    the id batch is the body.
  - **`GET /health`** — DB connectivity check. Unversioned on purpose: probes
    are wired once at deploy time and must not track API versions.
- Depends on `agent-dna` (for `Provenance`, `AgentCard`, `IntentWorkflow`, `id`).

## Architecture

```
user -> agent -> gateway (cbac/ guard) --HTTP--> cbac_service (FastAPI) -> tool runs
                                                   |
                         ┌─────────────────────────┼─────────────────────┐
                    OpenFGA v1.22.0         PostgreSQL 18           parser LLM
                    (allow/deny tuples,     ├── pgvector 0.8.6      (Ollama or
                     datastore: the         ├── pg_textsearch 1.4.0  Claude API)
                     `openfga` database     ├── policy_chunks
                     in this Postgres)      ├── policy_meta
                                            ├── policy_versions (v2 policy + catalog)
                                            ├── cbac_intents (intent -> query, grants)
                                            ├── cbac_decisions (audit log)
                                            └── lhi_records (trust history)
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
- `policy_hash` — the hash of the policy text these chunks came from
- `section` / `chunk_index` — ordering and provenance

**`policy_meta`** — one row per agent, written by `index_policy`:
- `policy_hash` — which policy text the indexed chunks came from. Recorded, not
  compared: nothing on the decision path re-reads the chain to check it
- `encoder_model` / `nli_model` — detect if models changed
- `chunk_count` / `cached_at` — operational metadata

**`policy_versions`** — one row per v2 policy upload, append-only:
- `policy` / `policy_hash` — the document's instructions section (what gets
  chunked for policy drift), and the hash of the whole document
- `tool_schemas` / `trusted_data` — the document's tool schemas, and the
  upload's trusted data (`{}` when none)
- `catalog` — tool name → `{resources: [arg], free_text: [arg], description}`,
  validated
- `deny_rules` — `[{tool, match: exact|glob, resource, source}]`
- `rejected` / `unenforceable` — what validation threw out, and the policy
  statements no tuple can express
- `status` — `pending` | `active` | `superseded`; the active version is the
  agent's latest `active` row
- `parser_model`, `created_at`, `approved_at`

**`cbac_intents`** — one row per `intent_id`, written by the first v2 call that
carries it and never updated: `agent_id`, `user_query`, `policy_version_id`,
`grants` (readable form of the allow tuples), `dropped`, `parser_model`,
`created_at`, `expires_at`.

**`cbac_decisions`** — one row per verdict of either gate, the **audit log**:
- `agent_id` / `decision` / `reason` / `error_code` — who, what, why, which check
- `intended_action` — v1: the *flattened* action text the scorers saw; v2: the
  call rendered by `render_intent`
- `user_intent` / `intent_id` / `callee_name` / `callee_type` — context; NULL
  when not supplied
- `interaction_hash` — salted per-row id, returned to the caller
- `created_at`

**`lhi_records`** — one row per decision, the **trust history** (see the LHI
section below). Also carries `mcp_did` — which MCP server the call was bound
for, supplied on `/authorize` and returned by `/lhi-scores`. Descriptive only:
it is not part of the edge key and enters none of the trust arithmetic, so a
callee that moves servers keeps its accumulated trust.

⚠️ **These two are not interchangeable, and the difference is easy to get
wrong.** `cbac_decisions` is *complete*: every verdict `verify_cbac` reaches is
recorded, including the infrastructure-failure denies and calls with no callee.
`lhi_records` is deliberately *skipped* in exactly those cases (no
`callee_name`, no measured component, the infra-failure early returns), because
a trust score must not be moved by things that are not evidence about the agent.
So only `cbac_decisions` can answer "what did we decide, and why".

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
  `cbac` is the exception: it is a workspace member (`[tool.uv.workspace]`,
  `[tool.uv.sources]`), installed from `cbac/` in this checkout, so guard edits
  *are* live.
- `uv build --package cbac` — build the guard's sdist + wheel into `dist/`. It
  declares one dependency (`requests`) and ships only `cbac/src/cbac/`; the
  service is not published (`[tool.uv] package = false`).
- `uv run pytest` — runs `cbac_service/tests/` (`[tool.pytest.ini_options]`,
  `pythonpath = ["."]`).
- `ruff` and `pyright` are pinned **exactly**, not floored: CI and the local
  `PostToolUse` hook must run the same linter. Bump them in `pyproject.toml`.
- `transformers` is pinned `<5` on purpose — HHEM-2.1's remote code
  (`hallucination_score`) breaks on transformers 5.x. Don't loosen it.

Alembic and docker-compose are the exception: both read config relative to
`cbac_service/`, so `cd cbac_service` first for those two.

`docker compose` starts Postgres, a one-shot `openfga-migrate`, and OpenFGA
(`openfga/openfga:v1.22.0`, HTTP API on `127.0.0.1:$OPENFGA_HTTP_PORT`).
OpenFGA's datastore is a separate `openfga` database in the same Postgres,
created by `init-extensions.sql` on a fresh volume — an existing volume needs
`CREATE DATABASE openfga;` once by hand. OpenFGA requires a pre-shared key
(`OPENFGA_API_TOKEN`) and compose refuses to start it without one.

### Required environment variables

See `.env.sample` for the full list. Key ones:
- `DATABASE_URL` — async Postgres connection string
- `AGENTDNA_API_KEY` — Provenance Layer access
- `CBAC_SERVICE_HOST` / `CBAC_SERVICE_PORT` — binding
- `HYBRID_SEARCH_ENABLED` — toggle BM25 fusion (default: false)
- `OPENFGA_API_URL` / `OPENFGA_STORE_NAME` / `OPENFGA_API_TOKEN` — the v2 tuple
  store; one store name per environment, and no default for it — an env file
  that forgets it denies every v2 call rather than writing into another
  environment's store
- `POLICY_PARSER_MODEL` / `QUERY_PARSER_MODEL` — `"<backend>:<model>"`, backend
  `ollama` or `anthropic` (default `ollama:qwen3:14b`); `OLLAMA_URL`,
  `ANTHROPIC_API_KEY`
- `CBAC_POLICY_KEY` — the v2 policy endpoints' shared secret
- `INTENT_TTL_SECONDS` (default 3600)

Whether a policy upload waits for human review is not deployment config: it is
the upload's own `review_required` field, decided per agent by the admin.

## The v2 pipeline (`CBAC.authorize_call`)

**Trusted inputs:** the admin's policy, the raw tool schemas, the agent's
trusted data, and the user's query. **Untrusted:** everything the agent
produces — which tool it calls and every argument. The query arrives in the
`X-CBAC-User-Intent` header, set by the agent's runtime rather than its model.

**Policy upload** (`upload_policy`, once per version). The policy is one
Markdown document, split deterministically by
`parsers.split_policy_document` — sections separated by lines that are exactly
`---`, each opening with its title (a leading `#` is allowed):

```
---
Policy Instructions
<the policy, in prose>
---
Tool Schema
<JSON: a list of MCP tools, or a tools/list result {"tools": [...]}>
---
```

Both sections are required, each once; the JSON may be in a fenced code block.
Then `parse_policy` makes one LLM pass over the instructions and the tools that
returns the **catalog** (per tool: which top-level arguments are *resources* —
values that name what the call acts on, checked exactly — and which are *free
text*), the **deny rules** (per tool: an exact value or a glob over its
resource values, `*` for the whole tool), and the **unenforceable** statements.
`ibac.validate_catalog` then checks the catalog's shape in code: every schema
property classified exactly once (coverage — a forgotten argument is an open
channel). Which class an argument belongs to is the parser's judgment; nothing
second-guesses it, so an identifier misclassified as free text gets only the
semantic checks — `review_required` is the mitigation. A tool that fails is
left out of the catalog, so calls to it are denied. Each deny rule's `source`
must be a verbatim quote of the policy.

The capability is IBAC's *(tool, resource)*, and the system is deliberately
domain-neutral: no resource types, no per-domain normalization. A resource
value is compared in one canonical form (`ibac.canonical`: strings NFC +
trimmed, numbers and booleans as JSON, so `3` and `"3"` agree), and any other
spelling is a different resource. Under exact allow-listing that can only
produce a false deny, never a false allow.
Activation (`activate_policy_version`) re-indexes the policy chunks
(`index_policy`, for policy drift), swaps the exact deny tuples in OpenFGA, and
only then flips the rows. An upload with `review_required: true` stays
`pending` — the previous active version keeps deciding — until
`POST /cbac/v2/policies/{agent_id}/approve` activates it.

**Per call**, in order:

1. **Intent.** Empty `intent_id`/`user_query` → 4001. Under a transaction-scoped
   advisory lock on the intent id: an unseen id runs `parse_query` (query +
   trusted data → grants of `{tool, argument, resource}`, one value each; the
   argument must be one of the tool's resources and the value grounded in the
   query or trusted data — the tuple itself is *(tool, resource)*),
   writes the allow tuples, and stores the `cbac_intents` row; a seen id must
   carry the same query and agent (4002) and be unexpired (4003). A failed
   parse stores nothing (4004), so the next call retries.
2. **Mapping.** The active version's catalog: unknown tool 4005, an argument
   that is neither resource nor free text 4006, a resource value with no
   canonical form (an object, an empty string) 4007, no resource value 4008, no
   active version 4009. A tool with no resource arguments has the single
   resource `*`. A list-valued resource argument is checked item by item.
3. **Glob deny rules**, matched in the service (OpenFGA cannot wildcard an
   object id) → 4101.
4. **OpenFGA**: each resource needs a live allow tuple and no deny tuple —
   4102 deny tuple, 4103 not granted, 4104 OpenFGA error.
5. **Semantic checks** on free-text arguments, **deny-only**: every word window
   (`CHUNK_MAX_WORDS`) is scored, so nothing hides past the models' 512-token
   cut. Intent drift (NLI contradiction vs the query) → 4201; policy drift
   Tier 1 (gap < −`DENY_GAP`) → 4202, Tier 2 (contradiction vs the top allowed
   chunk) → 4203; a model or search error → 4204. Anything short of a clear
   deny passes — the call already holds an exact grant. HHEM is attached,
   never gating.
6. **Allow**: 4301 (no free text) or 4302 (free text passed).

Tuple shapes (`fga_model.json`): allow `(user:<sha256(intent_id)>, can_invoke,
tool_invocation:<sha256(agent_id, tool#resource)>)` with a
`non_expired_grant` condition; deny `(user:*, blocked, …)`. Object ids are
hashed so OpenFGA's id limits never meet a raw value, and agent-scoped so one
agent's deny never matches another's call.

Known limit (also IBAC's): each resource argument is checked on its own, so a
tool with two resource arguments accepts any combination of granted values. The readable forms are in
`policy_versions` and `cbac_intents`.

`authorize_call` wraps `_authorize_call` the way `verify_cbac` wraps `_decide`:
the trust fold and the audit write happen in one place for every path.

## The v1 pipeline (`CBAC.verify_cbac`)

Class `CBAC`. On each request:

1. **Check 1 — NLI drift** (when `user_intent` supplied): NLI cross-encoder
   checks if the agent's intended action contradicts the user intent.
   Contradiction ≥ 0.60 → immediate deny.

2. **Policy load**: Reads the agent's chunks from Postgres. On a hit that is the
   whole step — no network call, and decisions keep working while the Provenance
   Layer is down. Only an agent with nothing indexed reaches the chain, and then
   `index_policy` (chunk, classify, encode, store) runs before the tiers.
   **Nothing re-reads the chain afterwards**: the indexed copy decides until
   `POST /cbac/v1/policies/precompute` replaces it, so publishing a new policy
   card is half of a policy update and re-indexing is the other half.

3. **Tier 1 — Cosine gap** (via pgvector, `_tier1_gap`): Encodes the intent, runs
   `vector_search(allowed)` and `vector_search(forbidden)`. If
   `max_allowed − max_forbidden > ALLOW_GAP` → allow. If gap < −`DENY_GAP` → deny.
   Otherwise escalate.

4. **Tier 2 — NLI entailment** (`_tier2_nli`): Uses `hybrid_search` (pgvector +
   pg_textsearch RRF fusion, when `HYBRID_SEARCH_ENABLED`) to find the best
   allowed chunk candidate, then runs the NLI
   cross-encoder. Entailment ≥ 0.55 → allow, contradiction ≥ 0.60 → deny.

5. **Tier 3 — LLM backend** (optional): If configured, sends intent + full
   policy text to an LLM for judgment. Otherwise returns `"deny"`. A
   configured backend that itself returns `"advise"` is also folded to
   `"deny"` — the pipeline has no caller-must-decide state.

6. **Hallucination score** (HHEM): Attached after the decision, never gates it.

7. **LHI trust fold** (`_fold_trust` → `compute_lhi`): once a decision is
   *reached*, its component scores are folded into the (agent → callee) trust
   score and the new value is attached to the result. Also never gates the
   decision — a failed trust update is logged, not raised. Skipped when no
   `callee_name` was supplied, or when no component was measured. The early
   returns *above* the decision (policy lookup down, no policy, no chunks) are
   infrastructure failures, not evidence about the agent, and fold no trust.

8. **Decision recorded** (`_record_decision` → `insert_cbac_decision`): the
   verdict plus its action context is appended to `cbac_decisions`. Unlike the
   trust fold this has **no skip conditions** — every path is audited, including
   the infra-failure denies above. Also never gates: failing to write down what
   was decided must not change what was decided.

`verify_cbac` is a thin wrapper that calls `_decide` (the pipeline above) and
then records. `_decide` has seven return paths and the wrapper exists so the
audit write happens in exactly one place — threading it through each return is
how a path silently stops being audited.

Decisions are `"allow" | "deny"` and the pipeline is **fail-closed** (any
error, or an inconclusive/misbehaving Tier 3, → `deny`).

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
  records** — allow and deny alike — so an agent probing forbidden actions
  loses trust instead of keeping a pristine record.

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
  `callee_name`, `callee_type`); `mcp_did` rides along on the row but is *not*
  part of the key — and stored in the **`lhi_records` table, one row per
  decision**. The table *is* the trust history: rows are append-only, an
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

  v2 folds trust the same way, from its semantic scores: a call decided at the
  gate measures no component and writes no trust row.

## v2 tests

No OpenFGA, Postgres or LLM needed: `test_ibac.py` (normalization, catalog
validation), `test_parsers.py` (validation behind a fake `chat_json`), and
`test_authorize_v2.py` (the talk's injection replayed through `authorize_call`
against in-memory fakes of OpenFGA and the repository, plus the guard↔route
contract and the policy-key check).
