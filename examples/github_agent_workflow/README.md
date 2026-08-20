# Guarded GitHub Agent

A minimal demo of this repo's CBAC guard: an agent with two tools, where every
tool call is authorized by the CBAC decision service before it runs. One tool
the policy permits, one it forbids.

No AgentDNA, no provenance chain, no identity registration — the agent's policy
is a markdown file the service reads from disk.

```
        "Draft a summary…"            "Close issue #3…"
                │                            │
                ▼                            ▼
        ┌──────────────────────────────────────────┐
        │   cbac_context(agent_id, user_intent)    │
        └────────────────┬─────────────────────────┘
                         ▼
                 agent picks a tool
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     draft_summary                  close_issue
     @cbac_guard                    cbac_intercept
     (in-process)                   (outgoing MCP call)
          │                             │
          └──────────► POST ◄───────────┘
                  /authorize-cbac
                         │
              allow ─────┴───── deny
                │                 │
           tool runs        short-circuit,
                            never reaches
                              GitHub
```

## The two paths

| Tool | Where it runs | Guarded by | Policy says |
| --- | --- | --- | --- |
| `draft_summary` | this process | `@cbac_guard` decorator | allowed |
| `close_issue` | the MCP server | `cbac_intercept` on the client | forbidden |

The MCP server is deliberately CBAC-unaware — it stands in for a third-party
server you don't own and can't decorate. It exposes `close_issue` happily; the
policy is what stops the call.

## Files

```
agent.py                    the whole demo — MCP server + agent, one file
policies/github-worker.md   the policy CBAC decides against
requirements.txt
.env.sample
```

## Run it

### 1. Start the decision service

From the repo root:

```bash
cd cbac_service && docker compose up -d && alembic upgrade head && cd ..

DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac" \
HYBRID_SEARCH_ENABLED=true \
CBAC_LOCAL_POLICY_DIR=$PWD/examples/github_agent_workflow/policies \
  uv run uvicorn cbac_service.main:app --port 8767
```

`CBAC_LOCAL_POLICY_DIR` is what makes this demo standalone: the service reads
`<dir>/{agent_id}.md` instead of resolving the policy from the Provenance
Layer. Leave it unset and the service behaves exactly as it does in production.

`HYBRID_SEARCH_ENABLED=true` disables BM25 fusion — the flag's comparison is
inverted in `cbac_service/config.py`. It has to be off because Tier 2's BM25
half needs `pg_textsearch`, which the `pgvector/pgvector:pg17` image in the
compose file does not ship.

Check it decides correctly before involving an LLM:

```bash
curl -sD- -o /dev/null -XPOST localhost:8767/authorize-cbac \
  -H 'content-type: application/json' \
  -d '{"agent_id":"github-worker","intended_action":"The agent wants to draft summary, with repo = owner/repo, notes = rewrote the parser"}'
# X-CBAC-Decision: allow

curl -sD- -o /dev/null -XPOST localhost:8767/authorize-cbac \
  -H 'content-type: application/json' \
  -d '{"agent_id":"github-worker","intended_action":"The agent wants to close issue, with repo = owner/repo, number = 3"}'
# X-CBAC-Decision: deny
```

The first call is slow — the service loads three transformer models and embeds
the policy. Later calls hit the cached embeddings in Postgres.

### 2. Install and configure the example

```bash
cd examples/github_agent_workflow
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env    # then fill in an LLM key
```

### 3. Start the MCP server

```bash
python agent.py serve
```

### 4. Run the two scenarios

```bash
# ALLOWED — the guard authorizes it and the tool runs
python agent.py "Draft a summary for owner/repo from notes: rewrote the parser, fixed two crashes"

# DENIED — blocked at the client; no HTTP request reaches GitHub
python agent.py "Close issue #3 in owner/repo"
```

The `[cbac]` line is the guard's verdict for that call.

Allowed:

```
[call] draft_summary({'repo': 'owner/repo', 'notes': 'rewrote the parser, fixed two crashes'})
[cbac] ALLOW — authorized by policy, tool executed
[tool] {"status": "ok", "repo": "owner/repo", "summary": "rewrote the parser, fixed two crashes"}
[ai]   Here's the condensed summary for owner/repo: ...
```

Denied — the denial comes back as a readable tool result rather than an
exception, so the model can see it and report it:

```
[call] close_issue({'repo': 'owner/repo', 'number': 3})
[cbac] DENIED — Tier 1 cosine gap -0.157 < -0.08 (intent closer to forbidden than allowed policy)
[tool] {"status": "denied", "error": "Tier 1 cosine gap -0.157 < -0.08 (...)"}
[ai]   I'm unable to close the issue due to a policy restriction.
```

The model must support tool calling. If you see an `[ai]` line containing
`{"name": …, "arguments": …}` with no `[call]` line, no tool was invoked and
CBAC was never consulted — that is a model problem, not a guard problem.

### 5. Check the guard wiring without an LLM

```bash
python agent.py check
```

Drives both paths directly and asserts the outcome — same interceptor, same
`/authorize-cbac` call, no model in the loop:

```
[@cbac_guard    draft_summary] {'status': 'ok', 'repo': 'owner/repo', 'summary': 'fixed the parser'}
[cbac_intercept close_issue  ] {"status": "denied", "error": "..."}

OK — permitted tool ran, forbidden tool blocked before the GitHub call.
```

## What the guard sends

One POST per tool call, carrying only what the decision needs:

```json
{
  "agent_id":        "github-worker",
  "intended_action": "The agent wants to close issue, with repo = owner/repo, number = 3",
  "user_intent":     "Close issue #3 in owner/repo"
}
```

`intended_action` is rendered as prose from a verb phrase plus the call's
arguments — NLI scores prose far better than `close_issue(repo=...)` call
syntax. The verb phrase comes from the function's docstring under
`@cbac_guard`, and from the de-snaked tool name under `cbac_intercept`, since
the MCP request carries no description across the process boundary.

`user_intent` is the root request, which lets the service also check the tool
call for drift away from what the user actually asked for.

After the call finishes, the guard reports the outcome to `/compute-lhi`, which
folds it into a per-edge trust score. That is fire-and-forget: it never affects
the result you get back.

## Editing the policy

Change `policies/github-worker.md` and re-run. The service hashes the policy
text, notices the change, and re-embeds it on the next authorization — no
restart, no explicit invalidation.

Two things to keep in mind when writing bullets:

- Phrase prohibitions as `<gerund> ... is prohibited and forbidden`. The
  service classifies each chunk by NLI entailment, and "the agent must not X"
  classifies as *allowed*.
- Keep one capability per bullet and leave editorial prose out — every
  sentence in the file becomes a chunk that an intent can match against.

Inspect the split after any edit:

```sql
SELECT chunk_type, chunk_text FROM policy_chunks
WHERE agent_id = 'github-worker' ORDER BY chunk_type, chunk_index;
```
