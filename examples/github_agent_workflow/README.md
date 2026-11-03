# Guarded GitHub Agent

A demo of CBAC enforced where the agent developer **cannot switch it off**: a
gateway between the MCP client and the MCP server.

Every MCP tool call the agent makes is authorized at that gateway. The agent
process holds no policy, makes no decision, and cannot opt out.

No AgentDNA, no provenance chain, no identity registration — the agent's policy
is a markdown file, uploaded to the service as one policy document together
with the tool schemas the gateway serves.

```
      "Close issue #3…"                    "Draft a summary…"
              │                                     │
              └─────────────────┬───────────────────┘
                                ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │ agent process                                                      │
 │ cbac_context(agent_id, user_intent, intent_id) — once, ambient     │
 │                                                                    │
 │   ┌────────────────────────────────────────────────────────────┐   │
 │   │                       LLM tool loop                        │   │
 │   └──────────────┬─────────────────────────────┬───────────────┘   │
 │      call tool   │                             │   call tool       │
 │                  ▼                             ▼                   │
 │      ┌───────────────────────┐     ┌───────────────────────┐       │
 │      │      MCP client       │     │     draft_summary     │       │
 │      │   + cbac_propagate    │     │      in-process       │       │
 │      │ labels the call —     │     │ never becomes MCP     │       │
 │      │ makes NO decision     │     │ traffic, so no        │       │
 │      │                       │     │ gateway ever sees it  │       │
 │      └───────────┬───────────┘     └───────────────────────┘       │
 └──────────────────┼─────────────────────────────────────────────────┘
                    │  tools/call (MCP) + X-CBAC-* headers
                    ▼
    ┌───────────────────────────────┐
    │         CBAC gateway          │ :8766
    │ MCP server to the agent,      │ ← the PEP
    │ MCP client to each upstream   │
    └───────────────┬───────────────┘
                    │  POST /cbac/v2/authorize
                    ▼
  ┌───────────────────────────────────┐
  │   CBAC decision service   :8767   │
  │           allow · deny            │
  └─────────────────┬─────────────────┘
                    │
        deny ───────┴──────────── allow
          │                         │
  nothing is forwarded;   relayed upstream, with the
  the call dies at        gateway's credential attached
  the gateway                       │
                       ┌────────────┴────────────┐
                       ▼                         ▼
               github-mcp :8765          mcp.deepwiki.com
               (local stand-in)         (real third party)
                       │
                       ▼  PATCH /repos/…/issues/3
                api.github.com
```

## What is governed

| Capability | MCP primitive | Where it runs | Decided by | Policy says |
| --- | --- | --- | --- | --- |
| `close_issue` | tool | `github-mcp`, a local stand-in for a server you don't own | the **gateway** | forbidden |
| `collaborators` | **resource** | `github-mcp` | the **gateway** | forbidden |
| `review_checklist` | **prompt** | `github-mcp` | nothing — prompts are not gated | n/a |
| `deepwiki_*` | tool | `mcp.deepwiki.com`, an actual third-party server | the **gateway** | one of each |
| `draft_summary` | none | this process | nothing — it never becomes MCP traffic | n/a |

Both MCP servers are CBAC-unaware. `github-mcp` exposes `close_issue` happily
and DeepWiki has never heard of this repo; the gateway is what governs them.

**Tools and resources go through the same gate.** A server exposes both, and
the client picks which to use — so a gate wired to `tools/call` alone is not a
gate: the same collaborator list is reachable as a resource. `CBACMiddleware`
implements `on_call_tool` and `on_read_resource`; only the exception type
differs between them.

**Prompts are deliberately not gated.** They are user-controlled in MCP's
model — the person picks one, the model does not reach for it — so checking a
prompt for drift from what the user asked is circular. A prompt also returns
text rather than acting, and the actions that follow it are tool calls, which
are gated. The residual risk, if you want it later: a server free to do real
work while building a template does that work ungated.

`draft_summary` is the boundary marker. A gateway sees MCP traffic, so an
in-process call is out of its reach by construction — which is the argument for
putting anything that matters behind the MCP boundary rather than inlining it.

## Why a gateway and not an interceptor

An interceptor inside the agent process is opt-in, and a security control the
developer can delete is not a control. So the demo splits the job:

- The agent **labels** its calls with the governance context — who is acting
  (`X-CBAC-Agent-Id`), what the user actually asked for
  (`X-CBAC-User-Intent`), and which request this is (`X-CBAC-Intent-Id`).
  `cbac_propagate` does only that. It calls no service, makes no decision, and
  cannot deny anything.
- The gateway **decides**. It reads the label, calls `/cbac/v2/authorize`, and
  either forwards the call upstream or returns a denial.

Delete `cbac_propagate` and calls still go through the gateway — they just
arrive with no request to grant anything from, so every one is denied.
`python agent_check.py` proves this: probe 2 is a client with the interceptor
removed.

One more thing falls out of the split: **credential custody.** The gateway is
also where the upstream credential lives. That is what makes bypass pointless
rather than merely inconvenient — routing around the gateway routes around the
only thing that can authenticate upstream. See the next section.

## Pointing straight at the MCP server

The obvious objection: the developer controls the URL, so what stops them
writing `http://127.0.0.1:8765/mcp/` and skipping the gateway?

Not the routing — routing is exactly what they control. What stops them is that
**there is nothing at that address to reach.** The gateway is the only service
with a public address; every upstream sits on a private network behind it.

This is a deployment property, not application code. The MCP servers carry no
check of their own:

| Situation | What stops the bypass | Bypass gets you |
| --- | --- | --- |
| MCP server you run | no route from the agent's network (NetworkPolicy, private subnet, mTLS) | connection refused |
| Third-party remote MCP (GitHub, Notion, Slack) | deny-all egress for agent workloads, and the gateway holds the OAuth token | no route, and a 401 if there were one |

Probe 3 of `python agent_check.py` makes the assumption visible: pointed
straight at `:8765`, the MCP server lists its tools to anyone who asks. It has
no opinion about who is calling, which is why the address has to be private.

The principle in both rows: **the agent's process can neither route to an
upstream nor authenticate to one.** Everything else — the interceptor, the
gateway's URL, the config — is a convenience the developer may break, and
breaking it costs them their own agent, not your policy.

⚠️ The demo runs everything on `127.0.0.1`, so all three ports are reachable
from anywhere on the machine. The network boundary is the one part of this you
configure rather than run.

### Trust boundary

`X-CBAC-User-Intent` is set by the agent's runtime, never by its model — the
LLM emits tool calls, not headers. It is the source of the request's grants:
the service parses it once, on the first call carrying a new `X-CBAC-Intent-Id`,
and denies any later call that pairs that id with a different query. A runtime
that lies about it can ask for anything the policy allows; a model steered by
injected content cannot change it.

`X-CBAC-Agent-Id` decides *whose policy applies*. The gateway reads it off the
call and falls back to `CBAC_AGENT_ID` when the call carries none — fine for a
demo, where the only client is the agent next door. A compromised client is
exactly what would lie about this header, so a gateway that must not take the
caller's word for it maps its authenticated principal — OAuth subject, mTLS
SAN, API key — to an agent id in the two lines where the hooks read the header.

## Files

```
agent.py                    the agent + the (CBAC-unaware) MCP server
agent_check.py              drives every path directly and asserts — no agent LLM
gateway.py                  the CBAC gateway — the enforcement point, fronting
                            github-mcp and the public mcp.deepwiki.com
service.py                  the CBAC decision service, started against the
                            policy files below instead of the Provenance Layer
upload_policy.py            uploads the policy document: the policy + the
                            gateway's tool schemas (v2)
policies/github-worker.md   the policy CBAC decides against
requirements.txt
.env.sample
```

## Run it

### 1. Start the decision service

From the repo root:

```bash
cd cbac_service
docker compose --env-file .env.dev up -d
uv run --env-file .env.dev alembic upgrade head
cd ..

HYBRID_SEARCH_ENABLED=true \
  uv run --env-file cbac_service/.env.dev python examples/github_agent_workflow/service.py
```

`service.py` is what makes this demo standalone. It starts the real
`cbac_service` app — same pipeline, same endpoints, same Postgres — with one
substitution: wherever the service would resolve a policy from the Provenance
Layer, which would need an API key and a registered actor id, it reads
`policies/{agent_id}.md` instead. (The v2 gate the gateway uses gets its policy
from `upload_policy.py`, which sends the file's text explicitly.) The
substitution lives in the demo, not in the service: run
`uvicorn cbac_service.main:app` directly and you get the production behaviour,
because that is the only policy source the service itself knows.

Run it from the **repo root** with the root venv — the service's ML
dependencies live there, not in this directory's venv.

`HYBRID_SEARCH_ENABLED=true` turns on Tier 2's BM25 fusion. The compose image
is built from `Dockerfile.postgres` and ships `pg_textsearch`, so it works;
set it to `false` if you point `DATABASE_URL` at a plain pgvector server.

The v2 gate also needs OpenFGA (started by the same `docker compose`) and a
parser model. The env file the service reads must set `OPENFGA_API_TOKEN`,
`CBAC_POLICY_KEY`, and `POLICY_PARSER_MODEL` / `QUERY_PARSER_MODEL` (default
`ollama:qwen3:14b`, which needs a local Ollama with that model pulled) — see
`cbac_service/.env.sample`. Pass it with `uv run --env-file …`.

The service has no policy for `github-worker` until step 4 uploads one, and
until then every call through the gateway is denied (`4009`).

### 2. Install and configure the example

```bash
cd examples/github_agent_workflow
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env    # then fill in an LLM key
```

### 3. Start the MCP server and the gateway

Two more terminals:

```bash
python agent.py serve    # the CBAC-unaware MCP server   :8765
python gateway.py        # the CBAC gateway              :8766
```

The gateway fronts two upstreams: the local `github-mcp` and the public
`mcp.deepwiki.com`, so the demo covers both "a server you run" and "a server
you have no control over at all". The agent only ever talks to `:8766` — in a
real deployment `:8765` has no route from the agent's network at all, which is
what "Pointing straight at the MCP server" is about.

### 4. Upload the policy

With the gateway up:

```bash
CBAC_POLICY_KEY=… python upload_policy.py
```

It builds the v2 policy document — the `Policy Instructions` section from
`policies/github-worker.md`, the `Tool Schema` section from every tool and
resource template listed through the gateway, names exactly as the gateway
serves them, `github_close_issue` included — and posts it to
`/cbac/v2/policies`. The
response is the parsed version. Read it before running anything:

- `catalog` — for each tool, which arguments are resources (the repository, the
  issue number) and which are free text (a question).
- `deny_rules` — each forbidden capability as a rule, with the policy sentence
  it came from.
- `rejected` — tools or rules that failed validation. A rejected tool is denied
  on every call.
- `unenforceable` — policy statements no rule can express.

The policy parse is one LLM call over every schema; on a laptop-class local
model it takes minutes.

### 5. Run it

```bash
# DENIED at the gateway — the call never reaches the MCP server
python agent.py "Close issue #3 in owner/repo"

# ALLOWED — relayed through the gateway to the real third-party server
python agent.py "What docs exist for modelcontextprotocol/python-sdk?"

# DENIED — the third party never sees it
python agent.py "Ask deepwiki how auth works in modelcontextprotocol/python-sdk"
```

The `[cbac]` line is the gateway's verdict for that call. The first request
asks to close an issue; the user did ask for it, so it is granted, and the
policy's deny rule for closing issues still blocks it (`4101`). The second is
granted from the request and relayed (`4301`). The third is denied: the policy
forbids asking an external AI service. The denial comes back as a readable
tool result rather than an exception, so the model can see it and report it.

The model must support tool calling. If you see an `[ai]` line containing
`{"name": …, "arguments": …}` with no `[call]` line, no tool was invoked and
CBAC was never consulted — that is a model problem, not a gateway problem.

### 6. Check the wiring, and the bypass, without an agent LLM

```bash
python agent_check.py
```

Every path an agent can take, driven directly and asserted on. Each probe opens
its own request (`cbac_context` with a fresh `intent_id`), so the service parses
each user request once:

| # | probe | expected |
| --- | --- | --- |
| 1 | `close_issue`, labelled | denied — the policy's deny rule |
| 2 | `close_issue`, client skipped `cbac_propagate` | denied — no intent (`4001`) |
| 3 | `tools/list` straight at `:8765` | answers — the server has no check of its own |
| 4 | `read_wiki_structure` on DeepWiki | relayed upstream, real content back |
| 5 | `ask_question` on DeepWiki | denied — forbidden by policy |
| 6 | the `collaborators` resource | denied |
| 7 | the `review_checklist` prompt | comes back — prompts are not gated |

plus `draft_summary`, in-process and never seen by the gateway.

Probe 2 is the point of the demo: an agent that never installed the CBAC wiring
still gets nothing through. Probe 3 shows what the MCP server does on its own,
which is nothing — the network keeps it out of reach, not the application.
`draft_summary` is the honest limit: CBAC governs the MCP boundary, so a
capability that never crosses it is never seen.

Probes 4 and 5 close the gap between the stand-in and the real thing. DeepWiki
was added to the gateway by URL alone — no decorator, no middleware, no
cooperation from its operator — and the policy still decides its calls.

## What crosses the wire

Per tool call, agent → gateway (MCP `tools/call`, plus three headers):

```
X-CBAC-Agent-Id:    github-worker
X-CBAC-User-Intent: Close%20issue%20%233%20in%20owner/repo
X-CBAC-Intent-Id:   9f1c…
```

Gateway → decision service, one POST per call:

```json
{
  "agent_id":    "github-worker",
  "intent_id":   "9f1c…",
  "user_query":  "Close issue #3 in owner/repo",
  "tool":        "github_close_issue",
  "arguments":   {"repo": "owner/repo", "number": 3},
  "callee_type": "mcp",
  "mcp_did":     "http://127.0.0.1:8765/mcp/"
}
```

Facts, not a sentence. On the first call of a request the service parses
`user_query` into exact grants; on every call it maps `arguments` to resources
through the uploaded catalog and checks them against those grants and the
policy's deny rules, then runs the deny-only semantic checks on any free-text
argument. `tool` is the name the gateway serves, server prefix included, which
is why the policy upload lists tools through the gateway.

Reaching a decision also folds the semantic scores into the (agent → callee)
trust score, service-side. There is no second call and nothing to report
afterwards: one HTTP round-trip per action.

## Other deployments

The gateway is the same process in each; only where it sits changes. What
forces traffic through it is covered above — this is just placement.

| Situation | Where the gateway goes |
| --- | --- |
| Third-party remote MCP (GitHub, Notion, Slack) | shared egress gateway in your network, holding the upstream OAuth token |
| On-prem MCP server you own | in front of it, or as a sidecar |
| Desktop agent spawning **stdio** MCP servers | a local proxy in the `mcpServers` entry: `cbac-mcp-proxy -- <original command>` |
| MCP server you own and want zero extra hops | the same middleware mounted directly on your `FastMCP` app |

The stdio case is the weak one: a local proxy in an MDM-managed config protects
against a confused agent and an unmodified machine, not against a developer
with a shell. For anything that matters, the credential has to live behind a
network gateway.

## Editing the policy

Change `policies/github-worker.md` and run `python upload_policy.py` again. Each
upload is a new policy version, parsed and validated from scratch, and the
service decides against it from the moment it activates — immediately, unless
the upload sets `"review_required": true`, in which case it waits until
approved with `POST /cbac/v2/policies/github-worker/approve`.

Write prohibitions as plain sentences, one capability per sentence. The parser
turns each into a deny rule and must quote the sentence it came from, so a rule
can always be traced back to the line that produced it. Limits it cannot
express as a rule — numbers, "only X" — come back under `unenforceable` rather
than being dropped.

The same text is also indexed as chunks for the policy-drift check on
free-text arguments. Inspect them after any edit:

```sql
SELECT chunk_type, chunk_text FROM policy_chunks
WHERE agent_id = 'github-worker' ORDER BY chunk_type, chunk_index;
```
