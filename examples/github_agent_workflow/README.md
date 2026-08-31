# Guarded GitHub Agent

A demo of CBAC enforced where the agent developer **cannot switch it off**: a
gateway between the MCP client and the MCP server.

Every MCP tool call the agent makes is authorized at that gateway. The agent
process holds no policy, makes no decision, and cannot opt out.

No AgentDNA, no provenance chain, no identity registration — the agent's policy
is a markdown file the service reads from disk.

```
      "Close issue #3…"                    "Draft a summary…"
              │                                     │
              └─────────────────┬───────────────────┘
                                ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │ agent process                                                      │
 │ cbac_context(agent_id, user_intent) — opened once, ambient         │
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
                    │  POST /cbac/v1/authorize
                    ▼
  ┌───────────────────────────────────┐
  │   CBAC decision service   :8767   │
  │       allow · deny · advise       │
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
  (`X-CBAC-Agent-Id`) and what the user actually asked for
  (`X-CBAC-User-Intent`). `cbac_propagate` does only that. It calls no service,
  makes no decision, and cannot deny anything.
- The gateway **decides**. It reads the label, calls `/cbac/v1/authorize`, and
  either forwards the call upstream or returns a denial.

Delete `cbac_propagate` and calls still go through the gateway — they just
arrive with nothing to authorize, and the gateway denies them (fail-closed,
`CBAC_REQUIRE_CONTEXT=true`). `python agent_check.py` proves this: probe 3 is a
client with the interceptor removed.

Two more things fall out of the split:

- **Better intent prose.** The gateway holds the upstream tool's real
  description, so it asks CBAC about *"the agent wants to close an existing
  GitHub issue, with repo = owner/repo, number = 3"*. A client-side interceptor
  only sees the tool name and can manage no better than *"close issue"* — and
  NLI scores the two very differently.
- **Credential custody.** The gateway is also where the upstream credential
  lives. That is what makes bypass pointless rather than merely inconvenient —
  routing around the gateway routes around the only thing that can
  authenticate upstream. See the next section.

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

`X-CBAC-User-Intent` is client-supplied and unverifiable — only the client
knows what the user typed. CBAC treats it as *evidence* (a drift signal), never
as identity, so a client that lies about it still gets its action scored
against the policy.

`X-CBAC-Agent-Id` decides *whose policy applies*, which is exactly what a
compromised client would lie about. The gateway therefore **pins** it
(`CBAC_AGENT_ID`) rather than reading the header. A multi-tenant gateway maps
its authenticated principal — OAuth subject, mTLS SAN, API key — to an agent id
at the same spot.

## Files

```
agent.py                    the agent + the (CBAC-unaware) MCP server
agent_check.py              drives every path directly and asserts — no LLM
gateway.py                  the CBAC gateway — the enforcement point, fronting
                            github-mcp and the public mcp.deepwiki.com
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

`HYBRID_SEARCH_ENABLED=true` turns on Tier 2's BM25 fusion. The compose image
is built from `Dockerfile.postgres` and ships `pg_textsearch`, so it works;
set it to `false` if you point `DATABASE_URL` at a plain pgvector server.

Check it decides correctly before involving an LLM:

```bash
curl -s -XPOST localhost:8767/cbac/v1/authorize \
  -H 'content-type: application/json' \
  -d '{"agent_id":"github-worker","intended_action":"The agent wants to get a list of documentation topics for a GitHub repository, with repoName = modelcontextprotocol/python-sdk."}'
# {"success":true,"message":"Tier 1 cosine gap 0.121 > 0.08 ...",
#  "data":{"decision":"allow","error_code":3201}}

curl -s -XPOST localhost:8767/cbac/v1/authorize \
  -H 'content-type: application/json' \
  -d '{"agent_id":"github-worker","intended_action":"The agent wants to close an existing GitHub issue, with repo = owner/repo, number = 3."}'
# {"success":true,"message":"Tier 1 cosine gap -0.182 < -0.08 ...",
#  "data":{"decision":"deny","error_code":3202}}
```

Note `success` is `true` on the deny: it reports that the call was *processed*,
and the verdict is `data.decision`. The same verdict is on the
`X-CBAC-Decision` header if you would rather not parse a body.

The first call is slow — the service loads three transformer models and embeds
the policy. Later calls hit the cached embeddings in Postgres.

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

### 4. Run it

```bash
# DENIED at the gateway — the call never reaches the MCP server
python agent.py "Close issue #3 in owner/repo"

# ALLOWED — relayed through the gateway to the real third-party server
python agent.py "What docs exist for modelcontextprotocol/python-sdk?"

# DENIED — the third party never sees it
python agent.py "Ask deepwiki how auth works in modelcontextprotocol/python-sdk"
```

The `[cbac]` line is the gateway's verdict for that call. The denial comes back
as a readable tool result rather than an exception, so the model can see it and
report it:

```
[call] close_issue({'repo': 'owner/repo', 'number': 3})
[cbac] DENIED — Tier 1 cosine gap -0.182 < -0.08 (intent closer to forbidden than allowed policy)
[tool] {"status": "denied", "error": "Tier 1 cosine gap -0.182 < -0.08 (...)"}
[ai]   I'm unable to close the issue due to a policy restriction.
```

The model must support tool calling. If you see an `[ai]` line containing
`{"name": …, "arguments": …}` with no `[call]` line, no tool was invoked and
CBAC was never consulted — that is a model problem, not a gateway problem.

### 5. Check the wiring, and the bypass, without an LLM

```bash
python agent_check.py
```

Every path an agent can take, driven directly and asserted on:

```
[gateway      close_issue        ] {"status": "denied", "error": "Tier 1 cosine gap -0.182 < -0.08 (...)"}
[gateway      close_issue        ] {"status": "denied", "error": "CBAC: call arrived with no governance
                                    context (X-CBAC-User-Intent missing) …"}
                                   ^ client skipped cbac_propagate
[direct :8765 tools/list         ] ['close_issue']
                                   ^ no check of its own — keeping this
                                     port unroutable is the deployment's job
[in-process   draft_summary      ] {'status': 'ok', 'repo': 'owner/repo', 'summary': 'fixed the parser'}
                                   ^ never became MCP traffic — ungoverned
[gateway      read_wiki_structure] [{'type': 'text', 'text': 'Available pages for modelcontext…
[gateway      ask_question       ] {"status": "denied", "error": "Tier 1 cosine gap -0.103 < -0.08 (...)"}
                                   ^ real third-party server, same policy
[gateway      collaborators      ] {"status": "denied", "error": "Tier 1 cosine gap -0.113 < -0.08 (...)"}
[gateway      review_checklist   ] text='Review the open pull requests in owner/repo. For each one…'
                                   ^ a resource, gated; a prompt, not

OK — permitted calls relayed, forbidden ones blocked at the gateway:
     by policy, and by the fail-closed check on a call that arrived
     with no governance context. Tools and resources alike, and the
     third-party server on the same policy as the local one.
```

Probe 2 is the point of the demo: an agent that never installed the CBAC wiring
still gets nothing through. Probe 3 shows what the MCP server does on its own,
which is nothing — the network keeps it out of reach, not the application.
`draft_summary` is the honest limit: CBAC governs the MCP boundary, so a
capability that never crosses it is never seen.

Probes 5 and 6 close the gap between the stand-in and the real thing. DeepWiki
was added to the gateway by URL alone — no decorator, no middleware, no
cooperation from its operator, who has never heard of this repo — and the
policy still decides its calls: `read_wiki_structure` is relayed and comes back
with real content, `ask_question` dies at the gateway.

DeepWiki's third tool, `read_wiki_contents`, is deliberately left unnamed by
the policy. It escalates past Tier 1 and comes back `advise` — the gray zone,
which is what an unconfigured Tier 3 returns. Name it in the policy and it
resolves at Tier 1 like the others.

## What crosses the wire

Per tool call, agent → gateway (MCP `tools/call`, plus two headers):

```
X-CBAC-Agent-Id:    github-worker
X-CBAC-User-Intent: Close%20issue%20%233%20in%20owner/repo
```

Gateway → decision service, one POST per call:

```json
{
  "agent_id":           "github-worker",
  "callee_name":        "github_close_issue",
  "callee_type":        "mcp",
  "callee_description": "Close an existing GitHub issue.",
  "arguments":          {"repo": "owner/repo", "number": "3"},
  "user_intent":        "Close issue #3 in owner/repo"
}
```

Facts, not a sentence. The service turns them into the prose its scorers
actually see — *"The agent wants to close an existing GitHub issue, with repo =
owner/repo, number = 3."* — because NLI scores prose far better than
`close_issue(repo=...)` call syntax, and because the wording is therefore part
of the decision. Rendering it service-side is what stops two enforcement points
from wording the same call differently and getting different verdicts; it also
means a gateway written in another language has nothing to port.

`callee_description` is the tool's own description, which the gateway looks up
locally — a client-side interceptor only has what was on the wire, and without
the description the de-snaked name lands in the verb slot instead. It survives
the gateway's server-name prefixing, so `github_close_issue` still reads as
*"close an existing GitHub issue"*; the prefix only shows up in `callee_name`,
which is what the trust edge is keyed on — one edge per (agent, upstream tool),
which is what you want.

A caller that phrases the action itself sends `intended_action` instead, and
the service scores that verbatim.

`user_intent` is the root request, which lets the service also check the tool
call for drift away from what the user actually asked for.

Reaching a decision also folds the component scores into the (agent → callee)
trust score, service-side. There is no second call and nothing to report
afterwards: one HTTP round-trip per action, and a denied call still leaves a
record of having been denied.

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
