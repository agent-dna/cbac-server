# agentdna_local

Runs agentdna and the CBAC v2 gate on one machine, with no hosted services.
The examples in this folder import it; `agentdna/` itself is not changed.

```python
import agentdna_local

agentdna_local.install()  # before the first AgentDNA(...)
```

## What `install()` replaces

agentdna reaches four hosted services, each through one function or class.
`install()` swaps each at runtime (`patches.py`):

| Hosted | Called through | Local stand-in |
|---|---|---|
| Rubix chain (`chain-connector-2.rubix.net`) | `agentdna.core.Provenance` | `LocalProvenance` (`provenance.py`) |
| Registration (`/core/v1/register-user`, `-agent`, `-tool`) | `AgentDNA.__register_user` / `__register_agent` / `__register_tool` | rows in the `registry` table; an agent's policy file becomes its agent card |
| Admin whitelist (`agentdna-admin…`) | `agentdna.mcp.server.checks.request_agent_whitelist_check` | registered agents, narrowed by `AGENTDNA_LOCAL_WHITELIST` |
| Login (`user-auth.agentdna.io`) | `agentdna.core.originate` | a new run id, and `AGENTDNA_LOCAL_USER_EMAIL` (default `<login>@localhost`) |
| Auth-evidence post (only with `AGENTDNA_AUTH_EVIDENCE` on) | `agentdna.auth.sink.send` | rows in the `auth_evidence` table (the `AGENTDNA_AUTH_EVIDENCE_FILE` append is kept) |

It also points `get_default_config_dir` at the local directory, so the actor
registry (`actor_info.json`), keys and the store share one place.

## Identities and the shared store

- **Keys.** Each actor name gets an Ed25519 key, stored as
  `<dir>/keys/<name>-<hash>.pem`. The same name always gets the same key.
- **DIDs.** The DID is a CIDv1 (dag-pb, sha2-256, base32) of the raw public
  key, so it starts with `bafy` like a Rubix DID.
- **Store.** `<dir>/agentdna-local.sqlite3` holds every DID's public key, the
  registry, provenance cards and their states, and auth evidence. The agent,
  the MCP server and the CBAC service open the same file, which is how one
  process verifies another's signatures.

`<dir>` is `AGENTDNA_LOCAL_DIR`, default `~/.agentdna-local`. Every process of
one run must use the same value.

`LocalProvenance` keeps the original's methods and the errors callers depend
on: `AgentDNA.record` appends to an existing workflow card when creation fails
with `child nftId <id> already exists locally`, and that is the text it raises.

## The v2 `cbac_fn`

`make_cbac_fn(cbac_url=None, cbac_timeout=None)` (`cbac_v2.py`) returns an
agentdna `CbacFn` that calls `cbac.authorize_call`, the guard for
`POST /cbac/v2/authorize`:

```python
from agentdna.mcp.server.fastmcp import AgentDNAMCPMiddleware
from agentdna_local import make_cbac_fn

mcp.add_middleware(AgentDNAMCPMiddleware(dna, cbac_fn=make_cbac_fn()))
```

- **URL.** `cbac_url`, else `CBAC_URL`, else `http://127.0.0.1:8767`.
  Timeout: `CBAC_TIMEOUT`, else 600 s.
- **`agent_id`** is what the middleware sends: the actor of the workflow's
  latest envelope, the calling agent's DID. Upload each agent's policy under
  that DID.
- **`user_query`** is the root envelope's payload, the user's request.
- **`intent_id`** is `<workflow id>:<agent DID>`. v2 binds an intent id to one
  agent (a second agent gets `4002`), and every member of a multi-agent
  workflow carries the same workflow id, so each (request, agent) pair gets
  its own intent, parsed against that agent's own policy.
- **Verdict.** Only status `4301` and `4302` allow. Every other code — a deny,
  or one of the guard's `9100`s when no verdict was reached — is a refusal.

## Uploading a policy

`policy.py` builds the v2 document `cbac_service.parsers.split_policy_document`
reads:

```
---
Policy Instructions
<the instructions file>
---
Tool Schema
<JSON list of the MCP server's tools>
---
```

- `server_tools(mcp)` lists a FastMCP server's tools in process (needs
  `fastmcp`), so the catalog names match what the middleware reports.
- `actor_did(name)` is the DID an agent with that name has, or will have.
- `upload_policy(agent_id, document, trusted_data=…, review_required=…)` posts
  to `/cbac/v2/policies` with `X-CBAC-Key: $CBAC_POLICY_KEY`.
- `upload_agent_policy(name, instructions, mcp, trusted_data=…)` does all
  three for one agent. Each example's `upload_policy.py` calls it.

## The CBAC service runner

`service.py` starts `cbac_service.main.app` with `main._cbac` set to a `CBAC`
whose provenance is a `LocalProvenance`. Without it, `main._get_cbac()` would
build a Rubix `Provenance`. v2 never reads provenance; v1 reads an agent's
policy from its agent card, which registration writes to the store.

From the repo root, with the root venv:

```sh
uv run --env-file cbac_service/.env.dev python examples/agentdna_local/service.py
```

It needs Postgres and OpenFGA (`cd cbac_service && docker compose --env-file
.env.dev up -d`, then `uv run --env-file .env.dev alembic upgrade head`) and a
parser model; see `cbac_service/.env.sample`.

## Install and test

The examples install it from their `requirements.txt` (`-e ../agentdna_local`,
after `-e ../../agentdna` and `-e ../../cbac`). This directory is the package
itself (flat layout), so it imports the same way when `examples/` is on
`sys.path`.

The tests need nothing beyond the root venv and run with the root suite
(`uv run pytest`), or alone:

```sh
uv run pytest examples/agentdna_local/tests
```

They use a temporary `AGENTDNA_LOCAL_DIR`, stand up any HTTP stub on
`127.0.0.1`, and fail if anything opens a non-loopback connection. The tests
that need `fastmcp` — among them `test_e2e_relay.py`, a full user → agent →
Provenance Relay → MCP middleware → v2 `cbac_fn` run against a stub gate — are
skipped in the root venv; run them from an example's venv:

```sh
cd examples/github_repository_agent
.venv/bin/python -m pytest ../agentdna_local/tests
```

There the one test that needs `cbac_service` is skipped instead.
