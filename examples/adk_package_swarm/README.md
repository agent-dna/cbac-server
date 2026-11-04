# ADK Package Swarm

A swarm of four Google ADK agents does due diligence on Python packages: should we adopt this package, is this pinned version safe, what should we upgrade to? Gemini is the LLM, and the agents' tools come from a FastMCP server that uses public data needing no API keys. Every member has its own AgentDNA identity, every MCP tool call is signed as the member that made it, and every hand-off is a signed hop: the run ends as one verifiable chain of custody.

## Architecture

```text
              ┌──────────── hand-offs (any to any) ────────────┐
              ▼                                                 ▼
coordinator ⇄ package_agent ⇄ security_agent ⇄ ecosystem_agent      Google ADK + Gemini
                    │               │                 │
                    └──────── AgentDNA Provenance Relay ────┘       one session per turn
                                    │
                    package-intelligence MCP server                 FastMCP 3.4.7 + AgentDNA middleware + CBAC v2
                                    │
                    PyPI · pypistats · OSV.dev · deps.dev (OpenSSF Scorecard)
```

**The swarm.** `coordinator` takes the request. Any member can hand control to any other with ADK's `transfer_to_agent`: ADK allows transfers between a parent, its children, and the children as peers. Each specialist does its part with its own tools, then hands off to whichever specialist's part is still missing, or back to `coordinator`, which writes the final report with a verdict: adopt, adopt with care, or avoid.

| Agent | Does | MCP tools | Data source (no API key) |
|---|---|---|---|
| `coordinator` | Routes the work, writes the report | none | none |
| `package_agent` | Versions, licence, Python support, release activity, downloads | `package_info`, `release_history`, `download_stats` | PyPI JSON API, pypistats.org |
| `security_agent` | Known vulnerabilities and the versions that fix them | `find_vulnerabilities`, `vulnerability_details` | OSV.dev |
| `ecosystem_agent` | Dependency tree, repository health | `dependencies`, `project_health` | deps.dev, OpenSSF Scorecard |

Each specialist's MCP tools are filtered to its own (`swarm.py`).

## How AgentDNA fits in

All of the agent side is in `swarm.py`. The server side is `AgentDNAMCPMiddleware` in `mcp_server.py`: it verifies each tool call's chain, checks that the calling agent is whitelisted, has the CBAC v2 gate (`POST /cbac/v2/authorize`, through `agentdna_local.make_cbac_fn`) decide whether the user's request grants the call under the calling member's policy, and signs the answer.

Everything runs on one machine. AgentDNA's hosted services (the Rubix provenance chain, registration, the admin whitelist, the login server) are replaced at runtime by [`agentdna_local`](../agentdna_local/README.md), and CBAC is the service in this repo. Only Gemini and the tools' public data sources are reached over the network.

- **A CBAC policy per specialist.** Each specialist signs its own tool calls, so the CBAC `agent_id` is that member's DID, and each has its own v2 policy in `cbac_policies/`, uploaded by `upload_policy.py`. Each policy allows only the member's own tools, for the packages and versions the user's request names. All members share one workflow id, so the gate sees one intent per (request, member).

- **An identity per member.** `identities()` creates the four members' `AgentDNA` instances, each registered with its policy from `policies/`. `manual.py` has the user sign the request, and records the finished chain.
- **Each turn is a relay session.** `Custody.before_agent` is every member's `before_agent_callback`, so it runs whenever a member takes control. It opens a `ProvenanceRelay` session for that member on the workflow it just received: the user's request for the coordinator's first turn, a hand-off after that. Opening the session verifies the workflow, and the member's tool calls through it are signed as that member, on top of it.
- **Hand-offs are signed.** When control passes from A to B, A's session closes and A signs the hand-off on top of what its turn produced, addressed to B. B's session then starts from that hand-off.
- **The answer** is signed by the member that gave it, on top of its last turn.

The chain is one line, from the user's request through every hand-off, tool call and answer, exactly as the swarm ran; `tests/test_swarm.py` checks it envelope by envelope.

### Why the extra `Custody` and `SessionToolset` classes?

In the simplest AgentDNA setup, your own code calls each agent, so it can hand each one the workflow it should build on:

```py
async with PROVENANCE_RELAY.session(AGENT, workflow_it_received) as session:
    ...  # the agent's MCP client uses session.url(...)
```

A swarm doesn't work that way. ADK decides when control passes from one agent to another (`transfer_to_agent`), inside its own runner. There's no line of your code between "A hands off" and "B starts" where you could give B a workflow. That causes two problems, and each class solves one:

1. **Nobody signs the hand-off.** `Custody` hooks into ADK's `before_agent_callback`, which ADK calls every time an agent takes control. There it does what your own code would do at a hand-off: it closes A's session, has A sign the hand-off, and opens B's session on it.
2. **B's MCP client is stuck on an old URL.** Each relay session has its own URL, and B gets a new session on every turn. But ADK creates each agent's `McpToolset`, URL included, once, when the agents are built, before anything has been handed off. `SessionToolset` stands in for it: on each turn, `Custody` points it at the new session's URL, and it gives ADK the tools from there. It also turns off ADK's caching of tools for the whole run, which would otherwise keep B on its first, closed session.

One hand-off, step by step:

```text
ADK: package_agent transfers to security_agent
 └─ Custody.before_agent("security_agent")
      1. package_agent's MCP client disconnects, and its session closes: the chain of its turn
      2. package_agent signs the hand-off on top of that chain, addressed to security_agent
      3. a session opens for security_agent on the hand-off, which verifies it
      4. security_agent's SessionToolset switches to that session's URL
ADK: security_agent runs; each of its tool calls is signed on top of the hand-off
```

**Would another framework need them?** It depends on two things: who moves control between agents, and when the MCP client is created.

| If... | then... |
|---|---|
| your code moves control (an orchestrator calling workers, explicit graph nodes) | you need neither class: open a session per agent on the workflow you pass it |
| the framework moves control (ADK, OpenAI Agents SDK hand-offs, LangGraph swarm, AutoGen group chats, CrewAI delegation) | you need a `Custody`-like hook at each hand-off, in whatever callback the framework offers |
| the MCP client is created once, when the agent is built (as ADK's `McpToolset` is) | you need a `SessionToolset`-like stand-in that follows each turn's session URL |
| the MCP client is created inside each turn | you don't: build it on the session's URL |

## Files

| File | What it is |
|---|---|
| `package_tools.py` | The tools, as plain async functions; no MCP or ADK code |
| `mcp_server.py` | Serves the tools over MCP (Streamable HTTP), behind AgentDNA's middleware and the CBAC v2 gate |
| `swarm.py` | The agents, their instructions and tools, their AgentDNA identities and chain of custody, and `run_swarm()` |
| `policies/` | Each member's AgentDNA policy, registered with its identity (its agent card, which the v1 gate reads) |
| `cbac_policies/` | Each specialist's CBAC v2 policy instructions |
| `upload_policy.py` | Uploads the three specialists' v2 policies, with the server's tool schemas, under their DIDs |
| `manual.py` | Runs one request from the command line |
| `automated.py` | Runs built-in requests in a loop |

## Prerequisites

- Python 3.10–3.12, and [uv](https://docs.astral.sh/uv/) for the CBAC service
- Docker, for Postgres and OpenFGA
- A Gemini API key, for the swarm
- A parser model for the CBAC v2 gate: Ollama with `qwen3:14b` by default, or the Claude API (see `cbac_service/.env.sample`)

No AgentDNA account or API key: identities, signatures and provenance cards are local (see [agentdna_local](../agentdna_local/README.md)).

## Run, in order

Paths below are from the repo root unless a step says otherwise. Every process
must see the same `AGENTDNA_LOCAL_DIR` (default `~/.agentdna-local`): that is
how the MCP server and the CBAC service know each agent's DID and public key.

**1. Postgres and OpenFGA.** Copy `cbac_service/.env.sample` to
`cbac_service/.env.dev` and fill it in: `DATABASE_URL`, `OPENFGA_API_TOKEN`,
`OPENFGA_STORE_NAME`, `CBAC_POLICY_KEY`, and the parser models.

```sh
cd cbac_service
docker compose --env-file .env.dev up -d
uv run --env-file .env.dev alembic upgrade head
cd ..
```

**2. The CBAC service**, with the root venv (`uv sync --locked` first):

```sh
uv run --env-file cbac_service/.env.dev python examples/agentdna_local/service.py
```

**3. This example's venv**, from `examples/adk_package_swarm`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set GOOGLE_API_KEY and CBAC_POLICY_KEY
```

**4. The MCP server** (terminal 2, this directory):

```sh
python mcp_server.py
```

**5. Upload the policy** (once, and again after editing it):

```sh
python upload_policy.py
```

It uploads three policies, one per specialist, and prints each member's DID and parsed version. Read each reply's `catalog` (which arguments are resources) and
`deny_rules` before running the agent. Add `--review-required` to hold a
version until `POST /cbac/v2/policies/{agent_did}/approve` activates it.

**6. The agent** (terminal 3, this directory):

```sh
python manual.py "We're about to pin requests 2.31.0 in production. Is that safe, and should we upgrade?"
```

### Environment variables

**Gemini**

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini API key, from Google AI Studio. Required. | |
| `GOOGLE_GENAI_USE_VERTEXAI` | `FALSE` to use the Gemini API key rather than Vertex AI. | `FALSE` in `.env.example` |
| `GEMINI_MODEL` | Gemini model every member uses. | `gemini-3.5-flash-lite` |

**Package-intelligence MCP server**

| Variable | Description | Default |
|---|---|---|
| `PACKAGE_MCP_URL` | The URL the members reach the MCP server at. It must match `MCP_HOST` and `MCP_PORT`. | `http://127.0.0.1:8680/mcp` |
| `MCP_HOST` | Interface the MCP server listens on. | `127.0.0.1` |
| `MCP_PORT` | Port the MCP server listens on. | `8680` |
| `HTTP_TIMEOUT_SECONDS` | Timeout for the tools' requests to PyPI, OSV.dev and deps.dev, and for the members' requests to the MCP server. | `30` |

**AgentDNA (local) and CBAC**

| Variable | Description | Default |
|---|---|---|
| `AGENTDNA_LOCAL_DIR` | Keys and the shared SQLite store. Same value for every process. | `~/.agentdna-local` |
| `AGENTDNA_LOCAL_WHITELIST` | Comma-separated agent names or DIDs the MCP server accepts. Empty: every registered agent. | |
| `USER_NAME` | Name of the user identity that signs requests and records results. | `local-user@example.local` |
| `COORDINATOR_AGENT_NAME` | Name of the coordinator's identity. | `package-coordinator` |
| `PACKAGE_AGENT_NAME` | Name of `package_agent`'s identity. | `package-agent` |
| `SECURITY_AGENT_NAME` | Name of `security_agent`'s identity. | `package-security-agent` |
| `ECOSYSTEM_AGENT_NAME` | Name of `ecosystem_agent`'s identity. | `package-ecosystem-agent` |
| `MCP_SERVER_NAME` | Name of the MCP server's identity. | `package-intelligence-mcp` |
| `CBAC_URL` | The CBAC service the MCP server asks about each tool call. | `http://127.0.0.1:8767` |
| `CBAC_TIMEOUT` | Seconds per authorize call. | `600` |
| `CBAC_POLICY_KEY` | The service's policy key; only `upload_policy.py` reads it. | |

**Authentication evidence (optional)**

`AGENTDNA_AUTH_EVIDENCE`, `AGENTDNA_AUTH_EVIDENCE_FILE` and `AGENTDNA_FINGERPRINT_KEY` switch on AgentDNA's authentication evidence, which is off by default. See [Authentication evidence](../../agentdna/README.md#authentication-evidence) in the agentdna README.

## More ways to run

`manual.py` prints:
- the hand-offs the swarm made, and the tool calls each agent made;
- which agent answered;
- the report.

Add `--json` to print a structured result instead. `automated.py` runs continuously until you stop it (Ctrl+C): each run picks one of five built-in requests at random, prints the JSON result, and records the chain.

### Example prompts

```sh
# An old pinned version with known vulnerabilities: what to upgrade to
python3 manual.py "Is pyyaml 5.3 still OK to use? What would we need to upgrade to?"

# No version given: the latest release is not in the request, so per-version calls are refused
python3 manual.py "Due diligence on httpx: is the latest version safe to adopt, and how healthy is the project?"

# Two packages compared side by side
python3 manual.py "Compare flask and fastapi for a new internal service: maintenance, vulnerabilities and dependencies."

# A narrower question, mostly for one specialist
python3 manual.py "Our team wants to add pydantic. How many dependencies does it pull in, and is the project well maintained?"
```

The CBAC v2 gate parses the request once per specialist into grants of (tool, package or version), copied from the request's own words. A specialist calling a tool its policy forbids, or asking about a package or version the request does not name, is refused (`4101` for a tool its policy forbids, `4103` for a value the request does not grant) and the run's chain ends. That is why a request that gives no version is granted no per-version call: name the version to have it checked.

A run makes a dozen or so Gemini calls. The agents retry Gemini's transient 429 and 503 errors with backoff, but a free-tier key can still run out of quota; a different `GEMINI_MODEL` has its own.

`config.py` silences three kinds of ADK notice that don't affect the run: its announcements of the experimental features it uses internally, its suggestion to configure context caching, and its failed probe for Google Cloud mTLS on each MCP connection. Every other ADK warning still shows.

## Tests

From this directory, in its venv: `python -m pytest tests -q` needs no service, network or keys (it uses a temporary `AGENTDNA_LOCAL_DIR`). It checks:
- the tools' input validation, and how they parse each API's responses;
- the swarm's wiring: every member can reach every other, and a request driven by a scripted model flows through ADK's real runner, transfers and tool calls, from the coordinator through each specialist and back;
- the chain of custody of that run, with stand-in identities and relay: each turn is a session on the workflow that member received, every hand-off is signed by the sender and addressed to the receiver, and the chain records exactly the flow that happened.
- the CBAC v2 wiring: every specialist has a policy that names its own tools, and each member gets its own intent on the shared workflow.
