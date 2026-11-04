# SQLite Analytics Agent

A CrewAI agent that analyses a local SQLite database through a FastMCP server. The database and its sample data are created automatically, and the agent can only reach the database through the MCP server's read-only tools.

## Architecture

```text
sqlite-analytics-agent -> MCPServerAdapter -> AgentDNA Provenance Relay -> FastMCP + AgentDNA middleware -> SQLite
```

- `manual.py` / `automated.py`: the user signs the request, the agent runs, and the user records the finished chain on the provenance layer (here, the local store).
- `agents/sqlite_agent/agent.py`: the agent. It opens a Provenance Relay session on the user's workflow, and points CrewAI's MCP client (`MCPServerAdapter`, in `mcp_client.py`) at the session's URL, so every tool call is signed as the agent.
- `mcp_server.py`: the SQLite MCP server, behind `AgentDNAMCPMiddleware`. Every tool call must come from a whitelisted agent, pass CoCA verification, and be allowed by the CBAC v2 gate (`POST /cbac/v2/authorize`).
- `upload_policy.py`: uploads the agent's CBAC v2 policy — [cbac_policies/sqlite_agent.md](cbac_policies/sqlite_agent.md) plus the MCP server's tool schemas — under the agent's DID, with the table names as trusted data.

Everything runs on one machine. AgentDNA's hosted services (the Rubix provenance chain, registration, the admin whitelist, the login server) are replaced at runtime by [`agentdna_local`](../agentdna_local/README.md), and CBAC is the service in this repo. Only the LLM you pick is reached over the network.

```py
PROVENANCE_RELAY = ProvenanceRelay({"sqlite": settings.sqlite_mcp_url})

with PROVENANCE_RELAY.session(SQLITE_AGENT, adna_workflow) as session:   # verifies the incoming workflow
    with sqlite_tools(session.url("sqlite")) as tools:                   # MCPServerAdapter on the session URL
        output = Crew(...).kickoff()

# session.workflows: the user's workflow plus every tool call
reply_workflow = SQLITE_AGENT.build(payload=str(output), previous_workflows=session.workflows)
```

If the MCP server refuses a tool call (whitelist, CoCA or CBAC), it records the refusal and the run stops with an error saying the run's chain has ended, along with the reason. Nothing is recorded on the agent's side.

## Prerequisites

- Python 3.11 or 3.12, and [uv](https://docs.astral.sh/uv/) for the CBAC service
- Docker, for Postgres and OpenFGA
- An LLM for the agent: Ollama, OpenAI (or an OpenAI-compatible service), or Gemini
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

**3. This example's venv**, from `examples/sqlite_analytics_agent`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set the LLM and CBAC_POLICY_KEY
```

**4. The MCP server** (terminal 2, this directory):

```sh
python mcp_server.py
```

**5. Upload the policy** (once, and again after editing it):

```sh
python upload_policy.py
```

It prints the agent's DID and the parsed version. Read each reply's `catalog` (which arguments are resources) and
`deny_rules` before running the agent. Add `--review-required` to hold a
version until `POST /cbac/v2/policies/{agent_did}/approve` activates it.

**6. The agent** (terminal 3, this directory):

```sh
python manual.py "Explore the analytics database's tables and their schema, then identify high-revenue products and unusual pending-order trends."
```

The database (`SQLITE_DATABASE_PATH`) is created with its schema and sample data on the MCP server's first tool call. To create it beforehand, run `python3 seed_data.py` from this directory; it is safe to run more than once.

### Environment variables

**LLM**

| Variable | Description | Default |
|---|---|---|
| `LLM_BACKEND` | `ollama`, `openai` or `gemini`. | `ollama` |
| `LLM_TEMPERATURE` | Sampling temperature. | `0` |
| `OLLAMA_HOST` | Ollama server URL. | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Ollama model; pull it first (`ollama pull <model>`). | `llama3.1` |
| `OPENAI_API_KEY` | OpenAI API key. Required for `openai`. | |
| `OPENAI_MODEL` | OpenAI model. | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Base URL of the OpenAI(-compatible) API. Required for `openai`; `/v1` is added when missing, so `https://llm.agentdna.io` becomes `https://llm.agentdna.io/v1`. | |
| `GOOGLE_API_KEY` | Gemini API key, from Google AI Studio. Required for `gemini`. | |
| `GEMINI_MODEL` | Gemini model. | `gemini-3.5-flash-lite` |

**SQLite and MCP**

| Variable | Description | Default |
|---|---|---|
| `SQLITE_DATABASE_PATH` | The SQLite database file. It is created when missing. | `data/analytics.db` |
| `SQLITE_MCP_URL` | The URL the agent reaches the MCP server at. It must match `MCP_HOST` and `MCP_PORT`. | `http://127.0.0.1:8544/mcp` |
| `MCP_HOST` | Interface the MCP server listens on. | `127.0.0.1` |
| `MCP_PORT` | Port the MCP server listens on. | `8544` |

**AgentDNA**

| Variable | Description | Default |
|---|---|---|
| `AGENTDNA_LOCAL_DIR` | Keys and the shared SQLite store. Same value for every process. | `~/.agentdna-local` |
| `AGENTDNA_LOCAL_WHITELIST` | Comma-separated agent names or DIDs the MCP server accepts. Empty: every registered agent. | |
| `USER_NAME` | Name of the user identity that signs requests and records results. | `local-user@example.local` |
| `SQLITE_AGENT_NAME` | Name of the agent's identity. Its DID is the CBAC `agent_id`. | `sqlite-analytics-agent` |
| `MCP_SERVER_NAME` | Name of the MCP server's identity. | `sqlite-analytics-mcp` |
| `CBAC_URL` | The CBAC service the MCP server asks about each tool call. | `http://127.0.0.1:8767` |
| `CBAC_TIMEOUT` | Seconds per authorize call. | `600` |
| `CBAC_POLICY_KEY` | The service's policy key; only `upload_policy.py` reads it. | |

**Authentication evidence (optional)**

| Variable | Description | Default |
|---|---|---|
| `SQLITE_MCP_HEADERS` | HTTP headers the agent sends to the MCP server, as a JSON object, for example `{"Authorization": "Bearer <token>"}`. They give a run a credential for authentication evidence to observe; the SQLite tools don't need them. `python3 make_token.py <user>` prints a test token. | |
| `AGENTDNA_AUTH_EVIDENCE`, `AGENTDNA_AUTH_EVIDENCE_FILE`, `AGENTDNA_FINGERPRINT_KEY` | Switch on AgentDNA's authentication evidence, which is off by default. See [Authentication evidence](../../agentdna/README.md#authentication-evidence) in the agentdna README. | |

## More ways to run

`manual.py` prints a readable report. Add `--json` after the prompt to print the structured execution event instead. On Windows, use `python` instead of `python3`.

### Example prompts

```sh
# Top products by revenue, and orders stuck in pending
python3 manual.py "Explore the analytics database's tables and their schema, then identify high-revenue products and unusual pending-order trends."

# Customers ranked by spend
python3 manual.py "Explore the analytics database's tables and their schema, then compare customers by total order value and flag any with unusually high spending."

# Revenue by product category
python3 manual.py "Explore the analytics database's tables and their schema, then summarise revenue by product category and recommend which categories need a closer look."
```

Each prompt names exploring the tables and schema because the agent always starts there. The CBAC v2 gate parses the prompt once into grants of (tool, table): a prompt that does not ask to explore the tables is not granted `sqlite_list_tables` or `sqlite_describe_table`, and those calls are refused (`4103`). The SQL text of `sqlite_query` is free text, and passes only the deny-only semantic checks; the server's own validation keeps it to one `SELECT`.

### Automated

`automated.py` runs continuously until you stop it (Ctrl+C): each run picks one of ten built-in prompts at random, prints the JSON execution event, and records the chain.

```sh
# Terminal 1
python3 mcp_server.py

# Terminal 2
python3 automated.py
```

`scripts/run.sh` (and `scripts/run.ps1` / `scripts/run.cmd` on Windows) start the MCP server and run `automated.py`; stopping it stops the server too. Steps 1, 2 and 5 still come first.

## MCP tools

`sqlite_list_tables`, `sqlite_describe_table` and `sqlite_query`. The server validates table names, accepts a single `SELECT` statement only, opens the database read-only for queries, and rejects DDL, writes, pragmas, attachments and multiple statements. The schema isn't in the prompt: the agent discovers it through the tools.

The agent's skills and restrictions are in [agents/sqlite_agent/SKILLS.md](agents/sqlite_agent/SKILLS.md), which is the policy registered with its AgentDNA identity (its agent card, which the v1 gate reads). The v2 gate decides from [cbac_policies/sqlite_agent.md](cbac_policies/sqlite_agent.md), uploaded by `upload_policy.py`.

## Tests

From this directory, in its venv: `python -m pytest tests -q`. The tests check the database's automatic creation, that destructive queries are rejected, and the CBAC v2 wiring. No service, network or API key is needed: the tests use a temporary `AGENTDNA_LOCAL_DIR`.
