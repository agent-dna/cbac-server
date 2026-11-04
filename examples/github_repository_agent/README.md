# GitHub Repository Agent

A LangGraph ReAct agent that analyses one GitHub repository through a local FastMCP server. The agent has no direct GitHub API access: the MCP server holds the GitHub token and exposes read-only tools.

## Architecture

```text
github-repository-agent -> langchain_mcp_adapters -> AgentDNA Provenance Relay -> FastMCP + AgentDNA middleware -> GitHub REST API
```

- `manual.py` / `automated.py`: the user signs the request, the agent runs, and the user records the finished chain on the provenance layer (here, the local store).
- `agents/github_agent/agent.py`: the agent. It opens a Provenance Relay session on the user's workflow, and points `langchain_mcp_adapters` at the session's URL, so every tool call is signed as the agent.
- `mcp_server.py`: the GitHub MCP server, behind `AgentDNAMCPMiddleware`. Every tool call must come from a whitelisted agent, pass CoCA verification, and be allowed by the CBAC v2 gate (`POST /cbac/v2/authorize`).
- `upload_policy.py`: uploads the agent's CBAC v2 policy — [cbac_policies/github_agent.md](cbac_policies/github_agent.md) plus the MCP server's tool schemas — under the agent's DID, with the configured repository as trusted data.

Everything runs on one machine. AgentDNA's hosted services (the Rubix provenance chain, registration, the admin whitelist, the login server) are replaced at runtime by [`agentdna_local`](../agentdna_local/README.md), and CBAC is the service in this repo. Only the LLM you pick and the GitHub API are reached over the network.

```py
PROVENANCE_RELAY = ProvenanceRelay({"github": settings.github_mcp_url})

async with PROVENANCE_RELAY.session(GITHUB_AGENT, adna_workflow) as session:   # verifies the incoming workflow
    tools = await load_tools(session.url("github"))
    result = await workflow.ainvoke(...)

# session.workflows: the user's workflow plus every tool call
reply_workflow = GITHUB_AGENT.build(payload=..., previous_workflows=session.workflows)
```

If the MCP server refuses a tool call (whitelist, CoCA or CBAC), it records the refusal and the run stops with an error saying the run's chain has ended, along with the reason. Nothing is recorded on the agent's side.

## Prerequisites

- Python 3.11 or 3.12, and [uv](https://docs.astral.sh/uv/) for the CBAC service
- Docker, for Postgres and OpenFGA
- An LLM for the agent: Ollama, OpenAI (or an OpenAI-compatible service), or Gemini
- A parser model for the CBAC v2 gate: Ollama with `qwen3:14b` by default, or the Claude API (see `cbac_service/.env.sample`)
- A GitHub token with read access to the repository

## Run, in order

Paths below are from the repo root unless a step says otherwise. Every process
must see the same `AGENTDNA_LOCAL_DIR` (default `~/.agentdna-local`): that is
how the MCP server and the CBAC service know the agent's DID and public key.

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

**3. This example's venv**, from `examples/github_repository_agent`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set the LLM, GITHUB_* and CBAC_POLICY_KEY
```

**4. The MCP server** (terminal 2, this directory):

```sh
python mcp_server.py
```

**5. Upload the policy** (once, and again after editing it):

```sh
python upload_policy.py
```

It prints the agent's DID and the parsed version. Read `catalog` (which
arguments are resources) and `deny_rules` before running the agent. Add
`--review-required` to hold the version until
`POST /cbac/v2/policies/{agent_did}/approve` activates it.

**6. The agent** (terminal 3, this directory):

```sh
python manual.py "Review the repository's open issues and pull requests."
```

### Environment variables

**LLM**

| Variable | Description | Default |
|---|---|---|
| `LLM_BACKEND` | `ollama`, `openai` or `gemini`. | `ollama` |
| `LLM_TEMPERATURE` | Sampling temperature. | `0` |
| `OLLAMA_HOST` | Ollama server URL. | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Ollama model; pull it first (`ollama pull <model>`). | `llama3.1` |
| `OPENAI_API_KEY` | OpenAI API key. | |
| `OPENAI_MODEL` | OpenAI model. | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Base URL of an OpenAI-compatible service. Optional. | OpenAI's |
| `GOOGLE_API_KEY` | Gemini API key, from Google AI Studio. | |
| `GEMINI_MODEL` | Gemini model. | `gemini-3.5-flash-lite` |

**GitHub and MCP**

| Variable | Description | Default |
|---|---|---|
| `GITHUB_TOKEN` | GitHub token with read access to the repository. Only the MCP server uses it. Required. | |
| `GITHUB_REPOSITORY_OWNER` | Owner of the repository to analyse. Required. | |
| `GITHUB_REPOSITORY_NAME` | Name of the repository to analyse. Required. | |
| `GITHUB_API_URL` | GitHub REST API URL; change it for GitHub Enterprise Server. | `https://api.github.com` |
| `GITHUB_MCP_URL` | The URL the agent reaches the MCP server at. It must match `MCP_HOST` and `MCP_PORT`. | `http://127.0.0.1:8011/mcp` |
| `MCP_HOST` | Interface the MCP server listens on. | `127.0.0.1` |
| `MCP_PORT` | Port the MCP server listens on. | `8011` |
| `MCP_TOOL_TIMEOUT_SECONDS` | Timeout for each MCP request from the agent, and for the server's GitHub API calls. | `30` |
| `GITHUB_ANALYSIS_TASK` | The task used when the prompt given to the agent is empty. | `Analyse recent repository activity, notable issues, pull requests, changes, and risks.` |

**AgentDNA (local) and CBAC**

| Variable | Description | Default |
|---|---|---|
| `AGENTDNA_LOCAL_DIR` | Keys and the shared SQLite store. Same value for every process. | `~/.agentdna-local` |
| `AGENTDNA_LOCAL_WHITELIST` | Comma-separated agent names or DIDs the MCP server accepts. Empty: every registered agent. | |
| `USER_NAME` | Name of the user identity that signs requests and records results. | `local-user@example.local` |
| `GITHUB_AGENT_NAME` | Name of the agent's identity. Its DID is the CBAC `agent_id`. | `github-repository-agent` |
| `MCP_SERVER_NAME` | Name of the MCP server's identity. | `Github MCP` |
| `CBAC_URL` | The CBAC service the MCP server asks about each tool call. | `http://127.0.0.1:8767` |
| `CBAC_TIMEOUT` | Seconds per authorize call. | `600` |
| `CBAC_POLICY_KEY` | The service's policy key; only `upload_policy.py` reads it. | |

**Authentication evidence (optional)**

`AGENTDNA_AUTH_EVIDENCE`, `AGENTDNA_AUTH_EVIDENCE_FILE` and `AGENTDNA_FINGERPRINT_KEY` switch on AgentDNA's authentication evidence, which is off by default. See [Authentication evidence](../../agentdna/README.md#authentication-evidence) in the agentdna README.

## More ways to run

`manual.py` prints the analysis. Add `--json` after the prompt to print the structured execution event instead. On Windows, use `python` instead of `python3`.

### Example prompts

```sh
python3 manual.py "Review the repository's open issues and pull requests."
python3 manual.py "Review the repository's recent commits and summarise the changes."
python3 manual.py "Review the repository's top-level files and documentation."
```

The CBAC v2 gate parses the prompt once per run into grants of (tool, repository): a tool the prompt does not ask for is refused (`4103`), whatever the agent's reasons. A prompt that names what to read (issues, pull requests, commits, files) is granted those tools; one that asks for a broad judgement may be granted fewer. The policy asks the parser to treat `path` and `limit` as free text; read the uploaded `catalog` to confirm it did. Free-text arguments pass only the deny-only semantic checks.

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

`repository_metadata`, `directory_listing`, `file_contents`, `commits`, `issues` and `pull_requests`: read-only, validated tools over the configured repository only. The GitHub token stays in the MCP server process; it never reaches the LLM or a tool argument.

The agent's skills and restrictions are in [agents/github_agent/SKILLS.md](agents/github_agent/SKILLS.md), which is the policy registered with its AgentDNA identity (its agent card, which the v1 gate reads). The v2 gate decides from [cbac_policies/github_agent.md](cbac_policies/github_agent.md), uploaded by `upload_policy.py`.

## Tests

From this directory, in its venv: `python -m pytest tests -q`. No service, network or API key is needed: the tests use a temporary `AGENTDNA_LOCAL_DIR`.
