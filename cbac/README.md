# `cbac` — the guard

The client half of CBAC: the code that *asks* whether an action is allowed. The
deciding happens in a separate service (`cbac_service/` in this repo) that this
package reaches over HTTP, one call per action.

It imports none of the ML stack — `requests` (lazily, inside the call) and the
standard library are the whole of it, `cbac.mcp` included.

```bash
pip install cbac
```

Two entry points, one per gate the service runs:

| function | service route | decides by |
| --- | --- | --- |
| `authorize_call` | `POST /cbac/v2/authorize` | exact grants parsed from the user's query, OpenFGA, then deny-only semantic checks on free-text arguments |
| `authorize` | `POST /cbac/v1/authorize` | semantic scoring of the whole call against the policy text |

## Where the service is

Both take `cbac_url` and `cbac_timeout` as arguments:

| argument | default | environment fallback |
| --- | --- | --- |
| `cbac_url` | `https://cbac-service.agentdna.io` | `CBAC_URL`, read only when `cbac_url=""` |
| `cbac_timeout` | `300` (seconds) | `CBAC_TIMEOUT`, read only when `cbac_timeout=0` |

The route is fixed per function. `authorize` alone also honours `CBAC_PATH`
(default `/cbac/v1/authorize`), for a v1 service mounted under another prefix.
The environment is read **per call**, so a process that loads its `.env` after
importing this package still gets it. With no URL from either source the call
resolves to `"error"`, which every caller treats as *not allowed*.

## The result

Both return a tuple and **never raise**:

```python
decision, status_code, interaction_hash = await authorize_call(...)
```

- `decision` — `"allow"`, `"deny"`, or `"error"` when no verdict was reached.
- `status_code` — the service's code for the check that decided
  (`cbac_service.error_codes`: the 4000s for v2, the 3000s for v1), or one of
  this package's own when there was no verdict to report: `9101` no service
  configured, `9102` the response carried no decision, `9103` the request
  failed, `9104` a decision came back without a code.
- `interaction_hash` — the audit row's id, `""` when no row was written.

**Proceed only on `"allow"`** — or, stricter, only on the allow codes you know
(`4301`/`4302` for v2), which also refuses a code minted after you wrote the
check. Treating anything else as permission defeats the point of a fail-closed
gate.

## Authorizing a call (v2)

```python
from cbac import authorize_call

decision, code, _ = await authorize_call(
    agent_id,
    intent_id,  # one per user request, minted where the request arrives
    user_query,  # that request, verbatim
    tool,  # the tool's name as the policy's catalog lists it
    args,  # the call's arguments, as the tool will receive them
    mcp_did=MCP_DID,
    callee_type="mcp",
    cbac_url=CBAC_URL,
)
if decision != "allow":
    ...  # blocked — do not run the call
```

Send the same `intent_id` and `user_query` on every call made for one user
request. The first call with a new `intent_id` makes the service parse the
query into that request's grants; every later call reuses them and is denied
(`4002`) if it pairs the id with a different query. A new user request gets a
new `intent_id`.

## Authorizing a call (v1)

```python
from cbac import authorize

decision, code, _ = await authorize(
    agent_id,
    mcp_did,
    callee_name,
    args,
    user_intent,
    description,
    intent_id,
    callee_type="mcp",
    cbac_url=CBAC_URL,
)
```

`description` is the callee's own description — a schema field, a docstring's
first line. The service uses it as the verb phrase of the text it scores and
falls back to the de-snaked `callee_name` without it. `intent_id` is only a
correlation id here, threaded to the audit row and its hash.

In both, `callee_type` (`"mcp_tool"`, `"tool"`, `"agent"`) labels the other end
of the edge whose trust score the service updates while deciding, and `mcp_did`
names the MCP server the call is bound for — your own routing config, recorded
on the trust row and returned by `POST /cbac/v1/lhi-scores`, never scored.

## What goes on the wire

Mechanical facts, never a rendered sentence. v2:

```json
{
  "agent_id": "...", "intent_id": "...", "user_query": "...",
  "tool": "send_invite", "arguments": {"to": ["bob@company.com"], "body": "..."},
  "callee_type": "mcp", "mcp_did": "..."
}
```

`arguments` keeps its structure — the service checks each item of a
list-valued recipient — and anything JSON cannot encode (a `datetime`) is sent
as its `str`.

v1 sends `agent_id`, `mcp_did`, `callee_name`, `callee_type`,
`callee_description`, `arguments` (every value stringified), `user_intent` and
`intent_id`; the service phrases them into the text its scorers see, so an
enforcement point cannot word a call into a different verdict.

Either way the trust score (LHI) is folded in service-side while the decision
is reached, so there is exactly one HTTP call per action.

## MCP (`cbac.mcp`)

Enforcement is gateway-side: the agent's MCP client points at a CBAC gateway
instead of the MCP server, and its only job is to **label** each outgoing call.
The gateway decides.

Client side:

```python
from cbac.mcp import cbac_propagate

client = MultiServerMCPClient({...}, tool_interceptors=[cbac_propagate])
```

`cbac_propagate` attaches the ambient context as headers and nothing else — it
calls no service and can deny nothing. That context is opened once, where the
user's request arrives:

```python
import uuid

from cbac import cbac_context

with cbac_context(
    agent_id="github-worker",
    user_intent="Close issue #3 in acme/api",
    intent_id=uuid.uuid4().hex,
):
    ...  # every MCP call made in here goes out labelled
```

It lives in a `contextvars.ContextVar`, never a function argument, so an LLM can
neither supply nor forge it. Without an open scope `cbac_propagate` is a
passthrough — the calls still reach the gateway, just unlabelled, and the v2
gate denies a call that carries no intent.

Gateway side:

```python
from cbac import authorize_call
from cbac.mcp import context_from_headers, denial_body

ctx = context_from_headers(get_http_headers())  # None when unlabelled
result = await authorize_call(
    ctx.agent_id,
    ctx.intent_id,
    ctx.user_intent,
    tool_name,
    tool_args,
    mcp_did=MCP_DID,
    cbac_url=CBAC_URL,
)
if result[1] not in {4301, 4302}:
    raise ToolError(denial_body(*result))
```

`denial_body(decision, status_code, interaction_hash)` renders
`{"status": "denied"|"error", "error": ..., "decision": ..., "status_code": ...,
"hash": ...}` — the same shape every enforcement point returns.

### The headers

| header | carries |
| --- | --- |
| `X-CBAC-Agent-Id` | whose policy applies |
| `X-CBAC-User-Intent` | what the user actually asked for |
| `X-CBAC-Intent-Id` | the id of that request |

All are percent-encoded (headers are latin-1 and size-capped; a user intent is
arbitrary UTF-8), and the intent is capped at 4096 encoded characters without
ever cutting an escape in half. `cbac_headers()` produces them from the ambient
context and returns `{}` when governance is off, for clients whose transport
takes headers on the connection rather than per call. The intent id is sent
only when the workflow minted one.

**Trust boundary.** All three values are set by the agent's *runtime*, never by
its model: the LLM emits tool calls, not headers. That is what v2 rests on —
`user_intent` is the source of the request's grants, so whatever sends the
headers decides what can be granted, and a compromised runtime can ask for
anything its policy allows. `agent_id` decides whose policy applies; a gateway
that cannot trust its callers should derive it from its authenticated principal
(OAuth subject, mTLS SAN, API key) and treat the header as a fallback for a
trusted network only. `intent_id` binds a request's grants to its query, so
mint it unguessable (`uuid4`) and never reuse it across requests. `mcp_did`
travels in no header at all: the gateway routes the call, so it is the one
party that knows where the call went.

The context travels as headers because that is the only per-call channel MCP
client adapters expose today. `_meta` is the protocol-native place for it;
`context_from_headers` is the only piece that would change.
