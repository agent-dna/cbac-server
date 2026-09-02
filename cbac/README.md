# `cbac` — the guard

The client half of CBAC: the code that *asks* whether an action is allowed. The
deciding happens in a separate service (`cbac_service/` in this repo) that this
package reaches over HTTP, one call per action.

It imports none of the ML stack — `requests` (lazily, inside the call) and the
standard library are the whole of it, `cbac.mcp` included.

```bash
pip install cbac
```

## Where the service is

Every call needs a URL to POST to. Either pass it, or set it in the
environment:

| argument | environment | default |
| --- | --- | --- |
| `cbac_url` | `CBAC_URL` | *(none — no URL fails the call closed)* |
| `cbac_endpoint` | `CBAC_PATH` | `/cbac/v1/authorize` |
| `cbac_timeout` | `CBAC_TIMEOUT` | `100` (seconds) |

The environment is read **per call**, so a process that loads its `.env` after
importing this package still gets it. An enforcement point that already reads
its own settings passes them as arguments and skips the environment entirely.

There is no baked-in host. With no URL from either source the call resolves to
`decision="error"`, which every caller treats as *not allowed* — an agent's
actions never go to whatever host happened to be compiled in.

## The result

Every entry point returns an `AuthResult` and **never raises**:

```python
@dataclass
class AuthResult:
    decision: str  # "allow" | "deny" | "error"
    message: str  # why — the service's reason, or the failure
    status_code: int | None  # the service's pipeline error code
    hash: str | None  # the audit row's interaction_hash
```

`"error"` is this layer's own fail-closed value: no service configured, a
network failure, a response that carried no verdict. `status_code` and `hash`
are `None` on those paths, because there was no service-side verdict to carry
them.

**Proceed only on `"allow"`.** Treating anything else as permission defeats the
point of a fail-closed gate.

## Authorizing a call

`authorize` takes everything as arguments — for an enforcement point that
parsed the governance context off the wire (a gateway) or read it from a queue
message. Nothing is looked up ambiently.

```python
from cbac import authorize

result = await authorize(
    agent_id,
    callee_name,
    args,
    user_intent,
    description,
    callee_type="mcp",
    cbac_url=CBAC_URL,
    cbac_timeout=CBAC_TIMEOUT,
)
if result.decision != "allow":
    ...  # blocked — do not run the call
```

`description` is the callee's own description — a schema field, a docstring's
first line. Pass it: the service uses it as the verb phrase and falls back to
the de-snaked `callee_name` without it, and an enforcement point that has the
real description scores far better than one working from the name alone.

`callee_type` (`"tool"`, `"agent"`, `"mcp"`) labels the other end of the edge
whose trust score the service updates while deciding. `intent_id` is your own
opaque correlation id, threaded unchanged to the audit row and its hash — CBAC
never parses it.

## What goes on the wire

Mechanical facts, not a rendered sentence:

```json
{
  "agent_id": "...", "callee_name": "...", "callee_type": "tool",
  "callee_description": "...", "arguments": {"repo": "acme/api"},
  "user_intent": "...", "intent_id": null
}
```

The service phrases those into the text its scorers see, so an enforcement
point cannot word a call into a different verdict — and a port of this layer to
another language needs no port of the rendering. Argument values are stringified
first, so a `datetime` or a dataclass argument cannot fail the call closed on
JSON encoding.

The trust score (LHI) is folded in service-side while the decision is reached,
so there is exactly one HTTP call per action and no component score ever
round-trips through the client.

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
calls no service and can deny nothing. That context is opened once, at the
request entry point:

```python
from cbac import cbac_context

with cbac_context(agent_id="github-worker", user_intent="Close issue #3 in acme/api"):
    ...  # every MCP call made in here goes out labelled
```

It lives in a `contextvars.ContextVar`, never a function argument, so an LLM can
neither supply nor forge it. Without an open scope `cbac_propagate` is a
passthrough — the calls still reach the gateway, just unlabelled.

Gateway side:

```python
from cbac.mcp import context_from_headers, denial_body

ctx = context_from_headers(get_http_headers())  # None when unlabelled
result = await authorize(ctx.agent_id, ..., cbac_url=CBAC_URL)
if result.decision != "allow":
    raise ToolError(denial_body(result.decision, result.message))
```

`denial_body` renders `{"status": "denied"|"error", "error": ...}` — the same
shape every enforcement point returns, so a client cannot tell whether the
block came from an in-process gate, an interceptor, or a gateway several
hops away.

The developer holds no security logic and cannot switch the check off: deleting
`cbac_propagate` does not bypass the gateway, it only makes calls arrive
unlabelled. What a gateway does with an unlabelled call is the gateway's
policy — deny it outright, or decide it on policy alone with no drift signal.

### The headers

| header | carries |
| --- | --- |
| `X-CBAC-Agent-Id` | whose policy applies |
| `X-CBAC-User-Intent` | what the user actually asked for |
| `X-CBAC-Intent-Id` | the caller's correlation id, when there is one |

All are percent-encoded (headers are latin-1 and size-capped; a user intent is
arbitrary UTF-8), and the intent is capped at 4096 encoded characters without
ever cutting an escape in half — a truncated intent is still a usable drift
signal, a rejected request is not. `cbac_headers()` produces them from the
ambient context and returns `{}` when governance is off, for clients whose
transport takes headers on the connection rather than per call. The intent id
is sent only when the workflow minted one, so a deployment that uses no
correlation ids puts no empty header on every call.

**Trust boundary.** Both values are client-supplied. `user_intent` is
unverifiable by anyone — only the client knows what the user asked — so CBAC
scores it as *evidence* (a drift signal), never as identity. `agent_id` decides
whose policy applies, which is exactly what a compromised client would lie
about; a gateway that cannot trust its callers should derive it from its
authenticated principal (OAuth subject, mTLS SAN, API key) and treat the header
as a fallback for a trusted network only. `intent_id` is neither identity nor
evidence — CBAC never parses it, only threads it to the audit row and its hash,
so a client that forges one corrupts its own trace and nothing else.

The context travels as headers because that is the only per-call channel MCP
client adapters expose today. `_meta` is the protocol-native place for it;
`context_from_headers` is the only piece that would change.

## A working example

`examples/github_agent_workflow/` runs all of this end to end: an agent that
only labels its calls, a gateway that decides them, a CBAC-unaware MCP server,
and a real third-party MCP server on the same policy. `agent_check.py` drives
every path and asserts on the verdicts, with no LLM in the loop.
