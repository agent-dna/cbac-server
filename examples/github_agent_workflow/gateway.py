"""
The CBAC MCP gateway: a Policy Enforcement Point the agent developer does not
control.

It is an MCP server to the agent and an MCP client to the real MCP server, so
every ``tools/call`` passes through it. It reads the governance context off the
call, asks the CBAC decision service, and forwards or blocks. Nothing about
this depends on the agent, the framework, or the upstream server cooperating:

                                        ┌─> github-mcp ──> api.github.com
    agent ──MCP──> gateway (PEP) ──MCP──┤     (local stand-in)
                      │                 └─> mcp.deepwiki.com
                      │                       (real third party)
                      └── POST /cbac/v1/authorize  (CBAC decision service)

Why here and not in the agent process: an interceptor inside the agent is
opt-in, and a security control the developer can delete is not a control. The
agent's *only* job is to label its calls (``cbac_propagate``), which carries no
decision. Skip the label and the calls still reach the gateway — contextless,
so they are judged on policy alone, with no drift check to help them.

The gateway is also where the upstream credentials live, and it is the only
service with a public address — every upstream sits on a private network
behind it. Between them those two facts are what make bypass pointless rather
than merely inconvenient: the agent cannot route to an upstream, and could not
authenticate to it if it could.

Run:

    python gateway.py

Env: MCP_UPSTREAM_URL, MCP_THIRDPARTY_URL, CBAC_GATEWAY_HOST/PORT, CBAC_URL,
CBAC_AGENT_ID.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cbac import configure, get_config
from cbac.guard import authorize
from cbac.mcp import context_from_headers, denial_body

load_dotenv()

UPSTREAM_URL = os.environ.get(
    "MCP_UPSTREAM_URL",
    f"http://{os.environ.get('MCP_SERVER_HOST', '127.0.0.1')}"
    f":{os.environ.get('MCP_SERVER_PORT', '8765')}/mcp/",
)
# A real third-party MCP server: public, no auth, and nobody in this repo owns
# it. The local stand-in above only *represents* an undecoratable server; this
# one actually is one. Same gateway, same middleware, same policy.
THIRDPARTY_URL = os.environ.get("MCP_THIRDPARTY_URL", "https://mcp.deepwiki.com/mcp")
GATEWAY_HOST = os.environ.get("CBAC_GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("CBAC_GATEWAY_PORT", "8766"))
CBAC_URL = os.environ.get("CBAC_URL", "http://127.0.0.1:8767")
CBAC_TIMEOUT = float(os.environ.get("CBAC_TIMEOUT", "600"))
# Whose policy applies when the call carries no X-CBAC-Agent-Id — the labelled
# id wins. ponytail: a gateway that must not take the caller's word for it maps
# the authenticated principal (OAuth subject, mTLS SAN, API key) to an agent id
# instead, in the two lines where the hooks read the header.
AGENT_ID = os.environ.get("CBAC_AGENT_ID", "github-worker")

configure(cbac_url=CBAC_URL, cbac_timeout=CBAC_TIMEOUT)


async def decide(
    context, getter, agent_id, user_intent, fallback=None
) -> tuple[str, str]:
    """The gate itself. Returns ``authorize``'s ``(decision, detail)``.

    Every MCP primitive that acts on the world or returns the server's data
    comes through here — only the exception the caller raises on a non-``allow``
    differs, because that is all that differs. A gate wired to ``tools/call``
    alone is not a gate: the same capability is usually reachable as a resource
    or a prompt, and the client picks which one to use.

    ``getter`` (``get_tool`` or ``get_resource``) says which primitive
    ``context.message`` holds and looks the callee up on the local proxy for
    its registered name and first description line — a gateway-side advantage,
    since a client-side interceptor only has what was on the wire, and NLI
    scores "close an existing GitHub issue" far better than a de-snaked "close
    issue". ``get_resource`` resolves only URIs that were registered literally,
    so a concrete URI for a templated resource
    (``github://{owner}/{name}/collaborators``) is matched against the
    templates instead. Worth the extra lookup: the description carries the
    verb, and without it the key lands in the verb slot, where a forbidden read
    that should score a clear -0.113 cosine gap scores -0.054 and falls into
    the gray zone. ``fallback`` is the last resort, never the bare key.
    """
    message = context.message
    if getter == "get_tool":
        key, args = message.name, dict(message.arguments or {})
    else:
        # A URI carries its arguments inside it — github://acme/api/… — so it
        # is both the lookup key and the whole argument set.
        key = str(message.uri)
        args = {"uri": key}

    server = context.fastmcp_context.fastmcp
    name = description = None
    with contextlib.suppress(Exception):
        obj = await getattr(server, getter)(key)
        name = obj.name
        description = (obj.description or "").partition("\n")[0] or None
    if description is None and getter == "get_resource":
        with contextlib.suppress(Exception):
            for template in await server.list_resource_templates():
                if template.matches(key) is not None:
                    name = template.name
                    first = (template.description or "").partition("\n")[0]
                    description = first or None
                    break

    return await authorize(
        agent_id,
        name or key,
        args,
        user_intent,
        description or fallback,
        callee_type="mcp",
        cfg=get_config(),
    )


class CBACMiddleware(Middleware):
    """Route every acting MCP primitive crossing this gateway through the gate.

    Both hooks read the governance context off the call: ``user_intent``
    because only the client can know what the user asked (CBAC treats it as
    evidence — a drift signal — never as identity), and ``agent_id`` because
    the client says whose policy applies, falling back to ``AGENT_ID`` when it
    says nothing.
    """

    async def on_call_tool(self, context, call_next):
        ctx = context_from_headers(get_http_headers())
        agent_id = (ctx.agent_id if ctx else "") or AGENT_ID
        user_intent = ctx.user_intent if ctx else ""
        decision, detail = await decide(context, "get_tool", agent_id, user_intent)
        if decision != "allow":
            raise ToolError(denial_body(decision, detail))
        return await call_next(context)

    async def on_read_resource(self, context, call_next):
        """``resources/read`` returns the server's data, so it is an action.

        The trust edge is keyed on the resource's registered name, not its URI.
        A URI carries its arguments inside it — ``github://acme/api/…`` — so
        keying on it would mint a new edge per repository and the trust history
        would never accumulate anywhere.
        """
        ctx = context_from_headers(get_http_headers())
        agent_id = (ctx.agent_id if ctx else "") or AGENT_ID
        user_intent = ctx.user_intent if ctx else ""
        decision, detail = await decide(
            context,
            "get_resource",
            agent_id,
            user_intent,
            fallback="read the resource",
        )
        if decision != "allow":
            raise ResourceError(denial_body(decision, detail))
        return await call_next(context)

    # ``prompts/get`` is deliberately not gated. Prompts are user-controlled in
    # MCP's model — the person picks one, the model does not reach for it — so
    # checking it for drift from what the user asked is circular. It also
    # returns text rather than acting, and the actions that follow are tool
    # calls, which are gated. Residual risk, if you want it later: a server
    # free to do real work while building a template does that work ungated.


# Two upstreams behind one gateway — the shape a real egress PEP has. FastMCP
# prefixes each server's tools with its name (``github_close_issue``,
# ``deepwiki_ask_question``), so the agent can tell them apart and the gateway
# knows which credential to attach.
#
# A config, not a Client: FastMCP then gives each proxied session its own
# client rather than sharing one across concurrent callers.
gateway = FastMCP.as_proxy(
    {
        "mcpServers": {
            # No credential here either — the local stand-in needs none. A
            # real upstream's token (GitHub OAuth, …) goes in this dict, which
            # is the point: it lives in the gateway and never reaches the agent.
            "github": {"url": UPSTREAM_URL},
            # No credential of ours to attach — DeepWiki is open. When the
            # third party does need one (GitHub, Notion, Slack), the token goes
            # here, in the gateway, and never reaches the agent process.
            "deepwiki": {"url": THIRDPARTY_URL},
        }
    },
    name="cbac-gateway",
)
gateway.add_middleware(CBACMiddleware())


if __name__ == "__main__":
    print(f"[gateway] {GATEWAY_HOST}:{GATEWAY_PORT} -> {UPSTREAM_URL} (github)")
    print(f"[gateway] {GATEWAY_HOST}:{GATEWAY_PORT} -> {THIRDPARTY_URL} (deepwiki)")
    print(f"[gateway] cbac={CBAC_URL} agent_id={AGENT_ID}")
    gateway.run(transport="http", host=GATEWAY_HOST, port=GATEWAY_PORT)
