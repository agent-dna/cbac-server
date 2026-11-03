"""Request and response shapes for the HTTP API.

The wire contract, kept apart from the routing in ``main.py`` so the shape a
caller has to send can be read without reading the handlers. Nothing here
touches the database or the decision pipeline — the pipeline's own types live
in ``skills.py``.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

# One request may not ask about more agents than this. A batch read fans out to
# one indexed query, so the ceiling is about bounding the response, not the
# lookup.
MAX_LHI_AGENTS = 200


class AuthorizeRequest(BaseModel):
    """The decision gate's body.

    An enforcement point sends the call's mechanical facts and the service
    phrases them into the text its scorers see. ``intended_action`` is the
    alternative for a caller that has its own phrasing (or only a sentence,
    like a ``curl`` probe) — supply either, and it wins when both are present.
    """

    agent_id: str
    # The MCP server the call is bound for. Opaque to the decision — never
    # worded into the scored text — and stored on the trust record so
    # /lhi-scores can say which server an edge's trust was earned against.
    # Defaults to "" for a caller with no MCP server behind it (an in-process
    # tool guard, a bare curl probe); stored NULL rather than "".
    mcp_did: str = ""
    callee_name: str = ""
    callee_type: str = "mcp_tool"
    callee_description: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_intent: str | None = None
    # Any shape: flattened to text server-side. Not a str so a caller can post
    # a structured action and let the flattener walk it.
    intended_action: Any = None
    intent_id: str | None = None


class PrecomputePolicyRequest(BaseModel):
    """Trigger embedding precomputation for one agent.

    ``policy`` embeds that text directly; without it the agent's card is read
    from the Provenance Layer.
    """

    agent_id: str
    policy: str | None = None


class LHIScoresRequest(BaseModel):
    """Ask for the current trust of a batch of agents.

    A batch rather than one id per call because the caller is typically a
    dashboard rendering a whole fleet, and per-agent round trips are what it
    is trying to avoid.
    """

    agent_ids: Annotated[
        list[Annotated[str, StringConstraints(min_length=1)]],
        Field(min_length=1, max_length=MAX_LHI_AGENTS),
    ]


# ── v2 ──────────────────────────────────────────────────────────────────────


class PolicyUploadRequest(BaseModel):
    """A v2 policy upload: the policy document, and optionally the agent's
    trusted data.

    ``policy`` is one Markdown document holding both the instructions and the
    tool schemas they govern, as ``---``-separated ``Policy Instructions`` and
    ``Tool Schema`` sections (see ``parsers.split_policy_document``). Omitted,
    it is the agent's card on the Provenance Layer, in the same format.
    ``trusted_data`` is what the query parser may resolve names through — a
    contacts directory, known repositories — and is trusted because an admin
    supplied it here, never because an agent did.

    ``review_required`` holds the parsed version ``pending`` until an admin
    approves it (``POST /cbac/v2/policies/{agent_id}/approve``); without it the
    version activates at once. Chosen per upload, because whether a policy
    needs a second reader depends on the agent, not on the deployment.
    """

    agent_id: str = Field(min_length=1)
    policy: str | None = None
    trusted_data: dict[str, Any] = Field(default_factory=dict)
    review_required: bool = False


class PolicyApproveRequest(BaseModel):
    version_id: int


class AuthorizeCallRequest(BaseModel):
    """The v2 decision gate's body: one tool call plus the intent it serves.

    ``intent_id`` and ``user_query`` default to empty rather than being
    required, so a caller that omits them gets an audited deny (4001) in the
    usual envelope rather than a 422. ``arguments`` is the call's JSON as the
    tool will receive it — not stringified — because resources are extracted
    from it value by value.
    """

    agent_id: str = Field(min_length=1)
    intent_id: str = ""
    user_query: str = ""
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    callee_type: str = "mcp_tool"
    mcp_did: str = ""
