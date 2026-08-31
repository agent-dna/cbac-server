"""Request and response shapes for the HTTP API.

The wire contract, kept apart from the routing in ``main.py`` so the shape a
caller has to send can be read without reading the handlers. Nothing here
touches the database or the decision pipeline — the pipeline's own types live
in ``skills.py``.
"""

from typing import Any

from pydantic import BaseModel, Field


class AuthorizeRequest(BaseModel):
    """The decision gate's body.

    An enforcement point sends the call's mechanical facts and the service
    phrases them into the text its scorers see. ``intended_action`` is the
    alternative for a caller that has its own phrasing (or only a sentence,
    like a ``curl`` probe) — supply either, and it wins when both are present.
    """

    agent_id: str
    callee_name: str = ""
    callee_type: str = "tool"
    callee_description: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_intent: str | None = None
    # Any shape: flattened to text server-side. Not a str so a caller can post
    # a structured action and let the flattener walk it.
    intended_action: Any = None


class PrecomputePolicyRequest(BaseModel):
    """Trigger embedding precomputation for one agent.

    ``policy`` embeds that text directly; without it the agent's card is read
    from the Provenance Layer.
    """

    agent_id: str
    policy: str | None = None
