from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml

REQUIRED_FRONTMATTER_KEYS = (
    "agent-did",
    "issued-by",
    "issued-at",
    "expires-at",
    "allowed-actions",
)


@dataclass
class SkillsCard:
    """Parsed skill.md card. Frontmatter fields + the markdown body."""

    agent_did: str
    agent_name: str
    issued_by: str
    issued_at: datetime
    expires_at: datetime
    allowed_actions: list[str]
    forbidden_actions: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    can_delegate_to: list[str] = field(default_factory=list)
    requires: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)


@dataclass
class CardCheck:
    """Result of CBAC check for one layer in the chain."""

    layer_did: str
    card_id: str | None
    card: SkillsCard | None
    action: str | None
    passed: bool
    reasons: list[str] = field(default_factory=list)


VALID_LLM_DECISIONS = ("allow", "deny", "advise")


@dataclass
class LLMVerdict:
    """Structured Tier-3 output an ``llm_backend`` callable must return.

    Replaces free-text keyword matching (checking for "deny"/"allow" as
    substrings anywhere in a response — which misfires on a negated mention
    like "not prohibited," or a response that says "I'd normally deny this,
    but..." before concluding allow). An ``llm_backend`` is expected to force
    its underlying LLM into returning exactly this shape — e.g. via a
    tool-call/structured-output API — so ``_tiered_decision`` reads
    ``decision`` directly instead of guessing from prose.

    ``decision`` must be one of :data:`VALID_LLM_DECISIONS`; anything else is
    treated as malformed and resolves to ``"advise"`` rather than raising —
    a misbehaving backend degrades to "caller must decide," the same
    fail-open-to-a-human posture as "no backend configured at all," not a
    hard failure.
    """

    decision: str  # "allow" | "deny" | "advise"
    reason: str


@dataclass
class CBACResult:
    """Overall CBAC decision after walking the full chain."""

    decision: str  # "allow" | "deny" | "advise"
    reason: str = ""
    trace: list[CardCheck] = field(default_factory=list)
    hallucination_score: float | None = None
    intent_score: float | None = None
    policy_score: float | None = None
    # Post-decision LHI trust for the (agent → callee) edge. None when no
    # trust update ran — see CBAC._fold_trust for the skip conditions.
    trust: float | None = None
    # Numeric identifier for exactly which layer/condition produced `reason`
    # — see cbac_service.error_codes. Machine-stable where `reason` (free
    # text) is not; every construction site in cbac.py sets this explicitly,
    # so `None` here would mean a return path forgot to, not that no
    # decision was reached.
    error_code: int | None = None


def parse_skill_md(text: str) -> SkillsCard:
    """
    Parse a skill.md (YAML frontmatter + markdown body) into a Card.

    Raises ValueError for any structural problem.
    """
    if not text.startswith("---"):
        raise ValueError("skill.md must start with YAML frontmatter delimited by ---")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("skill.md must have a closing --- line after frontmatter")
    frontmatter_text = parts[1]
    body = parts[2].lstrip("\n")

    fm = yaml.safe_load(frontmatter_text)
    if not isinstance(fm, dict):
        # TRY004 suppressed below: the documented contract is ValueError for
        # every structural problem, matching the other three raise sites.
        raise ValueError("skill.md frontmatter must be a YAML object")  # noqa: TRY004

    missing = [k for k in REQUIRED_FRONTMATTER_KEYS if k not in fm]
    if missing:
        raise ValueError(f"skill.md missing required field(s): {', '.join(missing)}")

    return SkillsCard(
        agent_did=fm["agent-did"],
        agent_name=fm.get("agent-name", ""),
        issued_by=fm["issued-by"],
        issued_at=_parse_dt(fm["issued-at"]),
        expires_at=_parse_dt(fm["expires-at"]),
        allowed_actions=list(fm.get("allowed-actions") or []),
        forbidden_actions=list(fm.get("forbidden-actions") or []),
        constraints=dict(fm.get("constraints") or {}),
        can_delegate_to=list(fm.get("can-delegate-to") or []),
        requires=dict(fm.get("requires") or {}),
        body=body,
        raw_frontmatter=fm,
    )


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        s = value.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    raise ValueError(f"unparseable timestamp: {value!r}")


def _collect_text(obj: Any, *, include_keys: bool = True, _depth: int = 0) -> str:
    """Recursively gather every string found in an arbitrary blob.

    Format-agnostic: handles raw text, dicts, lists, and dataclasses/objects
    (e.g. :class:`Card`) alike.

    ``include_keys`` controls whether mapping *keys* are gathered alongside
    values. For a large policy blob, keys are useful searchable vocabulary
    (default). For a short intent, structural keys like ``action``/``params``
    are pure scaffolding that would dilute the coverage score, so the intent
    side passes ``include_keys=False`` to flatten values only.
    """
    if _depth > 6 or obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        parts: list[str] = []
        for k, v in obj.items():
            if include_keys:
                parts.append(str(k))
            parts.append(_collect_text(v, include_keys=include_keys, _depth=_depth + 1))
        return " ".join(parts)
    if isinstance(obj, (list, tuple, set)):
        return " ".join(
            _collect_text(v, include_keys=include_keys, _depth=_depth + 1) for v in obj
        )
    # Dataclass / arbitrary object → walk its __dict__.
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return _collect_text(data, include_keys=include_keys, _depth=_depth + 1)
    return str(obj)


def _action_summary(action_name: str, description: str | None = None) -> str:
    """Verb-only prose for the intended action: "The agent wants to <verb>."

    The verb phrase is the callee's own description (schema or docstring first
    line) when available, else the de-snaked callee name. This is the prefix of
    :func:`render_intent`'s full rendering; kept as its own function so a
    verb-only view stays available to callers that want one.
    """
    verb = (description or action_name.replace("_", " ")).strip().rstrip(".")
    return f"The agent wants to {verb[:1].lower()}{verb[1:]}."


def render_intent(
    action_name: str, kwargs: dict[str, Any], description: str | None = None
) -> str:
    """Render a call as the prose the scorers see: "The agent wants to <verb>,
    with k = v, …".

    Server-side because the phrasing *is* part of the decision function: the
    tiers score prose, so two enforcement points that word the same call
    differently get different verdicts. Keeping it here means an enforcement
    point supplies only mechanical facts — callee name, arguments, description
    — and the wording is versioned alongside the models tuned against it. An
    enforcement point in another language needs no port of this.

    NLI and the policy tiers score natural language far better than raw
    ``tool k=v`` call syntax — snake_case names hide their verb from the NLI
    model (measured: a direct "close" vs "do not close" contradiction scored
    0.01 on call syntax, 0.95 as prose).
    """
    text = _action_summary(action_name, description).rstrip(".")
    if kwargs:
        text += ", with " + ", ".join(f"{k} = {v!s}" for k, v in kwargs.items())
    return text + "."


def _intended_action_text(intended_action: Any) -> str:
    """Flatten an intended-action of *any* shape (str / dict / list / object)
    into one string for tokenizing.

    Format-agnostic by design — it reuses the same recursive flattener as the
    policy side (:func:`_collect_text`), so no fixed key schema is assumed.
    Common keys like ``action``/``description``/``params`` are picked up
    automatically because their string values are gathered. Field *names* are
    skipped (``include_keys=False``) — for a short intent they are structural
    scaffolding that would only dilute the coverage score.
    """
    return _collect_text(intended_action, include_keys=False)
