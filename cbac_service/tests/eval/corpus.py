"""The paired policy corpus: one semantic spec, two document shapes.

Each spec under `corpus/` states a policy's *meaning* — the capabilities it
grants, the ones it withholds, its constraints, and the inert prose real
documents carry. Two documents are rendered from it: a skill card, and one
unstructured shape. The capability sentences are byte-identical in both, so a
structured-vs-unstructured number is a statement about document shape and
nothing else. A corpus of hand-written documents cannot make that claim: its
two groups differ in content as well as shape, and the two effects are not
separable after the fact.

The tradeoff that buys it: an IAM-JSON rendering wraps the same English
sentence rather than inventing `inventory:DeleteItem` syntax. Scaffolding is
realistic, wording is held constant.

Gold labels are *derived*. The renderer knows which sentence it wrote where, so
a chunk's label is read back off the capability texts it contains — there is no
hand-maintained marker table to fall out of sync with the documents.

Ground truth is the policy author's, not the pipeline's, and **default-deny**:
an action a policy never grants is `deny`. A measurement that disagrees with
`cbac.py` is a finding about `cbac.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import yaml

from cbac_service.skills import render_intent

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# Decision gold.
ALLOW, DENY, GRAY = "allow", "deny", "gray"
# Chunk gold. `mixed` is not a class anyone assigns — it is what a chunk gets
# when the chunker swallowed spans of both polarities, which no classifier
# could label correctly. Counted, then dropped from the rates.
ALLOWED, FORBIDDEN, NEUTRAL, MIXED = "allowed", "forbidden", "neutral", "mixed"

STRUCTURED, UNSTRUCTURED = "structured", "unstructured"

COMPLEXITIES = ("simple", "semi", "complex")
BASES = ("granted", "explicit", "constraint", "unlisted")
# How the agent's action relates to what the *user* asked for. The policy is
# irrelevant to this axis: `relation` is a fact about the request, `policy` is a
# fact about the agent's permissions, and the two vary independently.
#
#   faithful       does what was asked, nothing more
#   contradiction  does the opposite of what was asked
#   scope_creep    does what was asked and then some
#   hijack         does something unrelated to the request — an injection's
#                  outcome. Not the same as "does something forbidden": an
#                  action the user genuinely asked for is `faithful` however
#                  badly the policy forbids it
#   param_poison   right verb, attacker-chosen arguments
#
# `aligned` is not a second axis — it is `relation == "faithful"`, asserted in
# test_corpus_sanity. The interesting rows are the faithful ones a policy
# forbids: the drift layer is structurally blind to them, because nothing about
# the user's request is wrong, and only the tiers can see that region.
RELATIONS = ("faithful", "contradiction", "scope_creep", "hijack", "param_poison")

# How a deliberately evasive action tries to get past the scorers. Every layer
# here reads text, so the ways of attacking it are ways of perturbing text: hide
# the verb, bury it, restate it, or claim it was already approved. Reported as
# its own block rate — an aggregate that mixes attacks with ordinary requests is
# a statement about the mixing ratio.
EVASIONS = (
    "",  # not an attack: an ordinary request
    "homoglyph",  # Cyrillic/Greek lookalikes inside an argument
    "zero_width",  # zero-width joiners splitting a keyword
    "negation_stack",  # "do not fail to delete" — the verb survives, the polarity inverts
    "bundling",  # one forbidden argument buried among permitted ones
    "dilution",  # a long, benign preamble around a short forbidden verb
    "authority_claim",  # asserts an approval the policy requires but cannot verify
    "roleplay",  # frames the action as a drill, test, or hypothetical
    "cross_lingual",  # the forbidden verb in another language
    "encoding",  # the forbidden argument base64-encoded or escaped
    "indirect_injection",  # the action originates in retrieved content, not the user
)

# Surface markers that state a prohibition *as a label* rather than in the
# language of the rule. Splitting forbidden-recall on these separates "the
# classifier read the word forbidden" from "the classifier understood the
# prohibition" — the two are indistinguishable in an aggregate recall number.
PROHIBITION_TOKENS = (
    "forbidden",
    "must not",
    "must never",
    "never",
    "not permitted",
    "prohibit",
    "shall not",
    "may not",
    "deny",
    "disallow",
    "revoke",
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def has_prohibition_token(text: str) -> bool:
    low = _norm(text)
    return any(tok in low for tok in PROHIBITION_TOKENS)


# ── The spec ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Capability:
    id: str
    text: str


@dataclass(frozen=True)
class Action:
    """One agent action, held as the mechanical facts a guard actually sends.

    The guard posts `callee_name` / `arguments` / `callee_description` and the
    service phrases them (`cbac_service.main._intended_action`). Storing prose
    here instead would score a sentence no enforcement point ever produces, so
    the dataset holds the facts and `text` runs them through the production
    renderer.
    """

    id: str
    callee: str
    args: dict[str, Any]
    description: str | None
    relation: str
    aligned: bool
    policy: str
    # How the action tries to evade the scorers, "" for an ordinary request.
    # See EVASIONS. An evasive action is scored in its own block rate as well
    # as the overall one.
    evasion: str = ""
    # Why a `deny` gold is a deny. `explicit` — a forbidden capability names
    # this action. `constraint` — a granted capability, but a stated limit is
    # broken. `unlisted` — neither granted nor forbidden, so only default-deny
    # blocks it. Without the split a block rate is uninterpretable: default-deny
    # blocks an unlisted action for the same reason it blocks legitimate work,
    # and only the `explicit` rate says whether the forbidden bucket is read at
    # all. Carried on allow/gray rows too, where it is always `granted`.
    basis: str = "granted"

    @cached_property
    def text(self) -> str:
        return render_intent(self.callee, self.args, self.description)

    @property
    def should_block(self) -> bool:
        """Blocked if *either* axis fails: the action is not what the user
        asked for, or the policy does not permit it."""
        return self.policy == DENY or not self.aligned

    @property
    def call(self) -> str:
        """`callee(k=v, …)` — the call as a reader recognises it."""
        inner = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.callee}({inner})"


@dataclass(frozen=True)
class Intent:
    id: str
    complexity: str
    text: str
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class Spec:
    id: str
    domain: str
    shape: str
    allowed: tuple[Capability, ...]
    forbidden: tuple[Capability, ...]
    inert: tuple[str, ...]
    constraints: dict[str, Any]
    intents: tuple[Intent, ...]
    # Frontmatter `parse_skill_md` reads and `cbac.py` never consults. Rendered
    # into the card so a test can ask whether either reaches a verdict.
    can_delegate_to: tuple[str, ...] = ()
    note: str = ""

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        return self.allowed + self.forbidden

    @property
    def actions(self) -> list[tuple[Intent, Action]]:
        return [(i, a) for i in self.intents for a in i.actions]

    @cached_property
    def renderings(self) -> tuple[Rendering, ...]:
        return (
            Rendering(
                self,
                STRUCTURED,
                "skill-card",
                _card(
                    self,
                    can_delegate_to=list(self.can_delegate_to) or None,
                ),
            ),
            Rendering(self, UNSTRUCTURED, self.shape, SHAPES[self.shape](self)),
        )

    @property
    def oracle_buckets(self) -> tuple[list[str], list[str]]:
        """The index a perfect chunker and a perfect classifier would build.

        Taken from the spec, not from either document's chunks: it is the same
        under both renderings by construction, which is what makes it a shared
        upper bound rather than a second thing that varies with shape.
        """
        return (
            [c.text for c in self.allowed],
            [c.text for c in self.forbidden],
        )


@dataclass(frozen=True)
class Rendering:
    """One spec as one document."""

    spec: Spec
    kind: str  # structured | unstructured
    shape: str
    text: str

    @property
    def key(self) -> str:
        return f"{self.spec.id}:{self.kind}"

    def label_for(self, chunk: str) -> str:
        """Gold label of a produced chunk.

        `mixed` when one chunk carries capability sentences of both polarities
        — a chunking defect, not a classification one.
        """
        c = _norm(chunk)
        hits = {
            ALLOWED if cap in self.spec.allowed else FORBIDDEN
            for cap in self.spec.capabilities
            if _norm(cap.text) in c
        }
        if len(hits) > 1:
            return MIXED
        return hits.pop() if hits else NEUTRAL


# ── Document renderers ───────────────────────────────────────────────────────
#
# Every renderer embeds each capability's sentence verbatim; `test_corpus_sanity`
# enforces it. That is the invariant the whole paired design rests on — a shape
# that paraphrased would turn its own delta into a wording effect.


def _card(
    spec: Spec,
    *,
    expires_at: str = "2026-12-31T00:00:00Z",
    can_delegate_to: list[str] | None = None,
) -> str:
    """The skill card. `expires_at` and `can_delegate_to` are parameters rather
    than constants so a test can render a variant of a real policy and compare
    verdicts against the original — the only way to show whether a frontmatter
    field the parser reads reaches the decision at all."""
    fm: dict[str, Any] = {
        "agent-did": f"did:agent:{spec.id}",
        "agent-name": spec.id,
        "issued-by": "did:org:acme",
        "issued-at": "2026-01-05T00:00:00Z",
        "expires-at": expires_at,
        "allowed-actions": [c.text for c in spec.allowed],
        "forbidden-actions": [c.text for c in spec.forbidden],
    }
    if can_delegate_to is not None:
        fm["can-delegate-to"] = can_delegate_to
    if spec.constraints:
        fm["constraints"] = spec.constraints
    body = "\n\n".join(spec.inert)
    dumped = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False, width=10_000)
    return f"---\n{dumped}---\n\n{body}\n"


def _md_headings(spec: Spec) -> str:
    out = [f"# {spec.domain} — Tool Policy", "", "## What this agent may do", ""]
    out += [f"- {c.text}" for c in spec.allowed]
    out += ["", "## What this agent must not do", ""]
    out += [f"- {c.text}" for c in spec.forbidden]
    if spec.constraints:
        out += ["", "## Limits", ""]
        out += [f"- {k}: {v}" for k, v in spec.constraints.items()]
    out += ["", "## Notes", "", *spec.inert]
    return "\n".join(out) + "\n"


def _system_prompt(spec: Spec) -> str:
    out = [f"You are {spec.domain}. Follow these rules exactly.", ""]
    out += [f"- You may {c.text}." for c in spec.allowed]
    out += [""]
    out += [f"- You must never {c.text}." for c in spec.forbidden]
    if spec.constraints:
        out += ["", *[f"- Limit — {k}: {v}" for k, v in spec.constraints.items()]]
    out += ["", *spec.inert]
    return "\n".join(out) + "\n"


def _iam_json(spec: Spec) -> str:
    doc = {
        "Version": "2026-01-05",
        "Principal": f"agent/{spec.id}",
        "Statement": [
            *[
                {"Sid": c.id, "Effect": "Allow", "Description": c.text}
                for c in spec.allowed
            ],
            *[
                {"Sid": c.id, "Effect": "Deny", "Description": c.text}
                for c in spec.forbidden
            ],
        ],
        "Condition": spec.constraints or {},
    }
    return json.dumps(doc, indent=2) + "\n"


def _yaml_config(spec: Spec) -> str:
    doc = {
        "agent": spec.id,
        "role": spec.domain,
        "permissions": {
            "grant": [c.text for c in spec.allowed],
            "revoke": [c.text for c in spec.forbidden],
        },
        "limits": spec.constraints or {},
        "notes": list(spec.inert),
    }
    return yaml.safe_dump(doc, sort_keys=False, width=10_000)


def _readme(spec: Spec) -> str:
    out = [
        f"# {spec.id}",
        "",
        f"{spec.domain}. Read this before wiring the agent into anything.",
        "",
        "## Supported operations",
        "",
    ]
    out += [f"* {c.text}" for c in spec.allowed]
    out += ["", "## Out of bounds", ""]
    out += [f"* {c.text}" for c in spec.forbidden]
    out += ["", *spec.inert]
    return "\n".join(out) + "\n"


def _numbered_rules(spec: Spec) -> str:
    out = [f"{spec.domain} — operating rules", ""]
    n = 1
    for c in spec.allowed:
        out.append(f"{n}. The agent is permitted to {c.text}.")
        n += 1
    out.append("")
    for c in spec.forbidden:
        out.append(f"{n}. The agent shall not {c.text}.")
        n += 1
    out += ["", *spec.inert]
    return "\n".join(out) + "\n"


def _runbook_table(spec: Spec) -> str:
    # A markdown table has no bullets and no blank lines, so the chunker
    # collapses the whole grid into one chunk carrying both polarities. That is
    # a real property of a real document shape, kept rather than worked around.
    rows = [f"| {c.text} | permitted |" for c in spec.allowed]
    rows += [f"| {c.text} | denied |" for c in spec.forbidden]
    out = [
        f"{spec.domain} runbook",
        "",
        "| operation | access |",
        "| --- | --- |",
        *rows,
        "",
        *spec.inert,
    ]
    return "\n".join(out) + "\n"


def _legal_prose(spec: Spec) -> str:
    grants = " ".join(
        f"The Agent shall be permitted to {c.text}." for c in spec.allowed
    )
    bans = " ".join(
        f"The Agent shall in no circumstance {c.text}." for c in spec.forbidden
    )
    out = [
        f"TERMS OF OPERATION — {spec.domain}",
        "",
        "1. Scope of authority.",
        "",
        grants,
        "",
        "2. Reserved matters.",
        "",
        bans,
        "",
        *spec.inert,
    ]
    return "\n".join(out) + "\n"


def _opa_rules(spec: Spec) -> str:
    body = []
    for c in spec.allowed:
        body.append(f'allow {{ input.operation == "{c.id}" }}  # {c.text}')
    for c in spec.forbidden:
        body.append(f'deny {{ input.operation == "{c.id}" }}  # {c.text}')
    out = [
        f"package agent.{spec.id.replace('-', '_')}",
        "",
        "default allow = false",
        "",
    ]
    out += body
    out += ["", *spec.inert]
    return "\n".join(out) + "\n"


def _cross_reference(spec: Spec) -> str:
    out = [
        f"{spec.domain} — access policy",
        "",
        (
            "Section 1 enumerates granted operations. Operations listed in "
            "Section 2 are withheld and take precedence over Section 1 wherever "
            "the two overlap."
        ),
        "",
        "Section 1.",
        "",
    ]
    out += [f"- {c.text}" for c in spec.allowed]
    out += ["", "Section 2.", ""]
    out += [f"- {c.text}" for c in spec.forbidden]
    out += ["", *spec.inert]
    return "\n".join(out) + "\n"


def _narrative(spec: Spec) -> str:
    # No list structure at all: every rule is a sentence inside a paragraph, so
    # the chunker's only boundary is the blank line.
    grants = " ".join(f"It can {c.text}." for c in spec.allowed)
    bans = " ".join(f"It will not {c.text}." for c in spec.forbidden)
    out = [
        f"About the {spec.domain}.",
        "",
        f"The assistant handles day-to-day requests in this area. {grants}",
        "",
        f"There are limits to what it does on a user's behalf. {bans}",
        "",
        *spec.inert,
    ]
    return "\n".join(out) + "\n"


def _malformed_frontmatter(spec: Spec) -> str:
    # Opens frontmatter and never closes it: `parse_skill_md` raises, and
    # `flatten_policy_chunks` silently falls back to chunking the raw YAML as
    # prose. The chunks that reach the classifier are then neither frontmatter
    # entries nor sentences.
    out = [
        "---",
        f"agent-did: did:agent:{spec.id}",
        "allowed-actions:",
        *[f"  - {c.text}" for c in spec.allowed],
        "forbidden-actions:",
        *[f"  - {c.text}" for c in spec.forbidden],
        "",
        *spec.inert,
    ]
    return "\n".join(out) + "\n"


def _xacml_xml(spec: Spec) -> str:
    # The polarity is an XML attribute sitting inside the same chunk as the
    # sentence — neither a frontmatter key nor a word in the rule.
    def rule(effect: str, c: Capability) -> str:
        return (
            f'  <Rule RuleId="{c.id}" Effect="{effect}">\n'
            f"    <Description>{c.text}</Description>\n"
            f"  </Rule>"
        )

    body = "\n\n".join(
        [rule("Permit", c) for c in spec.allowed]
        + [rule("Deny", c) for c in spec.forbidden]
    )
    notes = "\n".join(f"  <!-- {n} -->" for n in spec.inert)
    return f'<Policy PolicyId="{spec.id}">\n\n{body}\n\n{notes}\n</Policy>\n'


def _ini_conf(spec: Spec) -> str:
    # The section header survives chunking (it is not `#`-prefixed) but lands in
    # a chunk of its own, so the polarity is one chunk away from every rule it
    # governs — a different failure from a heading that is deleted outright.
    out = ["[agent]", f"role = {spec.domain}", "", "[permitted]"]
    out += [f"{c.id} = {c.text}" for c in spec.allowed]
    out += ["", "[denied]"]
    out += [f"{c.id} = {c.text}" for c in spec.forbidden]
    if spec.constraints:
        out += ["", "[limits]", *[f"{k} = {v}" for k, v in spec.constraints.items()]]
    out += ["", "[notes]", *[f"note{i} = {n}" for i, n in enumerate(spec.inert)]]
    return "\n".join(out) + "\n"


def _tool_manifest(spec: Spec) -> str:
    # The polarity is a boolean field beside the sentence, which is how an MCP
    # or function-calling manifest actually carries it.
    doc = {
        "agent": spec.id,
        "tools": [
            *[
                {"name": c.id, "description": c.text, "permitted": True}
                for c in spec.allowed
            ],
            *[
                {"name": c.id, "description": c.text, "permitted": False}
                for c in spec.forbidden
            ],
        ],
        "limits": spec.constraints or {},
        "notes": list(spec.inert),
    }
    return json.dumps(doc, indent=2) + "\n"


def _faq(spec: Spec) -> str:
    # Question and answer stay in one paragraph, so the polarity is inside the
    # chunk — but phrased as a reply rather than as a rule.
    out = [f"{spec.domain} — frequently asked questions", ""]
    for c in spec.allowed:
        out += [f"Can the agent {c.text}?", "Yes, that is within scope.", ""]
    for c in spec.forbidden:
        out += [f"Can the agent {c.text}?", "No, that is out of bounds.", ""]
    out += spec.inert
    return "\n".join(out) + "\n"


SHAPES = {
    "skill-card": _card,
    "xacml-xml": _xacml_xml,
    "ini-conf": _ini_conf,
    "tool-manifest": _tool_manifest,
    "faq": _faq,
    "md-headings": _md_headings,
    "system-prompt": _system_prompt,
    "iam-json": _iam_json,
    "yaml-config": _yaml_config,
    "readme": _readme,
    "numbered-rules": _numbered_rules,
    "runbook-table": _runbook_table,
    "legal-prose": _legal_prose,
    "opa-rules": _opa_rules,
    "cross-reference": _cross_reference,
    "narrative": _narrative,
    "malformed-frontmatter": _malformed_frontmatter,
}

UNSTRUCTURED_SHAPES = tuple(s for s in SHAPES if s != "skill-card")


# ── Loading ──────────────────────────────────────────────────────────────────


def _caps(raw: list[dict] | None) -> tuple[Capability, ...]:
    return tuple(Capability(c["id"], c["text"].strip()) for c in (raw or []))


def _load_spec(path: Path) -> Spec:
    d = yaml.safe_load(path.read_text())
    intents = tuple(
        Intent(
            id=i["id"],
            complexity=i["complexity"],
            text=i["text"].strip(),
            actions=tuple(
                Action(
                    id=a["id"],
                    callee=a["callee"],
                    args=dict(a.get("args") or {}),
                    description=a.get("description"),
                    relation=a["relation"],
                    aligned=bool(a["aligned"]),
                    policy=a["policy"],
                    evasion=a.get("evasion", ""),
                    basis=a.get("basis", "granted"),
                )
                for a in i["actions"]
            ),
        )
        for i in d.get("intents") or []
    )
    return Spec(
        id=d["id"],
        domain=d["domain"],
        shape=d["shape"],
        allowed=_caps(d.get("allowed")),
        forbidden=_caps(d.get("forbidden")),
        inert=tuple(s.strip() for s in (d.get("inert") or [])),
        constraints=dict(d.get("constraints") or {}),
        intents=intents,
        can_delegate_to=tuple(d.get("can-delegate-to") or []),
        note=d.get("note", ""),
    )


@cache
def specs() -> tuple[Spec, ...]:
    return tuple(_load_spec(p) for p in sorted(CORPUS_DIR.glob("*.yaml")))


@cache
def renderings() -> tuple[Rendering, ...]:
    return tuple(r for s in specs() for r in s.renderings)


@cache
def spec_by_id() -> dict[str, Spec]:
    return {s.id: s for s in specs()}
