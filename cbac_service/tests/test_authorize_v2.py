"""The v2 pipeline end to end, against in-memory OpenFGA, DB and models.

The scenario is the talk's indirect prompt injection: the user asks to tell
Bob the meeting is at 3pm, and a calendar entry the agent reads tells it to
invite an attacker and tell Bob the meeting is cancelled.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

import cbac_service.cbac as cbac_mod
from cbac_service import error_codes as ec
from cbac_service import main, parsers
from cbac_service.cbac import CBAC
from cbac_service.config import QUERY_PARSER_RETRY_SECONDS
from cbac_service.entity import PolicyApproveRequest, PolicyUploadRequest
from cbac_service.ibac import TUPLE_SCHEME, object_id

AGENT = "did:agent:calendar"
INTENT = "intent-1"
QUERY = "Check today's calendar and tell Bob the meeting is at 3pm"
TRUSTED = {
    "contacts": {"Bob": "bob@company.com", "CEO": "ceo@company.com"},
    "agenda": {"first item": 3},
}

CATALOG = {
    "read_calendar": {"resources": ["day"], "free_text": []},
    "send_invite": {"resources": ["to"], "free_text": ["body"]},
    "read_file": {"resources": ["path"], "free_text": []},
    "open_item": {"resources": ["item"], "free_text": []},
    "list_events": {"resources": [], "free_text": []},
    "transfer": {"resources": ["from", "to"], "free_text": ["memo"]},
    "deploy": {"resources": ["env"], "free_text": []},
}


def deny(tool, match, resource):
    return {"tool": tool, "match": match, "resource": resource, "source": "s"}


DENY_RULES = [
    deny("read_file", "glob", "/etc/*"),
    deny("send_invite", "exact", "ceo@company.com"),
    deny("transfer", "exact", "offshore"),
    deny("deploy", "exact", "production"),
]


def grant(tool, argument, resource):
    return {"tool": tool, "argument": argument, "resource": resource}


# What the (fake) query parser proposes, whatever the query. Grounding keeps
# the ones the query at hand or the trusted data names. The CEO grant is
# grounded in the trusted data; the exact deny tuple must still beat it.
GRANTS = [
    grant("read_calendar", "day", "today"),
    grant("send_invite", "to", "bob@company.com"),
    grant("send_invite", "to", "ceo@company.com"),
    grant("read_file", "path", "/etc/hosts"),
    grant("read_file", "path", "/ETC/hosts"),
    grant("open_item", "item", "3"),
    grant("list_events", "*", "*"),
    grant("transfer", "from", "savings"),
    grant("transfer", "to", "checking"),
    grant("transfer", "from", "offshore"),
    grant("transfer", "to", "offshore"),
    grant("deploy", "env", "Production"),
    grant("deploy", "env", "staging"),
]


class FakeFGA:
    """allow tuples with expiry, exact deny tuples — the model's semantics."""

    def __init__(self):
        self.grants: dict[tuple[str, str], datetime] = {}
        self.denies: set[str] = set()
        self.fail = False

    async def write_grants(self, grants, grant_time, ttl_seconds):
        for user, obj in grants:
            self.grants[(user, obj)] = grant_time + timedelta(seconds=ttl_seconds)

    async def replace_denies(self, add, remove):
        self.denies |= set(add)
        self.denies -= set(remove) - set(add)

    async def can_invoke(self, user, obj, now):
        if self.fail:
            raise ConnectionError("openfga down")
        expiry = self.grants.get((user, obj))
        return expiry is not None and now < expiry and obj not in self.denies

    async def is_blocked(self, user, obj):
        return obj in self.denies


class FakeSession:
    async def commit(self):
        pass

    async def rollback(self):
        pass


@pytest.fixture
def world(rows, decisions, monkeypatch):
    fga = FakeFGA()
    version = SimpleNamespace(
        id=1,
        agent_id=AGENT,
        catalog=CATALOG,
        deny_rules=DENY_RULES,
        trusted_data=TRUSTED,
        tuple_scheme=TUPLE_SCHEME,
    )
    # The deny tuples activation writes for this version.
    fga.denies.update(cbac_mod._exact_deny_objects(version))
    intents: dict[str, SimpleNamespace] = {}
    failures: dict[str, SimpleNamespace] = {}
    llm_calls: list[str] = []
    llm_failures: list[Exception] = []

    async def get_intent_failure(session, intent_id):
        return failures.get(intent_id)

    async def record_intent_failure(session, intent_id, reason):
        failures[intent_id] = SimpleNamespace(
            reason=reason, failed_at=datetime.now(timezone.utc)
        )

    async def get_active_policy_version(session, agent_id):
        return version if agent_id == AGENT else None

    async def lock_intent(session, intent_id):
        pass

    async def get_intent(session, intent_id):
        return intents.get(intent_id)

    async def insert_intent(session, **fields):
        intents[fields["intent_id"]] = SimpleNamespace(**fields)
        return intents[fields["intent_id"]]

    async def chat_json(model, system, user, schema, **limits):
        llm_calls.append(user)
        if llm_failures:
            raise llm_failures.pop()
        return {"grants": GRANTS}

    monkeypatch.setattr(cbac_mod, "get_intent_failure", get_intent_failure)
    monkeypatch.setattr(cbac_mod, "record_intent_failure", record_intent_failure)
    monkeypatch.setattr(cbac_mod, "fga", fga)
    monkeypatch.setattr(
        cbac_mod, "get_active_policy_version", get_active_policy_version
    )
    monkeypatch.setattr(cbac_mod, "lock_intent", lock_intent)
    monkeypatch.setattr(cbac_mod, "get_intent", get_intent)
    monkeypatch.setattr(cbac_mod, "insert_intent", insert_intent)
    monkeypatch.setattr(parsers, "chat_json", chat_json)

    cbac = CBAC(provenance=SimpleNamespace())

    def nli(pairs):
        # Contradiction when the free text changes the plan the user stated.
        return [
            {
                "contradiction": 0.97 if "cancelled" in h else 0.01,
                "entailment": 0.0,
                "neutral": 0.0,
            }
            for _, h in pairs
        ]

    monkeypatch.setattr(cbac, "_nli_scores_batch", nli)
    monkeypatch.setattr(cbac, "hallucination_score", lambda source, generated: 0.5)

    async def no_policy_drift(session, agent_id, text):
        return None, 0.6

    monkeypatch.setattr(cbac, "_policy_drift", no_policy_drift)

    def call(tool, arguments, query=QUERY, agent=AGENT, intent=INTENT):
        return asyncio.run(
            cbac.authorize_call(
                FakeSession(),
                agent_id=agent,
                intent_id=intent,
                user_query=query,
                tool=tool,
                arguments=arguments,
            )
        )

    return SimpleNamespace(
        call=call,
        cbac=cbac,
        fga=fga,
        version=version,
        intents=intents,
        failures=failures,
        llm_calls=llm_calls,
        llm_failures=llm_failures,
        decisions=decisions,
        rows=rows,
    )


def test_the_injection_replayed(world):
    """The slides' scenario: the exact gate stops the attacker, the semantic
    check stops the lie, the honest call goes through."""
    read = world.call("read_calendar", {"day": "today"})
    assert (read.decision, read.error_code) == ("allow", ec.ALLOW_NO_FREE_TEXT)

    attacker = world.call("send_invite", {"to": "attacker@evil.com", "body": "Hi"})
    assert (attacker.decision, attacker.error_code) == ("deny", ec.AUTHZ_NO_ALLOW_TUPLE)

    bcc = world.call(
        "send_invite",
        {"to": "bob@company.com", "bcc": "attacker@evil.com", "body": "At 3pm."},
    )
    assert (bcc.decision, bcc.error_code) == ("deny", ec.MAPPING_UNKNOWN_ARGUMENT)

    lie = world.call(
        "send_invite", {"to": "bob@company.com", "body": "The meeting is cancelled."}
    )
    assert (lie.decision, lie.error_code) == ("deny", ec.SEMANTIC_INTENT_DRIFT)
    assert lie.intent_score == pytest.approx(0.03)

    honest = world.call(
        "send_invite", {"to": " bob@company.com ", "body": "The meeting is at 3pm."}
    )
    assert (honest.decision, honest.error_code) == ("allow", ec.ALLOW_FREE_TEXT_PASSED)
    assert honest.hallucination_score == 0.5
    assert honest.policy_score == 0.6

    # One parse for the whole intent, however many calls it serves.
    assert len(world.llm_calls) == 1
    # Every call is audited under its intent id; the semantic outcomes fold trust.
    assert [d.error_code for d in world.decisions] == [4301, 4103, 4006, 4201, 4302]
    assert {d.intent_id for d in world.decisions} == {INTENT}
    assert all(d.user_intent == QUERY for d in world.decisions)
    assert [r.callee_name for r in world.rows] == ["send_invite", "send_invite"]


def test_exact_deny_tuple_beats_a_grant(world):
    result = world.call("send_invite", {"to": "ceo@company.com"})
    assert (result.decision, result.error_code) == ("deny", ec.AUTHZ_DENY_TUPLE)


def test_glob_deny_rule_beats_a_grant(world):
    result = world.call("read_file", {"path": "/etc/hosts"})
    assert (result.decision, result.error_code) == ("deny", ec.AUTHZ_GLOB_DENY)


def test_a_typed_resource_matches_its_granted_text(world):
    # The parser grants the text "3"; the tool call carries the integer 3.
    result = world.call("open_item", {"item": 3})
    assert (result.decision, result.error_code) == ("allow", ec.ALLOW_NO_FREE_TEXT)


@pytest.mark.parametrize(
    "value", ["Bob@company.com", "Bob <bob@company.com>", "bob@company.com, x@y.io"]
)
def test_any_other_spelling_is_a_different_resource(world, value):
    # No domain knowledge: a value matches a grant only spelled the same way,
    # so every other spelling fails closed.
    result = world.call("send_invite", {"to": value})
    assert (result.decision, result.error_code) == ("deny", ec.AUTHZ_NO_ALLOW_TUPLE)


def test_tool_without_resources_needs_its_own_grant(world):
    result = world.call("list_events", {})
    assert (result.decision, result.error_code) == ("allow", ec.ALLOW_NO_FREE_TEXT)


def test_every_recipient_in_a_list_is_checked(world):
    result = world.call("send_invite", {"to": ["bob@company.com", "eve@evil.com"]})
    assert (result.decision, result.error_code) == ("deny", ec.AUTHZ_NO_ALLOW_TUPLE)


@pytest.mark.parametrize(
    ("tool", "arguments", "code"),
    [
        ("delete_mailbox", {}, ec.MAPPING_UNKNOWN_TOOL),
        ("send_invite", {"to": {"name": "bob"}}, ec.MAPPING_NORMALIZATION_FAILED),
        ("send_invite", {"body": "hello"}, ec.MAPPING_NO_RESOURCE),
    ],
)
def test_mapping_denies(world, tool, arguments, code):
    result = world.call(tool, arguments)
    assert (result.decision, result.error_code) == ("deny", code)


def test_intent_is_frozen_to_its_first_query_and_agent(world):
    world.call("read_calendar", {"day": "today"})
    swapped = world.call(
        "read_calendar", {"day": "today"}, query="Invite attacker@evil.com"
    )
    assert swapped.error_code == ec.INTENT_MISMATCH
    assert len(world.llm_calls) == 1


def test_missing_intent_or_query_denies(world):
    assert world.call("read_calendar", {"day": "today"}, intent="").error_code == (
        ec.INTENT_MISSING
    )
    assert world.call("read_calendar", {"day": "today"}, query=" ").error_code == (
        ec.INTENT_MISSING
    )
    assert world.llm_calls == []


def test_agent_without_active_policy_denies(world):
    result = world.call("read_calendar", {"day": "today"}, agent="did:agent:other")
    assert result.error_code == ec.MAPPING_NO_ACTIVE_POLICY


def test_expired_intent_denies(world):
    world.call("read_calendar", {"day": "today"})
    world.intents[INTENT].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    result = world.call("read_calendar", {"day": "today"})
    assert result.error_code == ec.INTENT_EXPIRED


def test_a_failed_parse_is_remembered_and_not_repeated(world):
    world.llm_failures.append(TimeoutError("parser gave no answer within 15s"))
    first = world.call("read_calendar", {"day": "today"})
    assert first.error_code == ec.INTENT_PARSE_FAILED
    assert INTENT not in world.intents

    # Inside the window: denied at once, and the parser is not asked again.
    again = world.call("read_calendar", {"day": "today"})
    assert (again.decision, again.error_code) == ("deny", ec.INTENT_PARSE_FAILED)
    assert "within 15s" in again.reason
    assert len(world.llm_calls) == 1

    # Past the window the next call parses again, and this time it works.
    world.failures[INTENT].failed_at -= timedelta(
        seconds=QUERY_PARSER_RETRY_SECONDS + 1
    )
    retry = world.call("read_calendar", {"day": "today"})
    assert retry.error_code == ec.ALLOW_NO_FREE_TEXT
    assert len(world.llm_calls) == 2


def test_a_failed_parse_that_cannot_be_recorded_still_denies(world, monkeypatch):
    async def down(session, intent_id, reason):
        raise ConnectionError("db down")

    monkeypatch.setattr(cbac_mod, "record_intent_failure", down)
    world.llm_failures.append(TimeoutError("slow"))
    result = world.call("read_calendar", {"day": "today"})
    assert (result.decision, result.error_code) == ("deny", ec.INTENT_PARSE_FAILED)


# ── the argument is part of the grant ───────────────────────────────────────


def test_a_value_is_granted_for_one_argument_only(world):
    """savings is granted as the source and checking as the destination, so
    the same two values the other way round are not granted."""

    def transfer(source, destination):
        return world.call(
            "transfer",
            {"from": source, "to": destination},
            query="move $100 from savings to checking",
            intent="intent-transfer",
        )

    asked = transfer("savings", "checking")
    assert (asked.decision, asked.error_code) == ("allow", ec.ALLOW_NO_FREE_TEXT)
    swapped = transfer("checking", "savings")
    assert (swapped.decision, swapped.error_code) == ("deny", ec.AUTHZ_NO_ALLOW_TUPLE)


@pytest.mark.parametrize(
    "arguments",
    [{"from": "offshore", "to": "checking"}, {"from": "savings", "to": "offshore"}],
)
def test_an_exact_deny_blocks_its_value_in_every_argument(world, arguments):
    # The rule names no argument. Both calls hold a grant for every value.
    query = "move $100 from offshore to checking, then from savings to offshore"
    result = world.call("transfer", arguments, query=query, intent="intent-offshore")
    assert (result.decision, result.error_code) == ("deny", ec.AUTHZ_DENY_TUPLE)


# ── deny rules match loosely, allow tuples strictly ─────────────────────────


def test_another_spelling_does_not_get_past_a_deny_rule(world):
    # The rules are on "production" and "/etc/*". The user typed neither
    # spelling, so the grants are for "Production" and "/ETC/hosts".
    def call(tool, arguments):
        return world.call(
            tool,
            arguments,
            query="deploy to Production and to staging, then read /ETC/hosts",
            intent="intent-deploy",
        )

    exact = call("deploy", {"env": "Production"})
    assert (exact.decision, exact.error_code) == ("deny", ec.AUTHZ_DENY_TUPLE)
    glob = call("read_file", {"path": "/ETC/hosts"})
    assert (glob.decision, glob.error_code) == ("deny", ec.AUTHZ_GLOB_DENY)

    # Loose matching only ever denies: a grant still needs its exact spelling.
    assert call("deploy", {"env": "staging"}).error_code == ec.ALLOW_NO_FREE_TEXT
    assert call("deploy", {"env": "Staging"}).error_code == ec.AUTHZ_NO_ALLOW_TUPLE


def test_a_version_with_stale_tuples_denies_every_call(world):
    # Activated under another tuple scheme: its deny tuples are not on the
    # objects the gate checks, so nothing is decided on it.
    world.version.tuple_scheme = None
    result = world.call("read_calendar", {"day": "today"})
    assert (result.decision, result.error_code) == ("deny", ec.MAPPING_STALE_TUPLES)
    assert world.llm_calls == []


def test_openfga_failure_denies(world):
    world.call("read_calendar", {"day": "today"})
    world.fga.fail = True
    result = world.call("read_calendar", {"day": "today"})
    assert (result.decision, result.error_code) == ("deny", ec.AUTHZ_FGA_ERROR)


def test_semantic_model_failure_denies(world, monkeypatch):
    def boom(pairs):
        raise RuntimeError("nli down")

    monkeypatch.setattr(world.cbac, "_nli_scores_batch", boom)
    result = world.call("send_invite", {"to": "bob@company.com", "body": "At 3pm."})
    assert (result.decision, result.error_code) == ("deny", ec.SEMANTIC_ERROR)


def test_long_free_text_is_scored_to_the_end(world):
    # The contradiction sits past what one 512-token window would read.
    body = "Agenda item. " * 400 + "The meeting is cancelled."
    result = world.call("send_invite", {"to": "bob@company.com", "body": body})
    assert result.error_code == ec.SEMANTIC_INTENT_DRIFT


def test_structured_free_text_is_scored_whole(world):
    body = {"note": "fine", "postscript": "The meeting is cancelled."}
    result = world.call("send_invite", {"to": "bob@company.com", "body": body})
    assert result.error_code == ec.SEMANTIC_INTENT_DRIFT


# ── policy drift: Tier 1 and Tier 2 only ever deny ──────────────────────────


def drift_engine(monkeypatch, allowed, forbidden, contradiction):
    cbac = CBAC(provenance=SimpleNamespace())
    monkeypatch.setattr(
        cbac,
        "_get_encoder",
        lambda: SimpleNamespace(
            encode=lambda texts, normalize_embeddings=True: np.zeros((len(texts), 4))
        ),
    )

    async def search(session, agent_id, vec, *args, chunk_type="allowed", **kwargs):
        score = allowed if chunk_type == "allowed" else forbidden
        return [SimpleNamespace(score=score, chunk_text=f"{chunk_type} chunk")]

    monkeypatch.setattr(cbac_mod, "vector_search", search)
    monkeypatch.setattr(
        cbac,
        "_nli_scores",
        lambda p, h: {"contradiction": contradiction, "entailment": 0.0},
    )
    return cbac


@pytest.mark.parametrize(
    ("allowed", "forbidden", "contradiction", "code"),
    [
        (0.2, 0.6, 0.0, ec.SEMANTIC_POLICY_DRIFT_TIER1),
        (0.5, 0.5, 0.9, ec.SEMANTIC_POLICY_DRIFT_TIER2),
        # Clear allow-side gap and no contradiction: passes, as does the gray zone.
        (0.9, 0.1, 0.0, None),
        (0.5, 0.5, 0.3, None),
    ],
)
def test_policy_drift_is_deny_only(
    monkeypatch, allowed, forbidden, contradiction, code
):
    cbac = drift_engine(monkeypatch, allowed, forbidden, contradiction)
    denial, score = asyncio.run(cbac._policy_drift(None, AGENT, "some text"))
    assert (denial.error_code if denial else None) == code
    assert score is not None


# ── policy upload and activation ─────────────────────────────────────────────


@pytest.mark.parametrize("review_required", [False, True])
def test_upload_activates_unless_review_is_required(monkeypatch, review_required):
    fga = FakeFGA()
    old = SimpleNamespace(
        id=1,
        agent_id=AGENT,
        catalog={"send_invite": {"resources": ["to"], "free_text": ["body"]}},
        deny_rules=[deny("send_invite", "exact", "old@x.io")],
    )
    stored: list[SimpleNamespace] = []
    indexed: list[str] = []

    async def insert_policy_version(session, **fields):
        stored.append(
            SimpleNamespace(id=2, approved_at=None, tuple_scheme=None, **fields)
        )
        return stored[-1]

    async def get_active_policy_version(session, agent_id):
        return old

    async def mark_policy_version_active(session, version):
        version.status = "active"

    async def index_policy(session, agent_id, policy=None):
        indexed.append(policy)
        return 3

    async def chat_json(model, system, user, schema, **limits):
        return {
            "catalog": [
                {
                    "tool": "send_invite",
                    "resources": ["to"],
                    "free_text": ["body"],
                }
            ],
            "deny_rules": [
                {
                    "tool": "send_invite",
                    "match": "exact",
                    "resource": " ceo@company.com ",
                    "source": "Never invite the CEO (ceo@company.com).",
                }
            ],
            "unenforceable": [],
        }

    monkeypatch.setattr(cbac_mod, "fga", fga)
    fga.denies.add(object_id(AGENT, "send_invite", "to", "old@x.io"))
    monkeypatch.setattr(cbac_mod, "insert_policy_version", insert_policy_version)
    monkeypatch.setattr(
        cbac_mod, "get_active_policy_version", get_active_policy_version
    )
    monkeypatch.setattr(
        cbac_mod, "mark_policy_version_active", mark_policy_version_active
    )
    monkeypatch.setattr(parsers, "chat_json", chat_json)
    cbac = CBAC(provenance=SimpleNamespace())
    monkeypatch.setattr(cbac, "index_policy", index_policy)

    policy = "Never invite the CEO (ceo@company.com)."
    tools = [
        {
            "name": "send_invite",
            "inputSchema": {
                "properties": {
                    "to": {"type": "string", "format": "email"},
                    "body": {"type": "string"},
                }
            },
        }
    ]
    document = (
        f"---\nPolicy Instructions\n{policy}\n---\nTool Schema\n"
        f"{json.dumps(tools)}\n---\n"
    )
    version = asyncio.run(
        cbac.upload_policy(None, AGENT, document, TRUSTED, review_required)
    )

    assert version.deny_rules[0]["resource"] == "ceo@company.com"
    assert version.policy == policy
    assert version.tool_schemas == tools
    if review_required:
        # Held for an admin: nothing indexed, the old deny tuples untouched.
        assert version.status == "pending"
        assert indexed == []
        assert fga.denies == {object_id(AGENT, "send_invite", "to", "old@x.io")}
        assert version.tuple_scheme is None
    else:
        assert version.status == "active"
        # Only the instructions are chunked for policy drift, never the schema.
        assert indexed == [policy]
        assert fga.denies == {object_id(AGENT, "send_invite", "to", "ceo@company.com")}
        assert version.tuple_scheme == TUPLE_SCHEME


def test_reactivating_the_active_version_rewrites_its_deny_tuples(monkeypatch):
    """What an operator does after a tuple scheme change: the store holds no
    deny tuple the gate would find, and the version is not stamped current."""
    fga = FakeFGA()
    version = SimpleNamespace(
        id=1,
        agent_id=AGENT,
        policy="p",
        catalog=CATALOG,
        deny_rules=DENY_RULES,
        status="active",
        tuple_scheme=None,
    )

    async def get_active_policy_version(session, agent_id):
        return version

    async def mark_policy_version_active(session, version):
        version.status = "active"

    async def index_policy(session, agent_id, policy=None):
        return 1

    monkeypatch.setattr(cbac_mod, "fga", fga)
    monkeypatch.setattr(
        cbac_mod, "get_active_policy_version", get_active_policy_version
    )
    monkeypatch.setattr(
        cbac_mod, "mark_policy_version_active", mark_policy_version_active
    )
    cbac = CBAC(provenance=SimpleNamespace())
    monkeypatch.setattr(cbac, "index_policy", index_policy)

    asyncio.run(cbac.activate_policy_version(None, version))

    # One tuple per resource argument of each exact rule's tool; globs none.
    assert fga.denies == {
        object_id(AGENT, "send_invite", "to", "ceo@company.com"),
        object_id(AGENT, "transfer", "from", "offshore"),
        object_id(AGENT, "transfer", "to", "offshore"),
        object_id(AGENT, "deploy", "env", "production"),
    }
    assert version.tuple_scheme == TUPLE_SCHEME


# ── HTTP: the policy key ─────────────────────────────────────────────────────


def envelope(response):
    return response.status_code, json.loads(response.body)


UPLOAD = PolicyUploadRequest(agent_id=AGENT, policy="p")


def test_policy_endpoints_refuse_without_a_configured_key(monkeypatch):
    monkeypatch.setattr(main, "CBAC_POLICY_KEY", "")
    status, body = envelope(asyncio.run(main.upload_policy_v2(UPLOAD, "anything")))
    assert status == 503
    assert body["success"] is False


def test_policy_endpoints_refuse_a_wrong_key(monkeypatch):
    monkeypatch.setattr(main, "CBAC_POLICY_KEY", "s3cret")
    for key in (None, "", "wrong"):
        status, _ = envelope(asyncio.run(main.upload_policy_v2(UPLOAD, key)))
        assert status == 401
    status, _ = envelope(asyncio.run(main.read_policies_v2(AGENT, "wrong")))
    assert status == 401


# ── the guard's half of the v2 contract ─────────────────────────────────────


def test_guard_authorize_call_posts_what_the_v2_route_reads(monkeypatch):
    """Guard and endpoint live in separate packages; a renamed key would fail
    silently, as a 4001 deny on every call. Pin the two to each other."""
    from cbac import authorize_call
    from cbac_service.entity import AuthorizeCallRequest

    posted = []

    class _Response:
        text = ""

        @staticmethod
        def json():
            return {"data": {"decision": "allow", "status_code": 4301, "hash": "h"}}

    monkeypatch.setattr(
        "requests.post",
        lambda url, json, timeout: (posted.append((url, json)), _Response())[1],
    )
    when = datetime(2026, 11, 2, tzinfo=timezone.utc)
    result = asyncio.run(
        authorize_call(
            AGENT,
            INTENT,
            QUERY,
            "send_invite",
            {"to": ["bob@company.com"], "when": when},
            mcp_did="did:mcp:cal",
            cbac_url="http://svc:8767",
        )
    )
    assert result == ("allow", 4301, "h")

    url, body = posted[0]
    assert url == "http://svc:8767/cbac/v2/authorize"
    assert "/cbac/v2/authorize" in main.app.openapi()["paths"]
    parsed = AuthorizeCallRequest(**body)
    assert parsed.intent_id == INTENT
    assert parsed.user_query == QUERY
    # Structure survives — the service checks list items one by one.
    assert parsed.arguments == {"to": ["bob@company.com"], "when": str(when)}


class _Session:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


@pytest.mark.parametrize(
    ("status", "http_status"),
    [("pending", 200), ("active", 200), ("superseded", 409)],
)
def test_approve_activates_a_pending_or_an_active_version(
    monkeypatch, status, http_status
):
    version = SimpleNamespace(
        id=7,
        agent_id=AGENT,
        status=status,
        policy_hash="h",
        policy="p",
        catalog={},
        deny_rules=[],
        rejected=[],
        unenforceable=[],
        trusted_data={},
        parser_model="m",
        created_at=datetime.now(timezone.utc),
        approved_at=None,
    )
    activated: list[int] = []

    async def get_policy_version(session, agent_id, version_id):
        return version

    async def activate_policy_version(session, version):
        activated.append(version.id)

    monkeypatch.setattr(main, "CBAC_POLICY_KEY", "s3cret")
    monkeypatch.setattr(main, "get_session", _Session)
    monkeypatch.setattr(main, "get_policy_version", get_policy_version)
    monkeypatch.setattr(
        main,
        "_get_cbac",
        lambda: SimpleNamespace(activate_policy_version=activate_policy_version),
    )
    body = PolicyApproveRequest(version_id=7)
    answered, _ = envelope(asyncio.run(main.approve_policy_v2(AGENT, body, "s3cret")))
    assert answered == http_status
    assert activated == ([7] if http_status == 200 else [])


def test_a_document_that_does_not_split_is_a_422(monkeypatch):
    monkeypatch.setattr(main, "CBAC_POLICY_KEY", "s3cret")
    monkeypatch.setattr(main, "_get_cbac", lambda: CBAC(provenance=SimpleNamespace()))
    monkeypatch.setattr(main, "get_session", _Session)
    no_tools = PolicyUploadRequest(
        agent_id=AGENT, policy="---\nPolicy Instructions\nNever.\n---\n"
    )
    status, body = envelope(asyncio.run(main.upload_policy_v2(no_tools, "s3cret")))
    assert status == 422
    assert "Tool Schema" in body["message"]
