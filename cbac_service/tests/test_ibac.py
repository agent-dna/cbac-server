import pytest

from cbac_service.ibac import (
    NormalizationError,
    canonical,
    fga_user,
    object_id,
    resource_values,
    validate_catalog,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  some value ", "some value"),
        ("é", "é"),  # NFC: decomposed and composed spellings agree
        (3, "3"),
        (2.5, "2.5"),
        (True, "true"),
        (False, "false"),
    ],
)
def test_canonical(value, expected):
    assert canonical(value) == expected


def test_canonical_makes_numbers_and_their_text_agree():
    # The query parser emits text; the tool call carries the typed value.
    assert canonical(3) == canonical("3")
    assert canonical(True) == canonical("true")


@pytest.mark.parametrize("value", ["", "   ", None, {"a": 1}, ["x"]], ids=repr)
def test_canonical_rejects_non_resources(value):
    with pytest.raises(NormalizationError):
        canonical(value)


def test_canonical_interprets_nothing():
    # No domain knowledge: different spellings stay different resources, so a
    # value only ever matches a grant spelled the same way.
    assert canonical("Value") != canonical("value")
    assert canonical("a/../b") == "a/../b"
    assert canonical("x, y") == "x, y"


def test_object_id_is_agent_scoped_and_opaque():
    a = object_id("agent-a", "tool_x", "value-1")
    assert a == object_id("agent-a", "tool_x", "value-1")
    assert a != object_id("agent-b", "tool_x", "value-1")
    assert a != object_id("agent-a", "tool_y", "value-1")
    assert a != object_id("agent-a", "tool_x", "value-2")
    assert a.startswith("tool_invocation:")
    assert "value-1" not in a
    assert fga_user("intent-1") != fga_user("intent-2")
    assert fga_user("intent-1").startswith("user:")


def test_resource_values_checks_lists_item_by_item():
    args = {"many": ["a", "b"], "one": "c", "absent": None}
    assert resource_values(args, "many") == ["a", "b"]
    assert resource_values(args, "one") == ["c"]
    assert resource_values(args, "absent") == []
    assert resource_values(args, "missing") == []


TOOLS = [
    {
        "name": "act_on",
        "description": "Act on a target.\nMore detail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "count": {"type": "integer"},
                "note": {"type": "string"},
            },
        },
    },
    {"name": "no_args", "inputSchema": {"type": "object", "properties": {}}},
]


def entry(tool, resources=(), free_text=()):
    return {"tool": tool, "resources": list(resources), "free_text": list(free_text)}


GOOD = [entry("act_on", ["target", "count"], ["note"]), entry("no_args")]


def test_validate_catalog_accepts_a_covering_catalog():
    catalog, rejected = validate_catalog(GOOD, TOOLS)
    assert rejected == []
    assert catalog == {
        "act_on": {
            "resources": ["target", "count"],
            "free_text": ["note"],
            "description": "Act on a target.",
        },
        "no_args": {"resources": [], "free_text": [], "description": ""},
    }


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        # `note` left out: an unclassified argument would be an open channel.
        (entry("act_on", ["target", "count"]), "unclassified"),
        (entry("act_on", ["target", "count"], ["note", "target"]), "both"),
        (entry("act_on", ["target", "count", "extra"], ["note"]), "not in the schema"),
        (entry("act_on", ["target", "target", "count"], ["note"]), "listed twice"),
        ({"tool": "act_on", "resources": "target", "free_text": []}, "lists"),
        (entry("missing_tool"), "no such tool"),
    ],
)
def test_validate_catalog_rejects(bad, why):
    catalog, rejected = validate_catalog([bad], TOOLS)
    assert bad["tool"] not in catalog
    assert any(why in r["reason"] for r in rejected if r["tool"] == bad["tool"])


def test_validate_catalog_rejects_a_tool_classified_twice():
    catalog, rejected = validate_catalog([GOOD[0], GOOD[0]], TOOLS)
    assert "act_on" in catalog
    assert any("twice" in r["reason"] for r in rejected)


def test_validate_catalog_reports_tools_the_parser_skipped():
    catalog, rejected = validate_catalog(GOOD[:1], TOOLS)
    assert set(catalog) == {"act_on"}
    assert {r["tool"] for r in rejected} == {"no_args"}
