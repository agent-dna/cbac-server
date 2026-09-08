"""Fixtures for the evaluation suite.

These tests load the real encoder / NLI / HHEM models and are slow, so they are
opt-in:  `uv run pytest cbac_service/tests/eval --run-eval -s`
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbac_service.chunking import flatten_policy_chunks
from cbac_service.tests.eval.dashboard import build
from cbac_service.tests.eval.datasets import (
    ALL_POLICIES,
    STRESS_POLICIES,
    STRUCTURED_POLICIES,
    UNSTRUCTURED_POLICIES,
)
from cbac_service.tests.eval.harness import Report, build_index, patch_search
from cbac_service.tests.eval.intents import (
    ADVERSARIAL_CASES,
    E2E_CASES,
    INTENT_CASES,
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-eval",
        action="store_true",
        default=False,
        help="run the slow model-backed CBAC evaluation suite",
    )
    parser.addoption(
        "--eval-json",
        default=None,
        metavar="PATH",
        help=(
            "write the evaluation's metrics, per-test outcomes and full report "
            "to PATH as JSON (see cbac_service/tests/eval/README.md)"
        ),
    )
    parser.addoption(
        "--eval-report",
        default=None,
        metavar="PATH",
        help="write the consolidated text report to PATH as well as stdout",
    )
    parser.addoption(
        "--eval-html",
        default=None,
        metavar="PATH",
        help=(
            "write a self-contained per-case dashboard to PATH — every case as "
            "one cell, with the detail behind each verdict"
        ),
    )


# Model-free, so it runs in the ordinary `uv run pytest` too: it guards the gold
# labels against the policy files they are matched into, and those files live
# outside this package where an edit would otherwise go unnoticed.
_ALWAYS_RUN = ("test_datasets_sanity.py",)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-eval"):
        return
    skip = pytest.mark.skip(reason="needs --run-eval (loads encoder/NLI/HHEM models)")
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        if "tests/eval/" in path and not path.endswith(_ALWAYS_RUN):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def report(request) -> Report:
    rep = Report()
    request.config._cbac_eval_report = rep
    return rep


def pytest_terminal_summary(terminalreporter, config):
    rep = getattr(config, "_cbac_eval_report", None)
    if rep is None or not rep.sections:
        return
    terminalreporter.write_line(rep.dump())

    report_path = config.getoption("--eval-report")
    if report_path:
        Path(report_path).write_text(rep.dump())
        terminalreporter.write_line(f"wrote text report to {report_path}")

    json_path = config.getoption("--eval-json")
    html_path = config.getoption("--eval-html")
    if json_path or html_path:
        payload = {
            "corpus": {
                "policies": len(ALL_POLICIES),
                "structured": len(STRUCTURED_POLICIES),
                "unstructured": len(UNSTRUCTURED_POLICIES),
                "stress": len(STRESS_POLICIES),
                "intents": len(INTENT_CASES),
                "adversarial": len(ADVERSARIAL_CASES),
                "e2e": len(E2E_CASES),
            },
            # A failing assertion here is a finding about the pipeline, not a
            # broken test, so outcomes are data rather than an exit condition.
            "tests": {
                report.nodeid.split("::", 1)[-1]: outcome
                for outcome in ("passed", "failed", "skipped")
                for report in terminalreporter.stats.get(outcome, [])
                if getattr(report, "when", None) == "call"
            },
            "metrics": dict(sorted(rep.metrics.items())),
            "policies": rep.policies,
            "cases": rep.cases,
            "sections": [
                {"title": title, "lines": lines} for title, lines in rep.sections
            ],
        }
        if json_path:
            Path(json_path).write_text(json.dumps(payload, indent=2, default=str))
            terminalreporter.write_line(
                f"wrote {len(payload['metrics'])} metrics to {json_path}"
            )
        if html_path:
            # Round-trip through JSON so the page is built from exactly the
            # bytes --eval-json would have written, not from live objects.
            Path(html_path).write_text(
                build(json.loads(json.dumps(payload, default=str)))
            )
            cases = sum(len(v) for v in payload["cases"].values())
            terminalreporter.write_line(f"wrote {cases}-case dashboard to {html_path}")


@pytest.fixture(scope="session")
def cbac():
    pytest.importorskip("sentence_transformers")
    from types import SimpleNamespace

    from cbac_service.cbac import CBAC

    # Provenance is never reached: these tests drive the internals directly.
    return CBAC(provenance=SimpleNamespace())


@pytest.fixture(scope="session")
def chunked() -> dict[str, list[str]]:
    """policy_id -> the chunks the production chunker actually produces."""
    return {p.id: flatten_policy_chunks(p.text) for p in ALL_POLICIES}


@pytest.fixture(scope="session")
def classified(cbac, chunked, report) -> dict[str, tuple[list[str], list[str]]]:
    """policy_id -> (allowed_chunks, forbidden_chunks) from `_classify_chunks`.

    Computed once — this is the expensive NLI pass, and both the classifier
    test and the "tiers on classifier output" test need the same buckets.

    Also records each policy's full detail on the report. A verdict is only
    explicable next to the index it was made against: "allowed" usually means
    the prohibition is sitting in the allowed bucket, which is visible here and
    nowhere else.
    """
    out = {pid: cbac._classify_chunks(chunks) for pid, chunks in chunked.items()}
    report.policies = [
        {
            "id": p.id,
            "style": p.style,
            "filename": p.filename,
            "group": (
                "structured"
                if p in STRUCTURED_POLICIES
                else "unstructured"
                if p in UNSTRUCTURED_POLICIES
                else "stress"
            ),
            "text": p.text,
            "chunks": [
                {
                    "text": c,
                    "gold": p.label_for(c),
                    "classified": (
                        "forbidden" if c in set(out[p.id][1]) else "allowed"
                    ),
                }
                for c in chunked[p.id]
            ],
        }
        for p in ALL_POLICIES
    ]
    return out


@pytest.fixture(scope="session")
def indices(cbac, chunked, classified):
    """policy index under both labellings: as `_classify_chunks` bucketed it,
    and as a human would."""
    out = {}
    for p in ALL_POLICIES:
        a, f = classified[p.id]
        out[("classifier", p.id)] = build_index(
            cbac, list(a) + list(f), ["allowed"] * len(a) + ["forbidden"] * len(f)
        )
        ga, gf = p.oracle_buckets(chunked[p.id])
        out[("oracle", p.id)] = build_index(
            cbac, list(ga) + list(gf), ["allowed"] * len(ga) + ["forbidden"] * len(gf)
        )
    return out


@pytest.fixture(scope="session")
def e2e_signals(cbac, indices, request):
    """Every layer's raw output on every E2E case, computed once.

    -> list of dicts: case, contradiction, intent_score, hhem,
       decision_classifier, decision_oracle
    """
    import asyncio
    from collections import defaultdict

    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)

    rows = {i: {"case": c} for i, c in enumerate(E2E_CASES)}

    # user-alignment layers (policy-independent)
    for i, c in enumerate(E2E_CASES):
        drift, intent_score = asyncio.run(cbac._check1_drift(c.user_intent, c.action))
        rows[i]["contradiction"] = 1.0 - intent_score
        rows[i]["intent_score"] = intent_score
        rows[i]["drift_denied"] = drift is not None
        rows[i]["hhem"] = cbac.hallucination_score(c.user_intent, c.action)

    # capability layer, once per (arm, policy) so the patch is set up once
    by_policy = defaultdict(list)
    for i, c in enumerate(E2E_CASES):
        by_policy[c.policy_id].append(i)
    for arm in ("classifier", "oracle"):
        for pid, idxs in by_policy.items():
            patch_search(mp, indices[(arm, pid)])
            for i in idxs:
                decision, reason, code, score = asyncio.run(
                    cbac._tiered_decision(None, "eval", E2E_CASES[i].action)
                )
                rows[i][f"decision_{arm}"] = decision
                rows[i][f"reason_{arm}"] = reason
                rows[i][f"error_code_{arm}"] = code
                rows[i][f"policy_score_{arm}"] = score
    return [rows[i] for i in range(len(E2E_CASES))]
