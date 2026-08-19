"""Fixtures for the evaluation suite.

These tests load the real encoder / NLI / HHEM models and are slow, so they are
opt-in:  `uv run pytest cbac_service/tests/eval --run-eval -s`
"""

from __future__ import annotations

import pytest

from cbac_service.chunking import flatten_policy_chunks
from cbac_service.tests.eval.datasets import ALL_POLICIES, E2E_CASES
from cbac_service.tests.eval.harness import Report, build_index, patch_search


def pytest_addoption(parser):
    parser.addoption(
        "--run-eval",
        action="store_true",
        default=False,
        help="run the slow model-backed CBAC evaluation suite",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-eval"):
        return
    skip = pytest.mark.skip(reason="needs --run-eval (loads encoder/NLI/HHEM models)")
    for item in items:
        if "tests/eval/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def report(request) -> Report:
    rep = Report()
    request.config._cbac_eval_report = rep
    return rep


def pytest_terminal_summary(terminalreporter, config):
    rep = getattr(config, "_cbac_eval_report", None)
    if rep is not None and rep.sections:
        terminalreporter.write_line(rep.dump())


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
def classified(cbac, chunked) -> dict[str, tuple[list[str], list[str]]]:
    """policy_id -> (allowed_chunks, forbidden_chunks) from `_classify_chunks`.

    Computed once — this is the expensive NLI pass, and both the classifier
    test and the "tiers on classifier output" test need the same buckets.
    """
    return {pid: cbac._classify_chunks(chunks) for pid, chunks in chunked.items()}


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
                decision, reason, score = asyncio.run(
                    cbac._tiered_decision(None, "eval", E2E_CASES[i].action)
                )
                rows[i][f"decision_{arm}"] = decision
                rows[i][f"reason_{arm}"] = reason
                rows[i][f"policy_score_{arm}"] = score
    return [rows[i] for i in range(len(E2E_CASES))]
