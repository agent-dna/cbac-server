"""Fixtures and output plumbing for the evaluation suite.

The model-backed tests load the encoder, the NLI cross-encoder and HHEM, so
they are opt-in behind `--run-eval`. `test_corpus_sanity.py` is the exception:
it checks the ground truth itself, needs no models, and runs in the ordinary
`uv run pytest` so a broken spec is caught without a slow run.

Failures here are findings, not breakage. The consolidated report prints after
the summary line regardless of outcome — every test records its numbers before
it asserts — and the suite's exit code is not a useful gate on its own.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from cbac_service.tests.eval.corpus import specs
from cbac_service.tests.eval.dashboard import build
from cbac_service.tests.eval.harness import Report


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
        help="write metrics, per-test outcomes and every per-case row to PATH as JSON",
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
        help="write the self-contained dashboard to PATH",
    )
    parser.addoption(
        "--eval-db",
        action="store_true",
        default=False,
        help=(
            "also index every rendering into real Postgres and run verify_cbac "
            "against it, reporting where it diverges from the in-memory arm "
            "(needs DATABASE_URL and a migrated database)"
        ),
    )


# Model-free, so it runs in the ordinary `uv run pytest`: it guards the corpus
# specs, which everything else in this directory is derived from.
_ALWAYS_RUN = ("test_corpus_sanity.py",)


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


@pytest.fixture(scope="session")
def cbac():
    pytest.importorskip("sentence_transformers")
    from types import SimpleNamespace

    from cbac_service.cbac import CBAC

    # Provenance is never reached: the tests drive the internals directly and
    # every policy is handed in as text.
    return CBAC(provenance=SimpleNamespace())


@pytest.fixture(scope="session")
def run(cbac, request, report):
    """Every stage's output on every case. The one expensive thing in the suite;
    every test reads from it rather than re-driving the models."""
    from _pytest.monkeypatch import MonkeyPatch

    from cbac_service.tests.eval import runner

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    result = runner.run(cbac, mp)
    request.config._cbac_eval_run = result

    if request.config.getoption("--eval-db"):
        from cbac_service.tests.eval.db_arm import compare

        report.add(*compare(cbac, result))
    return result


# ── Output ───────────────────────────────────────────────────────────────────


def _payload(config, rep: Report) -> dict:
    result = getattr(config, "_cbac_eval_run", None)
    row = dataclasses.asdict
    return {
        "corpus": {
            "policies": len(specs()),
            "renderings": 2 * len(specs()),
            "intents": sum(len(s.intents) for s in specs()),
            "actions": sum(len(s.actions) for s in specs()),
        },
        "specs": [
            {
                "id": s.id,
                "domain": s.domain,
                "shape": s.shape,
                "note": s.note,
                "constraints": s.constraints,
                "capabilities": [
                    {"id": c.id, "text": c.text, "gold": gold}
                    for gold, group in (
                        ("allowed", s.allowed),
                        ("forbidden", s.forbidden),
                    )
                    for c in group
                ],
                "renderings": [
                    {"kind": r.kind, "shape": r.shape, "text": r.text}
                    for r in s.renderings
                ],
                "intents": [
                    {
                        "id": i.id,
                        "complexity": i.complexity,
                        "text": i.text,
                        "actions": [
                            {
                                "id": a.id,
                                "call": a.call,
                                "text": a.text,
                                "relation": a.relation,
                                "aligned": a.aligned,
                                "policy": a.policy,
                                "should_block": a.should_block,
                            }
                            for a in i.actions
                        ],
                    }
                    for i in s.intents
                ],
            }
            for s in specs()
        ],
        "caps": [row(c) for c in result.caps] if result else [],
        "chunks": [row(c) for c in result.chunks] if result else [],
        "cases": [row(c) for c in result.cases] if result else [],
        "bare": [row(b) for b in result.bare] if result else [],
        "buckets": (
            {f"{k[0]}|{k[1]}": v for k, v in result.buckets.items()} if result else {}
        ),
        "hybrid_flips": result.hybrid_flips if result else [],
        "metrics": dict(sorted(rep.metrics.items())),
        "tests": {},
        "sections": [{"title": t, "lines": ln} for t, ln in rep.sections],
    }


def pytest_terminal_summary(terminalreporter, config):
    rep = getattr(config, "_cbac_eval_report", None)
    if rep is None or not rep.sections:
        return
    terminalreporter.write_line(rep.dump())

    text_path = config.getoption("--eval-report")
    if text_path:
        Path(text_path).write_text(rep.dump())
        terminalreporter.write_line(f"wrote text report to {text_path}")

    json_path = config.getoption("--eval-json")
    html_path = config.getoption("--eval-html")
    if not (json_path or html_path):
        return

    payload = _payload(config, rep)
    # A failing assertion is a finding about the pipeline, so outcomes are data
    # rather than an exit condition.
    payload["tests"] = {
        r.nodeid.split("::", 1)[-1]: outcome
        for outcome in ("passed", "failed", "skipped")
        for r in terminalreporter.stats.get(outcome, [])
        if getattr(r, "when", None) == "call"
    }
    if json_path:
        Path(json_path).write_text(json.dumps(payload, indent=2, default=str))
        terminalreporter.write_line(
            f"wrote {len(payload['metrics'])} metrics to {json_path}"
        )
    if html_path:
        # Round-trip through JSON so the page is built from exactly the bytes
        # --eval-json would have written, not from live objects.
        Path(html_path).write_text(build(json.loads(json.dumps(payload, default=str))))
        terminalreporter.write_line(
            f"wrote {len(payload['cases'])}-action dashboard to {html_path}"
        )
