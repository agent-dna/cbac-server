"""
CBAC Adversarial Benchmark — Tier 3 (structured LLMVerdict) cases.

Place at: scripts/cbac_benchmark/run_llm_tier3_cases.py

LLM-01..LLM-04 need Tier 3 to actually run, which only happens when Tier 1
and Tier 2 are both inconclusive AND an llm_backend is configured. Forcing
the real embedding/NLI pipeline into that exact gray zone on demand isn't
reliable, so — following the same technique the team's own
cbac_service/tests/test_cbac_verify.py uses (monkeypatching the encoder/NLI
stubs) — this script forces Tier 1/2 to be inconclusive directly and
supplies a canned Tier-3 backend, to isolate and test *only* the
`LLMVerdict`-handling logic in `_tiered_decision`'s Tier 3 branch.

LLM-01 and LLM-02 use a conforming backend that returns a real `LLMVerdict`,
confirming the old free-text keyword-matching bug is now structurally
impossible. LLM-03 and LLM-04 deliberately misbehave — a backend returning a
bare string, and one returning an `LLMVerdict` with an invalid `decision` —
to confirm both degrade to "advise" via `TIER3_LLM_MALFORMED_ADVISE` rather
than raising or silently guessing.

This does not need Postgres or a running server — it calls
`cbac._tiered_decision` directly with the DB search functions monkeypatched
to force the gray zone, exactly like test_cbac_verify.py's pattern.

Usage (from repo root):
    python3 scripts/cbac_benchmark/run_llm_tier3_cases.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cbac_service.cbac as cbac_mod
from cbac_service.cbac import CBAC
from cbac_service.skills import LLMVerdict

BENCH_DIR = Path(__file__).resolve().parent
INTENTS_FILE = BENCH_DIR / "intents.yaml"
RESULTS_FILE = BENCH_DIR / "results_llm_tier3.json"


def build_gray_zone_cbac(llm_backend) -> CBAC:
    """CBAC wired so Tier 1 and Tier 2 are forced inconclusive (equal
    allowed/forbidden scores, zero entailment/contradiction) and Tier 3 is
    configured with the given llm_backend callable — isolating the
    LLMVerdict-handling branch under test."""

    cbac = CBAC(provenance=MagicMock(), llm_backend=llm_backend)

    # No real model loads needed — stub the encoder/NLI so this runs in
    # milliseconds with no GPU/network dependency.
    cbac._get_encoder = lambda: SimpleNamespace(
        encode=lambda texts, normalize_embeddings=True: np.zeros((len(texts), 4))
    )
    cbac._nli_scores = lambda premise, hypothesis: {
        "entailment": 0.0,
        "contradiction": 0.0,
    }

    async def _search(
        session, agent_id, intent_vec, *args, chunk_type="allowed", **kwargs
    ):
        # Equal scores both sides -> Tier 1 gap == 0 -> escalates to Tier 2.
        return [SimpleNamespace(score=0.5, chunk_text=f"{chunk_type} chunk")]

    async def _get_chunks(session, agent_id):
        return ["some policy chunk"]

    cbac_mod.vector_search = _search
    cbac_mod.hybrid_search = _search
    cbac_mod.get_policy_chunks = _get_chunks

    return cbac


def make_conforming_backend(decision: str, reason: str):
    """A well-behaved backend: always returns a real LLMVerdict."""

    async def llm_backend(intent_text: str, policy_text: str) -> LLMVerdict:
        return LLMVerdict(decision=decision, reason=reason)

    return llm_backend


def make_bare_string_backend(text: str):
    """LLM-03: a backend that hasn't been updated to the LLMVerdict contract
    — returns a plain string, the old free-text shape."""

    async def llm_backend(intent_text: str, policy_text: str) -> Any:
        return text

    return llm_backend


async def run() -> None:
    all_cases = {
        c["id"]: c
        for c in yaml.safe_load(INTENTS_FILE.read_text())
        if c["id"] in {"LLM-01", "LLM-02", "LLM-03", "LLM-04"}
    }

    results = []
    for case_id, case in all_cases.items():
        if case_id == "LLM-03":
            # Deliberately non-conforming: bare string, not an LLMVerdict.
            backend_desc = "<bare string, not an LLMVerdict>"
            llm_backend = make_bare_string_backend(
                "I would deny this, it's not prohibited though."
            )
        else:
            decision = case["llm_backend_decision"]
            reason = case["llm_backend_reason"].strip()
            backend_desc = f"LLMVerdict(decision={decision!r}, reason={reason!r})"
            llm_backend = make_conforming_backend(decision, reason)

        cbac = build_gray_zone_cbac(llm_backend)
        decision, reason, error_code, policy_score = await cbac._tiered_decision(
            session=None, agent_id="benchmark:llm-tier3", intent_text="the request text"
        )
        print(
            f"[{case_id}] backend={backend_desc}\n  -> decision={decision} | {reason} | error_code={error_code}"
        )
        results.append(
            {
                "id": case_id,
                "category": "LLM",
                "backend": backend_desc,
                "decision": decision,
                "reason": reason,
                "error_code": error_code,
                "policy_score": policy_score,
                "hypothesis": case["hypothesis"].strip(),
            }
        )

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run())
