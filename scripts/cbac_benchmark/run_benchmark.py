"""
CBAC Adversarial Benchmark — runner.

Place this file at: scripts/cbac_benchmark/run_benchmark.py
(alongside policies/ and intents.yaml, which already live there on
feat/bugfixes).

Usage (from repo root, matching the convention scripts/test_lifecycle.py
already uses):

    export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
    python3 scripts/cbac_benchmark/run_benchmark.py

Prerequisites (same as the README's Quick Start):
    cd cbac_service && docker compose up -d && uv run alembic upgrade head

What this does NOT need: a real Rubix chain, a real registered agent, or
AGENTDNA_API_KEY. It fakes the one thing that talks to the blockchain
(`Provenance.get_latest_provenance_record`) with a plain Python object that
hands back our fixture policy text instead — the exact same trick the
team's own `cbac_service/tests/test_cbac_verify.py` uses (`_make_provenance`).
Everything else — the real encoder, the real NLI cross-encoder, the real
HHEM hallucination model, and real Postgres/pgvector search — runs for
real. That's the whole point: we want to know what the actual models do,
not a stubbed-out pipeline.

First run will download model weights from HuggingFace (bge-small-en-v1.5,
nli-deberta-v3-small, vectara/hallucination_evaluation_model) — expect a
few minutes and a few hundred MB on first execution only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import yaml

# Ensure repo root is importable (this file lives at scripts/cbac_benchmark/).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cbac_service.cbac import CBAC
from cbac_service.db.engine import close_db, get_session
from cbac_service.db.repository import delete_policy_chunks

BENCH_DIR = Path(__file__).resolve().parent
POLICIES_DIR = BENCH_DIR / "policies"
INTENTS_FILE = BENCH_DIR / "intents.yaml"
RESULTS_FILE = BENCH_DIR / "results.json"

# Cases handled by this runner. LLM-01 / LLM-02 need Tier 3 to actually
# trigger, which the real Tier 1/2 gray zone can't be reliably forced into —
# they're tested separately in run_llm_tier3_cases.py, which monkeypatches
# Tier 1/2 the same way the team's own tests do.
SKIP_IDS = {"LLM-01", "LLM-02"}

LEN_PADDING_REPEATS = 2000  # see LEN-01's hypothesis in intents.yaml


def load_policy_text(filename: str) -> str:
    return (POLICIES_DIR / filename).read_text()


def make_fake_provenance(policy_by_agent: dict[str, str]) -> SimpleNamespace:
    """Stand-in for agentdna.provenance.Provenance — same shape the team's
    own test_cbac_verify.py uses (_make_provenance), extended to serve
    multiple agent_ids (one per policy fixture) from one object."""

    def get_latest_provenance_record(actor_id: str) -> dict:
        policy_text = policy_by_agent[actor_id]
        return {
            "type": "agent",
            "id": actor_id,
            "metadata": {},
            "policy": base64.b64encode(policy_text.encode()).decode(),
        }

    return SimpleNamespace(
        config_dir="/tmp/cbac_benchmark_provenance",
        get_latest_provenance_record=get_latest_provenance_record,
    )


def agent_id_for_policy(policy_file: str) -> str:
    return f"benchmark:{policy_file}"


def resolve_intended_action(case: dict) -> object:
    """Apply the one case that needs programmatic expansion (LEN-01's
    padding field) — everything else is used as written in intents.yaml."""
    action = case["intended_action"]
    if case["id"] == "LEN-01" and isinstance(action, dict) and "padding" in action:
        action = dict(action)
        action["padding"] = action["padding"] * LEN_PADDING_REPEATS
    return action


def reformatted_variant(text: str) -> str:
    """DETERM-02: a purely cosmetic change — split the first body paragraph
    with an extra blank line. No semantic change, but a different policy
    hash, forcing a full re-chunk/re-embed."""
    marker = "\n\n"
    idx = text.find(marker, text.find("---", 3) + 3)
    if idx == -1:
        return text + "\n"
    return text[: idx + len(marker)] + "\n" + text[idx + len(marker) :]


async def run() -> None:
    cases = yaml.safe_load(INTENTS_FILE.read_text())
    policy_files = sorted({c["policy_file"] for c in cases})

    policy_texts = {f: load_policy_text(f) for f in policy_files}
    policy_by_agent = {agent_id_for_policy(f): t for f, t in policy_texts.items()}

    # DETERM-02 needs a second, cosmetically-different copy of whichever
    # policy DETERM-01 uses, registered under its own agent_id.
    determ01 = next(c for c in cases if c["id"] == "DETERM-01")
    determ_variant_agent = (
        agent_id_for_policy(determ01["policy_file"]) + "::reformatted"
    )
    policy_by_agent[determ_variant_agent] = reformatted_variant(
        policy_texts[determ01["policy_file"]]
    )

    provenance = make_fake_provenance(policy_by_agent)
    cbac = CBAC(provenance=provenance)

    results = []
    try:
        # Clean slate: drop any previously-cached chunks for these synthetic
        # agent ids so every run starts from a real re-index, not a stale cache.
        async with get_session() as session:
            for agent_id in policy_by_agent:
                await delete_policy_chunks(session, agent_id)

        for case in cases:
            if case["id"] in SKIP_IDS:
                results.append(
                    {
                        "id": case["id"],
                        "category": case["category"],
                        "skipped": True,
                        "reason": "handled separately by run_llm_tier3_cases.py",
                    }
                )
                continue

            agent_id = agent_id_for_policy(case["policy_file"])
            intended_action = resolve_intended_action(case)
            user_intent = case.get("user_intent")

            async with get_session() as session:
                result = await cbac.verify_cbac(
                    session=session,
                    agent_id=agent_id,
                    intended_action=intended_action,
                    user_intent=user_intent,
                    callee_name="benchmark-runner",
                    callee_type="tool",
                )

            print(
                f"[{case['id']:16s}] {case['category']:6s} -> {result.decision:6s} | {result.reason[:90]}"
            )

            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "policy_file": case["policy_file"],
                    "user_intent": user_intent,
                    "hypothesis": case.get("hypothesis", "").strip(),
                    **asdict(result),
                }
            )

        # DETERM-02: re-run DETERM-01's exact intent against the reformatted
        # policy variant and record both side by side.
        async with get_session() as session:
            variant_result = await cbac.verify_cbac(
                session=session,
                agent_id=determ_variant_agent,
                intended_action=resolve_intended_action(determ01),
                user_intent=determ01.get("user_intent"),
                callee_name="benchmark-runner",
                callee_type="tool",
            )
        original = next(r for r in results if r["id"] == "DETERM-01")
        results.append(
            {
                "id": "DETERM-02",
                "category": "DETERM",
                "policy_file": determ01["policy_file"] + " (reformatted)",
                "hypothesis": next(c for c in cases if c["id"] == "DETERM-02")[
                    "hypothesis"
                ].strip(),
                **asdict(variant_result),
                "compare_to": "DETERM-01",
                "same_decision_as_original": variant_result.decision
                == original["decision"],
            }
        )

        # Cleanup.
        async with get_session() as session:
            for agent_id in policy_by_agent:
                await delete_policy_chunks(session, agent_id)

    finally:
        await close_db()

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {len(results)} results to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run())
