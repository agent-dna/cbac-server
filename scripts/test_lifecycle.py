"""
Usage (from repo root):
    python3 cbac_service/scripts/test_lifecycle.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cbac_service.cbac import CBAC, _policy_hash
from cbac_service.chunking import flatten_policy_chunks
from cbac_service.db.engine import close_db, get_session
from cbac_service.db.repository import (
    delete_policy_chunks,
    save_policy_chunks,
)
from cbac_service.db.search import bm25_search, hybrid_search, vector_search

# ── Sample policy ──────────────────────────────────────────

SAMPLE_POLICY = """\
---
agent-did: bafytest1234567890abcdef
agent-name: github-worker
issued-by: bafyissuer9876543210fedcba
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - read GitHub issues
  - create pull requests
  - comment on issues
  - search repositories
forbidden-actions:
  - delete repositories
  - modify branch protection rules
  - access secrets or credentials
constraints:
  max-file-size: 10MB
  rate-limit: 100 requests/minute
---

## Permissions

This agent is authorized to interact with GitHub repositories in a read-heavy
manner. It may read issues, pull requests, and code. It may create new issues
and comment on existing discussions.

## Restrictions

The agent must never delete any repository, branch, or tag. It must not modify
access controls or branch protection settings. Private credentials, API tokens,
and SSH keys are strictly off-limits.

## Delegation

The agent may delegate read-only tasks to sub-agents but must not delegate
any write operations without explicit user approval.
"""

AGENT_ID = "bafytest1234567890abcdef"


async def run_lifecycle():

    try:
        async with get_session() as session:
            # Cleanup from previous runs
            await delete_policy_chunks(session, AGENT_ID)
            print("\n✓ DB initialized\n")


        print("Step 1: Parse & chunk policy")
        from unittest.mock import MagicMock

        mock_provenance = MagicMock()
        mock_provenance.config_dir = "/tmp/cbac_test"
        cbac = CBAC(provenance=mock_provenance)

        chunks = flatten_policy_chunks(SAMPLE_POLICY)
        print(f"  Produced {len(chunks)} chunks:")
        for i, c in enumerate(chunks):
            print(f"    [{i}] {c[:80]}{'...' if len(c) > 80 else ''}")


        print("\n Step 2: NLI classification ")
        allowed_chunks, forbidden_chunks = cbac._classify_chunks(chunks)
        print(f"  Allowed:   {len(allowed_chunks)} chunks")
        print(f"  Forbidden: {len(forbidden_chunks)} chunks")
        for c in allowed_chunks[:3]:
            print(f"    [A] {c[:70]}")
        for c in forbidden_chunks[:3]:
            print(f"    [F] {c[:70]}")


        print("\n Step 3: Encode embeddings ")
        encoder = cbac._get_encoder()
        all_chunks = allowed_chunks + forbidden_chunks
        chunk_types = ["allowed"] * len(allowed_chunks) + ["forbidden"] * len(forbidden_chunks)
        embeddings = encoder.encode(all_chunks, normalize_embeddings=True)
        print(f"  Encoded {len(all_chunks)} chunks → shape {embeddings.shape}")


        print("\n Step 4: Store in Postgres ")
        policy_hash = _policy_hash(SAMPLE_POLICY)
        async with get_session() as session:
            count = await save_policy_chunks(
                session, AGENT_ID, all_chunks, chunk_types, embeddings, policy_hash
            )
        print(f"  Stored {count} chunks (hash: {policy_hash[:16]}...)")


        print("\n Step 5: Vector search ")
        test_queries = [
            ("read github issues", "allowed"),
            ("delete repository", "forbidden"),
            ("search code in repos", "allowed"),
        ]
        for query, expected_type in test_queries:
            query_vec = encoder.encode([query], normalize_embeddings=True)[0]
            async with get_session() as session:
                results = await vector_search(session, AGENT_ID, query_vec, top_k=3)
            top = results[0] if results else None
            status = "✓" if top and top.chunk_type == expected_type else "✗"
            print(f"  {status} \"{query}\" → [{top.chunk_type}] {top.chunk_text[:50]}... (score={top.score:.4f})" if top else f"  ✗ \"{query}\" → no results")


        print("\n Step 6: BM25 keyword search ")
        bm25_queries = ["GitHub issues", "credentials secrets", "branch protection"]
        for query in bm25_queries:
            async with get_session() as session:
                results = await bm25_search(session, AGENT_ID, query, top_k=3)
            top = results[0] if results else None
            if top:
                print(f"  \"{query}\" → [{top.chunk_type}] score={top.score:.4f} | {top.chunk_text[:50]}...")
            else:
                print(f"  \"{query}\" → no BM25 results")


        print("\n Step 7: Hybrid search (RRF) ")
        query = "read github issues"
        query_vec = encoder.encode([query], normalize_embeddings=True)[0]
        async with get_session() as session:
            results = await hybrid_search(session, AGENT_ID, query_vec, query, top_k=5)
        print(f"  Query: \"{query}\"")
        for r in results:
            print(f"    [{r.chunk_type}] rrf={r.score:.6f} | {r.chunk_text[:60]}...")


        print("\n Step 8: Tiered decision pipeline ")
        test_intents = [
            ("read all open GitHub issues in the repo", "should allow"),
            ("delete the main branch", "should deny"),
            ("search for authentication code", "should allow"),
            ("access the API secret keys", "should deny"),
        ]
        for intent, expectation in test_intents:
            async with get_session() as session:
                decision, reason, policy_score = await cbac._tiered_decision(
                    session, AGENT_ID, intent
                )
            icon = "✓" if ("allow" in expectation and decision == "allow") or \
                         ("deny" in expectation and decision == "deny") else "?"
            print(f"  {icon} \"{intent}\"")
            score_str = f"{policy_score:.4f}" if policy_score is not None else "N/A"
            print(f"      → {decision} (score={score_str}) | {reason[:80]}")

        
        print("\n Cleanup ")
        async with get_session() as session:
            await delete_policy_chunks(session, AGENT_ID)
        print("  Cleaned up test data.")

    finally:
        await close_db()

    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_lifecycle())
