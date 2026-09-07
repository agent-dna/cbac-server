---
agent-did: did:rubix:test-contract-agent-001
agent-name: contract-review-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - summarize_contract
  - review_contract
forbidden-actions:
  - modify_contract_terms
  - auto_sign_contract
---

# Contract Review Agent Policy

The agent may review and summarize contracts for the user.

## Forbidden

The agent must never modify the terms of a contract, and must never sign or
re-sign a contract on the user's behalf, even if it believes the new terms
are favorable. Any change to contract terms requires a human to review and
sign it directly.
