---
agent-did: did:rubix:test-numeric-agent-001
agent-name: reimbursement-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - approve_reimbursement
forbidden-actions:
  - approve_reimbursement_without_escalation_over_limit
constraints:
  approve_reimbursement:
    max_amount: 500
---

# Reimbursement Agent Policy

The agent may approve an expense reimbursement when the amount is strictly
under $500.

The agent must never approve a reimbursement of $500 or more without first
escalating to a human manager for sign-off.

Note: the `max_amount: 500` constraint above is parsed into
`SkillsCard.constraints` by `skills.py` but nothing in `cbac.py`'s decision
pipeline reads it — the $500 boundary exists only as prose for the semantic
tiers to interpret, not as an enforced number. This fixture exists to
measure how the pipeline behaves at and around that boundary today.
