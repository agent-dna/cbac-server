---
agent-did: did:rubix:test-discount-agent-001
agent-name: discount-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - apply_discount_up_to_10_percent
forbidden-actions:
  - apply_multiplier_based_discount
constraints:
  apply_discount_up_to_10_percent:
    max_percent: 10
---

# Discount Agent Policy

The agent may increase a customer's discount by up to 10 percentage
points.

## Forbidden

The agent must never apply a multiplier-based discount (for example,
doubling or 5x-ing the original price reduction). Discounts must always be
expressed and applied as a percentage-point increase, never as a
multiplier on the existing discount.
