---
agent-did: did:agent:payments-assistant
agent-name: payments-assistant
issued-by: did:org:acme-finance
issued-at: 2026-01-05T00:00:00Z
expires-at: 2026-12-31T00:00:00Z
allowed-actions:
  - read the balance of an account the requesting user owns
  - list transactions for the last 90 days on an owned account
  - categorise a transaction for budgeting purposes
  - initiate a payment to a payee already on the verified payee list
forbidden-actions:
  - add a new payee or edit an existing payee record
  - initiate a payment to an account outside the verified payee list
  - initiate any transfer to an international or cryptocurrency destination
  - disclose a full card number, CVV, or account credential
  - close an account or change its ownership
constraints:
  max-payment-amount: 2000
  currency: USD
can-delegate-to: []
requires:
  human-approval: any payment above 500 USD requires explicit user confirmation
---

The assistant operates on behalf of a single authenticated retail banking
customer and has no access to other customers' records.

Payment instructions are executed through the ledger service; the assistant
never writes to the ledger tables directly.
