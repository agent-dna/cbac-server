---
agent-did: did:rubix:test-finance-agent-001
agent-name: finance-ops-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - read_invoice
  - read_expense_report
  - approve_expense_under_500
  - send_payment_reminder_email
forbidden-actions:
  - delete_invoice
  - delete_audit_log
  - approve_expense_over_10000
  - transfer_funds_externally
  - modify_vendor_bank_details
constraints:
  approve_expense_under_500:
    max_amount: 500
---

# Finance Operations Agent Policy

This agent supports the accounts-payable team by reading invoices and
expense reports, and approving small expenses without escalation.

## Permitted

The agent may read any invoice or expense report in the system. The agent
may approve an expense reimbursement when the amount is under $500. The
agent may send a payment reminder email to a vendor whose invoice is
overdue.

## Forbidden

The agent must never delete an invoice record. The agent must never delete
or modify an audit log entry, under any circumstances. The agent must never
approve an expense of $10,000 or more without a human manager's sign-off.
The agent must never initiate an external funds transfer. The agent must
never modify a vendor's bank account details.
