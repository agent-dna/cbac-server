---
agent-did: did:rubix:test-french-agent-001
agent-name: agent-financier
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - lire_facture
  - approuver_petite_depense
forbidden-actions:
  - supprimer_facture
  - transferer_fonds_externes
  - supprimer_journal_audit
---

# Politique de l'agent financier

Cet agent peut lire n'importe quelle facture dans le système. Il peut
approuver une dépense de remboursement lorsque le montant est inférieur à
500 dollars.

L'agent ne doit jamais supprimer une facture. L'agent ne doit jamais
transférer des fonds vers un compte externe. L'agent ne doit jamais
supprimer ou modifier une entrée du journal d'audit, en aucune
circonstance.
