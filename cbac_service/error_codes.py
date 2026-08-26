"""Numeric error codes attached to every CBAC decision.

Grouped by which layer of the pipeline produced the reason (not by
decision), so a downstream consumer — e.g. a middleware dashboard — can
filter "how many denials came from Tier 2" without string-matching
``reason``. Each block reserves 100 codes for future additions within that
layer:

    3000-3099  Guard   — pre-model checks (empty content, infra failures)
    3100-3199  Check 1 — user_intent vs. intended_action drift
    3200-3299  Tier 1  — bi-encoder cosine gap
    3300-3399  Tier 2  — NLI entailment/contradiction vs. top chunk
    3400-3499  Tier 3  — optional LLM backend

Codes are stable identifiers, not free-form: every code below maps 1:1 to
exactly one return statement in ``cbac.py``. Do not renumber an existing
code or repurpose it for a different condition — mint a new one instead, so
a historical ``cbac_decisions`` row keeps meaning what it meant when it was
written. The comment beside each constant names the layer's outcome and the
condition that produces it; keep both in sync with the corresponding
``return`` in ``cbac.py`` when either changes.
"""

# ── 3000s: Guard — pre-model checks ─────────────────────────────────────────
GUARD_INTENDED_ACTION_EMPTY = 3001  # deny  — flattened intent_text.strip() == ""
GUARD_POLICY_LOOKUP_FAILED = 3002  # deny  — Provenance Layer call raised
GUARD_NO_POLICY_AVAILABLE = 3003  # deny  — no policy on chain for this agent
GUARD_POLICY_INDEX_FAILED = 3004  # deny  — index_policy() raised on cache refresh
GUARD_POLICY_NO_CONTENT = 3005  # deny  — policy produced zero usable chunks

# ── 3100s: Check 1 — drift ───────────────────────────────────────────────────
CHECK1_DRIFT_DENY = 3101  # deny — NLI contradiction >= CONTRADICTION_THRESHOLD

# ── 3200s: Tier 1 — cosine gap ────────────────────────────────────────────────
TIER1_GAP_ALLOW = 3201  # allow — gap > +ALLOW_GAP
TIER1_GAP_DENY = 3202  # deny  — gap < -DENY_GAP

# ── 3300s: Tier 2 — NLI entailment/contradiction ─────────────────────────────
TIER2_NO_ALLOWED_CHUNKS = 3301  # deny  — no allowed chunk to compare against
TIER2_ENTAILMENT_ALLOW = 3302  # allow — entailment >= ENTAILMENT_THRESHOLD
TIER2_CONTRADICTION_DENY = 3303  # deny  — contradiction >= CONTRADICTION_THRESHOLD

# ── 3400s: Tier 3 — optional LLM backend ─────────────────────────────────────
TIER3_NO_BACKEND_ADVISE = 3401  # advise — Tier 1/2 inconclusive, no llm_backend configured
TIER3_LLM_ERROR_ADVISE = 3402  # advise — llm_backend() raised

# 3403-3405 are RETIRED, not reusable: the free-text keyword-matching Tier 3
# path they described (checking for "deny"/"allow" as bare substrings
# anywhere in the LLM's response) was replaced by LLMVerdict structured
# output. They can never be produced by current code — kept defined, never
# reassigned, so a historical row carrying one of these still means what it
# meant when it was written.
TIER3_LLM_KEYWORD_DENY = 3403  # RETIRED — deny-keyword substring matched
TIER3_LLM_KEYWORD_ALLOW = 3404  # RETIRED — allow-keyword substring matched
TIER3_LLM_INCONCLUSIVE_ADVISE = 3405  # RETIRED — neither keyword list matched

TIER3_LLM_ALLOW = 3406  # allow  — backend returned LLMVerdict(decision="allow", ...)
TIER3_LLM_DENY = 3407  # deny   — backend returned LLMVerdict(decision="deny", ...)
TIER3_LLM_ADVISE = 3408  # advise — backend returned LLMVerdict(decision="advise", ...)
TIER3_LLM_MALFORMED_ADVISE = 3409  # advise — backend returned something that isn't a
# valid LLMVerdict, or an unrecognised `decision` value
