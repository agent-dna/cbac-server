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
    4000-4099  v2 intent & mapping — intent lifecycle, catalog, normalization
    4100-4199  v2 authorization    — glob deny rules and the OpenFGA check
    4200-4299  v2 semantic         — deny-only checks on free-text arguments
    4300-4399  v2 allow
    9000-9099  API     — failures in the HTTP handler, outside the pipeline

The 3000s belong to ``POST /cbac/v1/authorize`` (``CBAC.verify_cbac``), the
4000s to ``POST /cbac/v2/authorize`` (``CBAC.authorize_call``).

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
# 3401/3402/3408/3409 used to read ADVISE — the pipeline no longer has a
# caller-must-decide state, so each of these now resolves to deny. Renamed to
# match; the numeric codes are unchanged, so a historical row bearing the old
# name's number still means what it meant when it was written.
TIER3_NO_BACKEND_DENY = 3401  # deny — Tier 1/2 inconclusive, no llm_backend configured
TIER3_LLM_ERROR_DENY = 3402  # deny — llm_backend() raised

# 3403-3405 are RETIRED, not reusable: the free-text keyword-matching Tier 3
# path they described (checking for "deny"/"allow" as bare substrings
# anywhere in the LLM's response) was replaced by LLMVerdict structured
# output. They can never be produced by current code — kept defined, never
# reassigned, so a historical row carrying one of these still means what it
# meant when it was written.
TIER3_LLM_KEYWORD_DENY = 3403  # RETIRED — deny-keyword substring matched
TIER3_LLM_KEYWORD_ALLOW = 3404  # RETIRED — allow-keyword substring matched
TIER3_LLM_INCONCLUSIVE_ADVISE = 3405  # RETIRED — neither keyword list matched

TIER3_LLM_ALLOW = 3406  # allow — backend returned LLMVerdict(decision="allow", ...)
TIER3_LLM_DENY = 3407  # deny  — backend returned LLMVerdict(decision="deny", ...)
TIER3_LLM_ADVISE_DENY = (
    3408  # deny — backend returned LLMVerdict(decision="advise", ...)
)
TIER3_LLM_MALFORMED_DENY = 3409  # deny — backend returned something that isn't a
# valid LLMVerdict, or an unrecognised `decision` value

# ── 4000s: v2 intent & mapping ───────────────────────────────────────────────
INTENT_MISSING = 4001  # deny — empty intent_id or user_query
INTENT_MISMATCH = 4002  # deny — intent_id already bound to another query or agent
INTENT_EXPIRED = 4003  # deny — the intent's grants are past INTENT_TTL_SECONDS
INTENT_PARSE_FAILED = 4004  # deny — the intent's query parse failed, on this call
# or within QUERY_PARSER_RETRY_SECONDS before it; nothing is granted
MAPPING_UNKNOWN_TOOL = 4005  # deny — tool not in the agent's active catalog
MAPPING_UNKNOWN_ARGUMENT = 4006  # deny — argument neither a resource nor free text
MAPPING_NORMALIZATION_FAILED = 4007  # deny — a resource value has no canonical form
MAPPING_NO_RESOURCE = 4008  # deny — the call names no value for any resource argument
MAPPING_NO_ACTIVE_POLICY = 4009  # deny — the agent has no active policy version
MAPPING_STALE_TUPLES = 4010  # deny — the active version's deny tuples were not
# written under ibac.TUPLE_SCHEME; it has to be re-activated

# ── 4100s: v2 authorization ─────────────────────────────────────────────────
AUTHZ_GLOB_DENY = 4101  # deny — a glob deny rule matches a resource
AUTHZ_DENY_TUPLE = 4102  # deny — OpenFGA: an exact deny tuple blocks the resource
AUTHZ_NO_ALLOW_TUPLE = 4103  # deny — OpenFGA: no live allow tuple for the resource
AUTHZ_FGA_ERROR = 4104  # deny — OpenFGA call raised

# ── 4200s: v2 semantic checks (deny-only) ────────────────────────────────────
SEMANTIC_INTENT_DRIFT = 4201  # deny — free text contradicts the user query (NLI)
SEMANTIC_POLICY_DRIFT_TIER1 = 4202  # deny — cosine gap < -DENY_GAP
SEMANTIC_POLICY_DRIFT_TIER2 = 4203  # deny — NLI contradiction vs top allowed chunk
SEMANTIC_ERROR = 4204  # deny — a model or search call raised

# ── 4300s: v2 allow ──────────────────────────────────────────────────────────
ALLOW_NO_FREE_TEXT = 4301  # allow — every resource granted, no free text to check
ALLOW_FREE_TEXT_PASSED = 4302  # allow — granted, and every free-text check passed

# ── 9000s: API — failures outside the decision pipeline ─────────────────────
# Produced by main.authorize_cbac's own exception handler, not by anything in
# cbac.py — thrown before or around _decide (e.g. the DB session never opens),
# so no CBACResult and no cbac_decisions row exist to carry a pipeline code.
API_UNHANDLED_EXCEPTION = 9001  # error — exception raised outside verify_cbac
