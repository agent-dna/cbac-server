# Tunables for the CBAC decision pipeline (cbac_service/cbac.py).
#
# Every environment variable this service reads is read *here*, once, and each
# constant is named after the variable it comes from — so `.env.sample` and this
# file are the same list, bar the block `.env.sample` marks as docker-compose-only,
# and nothing has to be grepped for. The guard (`cbac/`)
# is the exception by construction: it is a separate distribution that cannot
# import this module, so it reads its own three variables per call.

import os

# ── Provenance Layer ──────────────────────────────────────────────────────────
# Rubix chain connector the policy is fetched from. Pinned here rather than left
# to the agent-dna client's own default, so which chain this service talks to is
# visible in this file; a deployment points it elsewhere (a staging connector, a
# self-hosted node) without touching code.
PROVENANCE_URL: str = os.environ.get(
    "PROVENANCE_URL", "https://chain-connector-2-dev.rubix.net"
)

# Provenance Layer credential. Empty is a valid deployment: the client is
# constructed either way and fails on use, not on import.
AGENTDNA_API_KEY: str = os.environ.get("AGENTDNA_API_KEY", "")

# ── Database ──────────────────────────────────────────────────────────────────
# Async Postgres connection string (asyncpg driver).
# Override via environment variable for deployment.
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac",
)

# pgvector index type: "hnsw" (low-latency, moderate data) or "ivfflat" (large scale).
VECTOR_INDEX_TYPE: str = os.environ.get("VECTOR_INDEX_TYPE", "hnsw")

# Hybrid search: enable BM25 fusion alongside vector cosine.
HYBRID_SEARCH_ENABLED: bool = (
    os.environ.get("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
)

# Reciprocal Rank Fusion constant (higher = less aggressive re-ranking).
RRF_K: int = int(os.environ.get("RRF_K", "60"))

# Models
ENCODER_MODEL = "BAAI/bge-small-en-v1.5"  # bi-encoder for Tier 1 cosine
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"  # NLI for classify + Tier 2
HHEM_MODEL = (
    "vectara/hallucination_evaluation_model"  # hallucination scoring (1 = grounded)
)

# Tier 1 gap thresholds: allow when gap > +allow_gap, deny when gap < -deny_gap,
# else escalate. Model-agnostic (relative difference, not absolute scores).
ALLOW_GAP = 0.12
DENY_GAP = 0.08

# NLI thresholds for Check 1 drift and Tier 2 entailment.
ENTAILMENT_THRESHOLD = 0.55
CONTRADICTION_THRESHOLD = 0.60

# LHI (Local Heuristic Intelligence): weighted arithmetic mean of the
# (intent, policy, hallucination) scores — expected interaction quality; the
# allow/deny gates already enforce the hard constraints — then an asymmetric
# EMA against the stored trust: slow to build, fast to lose.
# The mean renormalizes over whichever components were actually observed
# (s = Σ wᵢxᵢ / Σ wᵢ), so these weights are scale-invariant — only their ratio
# matters, and adding/removing a component needs no retuning of the rest.
LHI_WEIGHTS: tuple[float, float, float] = (0.3, 0.3, 0.2)
LHI_LAMBDA_UP = 0.95
LHI_LAMBDA_DOWN = 0.70


# Structure-aware chunking: paragraphs / list items are the primary unit,
# split further only past this word-count budget (~1.3 tokens/word for English).
CHUNK_MAX_WORDS = 120
NLI_BATCH_SIZE = 64

# ── v2 gate: OpenFGA ──────────────────────────────────────────────────────────
# The OpenFGA server holding the allow/deny tuples, and the store name the
# service finds or creates at startup. One store per environment, so dev and
# test-prod never share tuples. No default for the store name: an env file
# that forgets it must fail closed, not quietly write into another
# environment's store.
OPENFGA_API_URL: str = os.environ.get("OPENFGA_API_URL", "http://localhost:8080")
OPENFGA_STORE_NAME: str = os.environ.get("OPENFGA_STORE_NAME", "")
# OpenFGA's pre-shared key (its OPENFGA_AUTHN_PRESHARED_KEYS). Whoever can write
# to the store can mint grants, so docker-compose refuses to start OpenFGA
# without one; empty here only suits an OpenFGA run with authn off.
OPENFGA_API_TOKEN: str = os.environ.get("OPENFGA_API_TOKEN", "")

# How long an intent's allow tuples stay live, from the first call that parsed
# its query. Enforced by OpenFGA itself (the non_expired_grant condition).
INTENT_TTL_SECONDS: int = int(os.environ.get("INTENT_TTL_SECONDS", "3600"))

# ── v2 gate: parsers ──────────────────────────────────────────────────────────
# "<backend>:<model>", backend one of "ollama" | "anthropic". The policy parser
# runs once per upload and can afford a large model; the query parser runs on
# the first call of every intent, so it sits on the latency path.
POLICY_PARSER_MODEL: str = os.environ.get("POLICY_PARSER_MODEL", "ollama:qwen3:14b")
QUERY_PARSER_MODEL: str = os.environ.get("QUERY_PARSER_MODEL", "ollama:qwen3:14b")
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Empty lets the Anthropic SDK resolve credentials itself (an `ant auth login`
# profile, ANTHROPIC_AUTH_TOKEN, ...).
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# Shared secret for the v2 policy endpoints, sent as X-CBAC-Key. Those
# endpoints write deny tuples and the argument mapping, so an unset key makes
# them refuse every request rather than accept anyone's.
CBAC_POLICY_KEY: str = os.environ.get("CBAC_POLICY_KEY", "")

# ── Service binding & logging ─────────────────────────────────────────────────
CBAC_SERVICE_HOST: str = os.environ.get("CBAC_SERVICE_HOST", "127.0.0.1")
CBAC_SERVICE_PORT: int = int(os.environ.get("CBAC_SERVICE_PORT", "8767"))
CBAC_SERVICE_LOG_LEVEL: str = os.environ.get("CBAC_SERVICE_LOG_LEVEL", "INFO").upper()
