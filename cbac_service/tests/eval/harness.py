"""Test harness: a faithful in-memory stand-in for the DB search layer, plus
metric helpers and a report collector.

The search fakes are not simplifications of the production query — they compute
the same quantities. `vector_search` returns cosine similarity over the same
normalized embeddings pgvector would store (for unit vectors, `1 - cosine
distance` is exactly the dot product). `hybrid_search` runs the same Reciprocal
Rank Fusion over a vector list and a BM25 list, with the same `RRF_K` and the
same `fetch_k = top_k * 3`. Only the storage engine differs, so Tier 1 and
Tier 2 behave as they do in production.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from cbac_service.config import RRF_K

_TOKEN_RE = re.compile(r"[a-z0-9_.:*/-]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """Okapi BM25. pg_textsearch's exact scoring differs in its tokenizer and
    saturation constants, so absolute scores are not comparable — but RRF only
    consumes the *ranking*, which is what this reproduces.
    """

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokens(d) for d in docs]
        self.lens = [len(d) for d in self.docs]
        self.avglen = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = _tokens(query)
        out = []
        for i in range(len(self.docs)):
            s = 0.0
            for t in q:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.lens[i] / (self.avglen or 1.0)
                )
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


@dataclass
class PolicyIndex:
    """Everything the search layer would have stored for one agent."""

    chunks: list[str]
    chunk_types: list[str]
    embeddings: np.ndarray
    _bm25: _BM25 = field(init=False)

    def __post_init__(self):
        self._bm25 = _BM25(self.chunks)

    def _rows(self, chunk_type: str | None) -> list[int]:
        if chunk_type is None:
            return list(range(len(self.chunks)))
        return [i for i, t in enumerate(self.chunk_types) if t == chunk_type]

    def _result(self, i: int, score: float):
        return SimpleNamespace(
            chunk_id=i,
            agent_id="eval",
            chunk_text=self.chunks[i],
            chunk_type=self.chunk_types[i],
            score=float(score),
            chunk_index=i,
            section=None,
        )

    def vector(self, query_vec: np.ndarray, top_k: int, chunk_type: str | None):
        idx = self._rows(chunk_type)
        if not idx:
            return []
        sims = self.embeddings[idx] @ query_vec
        order = np.argsort(-sims)[:top_k]
        return [self._result(idx[j], sims[j]) for j in order]

    def bm25(self, query_text: str, top_k: int, chunk_type: str | None):
        idx = self._rows(chunk_type)
        if not idx:
            return []
        all_scores = self._bm25.scores(query_text)
        scored = sorted(idx, key=lambda i: -all_scores[i])[:top_k]
        return [self._result(i, all_scores[i]) for i in scored]

    def hybrid(
        self, query_vec: np.ndarray, query_text: str, top_k: int, chunk_type: str | None
    ):
        fetch_k = top_k * 3
        rrf: dict[int, float] = {}
        seen: dict[int, object] = {}
        for lst in (
            self.vector(query_vec, fetch_k, chunk_type),
            self.bm25(query_text, fetch_k, chunk_type),
        ):
            for rank, r in enumerate(lst, start=1):
                rrf[r.chunk_id] = rrf.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                seen.setdefault(r.chunk_id, r)
        top = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:top_k]
        return [self._result(c, rrf[c]) for c in top]


def build_index(cbac, chunks: list[str], chunk_types: list[str]) -> PolicyIndex:
    embeddings = np.asarray(
        cbac._get_encoder().encode(chunks, normalize_embeddings=True)
    )
    return PolicyIndex(list(chunks), list(chunk_types), embeddings)


def patch_search(monkeypatch, index: PolicyIndex) -> None:
    """Point cbac.py's module-level search names at `index`."""
    import cbac_service.cbac as cbac_mod

    async def _vector(session, agent_id, vec, top_k=5, chunk_type=None, **kw):
        return index.vector(vec, top_k, chunk_type)

    async def _hybrid(session, agent_id, vec, text, top_k=5, chunk_type=None, **kw):
        return index.hybrid(vec, text, top_k, chunk_type)

    async def _chunks(session, agent_id, chunk_type=None):
        if chunk_type is None:
            return list(index.chunks)
        return [c for c, t in zip(index.chunks, index.chunk_types) if t == chunk_type]

    monkeypatch.setattr(cbac_mod, "vector_search", _vector)
    monkeypatch.setattr(cbac_mod, "hybrid_search", _hybrid)
    monkeypatch.setattr(cbac_mod, "get_policy_chunks", _chunks)


# ── metrics ──────────────────────────────────────────────────────────────────


@dataclass
class Binary:
    """Counts for one binary target, e.g. 'is this chunk forbidden'."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if actual and predicted:
            self.tp += 1
        elif actual and not predicted:
            self.fn += 1
        elif predicted:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or not (p + r):
            return float("nan")
        return 2 * p * r / (p + r)

    def row(self) -> str:
        return (
            f"n={self.n:<4} tp={self.tp:<4} fp={self.fp:<4} fn={self.fn:<4} "
            f"tn={self.tn:<4} P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f}"
        )


# The numbers worth reading first, in the order a reader needs them: can the
# classifier find the prohibitions, is that a property of the document shape,
# do the tiers act on what it found, and do the user-alignment layers see what
# the policy cannot. `None` renders as "-" so a partial run (one test selected
# with -k) still prints a well-formed block.
#
# (metric key, label, format). Everything else stays in `metrics` for the JSON.
HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("classify.structured.forbidden_recall", "forbidden recall, skill card", ".2f"),
    ("classify.unstructured.forbidden_recall", "forbidden recall, other shape", ".2f"),
    ("classify.paired.mean_delta", "  ...paired delta, same policies", "+.2f"),
    ("classify.structured.recall_frontmatter_key", "  ...on frontmatter keys", ".2f"),
    ("classify.unstructured.recall_prose", "  ...on marked prose", ".2f"),
    ("classify.forbid_e.above_floor", "prohibitions clearing the 0.40 floor", ".2f"),
    (
        "classify.unstructured.empty_forbidden_buckets",
        "policies with no forbidden bucket",
        "d",
    ),
    ("policy.structured.block_rate", "block rate, skill-card index", ".2f"),
    ("policy.oracle.block_rate", "block rate, oracle index", ".2f"),
    ("policy.oracle.allow_rate", "allow rate, oracle index", ".2f"),
    ("policy.oracle.block_rate.explicit", "  ...on stated prohibitions", ".2f"),
    ("policy.oracle.block_rate.unlisted", "  ...on unlisted actions (free)", ".2f"),
    ("policy.oracle.fallback_share", "decisions from the no-LLM fallback", ".2f"),
    ("drift.recall", "drift recall on misaligned actions", ".2f"),
    ("drift.auc.contradiction", "  ...AUC of the gated signal", ".2f"),
    ("hhem.auc.aligned", "HHEM AUC on the same actions", ".2f"),
    ("final.oracle.accuracy", "final accuracy, oracle index", ".2f"),
    ("layers.oracle.coverage_union", "blocked by drift∪policy∪HHEM", ".2f"),
)


class Report:
    """Collects the numbers and prints one consolidated block at session end,
    so the findings survive pytest's per-test output capture.

    Two audiences. `dump()` is the human read — a headline block first, then
    every section in full. `metrics` is the machine read: a flat
    `dotted.key -> number` mapping that `--eval-json` writes out, so a caller
    can track a number across runs without parsing the prose.
    """

    def __init__(self) -> None:
        self.sections: list[tuple[str, list[str]]] = []
        self.metrics: dict[str, float] = {}
        self.cases: dict[str, list[dict]] = {}
        self.policies: list[dict] = []

    def add(
        self,
        title: str,
        lines: list[str],
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.sections.append((title, lines))
        if metrics:
            self.metrics.update(metrics)

    def add_cases(self, kind: str, rows: list[dict]) -> None:
        """Per-case records, for consumers that need to ask *which* case rather
        than *how many*. An aggregate says a rate moved; only these say what
        moved it."""
        self.cases[kind] = rows

    def headline_lines(self) -> list[str]:
        width = max(len(label) for _, label, _ in HEADLINE)
        out = []
        for key, label, spec in HEADLINE:
            value = self.metrics.get(key)
            shown = "-" if value is None else format(value, spec)
            out.append(f"  {label:<{width}}  {shown:>6}")
        return out

    def dump(self) -> str:
        out = ["", "=" * 78, "CBAC PIPELINE EVALUATION", "=" * 78]
        if self.metrics:
            out += ["", "HEADLINE", "-" * 8, *self.headline_lines()]
        for title, lines in self.sections:
            out += ["", title, "-" * len(title), *lines]
        out.append("")
        return "\n".join(out)


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a positive scores above a negative), ties at 0.5 — the Mann-Whitney
    statistic. Threshold-free, so it measures the signal in a score separately
    from whether the shipped cutoff exploits it."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(scores, labels, *, higher_flags: bool):
    """(threshold, Binary) maximising F1 over the observed values."""
    best_t, best_f1, best_b = float("nan"), -1.0, Binary()
    for t in sorted(set(scores)):
        b = Binary()
        for s, y in zip(scores, labels):
            b.add(s >= t if higher_flags else s <= t, y)
        if b.f1 == b.f1 and b.f1 > best_f1:
            best_t, best_f1, best_b = t, b.f1, b
    return best_t, best_b
