# scripts/

Developer tools. None of them are part of the service or run in CI.

| Script | What it does |
|---|---|
| `inspect_policy.py` | Walks one eval policy through chunking and chunk classification. |
| `inspect_drift.py` | Walks one eval policy's intents through Check 1, the drift layer. |
| `inspect_tiers.py` | Walks one eval policy's actions through Tiers 1–3, the policy layer. |
| `eval_dashboard.py` | Rebuilds the eval HTML dashboard from a `--eval-json` file. |
| `test_lifecycle.py` | End-to-end smoke test against the live Docker Postgres. |

Run everything from the repo root. Each script inserts the repo root on
`sys.path` itself, so no `PYTHONPATH` is needed — except `test_lifecycle.py`.

## The three inspectors

Where the eval suite reports *what* each stage scored across the whole corpus,
these print *why* for one policy. They load the encoder, the NLI cross-encoder
and (for drift) HHEM, so the first run downloads weights. No database, no
`AGENTDNA_API_KEY` — the Provenance Layer is stubbed out, and the policies come
from `cbac_service/tests/eval/corpus.py`.

```bash
uv run python scripts/inspect_policy.py --list        # the policy ids
uv run python scripts/inspect_policy.py payments
uv run python scripts/inspect_drift.py payments --intent trip
uv run python scripts/inspect_tiers.py payments --oracle
```

All three take a policy id (default `payments`) and `--list`. `inspect_drift`
and `inspect_tiers` also take `--intent ID` to narrow to one user request.
`inspect_tiers --oracle` searches the index a correct chunker and classifier
would have built, which separates classification error from tier error.

Each script's `--help` carries the full explanation of what its columns mean.

## eval_dashboard.py

```bash
uv run python scripts/eval_dashboard.py eval.json -o eval.html
```

A normal eval run writes the page directly with `--eval-html`. This is for
rebuilding a page from a stored JSON without paying for the models again. The
JSON must come from a run that exported per-case rows (`cases`, `specs`,
`caps`, `chunks`), otherwise it exits and tells you to re-run the eval.

## test_lifecycle.py

Exercises the full pipeline — chunking, NLI classification, embedding,
vector/BM25/hybrid search, tiered decisions — against a real Postgres, then
deletes what it wrote. Needs Docker Postgres up (see the root README) and
`PYTHONPATH`:

```bash
export DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac"
PYTHONPATH=. uv run python scripts/test_lifecycle.py
```
