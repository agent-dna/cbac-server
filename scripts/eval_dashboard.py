"""Rebuild the eval dashboard from a JSON file written by `--eval-json`.

    uv run python scripts/eval_dashboard.py eval.json -o eval.html

A normal run needs neither step — `--eval-html PATH` writes the page directly:

    uv run pytest cbac_service/tests/eval --run-eval -q --eval-html eval.html

This exists for rebuilding an *old* run's page, or a new page from a stored
JSON, without paying for the models again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbac_service.tests.eval.dashboard import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", type=Path, help="the file written by --eval-json")
    ap.add_argument("-o", "--out", type=Path, default=Path("eval.html"))
    args = ap.parse_args()

    data = json.loads(args.json.read_text())
    missing = [k for k in ("cases", "policies", "metrics") if k not in data]
    if missing:
        raise SystemExit(
            f"{args.json} has no {', '.join(missing)} — it predates the per-case "
            f"export. Re-run the eval with --eval-json."
        )
    args.out.write_text(build(data))
    total = sum(len(v) for v in data["cases"].values())
    print(f"wrote {args.out} — {total} cases, {len(data['policies'])} policies")


if __name__ == "__main__":
    main()
