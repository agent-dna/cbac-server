"""The CBAC decision service, run with no hosted dependencies.

``cbac_service.main`` builds its engine on first use with a Rubix
``Provenance``. This runner assigns the engine first, with a
``LocalProvenance`` reading the shared local store instead, so the service
never reaches the chain:

- v2 (``/cbac/v2/*``) takes its policy from the upload and never reads
  provenance at all;
- v1 (``/cbac/v1/authorize``) reads an agent's policy from its agent card,
  which ``agentdna_local`` writes when the agent registers.

Same app, same endpoints, same Postgres and OpenFGA. Run it from the **repo
root** with the root venv — the service's ML dependencies live there:

    uv run --env-file cbac_service/.env.dev python examples/agentdna_local/service.py

or ``python -m agentdna_local.service`` wherever this package is installed
next to the service's dependencies.

Env: everything ``cbac_service/.env.sample`` lists, plus
``AGENTDNA_LOCAL_DIR`` (default ``~/.agentdna-local``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

# The repo root (for cbac_service) and examples/ (for this package).
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]

import uvicorn
from agentdna.provenance import Provenance

import agentdna_local
from cbac_service import main as service
from cbac_service.cbac import CBAC

HOST = os.environ.get("CBAC_SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CBAC_SERVICE_PORT", "8767"))


def build_engine(directory: str | None = None) -> CBAC:
    """A ``CBAC`` whose provenance is the local store. Cheap: the models load
    lazily, on the first decision."""
    local_dir = agentdna_local.install(directory)
    provenance = agentdna_local.LocalProvenance("cbac-service", config_path=local_dir)
    return CBAC(provenance=cast(Provenance, provenance))


def run() -> None:
    service._cbac = build_engine()
    print(f"[service] {HOST}:{PORT}, agentdna store {agentdna_local.default_dir()}")
    uvicorn.run(service.app, host=HOST, port=PORT, log_config=None)


if __name__ == "__main__":
    run()
