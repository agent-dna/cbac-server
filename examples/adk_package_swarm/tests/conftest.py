import os
import sys
import tempfile
from pathlib import Path

# Let the tests import the example's modules (package_tools, swarm, ...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing the example creates its AgentDNA identities; keep them, and the
# local store, out of the real AGENTDNA_LOCAL_DIR.
os.environ["AGENTDNA_LOCAL_DIR"] = tempfile.mkdtemp(prefix="agentdna-local-")
