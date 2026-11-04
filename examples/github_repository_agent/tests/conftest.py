import os
import sys
import tempfile
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT.parent))

# Importing the example creates its AgentDNA identities; keep them, and the
# local store, out of the real AGENTDNA_LOCAL_DIR.
os.environ["AGENTDNA_LOCAL_DIR"] = tempfile.mkdtemp(prefix="agentdna-local-")
