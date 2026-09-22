"""Make ``src/`` and the repo root importable without installing the package.

The root matters as much as ``src``: several tests import ``scripts.<module>``
directly. Locally that resolved through pytest's own path handling and in CI
it did not, so twenty-nine tests failed on a machine nobody was watching
while passing on the one everybody was. Putting both on the path removes the
difference instead of relying on it.
"""

import os
import sys
from pathlib import Path

# No test may talk to Weights & Biases. One did, forked a wandb-core helper,
# and blocked the CI job until its own timeout killed it - at 96 percent,
# with no failing test to point at. Disabling it here covers every runner and
# every laptop, instead of depending on whoever set the variable.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
