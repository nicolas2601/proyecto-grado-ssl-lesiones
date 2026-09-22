"""Make ``src/`` and the repo root importable without installing the package.

The root matters as much as ``src``: several tests import ``scripts.<module>``
directly. Locally that resolved through pytest's own path handling and in CI
it did not, so twenty-nine tests failed on a machine nobody was watching
while passing on the one everybody was. Putting both on the path removes the
difference instead of relying on it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
