from __future__ import annotations

import hashlib
from pathlib import Path


def test_training_recipe_sources_are_the_recovered_immutable_versions():
    root = Path(__file__).resolve().parents[1] / "scripts/training_moco"
    expected = {
        "moco_v3_20261029_runtime.py": "37fb14ac0883316d632d53ce0be9c3b8bacdbdb153f823df1bec986b248e1cf9",
        "moco_v3_20261029_helpers.py": "adee5b1583b11b251156a57694761a58c5ca1b3c98ea2e2452371ccd374083ee",
        "moco_v4_20261107_runtime.py": "389221fa30e7bef2f35049ee1c133182320ffcd3a92db2289941576c6bad711b",
        "moco_v4_20261107_helpers.py": "5351f5d4cb473a07ddc83ab486e53b85897248aa81024709611c224181f4a3c7",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
