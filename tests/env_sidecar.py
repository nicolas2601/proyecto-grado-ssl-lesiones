"""Test helper: the environment block of a result, read from its sidecar.

Asserts the contract of dermssl.run_environment.write_sidecar: the result
itself carries no block, and the sidecar names it and pins its bytes.
"""

import hashlib
import json
from pathlib import Path

from dermssl import run_environment


def sidecar_env(path) -> dict:
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    inside = isinstance(result, dict) and isinstance(result.get("environment"), dict) \
        and "schema" in result["environment"]  # run_segnorm_baseline has its own, older 'environment'
    assert not inside, f"{path.name} carries the block inside"
    record = json.loads(run_environment.sidecar_path(path).read_text(encoding="utf-8"))
    assert record["describes"] == path.name
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    return record["environment"]
