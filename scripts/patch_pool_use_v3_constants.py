"""Add V3 pool pins to the library additively and point the 026 trainer at them.

The library pins pool identity in reviewed source, not in configuration, so a
different training pool cannot be used without editing it. That guard is
deliberate. This change is therefore strictly ADDITIVE: the V2 constants stay
byte-identical and every caller that uses them keeps its behaviour, including the
completed 019-025 runs and the four-method line that shares this module.

Only ``train_dermlip_semi_026.py`` is repointed to the V3 pins.

Idempotent: re-running detects the constants are already present and stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HEADER = "# --- v3: source-cohort-disjoint pool (additive; v2 above is unchanged) ---"


def to_literal(value, indent: int = 4) -> str:
    text = json.dumps(value, indent=indent, sort_keys=True)
    return text.replace(": true", ": True").replace(": false", ": False").replace(": null", ": None")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--pool-use", type=Path, action="append", required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    args = parser.parse_args()

    payload = args.attestation.read_bytes()
    attestation = json.loads(payload)
    attestation_sha = hashlib.sha256(payload).hexdigest()
    bindings = attestation["bindings"]

    block = (
        f"\n\n{HEADER}\n"
        f"# Attestation: {args.attestation.resolve()}\n"
        f"# Derivation: mechanical re-derivation of v2, policy unchanged;\n"
        f"# patient_independence_accredited stays False.\n"
        f'V3_POOL_USE_ATTESTATION_SHA256 = "{attestation_sha}"\n'
        f"V3_POOL_USE_BINDINGS = {to_literal(bindings)}\n"
    )

    for path in args.pool_use:
        text = path.read_text(encoding="utf-8")
        if "V3_POOL_USE_ATTESTATION_SHA256" in text:
            print(f"  {path}: already patched, skipping")
            continue
        if "V2_POOL_USE_BINDINGS" not in text:
            raise SystemExit(f"{path} does not look like pool_use.py")
        before = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path.write_text(text + block, encoding="utf-8")
        print(f"  {path}")
        print(f"    sha256 {before} -> {hashlib.sha256(path.read_bytes()).hexdigest()}")
        # The V2 pins must survive untouched: other runs depend on them.
        new_text = path.read_text(encoding="utf-8")
        assert 'V2_POOL_USE_ATTESTATION_SHA256 = "aeaa24beae7857472ad07ca6b5e9e92478b7bbc66d1eb98ad7be1bb7c6f5385b"' in new_text
        assert new_text.startswith(text)

    trainer_text = args.trainer.read_text(encoding="utf-8")
    replaced = 0
    for old, new in (
        ("V2_POOL_USE_ATTESTATION_SHA256", "V3_POOL_USE_ATTESTATION_SHA256"),
        ("V2_POOL_USE_BINDINGS", "V3_POOL_USE_BINDINGS"),
    ):
        count = trainer_text.count(old)
        trainer_text = trainer_text.replace(old, new)
        replaced += count
        print(f"  trainer: {old} -> {new} ({count} occurrences)")
    if replaced == 0:
        raise SystemExit("trainer already repointed or unexpected content")
    args.trainer.write_text(trainer_text, encoding="utf-8")
    print(f"  trainer sha256: {hashlib.sha256(args.trainer.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
