#!/usr/bin/env python3
"""Verify published-runtime Docker predictions using synthetic fixtures only."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def compare(actual, expected, classes):
    if actual.get("classes") != classes or set(actual.get("probs", {})) != set(classes):
        raise ValueError("Runtime class order or probability keys differ")
    probabilities = [float(actual["probs"][name]) for name in classes]
    if any(not 0 <= value <= 1 for value in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("Runtime probabilities are invalid")
    maximum = max(abs(a - b) for a, b in zip(probabilities, expected, strict=True))
    predicted = max(range(len(classes)), key=lambda i: probabilities[i])
    reference = max(range(len(classes)), key=lambda i: expected[i])
    return {"passed": maximum <= 1e-4 and predicted == reference,
            "max_abs_diff": maximum, "argmax_equal": predicted == reference, "tolerance": 1e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    cases = json.loads((root / "parity-fixtures.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    checks = []
    for case in cases:
        path = (root / case["image"]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError("Synthetic fixture path/hash mismatch")
        for task in ("HAM", "PAD"):
            command = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
                       "--cpus", "2", "--memory", "1500m", "-v", f"{path}:/input/image.png:ro", args.image,
                       "--task", task, "--image", "/input/image.png"]
            started = time.monotonic()
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
            actual = json.loads(result.stdout)
            check = compare(actual, case["expected"][task], manifest["tasks"][task]["classes"])
            check.update(task=task, fixture=path.name, seconds=time.monotonic() - started)
            checks.append(check)
            if not check["passed"]:
                raise ValueError(f"Docker parity failed: {check}")
    report = {"passed": True, "checks": checks, "network": "none", "non_root": True,
              "read_only": True, "image": args.image, "cases": len(cases)}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed": True, "checks": len(checks),
                      "max_abs_diff": max(c["max_abs_diff"] for c in checks)}))


if __name__ == "__main__":
    main()
