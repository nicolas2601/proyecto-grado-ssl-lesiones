#!/usr/bin/env python3
"""Bounded synthetic-only HTTP conformance/parity check for candidate or production V10."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode


def request(url, *, image=None, content_type="image/png", method=None, headers=()):
    with tempfile.TemporaryDirectory(prefix="dermssl-http-") as directory:
        header_path = Path(directory) / "headers"
        command = ["curl", "--silent", "--show-error", "--max-time", "120", "--dump-header", str(header_path),
                   "--write-out", "\n%{http_code}"]
        if method:
            command += ["-X", method]
        for header in headers:
            command += ["-H", header]
        if image is not None:
            command += ["-F", f"file=@{image};type={content_type}"]
        command.append(url)
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=125)
        body, status = result.stdout.rsplit("\n", 1)
        try:
            payload = json.loads(body) if body.strip() and method != "OPTIONS" else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Non-JSON API response, status={status}") from exc
        parsed = {}
        for line in header_path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                parsed[key.lower()] = value.strip()
        return int(status), payload, parsed


def compare(actual, expected, classes):
    if actual.get("classes") != classes or set(actual.get("probs", {})) != set(classes):
        raise ValueError("API class contract mismatch")
    values = [actual["probs"][name] for name in classes]
    if any(not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-5:
        raise ValueError("API returned invalid normalized probabilities")
    maximum = max(abs(a - b) for a, b in zip(values, expected, strict=True))
    winner = max(range(len(values)), key=values.__getitem__)
    reference = max(range(len(expected)), key=expected.__getitem__)
    if maximum > 1e-4 or winner != reference or actual.get("pred") != classes[winner]:
        raise ValueError(f"API probability parity failed: max_abs={maximum}")
    return {"passed": True, "max_abs_diff": maximum, "argmax_equal": True, "tolerance": 1e-4}


def require_get(base, path, check):
    status, payload, _ = request(base + path)
    if status != 200 or not check(payload):
        raise ValueError(f"GET contract failed: {path}, status={status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-base")
    parser.add_argument("--legacy-reference", type=Path)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    root = args.bundle.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    cases = json.loads((root / "parity-fixtures.json").read_text())
    require_get(base, "/health", lambda d: d.get("status") == "ready" and d.get("release") == "v10.0.0")
    require_get(base, "/api/status", lambda d: d.get("classification_ready") and d.get("model_key") == "MOCO_HAM7"
                and d.get("identity", {}).get("model_sha256") == manifest["files"]["model.onnx"])
    require_get(base, "/openapi.json", lambda d: d.get("info", {}).get("version") == "10.0.0")
    require_get(base, "/api/report", lambda d: d.get("release") == "v10.0.0")
    require_get(base, "/api/models", lambda d: d.get("default_model") == "MOCO_HAM7")
    checks = []
    for case in cases:
        image = root / case["image"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError("Synthetic image digest mismatch")
        for task, declaration in manifest["tasks"].items():
            query = urlencode({"model": declaration["model_key"], "mode": "base"})
            started = time.monotonic()
            status, actual, _ = request(base + "/api/predict?" + query, image=image)
            if (status != 200 or actual.get("views") != 1 or actual.get("feature_dim") != 3072
                    or actual.get("model_key") != declaration["model_key"]
                    or actual.get("identity", {}).get("release") != "v10.0.0"
                    or actual.get("identity", {}).get("model_sha256") != manifest["files"]["model.onnx"]):
                raise ValueError(f"MoCo API identity/mode failure: {status}, {actual}")
            checks.append({"task": task, "fixture": image.name, "seconds": time.monotonic() - started,
                           **compare(actual, case["expected"][task], declaration["classes"])})
    legacy = legacy_checks(args, base, root, cases[0])
    invalid_checks(base, root / cases[0]["image"])
    require_get(base, "/health", lambda d: d.get("status") == "ready")
    report = {"passed": True, "base": base, "checks": checks, "legacy": legacy, "error_cases": 5,
              "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed": True, "MoCo_checks": len(checks), "legacy_checks": len(legacy),
                      "max_abs_diff": max(c["max_abs_diff"] for c in checks)}))


def legacy_checks(args, base, root, case):
    if args.legacy_reference:
        references = json.loads(args.legacy_reference.read_text())["legacy"]
    else:
        references = [{"model_key": key, "mode": mode} for key in ("A71", "B71") for mode in ("base", "tta")]
    results = []
    for reference in references:
        query = urlencode({"model": reference["model_key"], "mode": reference["mode"]})
        url = "/api/predict?" + query
        if args.legacy_base:
            status, old, _ = request(args.legacy_base.rstrip("/") + url, image=root / case["image"])
            if status != 200:
                raise ValueError("Legacy production reference failed")
            reference = {**reference, "classes": old["classes"],
                         "expected": [old["probs"][name] for name in old["classes"]]}
        status, actual, _ = request(base + url, image=root / case["image"])
        if status != 200 or actual.get("identity", {}).get("release") != "v9.0.0" or actual.get("views") != 8:
            raise ValueError("Legacy identity or D4 behavior changed")
        results.append({**reference, **compare(actual, reference["expected"], reference["classes"])})
    return results


def invalid_checks(base, image):
    with tempfile.TemporaryDirectory(prefix="dermssl-errors-") as directory:
        corrupt = Path(directory) / "invalid.png"
        corrupt.write_bytes(b"not an image")
        cases = [("/api/predict", {"image": corrupt}, 400),
                 ("/api/predict", {"image": corrupt, "content_type": "text/plain"}, 415),
                 ("/api/predict?model=UNKNOWN", {"image": image}, 422),
                 ("/api/predict?mode=tta", {"image": image}, 422),
                 ("/api/features", {"method": "POST"}, 501)]
        for path, kwargs, expected in cases:
            status, _, _ = request(base + path, **kwargs)
            if status != expected:
                raise ValueError(f"HTTP error contract changed: {path}, {status} != {expected}")
    status, _, headers = request(base + "/api/predict", method="OPTIONS", headers=(
        "Origin: https://dermssl-phone.vercel.app", "Access-Control-Request-Method: POST",
        "Access-Control-Request-Headers: content-type"))
    if status != 200 or headers.get("access-control-allow-origin") != "https://dermssl-phone.vercel.app":
        raise ValueError("Explicit frontend CORS contract failed")


if __name__ == "__main__":
    main()
