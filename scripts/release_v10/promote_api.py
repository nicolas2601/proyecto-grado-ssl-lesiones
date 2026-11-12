#!/usr/bin/env python3
"""Scoped V10 promotion on the existing port, preserving the stopped V9 container."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("DERMSSL_API_RELEASE_DIR", ""))
V9_MODELS = Path(os.environ.get("DERMSSL_V9_MODELS_DIR", ""))
OLD_NAME = "derm-api-v9-rollback-20261108"
NEW_IMAGE = "sha256:aa55c4e8bfeb6276dce0fe811e306e9531111dba85acc9965d9d89d67b399008"
OLD_IMAGE = "sha256:2a67b4d785da9c23c2db72caa619a0a1341774c61b9bf1195b2ef62002838748"
MANIFEST = "7e54de3b0aac476ca7e55c029f99547c55c59f2c49fe53391b390d8213fb8664"


def run(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=180).stdout.strip()


def main():
    if not (ROOT.parts and V9_MODELS.parts):
        raise SystemExit("set DERMSSL_API_RELEASE_DIR and DERMSSL_V9_MODELS_DIR")
    candidate = json.loads((ROOT / "candidate-http-validation.json").read_text())
    if not candidate["passed"] or candidate["manifest_sha256"] != MANIFEST:
        raise ValueError("Candidate gate is incomplete")
    old_id = run("docker", "inspect", "derm-api", "--format", "{{.Image}}")
    image_id = run("docker", "image", "inspect", "derm-api:v10-20261108", "--format", "{{.Id}}")
    candidate_id = run("docker", "inspect", "derm-api-v10-candidate-20261108", "--format", "{{.Image}}")
    if old_id != OLD_IMAGE or image_id != NEW_IMAGE or candidate_id != NEW_IMAGE:
        raise ValueError("Production or candidate image identity changed")
    started = time.time()
    renamed = False
    try:
        run("docker", "stop", "derm-api-v10-candidate-20261108")
        run("docker", "stop", "derm-api")
        run("docker", "rename", "derm-api", OLD_NAME)
        renamed = True
        run("docker", "run", "-d", "--name", "derm-api", "--restart", "unless-stopped", "--cpus", "2",
            "--memory", "2500m", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "-p", "127.0.0.1:3200:3200",
            "-v", str(ROOT / "models/bundle") + ":/models/v10:ro",
            "-v", f"{V9_MODELS}:/models/v9:ro",
            "-e", "MOCO_MANIFEST_SHA256=" + MANIFEST, NEW_IMAGE)
        for attempt in range(30):
            try:
                with urllib.request.urlopen("http://127.0.0.1:3200/health", timeout=15) as response:
                    health = json.load(response)
                if health.get("release") != "v10.0.0" or health.get("status") != "ready":
                    raise ValueError("Wrong live identity")
                break
            except Exception:
                if attempt == 29:
                    raise
                time.sleep(1)
        run("python3", str(ROOT / "verify_http.py"), "--base", "https://derm.nicolasmoreno.site",
            "--bundle", str(ROOT / "models/bundle"), "--legacy-reference", str(ROOT / "candidate-http-validation.json"),
            "--output", str(ROOT / "production-https-validation.json"))
        report = {"status": "COMPLETE", "image": NEW_IMAGE, "previous_image": OLD_IMAGE,
                  "rollback_container": OLD_NAME, "elapsed_seconds": time.time() - started,
                  "manifest_sha256": MANIFEST, "origin": "127.0.0.1:3200", "tunnel_changed": False}
    except Exception as exc:
        if renamed:
            subprocess.run(["docker", "stop", "derm-api"], capture_output=True, text=True, timeout=60)
            subprocess.run(["docker", "rename", "derm-api", "derm-api-v10-failed-20261108"], capture_output=True, text=True)
            run("docker", "rename", OLD_NAME, "derm-api")
        run("docker", "start", "derm-api")
        report = {"status": "ROLLED_BACK", "error": str(exc), "previous_image": OLD_IMAGE}
        with (ROOT / "deployment-failure.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        raise
    with (ROOT / "deployment.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
