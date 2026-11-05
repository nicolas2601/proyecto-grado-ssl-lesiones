"""Download only preapproved ungated public vision weights with published integrity checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"facebook/dinov2-large", "google/siglip2-so400m-patch14-384", "google/siglip2-base-patch16-224",
           "facebook/dinov2-small", "facebook/dinov2-giant"}  # ampliación autorizada 2026-10-27 (ledger)
FILES = ("config.json", "preprocessor_config.json", "README.md", "model.safetensors")


def check_access(metadata):
    if (metadata.get("repo_id") not in ALLOWED or metadata.get("private") is not False
            or metadata.get("gated") is not False or metadata.get("license_tag") is not True):
        raise ValueError("Public ungated permissive access is not established")


def check_download(path, metadata):
    path = Path(path)
    if path.stat().st_size != metadata["bytes"]:
        raise ValueError("Downloaded file size mismatch")
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    published = metadata.get("published_lfs_sha256")
    if published:
        if sha != published:
            raise ValueError("Published LFS SHA256 hash mismatch")
        verification = "Published LFS SHA256"
    else:
        if path.suffix == ".safetensors":
            raise ValueError("Weight hash must be published upstream")
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != metadata["git_blob_sha1"]:
            raise ValueError("Published Git blob hash mismatch")
        verification = "Published Git blob SHA1 plus local SHA256"
    return {"sha256": sha, "bytes": path.stat().st_size, "verified_against": verification}


def main(metadata_path):
    metadata = json.loads(Path(metadata_path).read_text())
    check_access(metadata)
    if not re.fullmatch("[0-9a-f]{40}", metadata.get("revision", "")):
        raise ValueError("Full immutable model revision required")
    environment = ROOT / "environments/public-encoders-20261025-v1"
    os.environ.update(HF_HOME=str(environment / "hf-cache"), HF_ENDPOINT="https://huggingface.co",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DOWNLOAD_TIMEOUT="60", HF_HUB_ETAG_TIMEOUT="30")
    os.environ.pop("HF_TOKEN", None)
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    fresh = HfApi(endpoint="https://huggingface.co", token=False).model_info(
        metadata["repo_id"], revision=metadata["revision"], token=False, timeout=30)
    check_access({"repo_id": metadata["repo_id"], "private": fresh.private,
                  "gated": fresh.gated, "license_tag": "license:apache-2.0" in (fresh.tags or [])})
    parent = ROOT / "inputs/public-frozen-encoder-weights-20261025-v1"
    parent.mkdir(exist_ok=True)
    if not parent.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Model directory escapes project")
    output = parent / metadata["repo_id"].replace("/", "--")
    output.mkdir(exist_ok=False)
    records = {row["name"]: row for row in metadata["files"]}
    if not set(FILES) <= set(records):
        raise ValueError("Expected safe single-file weights and native processor metadata")
    started, status = time.monotonic(), "INCOMPLETE"
    checked, failure, phase = {}, None, "download"
    try:
        for name in FILES:
            phase = "download"
            downloaded = Path(hf_hub_download(metadata["repo_id"], name,
                                             revision=metadata["revision"], local_dir=str(output), token=False))
            if not downloaded.resolve().is_relative_to(output.resolve()):
                raise ValueError("Downloaded file outside dedicated model directory")
            phase = "verify"
            checked[name] = check_download(downloaded, records[name])
            if name == "README.md" and not re.search(r"(?m)^license:\s*apache-2\.0\s*$", downloaded.read_text()):
                raise ValueError("Apache2 model-card license confirmation failed")
            print(json.dumps({"repo": metadata["repo_id"], "verified_file": name, **checked[name]}), flush=True)
        status = "COMPLETE"
    except Exception as error:
        failure = {"error_type": type(error).__name__, "phase": phase, "file": name,
                   "http_status": getattr(getattr(error, "response", None), "status_code", None)}
        raise RuntimeError("Public anonymous download failed; see structured receipt") from None
    finally:
        receipt = {"status": status, "repo_id": metadata["repo_id"], "revision": metadata["revision"],
                   "private": False, "gated": False, "license": "Apache-2.0", "files": checked,
                   "authentication": "Explicit token=False; no login or terms acceptance",
                   "seconds": time.monotonic() - started, "remote_code_downloaded": False,
                   "model_load_policy": "Safetensors only; concrete installed vision classes; no text inputs or remote repository code",
                   "failure": failure}
        with (output / "download-receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
        print(json.dumps({"repo_id": metadata["repo_id"], "status": status, "seconds": receipt["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    main(parser.parse_args().metadata)
