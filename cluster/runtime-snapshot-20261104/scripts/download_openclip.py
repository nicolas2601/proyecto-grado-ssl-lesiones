"""Download pinned, ungated, MIT-licensed OpenCLIP safetensors weights with published LFS hash verification."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"laion/CLIP-ViT-B-16-laion2B-s34B-b88K", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"}
WEIGHTS = {"laion/CLIP-ViT-B-16-laion2B-s34B-b88K": "open_clip_model.safetensors", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": "open_clip_pytorch_model.safetensors"}


def check_access(meta):
    if meta["repo_id"] not in ALLOWED or meta["private"] is not False or meta["gated"] is not False or meta["license_tag"] is not True:
        raise ValueError("Pinned ungated MIT public access not established")
    if not re.fullmatch("[0-9a-f]{40}", meta.get("revision", "")):
        raise ValueError("Full immutable revision required")


def check_file(path, record):
    raw = path.read_bytes()
    if len(raw) != record["bytes"]:
        raise ValueError("Downloaded size mismatch")
    sha = hashlib.sha256(raw).hexdigest()
    if record.get("published_lfs_sha256"):
        if sha != record["published_lfs_sha256"]:
            raise ValueError("Published LFS SHA256 mismatch")
    elif path.suffix == ".safetensors":
        raise ValueError("Weights must have a published LFS SHA256")
    else:
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != record["git_blob_sha1"]:
            raise ValueError("Published git blob SHA1 mismatch")
    return sha


def main(meta_path):
    meta = json.loads(Path(meta_path).read_text())
    check_access(meta)
    from huggingface_hub import hf_hub_download
    out = ROOT / "inputs/public-frozen-encoder-weights-20261025-v1" / meta["repo_id"].replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)
    records = {r["name"]: r for r in meta["files"]}
    checked = {}
    files = ("open_clip_config.json", WEIGHTS[meta["repo_id"]])
    for name in files:
        path = Path(hf_hub_download(meta["repo_id"], name, revision=meta["revision"], local_dir=str(out), token=False))
        checked[name] = {"sha256": check_file(path, records[name]), "bytes": records[name]["bytes"]}
    receipt = {"status": "COMPLETE", "repo_id": meta["repo_id"], "revision": meta["revision"], "gated": False,
               "license": "MIT", "files": checked, "remote_code_downloaded": False}
    (out / "download-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
