"""Stage reviewed V8 public weights. Default: show plan, without network or writes.

Downloads and local CPU conversions are separate opt-ins. This is not a training
launcher and never changes scientific configs, gates, datasets, or old checkouts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts/prepare_public_ssl_weights_v8.py"
HOSTS = {"dl.fbaipublicfiles.com", "storage.googleapis.com", "storage.cloud.google.com"}
ORIGINS = {"simclr", "mocov1", "mocov2", "mocov3", "dino"}
CHUNK = 8 * 1024 * 1024
CONVERTED_BUDGET = 256 * 1024 * 1024  # Fixed ResNet50 / ViT-S/16 encoder output.


def digest(path, limit=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while limit is None or limit > 0:
            block = stream.read(CHUNK if limit is None else min(CHUNK, limit))
            if not block:
                break
            h.update(block)
            if limit is not None:
                limit -= len(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def public_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.port not in (None, 443)):
        raise ValueError("Only allowlisted, credential-free public HTTPS weight URLs are supported")
    return url


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)  # Reject login pages, signed links, and unreviewed mirrors.
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def read_plan(path):
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1 or plan.get("protocol") != "public_ssl_staging_v8":
        raise ValueError("Unsupported weights plan")
    if not re.fullmatch(r"[a-f0-9]{40}", plan.get("adapter_revision", "")):
        raise ValueError("An exact adapter baseline commit is required")
    if not re.fullmatch(r"[a-f0-9]{64}", plan.get("adapter_sha256", "")):
        raise ValueError("An exact adapter script hash is required")
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("A nonempty artifact list is required")
    seen = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("review"), dict):
            raise ValueError("Each artifact and its review must be objects")
        aid = item.get("id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", aid) or aid in seen:
            raise ValueError("Artifact IDs must be unique safe directory names")
        seen.add(aid)
        if item.get("origin") not in ORIGINS:
            raise ValueError(f"Unknown SSL origin: {aid}")
        public_url(item["url"])
        if item.get("format") not in {"pytorch", "simclr_tensorflow"}:
            raise ValueError(f"Unsupported format: {aid}")
        if item["format"] == "simclr_tensorflow":
            if item["origin"] != "simclr" or not item.get("checkpoint_prefix"):
                raise ValueError("TensorFlow archive requires a SimCLR checkpoint prefix")
        for field in ("max_bytes", "max_unpacked_bytes"):
            if type(item.get(field)) is not int or item[field] <= 0:
                raise ValueError(f"Positive {field} required: {aid}")
        size = item.get("expected_bytes")
        if size is not None and (type(size) is not int or not 0 < size <= item["max_bytes"]):
            raise ValueError(f"Invalid expected size: {aid}")
        sha = item.get("expected_sha256")
        if sha is not None and not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError(f"Invalid expected SHA-256: {aid}")
    return plan


def require_review(item):
    review = item.get("review", {})
    required = ("reviewed_by", "reviewed_at", "license_id", "license_evidence_url",
                "artifact_license_scope", "source_selection_evidence", "normalization_evidence_url")
    if review.get("approved") is not True or any(not str(review.get(k) or "").strip() for k in required):
        raise ValueError("Artifact license, selection and normalization review is still pending")
    if review.get("input_normalization") not in {"unit_rgb", "imagenet"}:
        raise ValueError("Reviewed input normalization is required")
    return review


def room(path, needed, reserve):
    if shutil.disk_usage(path).free < needed + reserve:
        raise OSError("Insufficient disk budget; partial files were preserved")


def checkpoint_progress(meta_path, meta, stream, h):
    stream.flush()
    os.fsync(stream.fileno())
    meta.update(committed_bytes=stream.tell(), partial_sha256=h.hexdigest())
    atomic_json(meta_path, meta)


def download(item, folder, reserve, timeout, max_seconds):
    target = folder / ("source.zip" if item["format"] == "simclr_tensorflow" else "source.pth.tar")
    part = target.with_suffix(target.suffix + ".part")
    meta_path = folder / "transfer.json"
    spec = identity(item)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and meta.get("artifact_identity") != spec:
        raise ValueError("Artifact specification changed; use another staging directory")
    if target.exists():
        observed = digest(target)
        if (not meta.get("complete") or meta.get("sha256") != observed
                or meta.get("total_bytes") != target.stat().st_size):
            raise ValueError("Existing source is unrecorded or changed; refusing to overwrite it")
        return target, meta
    offset = part.stat().st_size if part.exists() else 0
    h = hashlib.sha256()
    if part.exists():
        if offset != meta.get("committed_bytes") or digest(part) != meta.get("partial_sha256"):
            raise ValueError("Partial file differs from durable progress; preserve it for review")
        with part.open("rb") as stream:
            for block in iter(lambda: stream.read(CHUNK), b""):
                h.update(block)
    if meta.get("total_bytes") == offset and part.exists():
        total = offset  # Recover a completed transfer interrupted before the rename.
    else:
        headers = {"Accept-Encoding": "identity", "User-Agent": "V8-public-weight-preparation/1"}
        if offset:
            etag = meta.get("strong_etag")
            if not etag:
                raise ValueError("Server gave no strong ETag; partial download cannot be resumed safely")
            headers.update(Range=f"bytes={offset}-", **{"If-Range": etag})
        request = urllib.request.Request(item["url"], headers=headers)
        opener = urllib.request.build_opener(PublicRedirect())
        started = time.monotonic()
        with opener.open(request, timeout=timeout) as response:
            public_url(response.geturl())
            content_type = response.headers.get("Content-Type", "").lower()
            if any(x in content_type for x in ("text/", "json", "xml")):
                raise ValueError("Server returned a document instead of checkpoint bytes")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Compressed HTTP transfer is not supported for byte-range resumption")
            length = response.headers.get("Content-Length")
            if length is None or not length.isdigit():
                raise ValueError("A known response length is required for the disk budget")
            length = int(length)
            etag = response.headers.get("ETag", "")
            strong = etag if re.fullmatch(r'"[^"\r\n]+"', etag) else None
            if offset:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match:
                    raise ValueError("Server did not honor Range/If-Range; existing partial is preserved")
                start, end, total = map(int, match.groups())
                if (start != offset or end != total - 1 or length != total - offset
                        or total != meta["total_bytes"] or strong != meta["strong_etag"]):
                    raise ValueError("Remote object changed or Content-Range is inconsistent")
            else:
                if response.status != 200:
                    raise ValueError("Expected a complete HTTP 200 response")
                total = length
            if total <= 0 or total > item["max_bytes"]:
                raise ValueError("Remote file exceeds its reviewed download budget")
            if item.get("expected_bytes") is not None and total != item["expected_bytes"]:
                raise ValueError("Remote size differs from the recorded candidate size")
            room(folder, total - offset, reserve)
            meta.update(artifact_identity=spec, source_url=item["url"], final_url=response.geturl(),
                        total_bytes=total, strong_etag=strong, complete=False)
            mode = "ab" if part.exists() else "xb"
            with part.open(mode) as stream:
                checkpoint_progress(meta_path, meta, stream, h)
                try:
                    while True:
                        if time.monotonic() - started > max_seconds:
                            raise TimeoutError("Transfer wall-time budget exceeded; resume with the same plan")
                        block = response.read(min(CHUNK, total - stream.tell() + 1))
                        if not block:
                            break
                        if stream.tell() + len(block) > total:
                            raise ValueError("Response exceeds its declared size")
                        room(folder, len(block), reserve)
                        stream.write(block)
                        h.update(block)
                        checkpoint_progress(meta_path, meta, stream, h)
                finally:
                    checkpoint_progress(meta_path, meta, stream, h)
            if part.stat().st_size != total:
                raise ValueError("Transfer is incomplete; partial file is available for resumption")
    observed = digest(part)
    if item.get("expected_sha256") and observed != item["expected_sha256"]:
        raise ValueError("Published/reviewed SHA-256 does not match; source was not promoted")
    meta.update(complete=True, sha256=observed,
                integrity="matches_expected_sha256" if item.get("expected_sha256") else "local_sha256_only")
    atomic_json(meta_path, meta)
    os.replace(part, target)
    return target, meta


def adapter_identity(plan):
    if digest(ADAPTER) != plan["adapter_sha256"]:
        raise ValueError("Adapter hash differs from the pinned version")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()
    revision = git("rev-parse", "HEAD")
    paths = ["scripts/prepare_public_ssl_weights_v8.py", "src"]
    # A newer commit may add this preparer/docs, but conversion code must match baseline.
    git("diff", "--exit-code", plan["adapter_revision"], "--", *paths)
    if git("ls-files", "--others", "--exclude-standard", "--", *paths):
        raise ValueError("Untracked conversion code exists; use a clean isolated checkout")
    return {"producer_revision": revision, "adapter_sha256": digest(ADAPTER)}


def extract_checkpoint(item, source, destination, reserve):
    prefix = item["checkpoint_prefix"]
    if prefix.startswith("/") or ".." in Path(prefix).parts or "\\" in prefix:
        raise ValueError("Unsafe TensorFlow archive prefix")
    with zipfile.ZipFile(source) as archive:
        selected = [entry for entry in archive.infolist()
                    if entry.filename == prefix + ".index"
                    or re.fullmatch(re.escape(prefix) + r"\.data-\d{5}-of-\d{5}", entry.filename)]
        names = [entry.filename for entry in selected]
        shards = [re.search(r"\.data-(\d{5})-of-(\d{5})$", name) for name in names if ".data-" in name]
        counts = {int(match[2]) for match in shards}
        if (len(names) != len(set(names)) or prefix + ".index" not in names or len(counts) != 1
                or {int(match[1]) for match in shards} != set(range(next(iter(counts), 0)))):
            raise ValueError("TensorFlow archive has missing or duplicate index/data shards")
        size = sum(entry.file_size for entry in selected)
        if size > item["max_unpacked_bytes"]:
            raise ValueError("TensorFlow shards exceed the extraction budget")
        room(destination, size, reserve)
        for entry in selected:
            if entry.is_dir() or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Checkpoint archive contains an unexpected symlink/directory")
            output = destination / Path(entry.filename).name
            with archive.open(entry) as src, output.open("xb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    return destination / Path(prefix).name


def convert(plan, item, source, transfer, folder, code, reserve):
    output_dir = folder / "converted"
    receipt_identity = identity({"artifact": item, "source_sha256": transfer["sha256"], "code": code})
    if output_dir.exists():
        receipt = json.loads((output_dir / "receipt.json").read_text())
        if receipt.get("identity") != receipt_identity:
            raise ValueError("Conversion identity changed; existing output is preserved")
        if not {"encoder.pt", "encoder.pt.json"}.issubset(receipt.get("files", {})):
            raise ValueError("Conversion receipt lacks the encoder/provenance hashes")
        for name, sha in receipt["files"].items():
            if Path(name).name != name:
                raise ValueError("Conversion receipt contains an unsafe file name")
            if digest(output_dir / name) != sha:
                raise ValueError("Converted output changed on disk")
        return output_dir
    room(folder, item["max_unpacked_bytes"] + CONVERTED_BUDGET, reserve)
    temp = Path(tempfile.mkdtemp(prefix="conversion-", dir=folder))
    local_source = source
    if item["format"] == "simclr_tensorflow":
        local_source = extract_checkpoint(item, source, temp, reserve)
    args = [sys.executable, str(ADAPTER), "--origin", item["origin"], "--source", str(local_source),
            "--format", item["format"], "--output", str(temp / "encoder.pt"),
            "--source-url", item["url"], "--license-id", item["review"]["license_id"],
            "--producer-revision", code["producer_revision"],
            "--input-normalization", item["review"]["input_normalization"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    with (temp / "adapter.log").open("w") as log:
        subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=600)
    provenance = json.loads((temp / "encoder.pt.json").read_text())
    if (temp / "encoder.pt").stat().st_size > CONVERTED_BUDGET:
        raise ValueError("Converted encoder exceeded the fixed architecture disk budget")
    if (provenance.get("protocol") != "public_ssl_initialization_v8"
            or provenance.get("method") != "ssl_" + item["origin"]
            or provenance.get("checkpoint_sha256") != digest(temp / "encoder.pt")
            or provenance.get("input_normalization") != item["review"]["input_normalization"]
            or provenance.get("source_license") != item["review"]["license_id"]):
        raise ValueError("Adapter provenance does not match the reviewed artifact")
    if item["format"] == "pytorch" and provenance.get("source_sha256") != transfer["sha256"]:
        raise ValueError("Adapter source hash differs from the staged bytes")
    # TF provenance records real shard paths: keep that directory stable after publication.
    if item["format"] == "simclr_tensorflow":
        provenance["source_files"] = [str(output_dir / Path(p).name) for p in provenance["source_files"]]
        atomic_json(temp / "encoder.pt.json", provenance)
    receipt = {"identity": receipt_identity, "code": code, "source_archive_sha256": transfer["sha256"],
               "files": {p.name: digest(p) for p in temp.iterdir() if p.is_file() and p.name != "adapter.log"},
               "status": "converted_strict", "scientific_gate_enabled": False}
    atomic_json(temp / "receipt.json", receipt)
    os.rename(temp, output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", action="append", help="Process only these IDs; repeat as needed")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--convert", action="store_true", help="Convert already complete sources on CPU")
    parser.add_argument("--reserve-bytes", type=int, default=1024 ** 3)
    parser.add_argument("--socket-timeout", type=float, default=30)
    parser.add_argument("--max-transfer-seconds", type=float, default=3600)
    args = parser.parse_args()
    if min(args.reserve_bytes, args.socket_timeout, args.max_transfer_seconds) <= 0:
        parser.error("Disk reserve and time limits must be positive")
    plan = read_plan(args.plan)
    selected = set(args.artifact or [item["id"] for item in plan["artifacts"]])
    if selected - {item["id"] for item in plan["artifacts"]}:
        parser.error("Unknown artifact ID")
    items = [item for item in plan["artifacts"] if item["id"] in selected]
    if not args.download and not args.convert:
        print(json.dumps({"execution_started": False, "workspace": str(args.workspace.expanduser().resolve()),
                          "artifacts": [{"id": i["id"], "review": i.get("review", {})} for i in items]}, indent=2))
        return 0
    workspace = args.workspace.expanduser().resolve()
    if (workspace == Path("/") or workspace == ROOT or ROOT in workspace.parents
            or any((p / ".git").exists() for p in (workspace, *workspace.parents))):
        raise ValueError("Choose a dedicated preparation directory outside the checkout")
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "weights.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = workspace / "weights-ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
            "protocol": plan["protocol"], "schema_version": 1, "artifacts": {}}
        if ledger.get("protocol") != plan["protocol"] or ledger.get("schema_version") != 1:
            raise ValueError("Existing ledger belongs to another protocol")
        failed = False
        for item in items:
            aid = item["id"]
            folder = workspace / aid
            if folder.is_symlink():
                raise ValueError("Artifact directories must not be symlinks")
            record = {"artifact_identity": identity(item), "plan_sha256": digest(args.plan),
                      "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            old = ledger["artifacts"].get(aid)
            if (old and old["artifact_identity"] != record["artifact_identity"]
                    and folder.exists() and any(folder.iterdir())):
                raise ValueError(f"Specification changed for {aid}; preserve this workspace and use another")
            try:
                require_review(item)
                folder.mkdir(exist_ok=True)
                target = folder / ("source.zip" if item["format"] == "simclr_tensorflow" else "source.pth.tar")
                if any(p.is_symlink() for p in (target, target.with_suffix(target.suffix + ".part"),
                                                folder / "transfer.json", folder / "converted")):
                    raise ValueError("Staging files must not be symlinks")
                if args.download:
                    target, transfer = download(item, folder, args.reserve_bytes,
                                                args.socket_timeout, args.max_transfer_seconds)
                else:
                    transfer = json.loads((folder / "transfer.json").read_text())
                    if (transfer.get("artifact_identity") != identity(item) or not transfer.get("complete")
                            or digest(target) != transfer.get("sha256")
                            or target.stat().st_size != transfer.get("total_bytes")):
                        raise ValueError("A complete, unchanged staged source is required before conversion")
                record.update(status="downloaded", source_sha256=transfer["sha256"],
                              source=str(target), integrity=transfer["integrity"])
                if args.convert:
                    code = adapter_identity(plan)
                    output = convert(plan, item, target, transfer, folder, code, args.reserve_bytes)
                    record.update(status="converted_strict", checkpoint=str(output / "encoder.pt"),
                                  provenance=str(output / "encoder.pt.json"))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
                    urllib.error.URLError, http.client.HTTPException, zipfile.BadZipFile) as error:
                failed = True
                record.update(status="blocked", error_type=type(error).__name__, error=str(error))
                if isinstance(error, urllib.error.HTTPError):
                    record["http_status"] = error.code
            ledger["artifacts"][aid] = record
            atomic_json(ledger_path, ledger)
            print(json.dumps({"id": aid, **record}), flush=True)
        return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
