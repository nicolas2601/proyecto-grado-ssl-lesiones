#!/usr/bin/env python3
"""Audit declared public-initialization metadata and bytes without loading tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit

BANK_ORIGINS = {frozenset(("simclr", "mocov1", "mocov2")): "usl_net",
                frozenset(("mocov3", "mocov2", "dino")): "rpi_net"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_METADATA_FILES = {
    "source-bank.rpi-net.reviewed-candidate.json",
    "source-bank.usl-net.reviewed-candidate.json",
    "usl-net-synthetic-forward-20261014.json",
    "usl-net.bank-verification.json",
}
EXPECTED_EVIDENCE_FILES = {
    "usl-net-synthetic-forward-20261014.json", "usl-net.bank-verification.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stat_identity(path):
    current = path.stat()
    return {"bytes": current.st_size, "mtime_ns": current.st_mtime_ns,
            "ctime_ns": current.st_ctime_ns, "inode": current.st_ino, "device": current.st_dev}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def closed_flags():
    return {"patient_lesion_independence": "unresolved", "freeze_blocked": True,
            "freeze_blocked_reason": "clinical_independence_unresolved",
            **{key: False for key in ("manifest_frozen", "training_consumable", "execution_ready",
                "execution_enabled", "launch_ready", "training_started", "patient_independence_accredited",
                "exposure_review_complete", "isolation_global_verified", "historical_exposure_verified",
                "scientific_gate_receipts_issued", "future_checkpoint_lineage_verified")}}


def publish(path, document):
    payload = (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def reject_learned_path(path):
    components = Path(path).as_posix().lower().replace("\\", "/").split("/")
    require(not any(re.search(r"(?:^|[-_])v7(?:$|[-_])|camino[-_]?b", item) for item in components),
            "Excluded learned V7/Camino B path: " + str(path))


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, path, expected=None, expected_bytes=None, binary=False):
        path = Path(path)
        require(path.is_absolute(), "Input paths must be absolute: " + str(path))
        key = str(path.resolve())
        if key in self.records:
            record = self.records[key]
            require(stat_identity(path) == record["stat"], "Input changed: " + key)
        else:
            before = stat_identity(path)
            actual = digest(path)
            require(before == stat_identity(path), "Input changed while hashing: " + key)
            record = {"path": key, "sha256": actual, "bytes": before["bytes"],
                      "stat": before, "binary_hashed_once": binary}
            self.records[key] = record
        require(expected is None or (isinstance(expected, str) and SHA256.fullmatch(expected)
                and record["sha256"] == expected), "Input SHA-256 mismatch: " + key)
        require(expected_bytes is None or record["bytes"] == expected_bytes, "Input size mismatch: " + key)
        return record

    def recheck(self):
        for record in self.records.values():
            path = Path(record["path"])
            require(stat_identity(path) == record["stat"], "Input changed before publication: " + str(path))
            if not record["binary_hashed_once"]:
                require(digest(path) == record["sha256"], "Metadata changed before publication: " + str(path))


def inspect_source(origin, source, inputs):
    require(isinstance(source, dict), "Source entry must be an object: " + origin)
    checkpoint = Path(source["checkpoint"])
    sidecar = Path(source["provenance"])
    reject_learned_path(checkpoint)
    reject_learned_path(sidecar)
    sidecar_identity = inputs.read(sidecar)
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    require(provenance.get("protocol") == "public_ssl_initialization_v8", "Unknown initialization protocol: " + str(sidecar))
    require(provenance.get("method") == "ssl_" + origin, "Origin/method mismatch: " + str(sidecar))
    require(provenance.get("uses_expert_masks") is False and provenance.get("uses_diagnoses") is False,
            "Expert-mask/diagnosis learning claims missing or incompatible: " + str(sidecar))
    require(provenance.get("lineage") is None, "Unexpected learned lineage on public initialization: " + str(sidecar))
    for field in ("producer_revision", "source_url", "source_license", "adapter"):
        require(isinstance(provenance.get(field), str) and provenance[field].strip(),
                "Missing declared " + field + ": " + str(sidecar))
    require(SHA256.fullmatch(provenance.get("source_sha256", "")) is not None,
            "Missing source SHA-256: " + str(sidecar))
    url = urlsplit(provenance["source_url"])
    require(url.scheme == "https" and bool(url.hostname) and url.username is None,
            "Source URL must be declared HTTPS: " + str(sidecar))
    expected_arch = "DINO_ViT-S/16_224" if origin == "dino" else "ResNet50_encoder_local_last_stride_1"
    require(provenance.get("architecture") == expected_arch, "Unsupported architecture: " + str(sidecar))
    require(provenance.get("input_normalization") in ("unit_rgb", "imagenet"), "Missing normalization: " + str(sidecar))
    count = provenance.get("mapped_tensor_count")
    require(type(count) is int and count > 0, "Invalid mapped tensor count: " + str(sidecar))
    if origin == "simclr":
        require(provenance.get("stem_pooling") == "tensorflow_same", "SimCLR requires TensorFlow SAME stem")
        require(provenance.get("upstream_numerical_equivalence_verified") is False,
                "SimCLR original numerical equivalence remains unverified")
    elif origin != "dino":
        require(provenance.get("stem_pooling", "symmetric") == "symmetric", "MoCo requires symmetric stem")
    for field, value in provenance.items():
        if field in ("uses_expert_masks", "uses_diagnoses"):
            require(value is False, "Incompatible learning claim: " + str(sidecar))
    weight_identity = inputs.read(checkpoint, provenance["checkpoint_sha256"], binary=True)
    return {"origin": origin, "checkpoint": weight_identity, "sidecar": sidecar_identity,
            "declared_provenance": provenance, "bank_source_declaration": source,
            "license_scope": "Declared source license only; no legal certification.",
            "verification_scope": "Declared initialization metadata and binary hash only; tensor payload not loaded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--inspection-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    created = False
    stage = "validate_inputs"
    inputs = Inputs()
    try:
        require(sys.platform.startswith("linux") and Path("/home/jovyan/work").is_dir(), "UNABIA runtime required")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created = True
        code = inputs.read(Path(__file__).resolve())
        inspection_identity = inputs.read(args.inspection, args.inspection_sha256)
        inspection = json.loads(args.inspection.read_text(encoding="utf-8"))
        weights = inspection["weights"]
        require(isinstance(weights, dict) and set(weights) == EXPECTED_METADATA_FILES,
                "Inspection must contain the four prescribed metadata basenames")
        bank_reports = {}
        prior_evidence = {}
        stage = "verify_declared_metadata_files"
        for label, entry in weights.items():
            require(isinstance(entry, dict) and entry.get("exists") is True,
                    "Declared metadata missing: " + str(label))
            path = Path(entry["path"])
            require(path.name == label, "Inspection key/path basename mismatch: " + str(label))
            require(type(entry.get("bytes")) is int and entry["bytes"] >= 0,
                    "Declared integer byte size required: " + str(path))
            reject_learned_path(path)
            metadata_identity = inputs.read(path, entry["sha256"], entry["bytes"])
            document = json.loads(path.read_text(encoding="utf-8"))
            sources = document.get("sources")
            if isinstance(sources, dict) and frozenset(sources) in BANK_ORIGINS:
                family = BANK_ORIGINS[frozenset(sources)]
                require(family not in bank_reports, "Duplicate declared source bank: " + family)
                stage = "verify_bank_" + family
                bank_reports[family] = {"bank": metadata_identity, "sources": {
                    origin: inspect_source(origin, source, inputs) for origin, source in sources.items()}}
            else:
                require(label in EXPECTED_EVIDENCE_FILES,
                        "Unrecognized source-bank origins or evidence basename: " + str(path))
                prior_evidence[label] = {"file": metadata_identity, "declared_document": document,
                    "scope": "Existing evidence inventory only; no rerun or new numerical verification."}
        require(set(bank_reports) == {"usl_net", "rpi_net"}, "Both reviewed candidate banks are required")
        require(len(prior_evidence) == 2, "Expected bank verification and synthetic-forward evidence inventory")
        stage = "recheck_inputs_before_publication"
        inputs.recheck()
        report = {"schema": "v8_declared_asset_lineage_audit.v1", "schema_version": 1,
            "status": "public_asset_metadata_verified", "code": code, "inspection": inspection_identity,
            "banks": bank_reports, "existing_evidence": prior_evidence,
            "scope": "Public initialization metadata only; no tensor load, forward, conversion, download or training.",
            "pending_lineage": {"sled": "Producer and pseudo-mask manifests not materialized.",
                "triderm": "Requires future pinned SLED teacher manifest.",
                "usl_net": "Future CCAM and iterative checkpoints require separate learned lineage.",
                "rpi_net": "Future cycle/phase checkpoints require separate learned lineage."},
            "limitations": ["Public source SHA is a sidecar declaration; original upstream bytes not revalidated here.",
                "Binary SHA computed once per unique path, then stat identity rechecked before publication.",
                "Metadata sidecars and inspection are rehashed before publication.",
                "No proof of historical exposure, clinical independence, future pseudo-mask lineage or tensor compatibility."],
            **closed_flags()}
        output = publish(args.output_dir / "declared_asset_lineage_audit.json", report)
        receipt = {"schema": "v8_declared_asset_lineage_receipt.v1", "schema_version": 1,
            "status": report["status"], "code": code, "inspection": inspection_identity,
            "inputs": list(inputs.records.values()), "report": output, **closed_flags()}
        final = publish(args.output_dir / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": final,
                          "unique_binary_files": sum(item["binary_hashed_once"] for item in inputs.records.values()), **closed_flags()}))
        return 0
    except Exception as error:
        failure = {"schema": "v8_declared_asset_lineage_receipt.v1", "schema_version": 1,
            "status": "failed_closed", "stage": stage, "error_type": type(error).__name__,
            "diagnostic": str(error) if isinstance(error, (ValueError, KeyError, OSError)) else "Input/schema/IO failure; inspect declared inputs at this stage",
            "inputs_verified_before_failure": list(inputs.records.values()), **closed_flags()}
        if created:
            publish(args.output_dir / "receipt.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
