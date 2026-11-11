"""Strict bindings to the approved MoCo development evidence; no sealed-test roles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PLAN_SHA256 = "e4589916523799fc653ce55a8c6a9754dea44827533985e0a6fd591b7cf88e1d"
ENCODER_SHA256 = "fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e"
CLASSES = {"HAM": ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),
           "PAD": ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")}
ROLES = {"HAM": "ham_dev", "PAD": "pad_adapt"}
META = {"PAD": ("/home/jovyan/work/pad_ufes_20/metadata.csv", "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"),
        "HAM": ("/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv", "06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5")}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_matrix(path, rows, dimension=3072):
    with np.load(path, allow_pickle=False) as cache:
        if (cache["image_ids"].tolist() != [r["image_id"] for r in rows]
                or cache["groups"].tolist() != [r["identity_component_id"] for r in rows]):
            raise ValueError("Feature identity/group alignment mismatch")
        x = np.asarray(cache["features"], dtype=np.float32)
    if x.shape != (len(rows), dimension):
        raise ValueError("Feature dimension mismatch")
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")
    return x


def _load_labels(task, rows):
    path, expected = META[task]
    if sha256_file(path) != expected:
        raise ValueError("Diagnostic metadata digest mismatch")
    wanted = {r["image_id"] for r in rows}
    if len(wanted) != len(rows):
        raise ValueError("Duplicate development identities")
    labels = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        records = csv.DictReader(stream)
        id_key = "img_id" if task == "PAD" else "image_id"
        label_key = "diagnostic" if task == "PAD" else ("label" if "label" in records.fieldnames else "dx")
        for record in records:
            identity = Path(record[id_key]).stem if task == "PAD" else record[id_key]
            if identity not in wanted:
                continue
            if identity in labels:
                raise ValueError("Duplicate development metadata identity")
            labels[identity] = record[label_key].strip()
    if set(labels) != wanted or set(labels.values()) != set(CLASSES[task]):
        raise ValueError("Development labels differ from registered classes")
    return np.asarray([labels[r["image_id"]] for r in rows])


def load_development(root, task):
    if task not in ROLES:
        raise ValueError("Only approved HAM/PAD development tasks are allowed")
    root = Path(root)
    campaign = root / "results/moco-v5-20261107"
    plan_path = campaign / "plan.json"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("MoCo v5 campaign plan digest mismatch")
    plan = json.loads(plan_path.read_text())
    role = ROLES[task]
    overlay = root / "results/content-exclusion-overlay-v1"
    manifest, receipt_path = overlay / f"{role}.jsonl", overlay / "receipt.json"
    for path in (manifest, receipt_path):
        if sha256_file(path) != plan["data"][str(path)]:
            raise ValueError("Approved development manifest/receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "COMPLETE" or sha256_file(manifest) != receipt["output_sha256"][manifest.name]:
        raise ValueError("Development overlay provenance mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    feature_path = campaign / f"jobs/quota4-s42/{role}-cls4.npz"
    feature_receipt = json.loads(feature_path.with_suffix(".json").read_text())
    if (feature_receipt["weights_sha256"] != ENCODER_SHA256
            or feature_receipt["features_sha256"] != sha256_file(feature_path)
            or feature_receipt["manifest_sha256"] != sha256_file(manifest)
            or feature_receipt["mode"] != "cls4"):
        raise ValueError("Feature cache provenance mismatch")
    features = load_feature_matrix(feature_path, rows)
    labels = _load_labels(task, rows)
    groups = np.asarray([r["identity_component_id"] for r in rows])
    order = [(r["image_id"], r["identity_component_id"]) for r in rows]
    provenance = {"task": task, "role": role, "classes": list(CLASSES[task]), "seed": 42,
                  "encoder_sha256": ENCODER_SHA256, "plan_sha256": PLAN_SHA256,
                  "manifest_sha256": sha256_file(manifest), "features_sha256": sha256_file(feature_path),
                  "metadata_sha256": META[task][1], "overlay_sha256": sha256_file(receipt_path),
                  "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                  "images": len(rows), "components": len(set(groups)),
                  "feature_precision": "CUDA BF16 extraction stored as float32"}
    return features, labels, groups, rows, provenance


EXTERNAL_RECIPE = {"mode": "cls4", "size": 224}
EXTERNAL_LABEL_FIELD = {"HAM": "class_ham", "PAD": "class_pad"}


def load_external_rows(manifest, features_dir, status_path, task):
    """Labelled external rows (e.g. ISIC 2019 non-HAM) usable only with the exact V10 CLS4@224 recipe."""
    if task not in EXTERNAL_LABEL_FIELD:
        raise ValueError("Only approved HAM/PAD external tasks are allowed")
    manifest, features_dir, status_path = Path(manifest), Path(features_dir), Path(status_path)
    status = json.loads(status_path.read_text())
    if (status.get("mode") != EXTERNAL_RECIPE["mode"] or status.get("size") != EXTERNAL_RECIPE["size"]
            or status.get("weights_sha256") != ENCODER_SHA256 or status.get("state") != "COMPLETE"
            or status.get("manifest_sha256") != sha256_file(manifest)):
        raise ValueError("External feature recipe mismatch: need CLS4 at 224 px, no TTA, same encoder")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if status.get("rows") != len(rows):
        raise ValueError("External feature row count differs from manifest")
    chunks = [np.load(features_dir / f"s{status['seed']}-{status['mode']}-{n:03d}.npy", allow_pickle=False)
              for n in range(status["chunks"])]
    x = np.concatenate(chunks).astype(np.float32)
    if x.shape != (len(rows), 3072) or not np.isfinite(x).all():
        raise ValueError("External feature matrix differs from CLS4 manifest rows")
    field = EXTERNAL_LABEL_FIELD[task]
    kept = [i for i, row in enumerate(rows) if row.get(field) in CLASSES[task]]
    labels = np.asarray([rows[i][field] for i in kept])
    groups = np.asarray([rows[i]["lesion_id"] or rows[i]["image_id"] for i in kept])
    provenance = {"source": "external", "task": task, "classes": list(CLASSES[task]),
                  "manifest_sha256": sha256_file(manifest), "status_sha256": sha256_file(status_path),
                  "encoder_sha256": ENCODER_SHA256, "rows_total": len(rows), "rows_used": len(kept),
                  "dropped_unmapped": len(rows) - len(kept), "components": len(set(groups))}
    return x[kept], labels, groups, provenance
