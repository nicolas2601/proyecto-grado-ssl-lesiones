from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

TASK_CLASSES = {
    "MOCO_HAM7": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    "MOCO_PAD6": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
}
OUTPUT_NAMES = ("ham7_probabilities", "pad6_probabilities")
THREADS = 2
RELEASE = "v10.0.0"
SEED = 42
BACKBONE_SHA256 = "fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e"
EXPECTED_RECIPE = {
    "feature": "cls4",
    "layers": [8, 9, 10, 11],
    "layer_norm": True,
    "l2_normalize": True,
    "n_views": 1,
    "tta": "none",
    "tau": 0.0,
    "color_constancy": False,
    "hue_jitter": 0.0,
}
REQUIRED_FILES = {
    "model.onnx",
    "report.json",
    "runner.py",
    "preprocess.py",
    "code/dermssl/release/moco_preprocess.py",
    "heads/HAM-reader.json",
    "heads/HAM-reader.npz",
    "heads/PAD-reader.json",
    "heads/PAD-reader.npz",
}


class V10ArtifactError(RuntimeError):
    """A V10 model manifest or release artifact is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(bundle_dir: Path, expected_sha256: str | None) -> dict[str, Any]:
    if expected_sha256 is None:
        raise V10ArtifactError("manifest SHA256 pin is required")
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise V10ArtifactError("manifest SHA256 pin is invalid")
    path = bundle_dir / "manifest.json"
    try:
        manifest_bytes = path.read_bytes()
    except OSError as exc:
        raise V10ArtifactError("V10 manifest is missing") from exc
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != expected_sha256:
        raise V10ArtifactError("manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V10ArtifactError("V10 manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise V10ArtifactError("V10 manifest must be an object")
    if manifest.get("schema") != "dermssl.moco-bundle.v1":
        raise V10ArtifactError("unsupported V10 manifest schema")
    _validate_contract(manifest)
    manifest["_manifest_sha256"] = digest
    return manifest


def _validate_contract(manifest: dict[str, Any]) -> None:
    if (manifest.get("release") != RELEASE or manifest.get("seed") != SEED
            or manifest.get("backbone_sha256") != BACKBONE_SHA256):
        raise V10ArtifactError("V10 manifest provenance mismatch")
    tasks = {
        "HAM": {"classes": TASK_CLASSES["MOCO_HAM7"], "output_name": OUTPUT_NAMES[0], "model_key": "MOCO_HAM7"},
        "PAD": {"classes": TASK_CLASSES["MOCO_PAD6"], "output_name": OUTPUT_NAMES[1], "model_key": "MOCO_PAD6"},
    }
    if manifest.get("tasks") != tasks:
        raise V10ArtifactError("V10 manifest tasks do not match HAM7/PAD6 contract")
    if manifest.get("recipe") != EXPECTED_RECIPE:
        raise V10ArtifactError("V10 manifest recipe does not match single-view CLS4")
    expected_input = {"name": "pixel_values", "dtype": "float32", "shape": ["batch", 3, 224, 224]}
    if manifest.get("feature_dim") != 3072 or manifest.get("input") != expected_input:
        raise V10ArtifactError("V10 manifest feature/input contract mismatch")
    if manifest.get("report") != "report.json":
        raise V10ArtifactError("V10 manifest report path does not match release contract")


def verify_bundle_file(bundle_dir: Path, manifest: dict[str, Any], relative_path: str) -> Path:
    files = manifest.get("files")
    if not isinstance(files, dict) or relative_path not in files:
        raise V10ArtifactError(f"V10 manifest has no SHA256 for {relative_path}")
    candidate = Path(relative_path)
    root = bundle_dir.resolve()
    path = (root / candidate).resolve()
    if candidate.is_absolute() or ".." in candidate.parts or not path.is_relative_to(root):
        raise V10ArtifactError("unsafe V10 bundle path")
    expected = files[relative_path]
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise V10ArtifactError(f"V10 manifest has invalid SHA256 for {relative_path}")
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise V10ArtifactError(f"V10 bundle file is missing: {relative_path}") from exc
    if digest != expected:
        raise V10ArtifactError(f"V10 SHA256 mismatch for {relative_path}")
    return path


def verify_bundle_files(bundle_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise V10ArtifactError("V10 manifest is missing file hashes")
    paths = {relative: verify_bundle_file(bundle_dir, manifest, relative) for relative in files}
    if not REQUIRED_FILES.issubset(files):
        raise V10ArtifactError("V10 manifest is missing required runtime files")
    if files["preprocess.py"] != files["code/dermssl/release/moco_preprocess.py"]:
        raise V10ArtifactError("V10 bundled preprocessing copies differ")
    return paths


def load_preprocessor(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dermssl_v10_verified_preprocess", path)
    if spec is None or spec.loader is None:
        raise V10ArtifactError("V10 canonical preprocessing source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - hash-verified runtime module boundary
        raise V10ArtifactError("V10 canonical preprocessing source failed to load") from exc
    if not callable(getattr(module, "preprocess_image", None)):
        raise V10ArtifactError("V10 preprocessing module has no preprocess_image function")
    return module.preprocess_image
