from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

TASK_CLASSES = {
    "MOCO_HAM7": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    "MOCO_PAD6": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
}
OUTPUT_NAMES = ("ham7_probabilities", "pad6_probabilities")
THREADS = 2
SEED = 42
BACKBONE_SHA256 = "fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e"
SCHEMA_V1 = "dermssl.moco-bundle.v1"
SCHEMA_V2 = "dermssl.moco-bundle.v2"
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
V2_RECIPE = {
    "features": "cls4",
    "layers": [8, 9, 10, 11],
    "layer_norm": True,
    "normalization": "row_l2",
    "prior": False,
    "resolution": 448,
    "softmax_temperature": 1.0,
    "standard_scaler": False,
    "tau": 0.0,
    "tta": "d4-mean",
    "views": 8,
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
V2_REQUIRED_FILES = frozenset({
    "model.onnx",
    "runner.py",
    "preprocess.py",
    "code/dermssl/release/moco_preprocess.py",
    "heads/HAM-reader.json",
    "heads/PAD-reader.json",
})


@dataclass(frozen=True)
class BundleSpec:
    """Per-schema runtime contract: recipe, input size, head format and required files."""

    schema: str
    release: str
    recipe: dict[str, Any]
    input_size: int
    inference_mode: str
    views: int
    head_ext: str
    required_files: frozenset[str]
    report_file: str | None
    recipe_error: str
    # Listed in the manifest but not shipped with the ONNX runtime (torch-only). Hash-checked if present.
    optional_files: frozenset[str] = frozenset()


BUNDLE_SPECS = {
    SCHEMA_V1: BundleSpec(
        schema=SCHEMA_V1,
        release="v10.0.0",
        recipe=EXPECTED_RECIPE,
        input_size=224,
        inference_mode="single_view",
        views=1,
        head_ext="npz",
        required_files=frozenset(REQUIRED_FILES),
        report_file="report.json",
        recipe_error="V10 manifest recipe does not match single-view CLS4",
    ),
    SCHEMA_V2: BundleSpec(
        schema=SCHEMA_V2,
        release="v10.1.0",
        recipe=V2_RECIPE,
        input_size=448,
        inference_mode="d4_tta_8view",
        views=8,
        head_ext="json",
        required_files=V2_REQUIRED_FILES,
        report_file=None,
        recipe_error="V10 manifest recipe does not match the v2 D4 contract",
        optional_files=frozenset({"classifier.pt"}),
    ),
}


class V10ArtifactError(RuntimeError):
    """A V10 model manifest or release artifact is invalid."""


def spec_for_manifest(manifest: dict[str, Any]) -> BundleSpec:
    spec = BUNDLE_SPECS.get(manifest.get("schema"))
    if spec is None:
        raise V10ArtifactError("unsupported V10 manifest schema")
    return spec


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
    _validate_contract(manifest, spec_for_manifest(manifest))
    manifest["_manifest_sha256"] = digest
    return manifest


def _validate_contract(manifest: dict[str, Any], spec: BundleSpec) -> None:
    if (manifest.get("release") != spec.release or manifest.get("seed") != SEED
            or manifest.get("backbone_sha256") != BACKBONE_SHA256):
        raise V10ArtifactError("V10 manifest provenance mismatch")
    tasks = {
        "HAM": {"classes": TASK_CLASSES["MOCO_HAM7"], "output_name": OUTPUT_NAMES[0], "model_key": "MOCO_HAM7"},
        "PAD": {"classes": TASK_CLASSES["MOCO_PAD6"], "output_name": OUTPUT_NAMES[1], "model_key": "MOCO_PAD6"},
    }
    if manifest.get("tasks") != tasks:
        raise V10ArtifactError("V10 manifest tasks do not match HAM7/PAD6 contract")
    if manifest.get("recipe") != spec.recipe:
        raise V10ArtifactError(spec.recipe_error)
    size = spec.input_size
    expected_input = {"name": "pixel_values", "dtype": "float32", "shape": ["batch", 3, size, size]}
    if manifest.get("feature_dim") != 3072 or manifest.get("input") != expected_input:
        raise V10ArtifactError("V10 manifest feature/input contract mismatch")
    if spec.report_file is not None and manifest.get("report") != spec.report_file:
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
    # Unknown schemas fall back to the strictest (v1) contract: fail closed.
    spec = BUNDLE_SPECS.get(manifest.get("schema"), BUNDLE_SPECS[SCHEMA_V1])
    paths = {
        relative: verify_bundle_file(bundle_dir, manifest, relative)
        for relative in files
        if relative not in spec.optional_files or (bundle_dir / relative).is_file()
    }
    if not spec.required_files.issubset(files):
        raise V10ArtifactError("V10 manifest is missing required runtime files")
    if files["preprocess.py"] != files["code/dermssl/release/moco_preprocess.py"]:
        raise V10ArtifactError("V10 bundled preprocessing copies differ")
    return paths


def bind_input_size(preprocess: Callable[..., Any], size: int) -> Callable[[Any], Any]:
    """v1 preprocessors take only the image (224 px); v2 take (image, size)."""
    if len(inspect.signature(preprocess).parameters) >= 2:
        return lambda image: preprocess(image, size)
    return preprocess


def load_preprocessor(path: Path, input_size: int) -> Any:
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
    return bind_input_size(module.preprocess_image, input_size)
