"""Capture real-model analyze and surrounding-skin luminance simulation outputs, in-process (no network, no shared API).

Every value written comes from the real classifier and segmenter artifacts given on the command line.
Missing or mismatched artifacts, a service that is not ready, or an unsupported image type fail the run.
Segmentation that the service reports as unavailable is recorded with its reason; nothing is filled in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO, REPO / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

DELTAS = (-9.73, -4.86, 4.86, 9.73)  # measured luminance spans, one and two bands; the API still accepts [-16, 16]
ANALYZE_MODEL = "MOCO_HAM7"
CAPTURE_SCHEMA = "dermssl.capture.v1"
MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
SHA256_PATTERN = r"[a-f0-9]{64}"


class CaptureError(RuntimeError):
    """The capture cannot run or a real service call failed; nothing partial is trusted."""


def analyze_name(stem: str) -> str:
    return f"analyze-{ANALYZE_MODEL}-{stem}.json"


def simulate_name(delta: int, stem: str) -> str:
    return f"simulate-tone-{delta}-{stem}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_pin(manifest: Path, pin: str) -> None:
    if not re.fullmatch(SHA256_PATTERN, pin or ""):
        raise CaptureError(f"{manifest.name} pin is missing or malformed")
    if not manifest.is_file():
        raise CaptureError(f"{manifest} is missing")
    if sha256_file(manifest) != pin:
        raise CaptureError(f"{manifest.name} SHA256 does not match the pin")


def check_artifacts(classifier_root: Path, classifier_pin: str, segmenter_dir: Path | None,
                    segmenter_pin: str | None, *, segmenter_check: bool = True) -> dict[str, str]:
    _check_pin(classifier_root / "manifest.json", classifier_pin)
    classifier_model = classifier_root / "model.onnx"
    if not classifier_model.is_file():
        raise CaptureError("model.onnx is missing under the classifier root")
    result = {"classifier_manifest_sha256": classifier_pin, "classifier_model_sha256": sha256_file(classifier_model)}
    if segmenter_check:
        _check_pin(segmenter_dir / "segmenter_manifest.json", segmenter_pin or "")
        segmenter_model = segmenter_dir / "segmenter.onnx"
        if not segmenter_model.is_file():
            raise CaptureError("segmenter.onnx is missing under the segmenter directory")
        result["segmenter_manifest_sha256"] = segmenter_pin or ""
        result["segmenter_model_sha256"] = sha256_file(segmenter_model)
    return result


def _write_json(path: Path, body: Any) -> None:
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _validate_images(images: Iterable[Path]) -> list[Path]:
    checked = []
    for image in images:
        if image.suffix.lower() not in MIME_BY_SUFFIX:
            raise CaptureError(f"unsupported image type for {image.name}; use one of {sorted(MIME_BY_SUFFIX)}")
        if not image.is_file():
            raise CaptureError(f"image {image} is missing")
        checked.append(image)
    return checked


def _check_ready(status: dict[str, Any]) -> None:
    segmenter = status.get("segmenter") or {}
    if not status.get("classification_ready"):
        raise CaptureError(f"classifier not ready: {status.get('reason')}")
    if not segmenter.get("ready"):
        raise CaptureError(f"segmenter not ready: {segmenter.get('reason')}")


def _simulate_image(client: Any, image: Path, mask_b64: str, out_dir: Path, stem: str) -> None:
    for delta in DELTAS:
        response = client.post(
            "/api/simulate-tone",
            files={"file": (image.name, image.read_bytes(), MIME_BY_SUFFIX[image.suffix.lower()])},
            data={"mask_png_base64": mask_b64, "delta_l": str(delta)},
        )
        if response.status_code != 200:
            raise CaptureError(f"simulate-tone delta {delta} failed for {image.name}: {response.status_code}")
        _write_json(out_dir / simulate_name(delta, stem), response.json())


def _capture_one(client: Any, image: Path, out_dir: Path) -> dict[str, Any]:
    stem = image.stem
    response = client.post(
        f"/api/analyze?model={ANALYZE_MODEL}",
        files={"file": (image.name, image.read_bytes(), MIME_BY_SUFFIX[image.suffix.lower()])},
    )
    if response.status_code != 200:
        raise CaptureError(f"analyze failed for {image.name}: {response.status_code}")
    body = response.json()
    _write_json(out_dir / analyze_name(stem), body)
    segmentation = body["segmentation"]
    entry: dict[str, Any] = {
        "name": image.name,
        "stem": stem,
        "sha256": sha256_file(image),
        "analyze": analyze_name(stem),
    }
    if segmentation.get("status") == "ok" and segmentation.get("mask_png_base64"):
        _simulate_image(client, image, segmentation["mask_png_base64"], out_dir, stem)
        entry.update(simulations="captured", simulation_skip_reason=None)
    else:
        entry.update(simulations="skipped", simulation_skip_reason=segmentation.get("reason"))
    return entry


def run_capture(client: Any, images: Iterable[Path], out_dir: Path, *, clock: Callable[[], datetime],
                git: dict[str, Any], pins: dict[str, str]) -> Path:
    checked = _validate_images(images)
    status = client.get("/api/status").json()
    _check_ready(status)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [_capture_one(client, image, out_dir) for image in checked]
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "captured_at": clock().isoformat(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "release": status.get("release"),
        "classifier": {**{k: v for k, v in pins.items() if k.startswith("classifier")},
                       "identity": status.get("identity")},
        "segmenter": {**{k: v for k, v in pins.items() if k.startswith("segmenter")},
                      "identity": (status.get("segmenter") or {}).get("identity")},
        "images": entries,
    }
    path = out_dir / "capture_manifest.json"
    _write_json(path, manifest)
    return path


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if commit.returncode != 0:
        raise CaptureError("cannot read the git commit for provenance")
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True,
                            check=False)
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _default_client(classifier_root: Path, classifier_pin: str, segmenter_dir: Path, segmenter_pin: str) -> Any:
    from fastapi.testclient import TestClient

    from demo.backend.segmenter_service import SegmenterService
    from demo.backend.v10_app import create_app
    from demo.backend.v10_service import MoCoModelService

    moco = MoCoModelService(bundle_dir=classifier_root, expected_manifest_sha256=classifier_pin)
    segmenter = SegmenterService(segmenter_dir=segmenter_dir, expected_manifest_sha256=segmenter_pin)
    return TestClient(create_app(moco_service=moco, segmenter_service=segmenter, report_path=classifier_root / "report.json"))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier-root", type=Path, required=True)
    parser.add_argument("--classifier-manifest-sha256", required=True)
    parser.add_argument("--segmenter-dir", type=Path, required=True)
    parser.add_argument("--segmenter-manifest-sha256", required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, clock: Callable[[], datetime] | None = None,
         git_reader: Callable[[], dict[str, Any]] | None = None,
         client_factory: Callable[..., Any] | None = None) -> int:
    args = _parse(argv)
    pins = check_artifacts(args.classifier_root, args.classifier_manifest_sha256, args.segmenter_dir,
                           args.segmenter_manifest_sha256)
    factory = client_factory or _default_client
    client = factory(args.classifier_root, args.classifier_manifest_sha256, args.segmenter_dir,
                     args.segmenter_manifest_sha256)
    path = run_capture(
        client,
        args.image,
        args.out,
        clock=clock or (lambda: datetime.now(timezone.utc)),
        git=(git_reader or _git_state)(),
        pins=pins,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
