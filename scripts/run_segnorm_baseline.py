"""Run SegRNorm/SegXNorm read-only over an image manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _records(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["records"] if isinstance(payload, dict) else payload
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_image_id(image_id: str) -> None:
    if not image_id or image_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", image_id):
        raise ValueError(f"image_id {image_id!r} must be a safe simple component")


def _normalise_stratification_column(column: str) -> str:
    return {"dataset": "dataset", "split": "split", "quality": "quality"}.get(column, column)


def _resolve_stratification_column(records: list[dict[str, str]], column: str) -> str:
    normalised = _normalise_stratification_column(column)
    if all(normalised in record for record in records):
        return normalised
    aliases = {"dataset": "dataset_id", "dataset_id": "dataset"}
    alias = aliases.get(normalised)
    if alias and all(alias in record for record in records):
        return alias
    return normalised


def _dataset_value(record: dict[str, str]) -> str:
    if "dataset" in record:
        return record["dataset"]
    if "dataset_id" in record:
        return record["dataset_id"]
    raise ValueError("manifest record is missing required dataset field: expected 'dataset' or 'dataset_id'")


def _image_path_value(record: dict[str, str]) -> str:
    if record.get("path"):
        return record["path"]
    if record.get("image_path"):
        return record["image_path"]
    raise ValueError("manifest record is missing required image path field: expected 'path' or 'image_path'")


def _resolve_image_path(record: dict[str, str], path_base: Path) -> Path:
    raw_path = Path(_image_path_value(record)).expanduser()
    image_path = raw_path if raw_path.is_absolute() else path_base / raw_path
    resolved_path = image_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"image path for image_id {record.get('image_id', '')!r} does not exist: {resolved_path}"
        )
    return resolved_path


def _stratified_sample(records: list[dict[str, str]], sample_size: int, seed: int, column: str) -> list[dict[str, str]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError(f"sample_size {sample_size} exceeds available records {len(records)}")
    strata: dict[str, list[dict[str, str]]] = {}
    for record in records:
        value = record.get(column, "")
        if not value:
            raise ValueError(f"stratification column {column!r} has blank or missing values")
        strata.setdefault(value, []).append(record)
    if sample_size < len(strata):
        raise ValueError("sample_size must include at least one image per stratum")

    generator = random.Random(seed)
    for group in strata.values():
        generator.shuffle(group)
    allocations = {value: 1 for value in strata}
    remaining = sample_size - len(strata)
    while remaining:
        eligible = [value for value in sorted(strata) if allocations[value] < len(strata[value])]
        if not eligible:
            break
        total_capacity = sum(len(strata[value]) - allocations[value] for value in eligible)
        for value in eligible:
            if not remaining:
                break
            share = max(1, remaining * (len(strata[value]) - allocations[value]) // total_capacity)
            amount = min(share, len(strata[value]) - allocations[value], remaining)
            allocations[value] += amount
            remaining -= amount
    selected = [record for value in sorted(strata) for record in strata[value][: allocations[value]]]
    generator.shuffle(selected)
    return selected


def _select_records(
    records: list[dict[str, str]],
    dataset_id: str,
    image_ids: list[str],
    sample_size: int | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
) -> list[dict[str, str]]:
    filtered = [record for record in records if not dataset_id or _dataset_value(record) == dataset_id]
    for record in filtered:
        _validate_image_id(record.get("image_id", ""))

    if len(image_ids) != len(set(image_ids)):
        duplicates = sorted({image_id for image_id in image_ids if image_ids.count(image_id) > 1})
        raise ValueError(f"duplicate requested image IDs: {', '.join(duplicates)}")
    for image_id in image_ids:
        _validate_image_id(image_id)

    counts: dict[str, int] = {}
    for record in filtered:
        image_id = record["image_id"]
        counts[image_id] = counts.get(image_id, 0) + 1
    ambiguous = sorted(image_id for image_id, count in counts.items() if count > 1)
    if ambiguous:
        raise ValueError(f"ambiguous image IDs: {', '.join(ambiguous)}")

    if image_ids:
        if sample_size is not None or stratify_by is not None:
            raise ValueError("--image-id cannot be combined with sampling options")
        missing = sorted(image_id for image_id in image_ids if image_id not in counts)
        if missing:
            raise ValueError(f"missing image IDs: {', '.join(missing)}")
        return [record for record in filtered if record["image_id"] in set(image_ids)]
    if sample_size is None:
        if seed is not None or stratify_by is not None:
            raise ValueError("--seed and --stratify-by require --sample-size")
        return filtered
    if seed is None:
        raise ValueError("--sample-size requires an explicit --seed")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size > len(filtered):
        raise ValueError(f"sample_size {sample_size} exceeds available records {len(filtered)}")
    if stratify_by is None:
        selected = filtered.copy()
        random.Random(seed).shuffle(selected)
        return selected[:sample_size]
    column = _resolve_stratification_column(filtered, stratify_by)
    if any(column not in record for record in filtered):
        raise ValueError(f"stratification column {column!r} is not present in every manifest record")
    return _stratified_sample(filtered, sample_size, seed, column)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_provenance(repo_root: Path, source_paths: list[Path]) -> dict[str, object]:
    root = repo_root.resolve()
    files = []
    for source_path in sorted((path.resolve() for path in source_paths), key=lambda path: path.as_posix()):
        relative_path = source_path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative_path,
                "size": source_path.stat().st_size,
                "sha256": _sha256_file(source_path),
            }
        )
    canonical_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "source_tree_hash": hashlib.sha256(canonical_files).hexdigest(),
        "files": files,
    }


def _implementation_source_paths(repo_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        repo_root / "src" / "dermssl" / "segnorm.py",
    ]


def _environment_fingerprint() -> dict[str, object]:
    packages = sorted(
        (
            {"name": str(distribution.metadata["Name"] or distribution.name), "version": str(distribution.version)}
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda package: (package["name"].casefold(), package["version"]),
    )
    package_bytes = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "implementation": platform.python_implementation(),
        },
        "packages": packages,
        "packages_sha256": hashlib.sha256(package_bytes).hexdigest(),
    }


def _code_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_status(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"], cwd=repo_root, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _created_parent_directories(path: Path) -> list[Path]:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
    return missing


def _cleanup_created_directories(directories: list[Path]) -> None:
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            break


def _ensure_inside(path: Path, directory: Path) -> None:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError("output path must remain inside the output directory") from error


def _downscaled(image, max_side: int):
    """Shrink so the longest side is ``max_side`` pixels; native size when zero."""
    from PIL import Image  # local: the script only imports Pillow inside run()

    if max_side <= 0 or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
    return image.resize(size, Image.BILINEAR)


def run(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--path-base",
        help="base directory for relative manifest paths; defaults to the manifest directory",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--method", choices=("rnorm", "xnorm", "both"), default="both")
    parser.add_argument("--adjustment", choices=("minmax", "none"), default="minmax")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--min-component-pixels", type=int, default=64)
    parser.add_argument("--polarity", choices=("bright", "dark"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stratify-by")
    parser.add_argument("--roi", choices=("circular", "full"), default="circular")
    parser.add_argument("--roi-black-threshold", type=float, default=0.02)
    parser.add_argument("--roi-margin", type=int, default=0)
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="downscale the longest image side to this many pixels before segmenting (0 = native)",
    )
    args = parser.parse_args(argv)
    if args.max_side < 0:
        parser.error("--max-side must be zero or positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    import numpy as np
    from PIL import Image
    from dermssl.segnorm import SegNormConfig, segment

    manifest = Path(args.manifest).expanduser().resolve()
    path_base = Path(args.path_base).expanduser().resolve() if args.path_base else manifest.parent
    path_resolution_policy = "explicit_path_base" if args.path_base else "manifest_parent"
    selected = _select_records(
        _records(manifest), args.dataset_id, args.image_id, args.sample_size, args.seed, args.stratify_by
    )
    if not selected:
        raise ValueError("manifest selection produced no images")

    methods = ("rnorm", "xnorm") if args.method == "both" else (args.method,)
    output_names = [f"{record['image_id']}_{method}.png" for record in selected for method in methods]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError("output filename collision detected for selected image IDs")
    created_parent_directories = _created_parent_directories(output.parent)
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        (staging / "masks").mkdir()
        diagnostics = []
        source_metadata = []
        decoded_hashes = {}
        sheet_images: dict[str, list[Image.Image]] = {method: [] for method in methods}
        polarity = args.polarity or "bright"
        config = SegNormConfig(
            roi=args.roi,
            roi_black_threshold=args.roi_black_threshold,
            roi_margin=args.roi_margin,
            adjustment=args.adjustment,
            sigma=args.sigma,
            kernel_size=args.kernel_size,
            min_component_pixels=args.min_component_pixels,
            polarity=polarity,
        )
        for record in selected:
            image_path = _resolve_image_path(record, path_base)
            with Image.open(image_path) as source:
                image = np.asarray(_downscaled(source.convert("RGB"), args.max_side))
            source_metadata.append(
                {
                    "image_id": record["image_id"],
                    "dataset": _dataset_value(record),
                    "stratum": record.get(_resolve_stratification_column(selected, args.stratify_by), "")
                    if args.stratify_by
                    else None,
                    "path": str(image_path.resolve()),
                    "file_sha256": _sha256_file(image_path),
                }
            )
            decoded_hashes[record["image_id"]] = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            for method in methods:
                result = segment(image, method, config)
                output_name = f"{record['image_id']}_{method}.png"
                mask_path = staging / "masks" / output_name
                _ensure_inside(mask_path, staging)
                Image.fromarray((result.mask.astype(np.uint8) * 255), mode="L").save(mask_path)
                if args.contact_sheet:
                    sheet_images[method].append(Image.fromarray(result.mask.astype(np.uint8) * 255, mode="L").resize((128, 128)))
                diagnostic = {
                    "dataset": _dataset_value(record),
                    "image_id": record["image_id"],
                    **result.diagnostics,
                    "mask_path": f"masks/{output_name}",
                    "max_side": args.max_side,
                }
                if "dataset_id" in record:
                    diagnostic["dataset_id"] = record["dataset_id"]
                diagnostics.append(diagnostic)
        if args.contact_sheet:
            for method, images in sheet_images.items():
                sheet = Image.new("L", (128 * len(images), 128), color=0)
                for index, image in enumerate(images):
                    sheet.paste(image, (128 * index, 0))
                    image.close()
                sheet_path = staging / f"contact_sheet_{method}.png"
                _ensure_inside(sheet_path, staging)
                sheet.save(sheet_path)
        diagnostics_path = staging / "diagnostics.json"
        _ensure_inside(diagnostics_path, staging)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        repo_root = Path(__file__).resolve().parents[1]
        source_tree = _source_tree_provenance(repo_root, _implementation_source_paths(repo_root))
        environment = _environment_fingerprint()
        code_revision = _code_revision(repo_root)
        provenance = {
            "manifest": {"path": str(manifest.resolve()), "sha256": _sha256_file(manifest)},
            "path_resolution": {
                "policy": path_resolution_policy,
                "base": str(path_base),
            },
            "selected_image_ids": [record["image_id"] for record in selected],
            "sampling": {
                "sample_size": args.sample_size,
                "seed": args.seed,
                "stratify_by": _resolve_stratification_column(selected, args.stratify_by) if args.stratify_by else None,
            },
            "sources": source_metadata,
            "decoded_image_hashes": decoded_hashes,
            "cli": {"argv": list(argv) if argv is not None else sys.argv[1:]},
            "configuration": {
                "method": args.method,
                "adjustment": args.adjustment,
                "sigma": args.sigma,
                "kernel_size": args.kernel_size,
                "min_component_pixels": args.min_component_pixels,
                "roi": args.roi,
                "roi_black_threshold": args.roi_black_threshold,
                "roi_margin": args.roi_margin,
            },
            "polarity": {
                "value": polarity,
                "status": "explicit" if args.polarity else "implementation_assumption",
                "rationale": "bright is retained as the implementation default for backward-compatible diagnostics; the paper does not disambiguate lesion polarity",
                "paper_ambiguity": "paper description does not define whether lesions are brighter or darker than the normalized background",
            },
            "roi": {
                "method": args.roi,
                "black_threshold": args.roi_black_threshold,
                "margin": args.roi_margin,
                "assumption": "circular ROI detects non-black field pixels and derives a centered circle from their bounding box; full mode intentionally preserves the comparison baseline",
            },
            "method": list(methods),
            "paper_variant": "diagnostic_baseline_not_exact_reproduction",
            "code_revision": code_revision,
            "git": {
                "head": code_revision,
                "status": _git_status(repo_root),
            },
            "python": sys.version,
            "packages": _package_versions(),
            "source_tree_hash": source_tree["source_tree_hash"],
            "source_files": source_tree["files"],
            "source_tree": source_tree,
            "environment": environment,
        }
        provenance_path = staging / "provenance.json"
        _ensure_inside(provenance_path, staging)
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return diagnostics
    except Exception:
        if staging is not None and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
        _cleanup_created_directories(created_parent_directories)
        raise


if __name__ == "__main__":
    run()
