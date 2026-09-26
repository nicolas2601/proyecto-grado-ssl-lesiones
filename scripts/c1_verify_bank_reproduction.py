"""Gate: condition (0) of C1 must reproduce the published bank features.

Condition (0) is, by definition, the bank's own pipeline with no mask applied.
So the cached features from the bank are the ground truth for it: if the C1
extraction path drifts in any way -- interpolation mode, resize edge, crop
offset, extraction point -- (0) stops landing on the cache, and every one of
the 24 cells is then being compared against a baseline that is not the
published one.

This attacks that at the root instead of by symptom. The nearest-vs-bilinear
defect found in review would have failed here on the first run; no downstream
metric would have looked wrong.

Run this BEFORE the 24 cells. A non-zero exit means C1 must not run.

Usage
-----
    python scripts/c1_verify_bank_reproduction.py \\
        --backbone dinov2_public_vitb14 \\
        --arch dinov2_vitb14 \\
        --source public \\
        --cache-dir ~/work/proyectonp/salidas/tone/bank/cache \\
        --image-dir ~/work/proyectonp/eval_ham10000_images \\
        --val-index ~/work/proyectonp/salidas/tone/bank_repro/predictions/val_index.json \\
        --masks ~/work/proyectonp/salidas/c1/masks/c1_pseudo_masks.npy \\
        --masks-manifest ~/work/proyectonp/salidas/c1/masks/c1_pseudo_masks_manifest.json \\
        --limit 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import c1_extraction as ex  # noqa: E402
from dermssl import c1_mask_export as mx  # noqa: E402
from dermssl import c1_mask_provenance as mp
from dermssl import c1_provenance as pv  # noqa: E402
from dermssl import cli_args  # noqa: E402
from dermssl import torch_c1_extraction as tex  # noqa: E402

DEFAULT_TOLERANCE = 1e-4
BASELINE_CONDITION = "none"


def _tolerance(raw: str) -> float:
    """argparse-level guard so a bypass value dies before anything runs."""
    value = float(raw)
    if not math.isfinite(value) or value <= 0 or value > pv.MAX_TOLERANCE:
        raise argparse.ArgumentTypeError(
            f"tolerance must be finite and within (0, {pv.MAX_TOLERANCE:g}]; got {raw!r}"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backbone", required=True, help="bank row name, e.g. dinov2_public_vitb14")
    parser.add_argument("--arch", required=True, help="hub arch, e.g. dinov2_vitb14")
    parser.add_argument("--source", required=True, help="'public' or a path to a .pt checkpoint")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--masks-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=cli_args.positive_int, default=64, help="how many val images to check")
    parser.add_argument(
        "--tolerance",
        type=_tolerance,
        default=DEFAULT_TOLERANCE,
        help=f"absolute bound on feature drift; 0 < tol <= {pv.MAX_TOLERANCE:g}",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    return parser.parse_args()


def load_bank_features(cache_dir: Path, backbone: str) -> np.ndarray:
    hits = sorted(cache_dir.glob(f"v1__{backbone}__val__r224__eval__*.features.npy"))
    if len(hits) != 1:
        raise SystemExit(
            f"ABORT: expected exactly 1 cached val feature file for {backbone}, found {len(hits)}"
        )
    return np.load(hits[0])


def build_backbone(arch: str, source: str, device):
    import cluster_frozen_bank as bank  # noqa: PLC0415

    model = bank.construct_backbone(arch, source, ex.RESOLUTION, device)
    model.eval()
    return model.to(device)


def extract_baseline(
    model, image_dir, image_ids, masks, mask_source, device
) -> tuple[np.ndarray, pv.FallbackTally]:
    """Condition (0) through the C1 code path, one image at a time.

    Returns the fallback tally alongside the features: how often the mask
    selected no patch is a result in itself, not noise to discard.
    """
    from PIL import Image  # noqa: PLC0415

    rows = []
    tally = pv.FallbackTally()
    for index, image_id in enumerate(image_ids):
        path = image_dir / f"{image_id}.jpg"
        if not path.exists():
            raise SystemExit(f"ABORT: no image for {image_id} at {path}")

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        tracked = mp.TrackedMask(
            array=masks[index].astype(bool),
            source=mask_source,
            image_id=image_id,
        )
        prepared = ex.build_eval_input(image, tracked, BASELINE_CONDITION)
        tensor = tex.to_normalized_tensor(prepared.image).to(device)
        feature, used_fallback = tex.extract_feature(
            model, tensor, BASELINE_CONDITION, prepared.mask
        )
        tally.record(used_fallback)
        rows.append(feature)
    return np.stack(rows), tally


def main() -> int:
    args = parse_args()
    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    manifest = json.loads(args.masks_manifest.expanduser().read_text(encoding="utf-8"))
    mask_source = mp.MaskSource.from_manifest(manifest)

    # The manifest is only worth anything if the file on disk is the one it
    # describes. Checking here means a regenerated .npy stops the gate instead
    # of quietly changing what C1 was verified against.
    mx.verify_artifact(args.masks.expanduser(), mask_source.masks_sha256)

    index = json.loads(args.val_index.expanduser().read_text(encoding="utf-8"))
    val_ids = list(index["val_ids"])[: args.limit]

    bank_features = load_bank_features(args.cache_dir.expanduser(), args.backbone)[: args.limit]
    if len(bank_features) != len(val_ids):
        raise SystemExit(
            f"ABORT: {len(bank_features)} cached rows against {len(val_ids)} val ids"
        )

    # Mask rows are indexed by the step-0 manifest order, not by the val order.
    position = {image_id: i for i, image_id in enumerate(mask_source.image_ids)}
    missing = [i for i in val_ids if i not in position]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} val images absent from the mask artifact")

    all_masks = np.load(args.masks.expanduser(), mmap_mode="r")
    masks = np.stack([np.asarray(all_masks[position[i]]) == 1 for i in val_ids])

    print(f"backbone : {args.backbone} ({args.arch})")
    print(f"device   : {device}")
    print(f"images   : {len(val_ids)}")
    print(f"tolerance: {args.tolerance:g}")

    model = build_backbone(args.arch, args.source, device)
    c1_features, tally = extract_baseline(
        model, args.image_dir.expanduser(), val_ids, masks, mask_source, device
    )

    report = pv.compare_to_bank(bank_features, c1_features, tolerance=args.tolerance)
    report["backbone"] = args.backbone
    report["condition"] = BASELINE_CONDITION
    report["image_ids"] = val_ids
    report["masks_sha256"] = mask_source.masks_sha256
    # The gate runs condition "none", which never pools patch tokens, so this
    # tally is structurally zero. Reporting it as a measurement would be a number
    # that looks earned and is not; the real count belongs to the 6x4 runner,
    # over mask_pool.
    assert tally.n_cls_fallback == 0, "condition 'none' cannot fall back to CLS"
    report["fallbacks"] = "not_applicable: gate runs condition 'none', which never pools"

    if args.report:
        args.report.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\nmax_abs_diff  : {report['max_abs_diff']:.3e}")
    print(f"mean_abs_diff : {report['mean_abs_diff']:.3e}")
    print(f"rows over tol : {report['n_rows_over_tolerance']}/{report['n_rows']}")
    print("cls fallbacks : n/a (this gate runs condition 'none')")

    if not report["reproduces"]:
        print("\n=== BANK REPRODUCTION FAILED — DO NOT RUN C1 ===")
        print("Condition (0) does not reproduce the published bank. Check, in order:")
        print("  1. image resize is BILINEAR (masks nearest) and the edge is 255, not 256")
        print("  2. CenterCrop(224) applied after the resize, same offsets as the bank")
        print("  3. extraction reads the pooled output, not patch tokens, for (0)")
        print("  4. the cached file belongs to this backbone and to the val split")
        return 1

    print("\n=== BANK REPRODUCTION OK — C1 may run ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
