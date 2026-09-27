"""Gate: condition (0) of C1 must reproduce the bank's own extraction pipeline.

Condition (0) is, by definition, the bank's pipeline with no mask applied. If
the C1 extraction path drifts in any way -- interpolation mode, resize edge,
crop offset, extraction point -- (0) stops landing on it, and all 24 cells are
then compared against a baseline that is not the published one.

Two comparisons, and they are NOT interchangeable:

* **Primary gate (decides).** C1's (0) against the bank pipeline RECOMPUTED in
  this process, at this precision, with these weights. Everything except C1's
  preprocessing is held fixed, so the bound can be tight (1e-5).
* **Secondary check (reports).** That recomputation against the stored cache.
  The cache was written under ``torch.amp.autocast("cuda")``, so a residue is
  expected: measured on the pod, the bank does not reproduce its own cache
  below 1.64e-02 even though the cosine stays at 1.000. Bounded loosely by
  magnitude and tightly by per-row cosine, and its failure is documented
  rather than decisive.

This split exists because the first version gated C1 at 1e-4 against the
stored cache. That bound came from the demo's PROBABILITIES re-extracted on
another device (7.85e-05) -- a different quantity measured a different way --
so the gate could not pass even with C1 perfect.

The nearest-vs-bilinear defect found in review, and the one-row centre-crop
offset found by running it, both fail here on the first run; no downstream
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
from dermssl.pretrain import tracking  # noqa: E402

# torch_c1_extraction imports torch at module level, so it is imported lazily
# inside extract_baseline. Two reasons: the W&B check has to be able to abort
# before anything heavy loads, and the pure parts of this file (the tracking
# config, the report logging) stay testable on a machine with no GPU stack.

DEFAULT_TOLERANCE = pv.DEFAULT_PRIMARY_TOLERANCE
BASELINE_CONDITION = "none"


def tracking_config(
    *,
    backbone: str,
    source: str,
    tolerance: float,
    code_revision: str,
    masks_sha256: str,
    n_images: int,
) -> dict:
    """Everything needed to tell one run of this gate from another.

    ``condition`` is in here so the zero fallback count is readable: without
    it, "0 CLS fallbacks" on a dashboard looks like a measurement of condition
    (c) rather than a structural property of (0), which never pools patches.
    """
    return {
        "gate": "bank_reproduction",
        "condition": BASELINE_CONDITION,
        "backbone": backbone,
        "source": source,
        "tolerance": tolerance,
        "code_revision": code_revision,
        "masks_sha256": masks_sha256,
        "n_images": n_images,
    }


def log_report(tracker, report: dict) -> None:
    """Log the verdict whether it passed or failed.

    A gate that only records its successes turns the dashboard into evidence
    for one side. ``reproduces`` goes in as 1,0/0,0 rather than a bool so it
    charts alongside the drift.
    """
    tracker.log_metric("max_abs_diff", float(report["max_abs_diff"]))
    tracker.log_metric("mean_abs_diff", float(report["mean_abs_diff"]))
    tracker.log_metric("n_rows_over_tolerance", float(report["n_rows_over_tolerance"]))
    tracker.log_metric("n_rows", float(report["n_rows"]))
    tracker.log_metric("reproduces", 1.0 if report["reproduces"] else 0.0)

    # The secondary check goes in under its own prefix. Same dashboard, clearly
    # separate numbers: reading "max_abs_diff 1.6e-02" without knowing which of
    # the two comparisons produced it is how the tolerance got miscalibrated in
    # the first place.
    cache = report.get("published_cache_check")
    if cache is not None:
        tracker.log_metric("cache_max_abs_diff", float(cache["max_abs_diff"]))
        tracker.log_metric("cache_min_cosine", float(cache["min_cosine"]))
        tracker.log_metric("cache_n_rows_below_cosine", float(cache["n_rows_below_cosine"]))
        tracker.log_metric("cache_agrees", 1.0 if cache["agrees"] else 0.0)


def build_gate_tracker(
    wandb_project: str,
    run: str,
    group: str,
    environ: dict[str, str] | None = None,
    importer=None,
):
    """A strict W&B tracker, or ``None`` when tracking was not requested.

    Strict because this gate's entire output is the verdict, and the verdict is
    reported as a link. Fail-safe is right for a 20-hour pretrain, where losing
    the sink is cheaper than losing the run; here losing the sink loses the
    result. Raises before anything expensive happens.
    """
    # ``importer`` is injected rather than defaulted at call time because
    # require_wandb captures its own default when it is DEFINED: patching
    # tracking.importlib afterwards does nothing, which made an earlier version
    # of the "missing package" test pass only because wandb happened to be
    # absent from this interpreter.
    extra = {} if importer is None else {"importer": importer}
    module = tracking.require_wandb(wandb_project, environ=environ, **extra)
    if module is None:
        return None
    return tracking.WandbTracker(
        module, project=wandb_project, run=run, group=group, strict=True
    )


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
    parser.add_argument(
        "--wandb-project",
        default="",
        help="proyecto de Weights & Biases; si se pide y no se puede registrar, el gate aborta",
    )
    parser.add_argument("--wandb-group", default="v7-c1", help="grupo del run en W&B")
    return parser.parse_args()


def code_revision(repo: Path) -> str:
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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


def extract_through_bank_pipeline(model, image_dir, image_ids, device) -> np.ndarray:
    """The BANK's own path -- its transform, its forward -- in THIS process.

    This is what the primary gate compares C1 against, and the reason the gate
    can be tight. The stored cache was written under
    ``torch.amp.autocast("cuda")`` on another day, so comparing against it
    measures the precision of two different runs on top of whatever C1 does;
    measured on the pod, the bank does not reproduce its own cache below
    1.64e-02. Recomputing here holds precision, device and library versions
    fixed, so any difference left is C1's preprocessing -- the thing under
    test.

    Uses ``bank.eval_transform``, not a reimplementation of it. Rebuilding the
    reference is how the one-row centre-crop offset got in.
    """
    from PIL import Image  # noqa: PLC0415

    import torch  # noqa: PLC0415

    import cluster_frozen_bank as bank  # noqa: PLC0415

    transform = bank.eval_transform(ex.RESOLUTION)
    rows = []
    for image_id in image_ids:
        path = image_dir / f"{image_id}.jpg"
        if not path.exists():
            raise SystemExit(f"ABORT: no image for {image_id} at {path}")
        tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            rows.append(model(tensor).float().cpu().numpy()[0])
    return np.stack(rows)


def extract_baseline(
    model, image_dir, image_ids, masks, mask_source, device
) -> tuple[np.ndarray, pv.FallbackTally]:
    """Condition (0) through the C1 code path, one image at a time.

    Returns the fallback tally alongside the features: how often the mask
    selected no patch is a result in itself, not noise to discard.
    """
    from PIL import Image  # noqa: PLC0415

    from dermssl import torch_c1_extraction as tex  # noqa: PLC0415

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

    # Before torch, before the backbone, before anything that costs time: if
    # tracking was requested and cannot be delivered, this gate has no output
    # worth producing, because its output is the link.
    tracker = build_gate_tracker(
        args.wandb_project, f"bank-gate-{args.backbone}", args.wandb_group
    )

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

    if tracker is not None:
        tracker.start(
            tracking_config(
                backbone=args.backbone,
                source=args.source,
                tolerance=args.tolerance,
                code_revision=code_revision(REPO),
                masks_sha256=mask_source.masks_sha256,
                n_images=len(val_ids),
            )
        )

    model = build_backbone(args.arch, args.source, device)
    c1_features, tally = extract_baseline(
        model, args.image_dir.expanduser(), val_ids, masks, mask_source, device
    )
    recomputed = extract_through_bank_pipeline(
        model, args.image_dir.expanduser(), val_ids, device
    )

    # PRIMARY GATE: C1 against the bank pipeline recomputed right here. Same
    # process, same precision, same weights -- so what is left is C1.
    report = pv.compare_to_bank(recomputed, c1_features, tolerance=args.tolerance)
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

    # SECONDARY CHECK: does the recomputation still match the stored cache?
    # Reported, never decisive. Its failure says the cache came from another
    # precision or device -- worth writing down, but it is not evidence about
    # C1, and letting it veto is how a gate ends up unfailable for a reason
    # nobody involved can act on.
    cache_check = pv.compare_to_published_cache(recomputed, bank_features)
    report["published_cache_check"] = cache_check

    if args.report:
        args.report.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")

    if tracker is not None:
        # Logged on BOTH outcomes. A dashboard that only holds the runs that
        # passed is evidence for one side, and a failing gate is exactly the
        # result someone needs to be able to look up afterwards.
        log_report(tracker, report)
        tracker.end()
        if tracker.run_url:
            print(f"\nW&B run: {tracker.run_url}")

    print("\n--- GATE PRINCIPAL: C1 (0) contra el banco recalculado aqui mismo ---")
    print(f"max_abs_diff  : {report['max_abs_diff']:.3e}  (tolerancia {args.tolerance:g})")
    print(f"mean_abs_diff : {report['mean_abs_diff']:.3e}")
    print(f"rows over tol : {report['n_rows_over_tolerance']}/{report['n_rows']}")
    print("cls fallbacks : n/a (this gate runs condition 'none')")

    print("\n--- CHEQUEO SECUNDARIO: recalculado contra el cache publicado ---")
    print(
        f"max_abs_diff  : {cache_check['max_abs_diff']:.3e}  "
        f"(margen {cache_check['max_abs_diff_bound']:g})"
    )
    print(
        f"min_cosine    : {cache_check['min_cosine']:.6f}  "
        f"(piso {cache_check['min_cosine_bound']:g})"
    )
    print(f"veredicto     : {'coincide' if cache_check['agrees'] else 'NO coincide'}")
    if not cache_check["agrees"]:
        # Printed, not returned as failure: this does not gate C1.
        print("  El cache viene de otra precision o dispositivo. Queda documentado:")
        for reason in cache_check["reasons"]:
            print(f"  - {reason}")

    if not report["reproduces"]:
        print("\n=== BANK REPRODUCTION FAILED — DO NOT RUN C1 ===")
        print("Condition (0) does not reproduce the bank pipeline. Check, in order:")
        print("  1. image resize is BILINEAR (masks nearest) and the edge is 255, not 256")
        print("  2. CenterCrop(224) offset is int(round(margin/2)), as torchvision does")
        print("  3. extraction reads the pooled output, not patch tokens, for (0)")
        print("  4. both paths ran at the same precision (autocast on one side only)")
        return 1

    print("\n=== BANK REPRODUCTION OK — C1 may run ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
