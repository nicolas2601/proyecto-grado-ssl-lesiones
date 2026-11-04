"""Before/after contact sheet for dermssl.tone_augment_lab_v2, stratified by realized delta.

Visual plausibility is the only thing blocking a policy cap on the
transform's magnitude (see src/dermssl/tone_augment_lab_v2.py, "INTEGRATION
REQUIREMENT"). Nobody has looked at what the transform actually does at the
extremes of its physically reachable range. This script does not decide
plausibility -- a human has to look at the panel grid -- it only builds it.

Stratified by REALIZED delta (``record["ita_delta_raw"]``), not by melanin or
dataset group: the two absolute extremes of darkening and lightening are
where an image is most likely to stop looking like skin, and that is
precisely the information melanin strata would average away.

Input is a manifest (JSONL, one object per line) with at least
``image_path`` and ``mask_path`` (paths resolved relative to the manifest's
own directory if not absolute). ``mask_path`` points to a PNG where any
nonzero pixel is lesion (``True``). This script does not know how to find
S-SYNTH or any other dataset's images on disk -- that indexing is a separate,
already-existing concern (see scripts/ssynth_stage_v1.py) and does not belong
here twice.

Usage::

    PYTHONPATH=src python scripts/make_tone_contact_sheet.py \\
        --manifest /path/to/manifest.jsonl \\
        --n-panels 20 \\
        --seed 42 \\
        [--policy-cap-degrees 25.0]

Writes a labeled PNG grid plus the JSON record of every selected panel (for
traceability: which image, which step, which realized numbers) under
``resultados/segmentacion/v8/tone_contact_sheet/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "resultados" / "segmentacion" / "v8" / "tone_contact_sheet"

import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from dermssl.fitzpatrick import estimate_fst  # noqa: E402
from dermssl.tone_augment_lab_v2 import augment_skin_tone_v2  # noqa: E402

# How many (image, step) attempts to make before giving up on filling the
# candidate pool. Most images abstain (mix gate, ita_indeterminate) by
# design, so this needs to be generous relative to --n-panels.
MAX_STEPS_PER_IMAGE = 8
MIN_POOL_SIZE_BEFORE_SELECTING = 60


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    base = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            row["image_path"] = str(image_path if image_path.is_absolute() else base / image_path)
            row["mask_path"] = str(mask_path if mask_path.is_absolute() else base / mask_path)
            rows.append(row)
    return rows


def build_candidate_pool(
    manifest_rows: list[dict],
    *,
    seed: int,
    policy_cap_degrees: float | None,
    min_pool_size: int,
) -> list[dict]:
    """Every (image, step) combination that actually gets transformed.

    One pool entry per applied case, carrying everything a panel needs to
    render and caption itself: the two images, the measured-before and
    measured-after ITA, and the realized delta.
    """
    pool = []
    for row in manifest_rows:
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        if not image_path.exists() or not mask_path.exists():
            continue
        img = _load_image(image_path)
        mask = _load_mask(mask_path)
        image_sha256 = _sha256_of_file(image_path)

        for step in range(MAX_STEPS_PER_IMAGE):
            out, record = augment_skin_tone_v2(
                img, mask, seed, step, image_sha256, policy_cap_degrees=policy_cap_degrees
            )
            if not record["applied"]:
                continue
            final = estimate_fst(out, mask=mask, border_mask=None)
            pool.append(
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "seed": seed,
                    "step": step,
                    "image_sha256": image_sha256,
                    "ita_before": record["ita_measured"],
                    "ita_objetivo": record["ita_objetivo"],
                    "ita_delta_realized": record["ita_delta_raw"],
                    "ita_after": final.ita if final.status == "ok" else None,
                    "ita_after_status": final.status,
                    "before_rgb": img,
                    "after_rgb": out,
                }
            )
            break  # one applied case per image is enough signal for the pool
        if len(pool) >= min_pool_size:
            break
    return pool


def select_stratified_panels(pool: list[dict], n_panels: int) -> list[dict]:
    """Percentiles of realized delta, forcing in the two absolute extremes."""
    if not pool:
        return []
    ordered = sorted(pool, key=lambda row: row["ita_delta_realized"])
    n = len(ordered)
    percentiles = np.linspace(0.0, 1.0, n_panels)
    indices = sorted({int(round(p * (n - 1))) for p in percentiles})
    indices[0] = 0
    indices[-1] = n - 1
    return [ordered[i] for i in indices]


def render_contact_sheet(panels: list[dict], output_path: Path) -> None:
    n = len(panels)
    cols = 2
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(6, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row_index, panel in enumerate(panels):
        ax_before, ax_after = axes[row_index]
        ax_before.imshow(np.clip(panel["before_rgb"], 0, 1))
        ax_before.set_title(f"antes  ITA={panel['ita_before']:.2f}", fontsize=8)
        ax_before.axis("off")
        after_label = (
            f"ITA={panel['ita_after']:.2f}" if panel["ita_after"] is not None else "ITA no medible"
        )
        ax_after.imshow(np.clip(panel["after_rgb"], 0, 1))
        ax_after.set_title(
            f"despues  {after_label}  delta={panel['ita_delta_realized']:+.2f}", fontsize=8
        )
        ax_after.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n-panels", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-cap-degrees", type=float, default=None)
    parser.add_argument("--min-pool-size", type=int, default=MIN_POOL_SIZE_BEFORE_SELECTING)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    manifest_rows = _read_manifest(args.manifest)
    pool = build_candidate_pool(
        manifest_rows,
        seed=args.seed,
        policy_cap_degrees=args.policy_cap_degrees,
        min_pool_size=args.min_pool_size,
    )
    if not pool:
        raise SystemExit("no applied cases found in the manifest; nothing to render")

    panels = select_stratified_panels(pool, args.n_panels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(panels, args.output_dir / "contact_sheet.png")

    record = [
        {k: v for k, v in panel.items() if k not in ("before_rgb", "after_rgb")}
        for panel in panels
    ]
    (args.output_dir / "contact_sheet_panels.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"pool size: {len(pool)}")
    print(f"panels written: {len(panels)} -> {args.output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
