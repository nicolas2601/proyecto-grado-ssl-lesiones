"""Figure A pod job: CIELAB histograms of the pixels the E5 perilesional ring actually reads.

``run`` re-executes the unchanged E5 runner (``scripts/e5_fitzpatrick.py``) with ``e5_roi`` instrumented:
every image whose ring estimate ends with status ``ok`` adds its valid ring pixels (the very array the
estimator reads) to per-process histograms, flushed to ``<out>/parts/part_<pid>.npz`` after each image.
Border-fallback estimates (``no_mask_border``) are NOT ring pixels and are ignored.
``aggregate`` sums the parts and writes the JSON contract read by ``scripts/figures/fig_a_cielab_ring.py``.

    python scripts/figA_ring_hist.py run --out OUT -- <the same arguments given to e5_fitzpatrick.py> --out-dir OUT/e5
    python scripts/figA_ring_hist.py aggregate --out OUT --dataset ham10000 --e5-json OUT/e5/e5_fitzpatrick_ham10000.json \\
        --code-revision <sha> --masks-sha256 <sha> --ids-sha256 <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

L_EDGES = np.arange(0, 101, 2)
AB_EDGES = np.arange(-128, 129, 4)
OUT_ENV = "FIGA_OUT"
CANONICAL = "lado corto 450 (PIL draft + resize bilineal), RGB float32 [0,1], sin constancia de color (entrada canónica de E5)"
RING_DEFINITION = ("anillo fuera de la máscara de lesión (dilatación de Chebyshev del 25 % del diámetro equivalente, mínimo 3 px, e3_geometry.ring_mask) "
                   "sin píxeles saturados, especulares, de borde negro ni de pelo; solo imágenes con estado ok del estimador roi")

_STATE = {"context": None, "L": None, "ab": None, "n_images": 0, "n_pixels": 0, "out_of_range": 0}


def _reset() -> None:
    _STATE.update(L=np.zeros(len(L_EDGES) - 1, dtype=np.int64), ab=np.zeros((len(AB_EDGES) - 1, len(AB_EDGES) - 1), dtype=np.int64),
                  n_images=0, n_pixels=0, out_of_range=0)


def accumulate(lab_pixels: np.ndarray) -> None:
    """Add one image's valid ring pixels (N x 3 CIELAB) to this process's histograms."""
    if _STATE["L"] is None:
        _reset()
    L, a, b = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]
    inside = (L >= L_EDGES[0]) & (L <= L_EDGES[-1]) & (a >= AB_EDGES[0]) & (a < AB_EDGES[-1]) & (b >= AB_EDGES[0]) & (b < AB_EDGES[-1])
    kept = lab_pixels[inside]
    # L == 100 is the last closed bin edge of np.histogram; a/b use half-open bins, so both histograms hold the same pixels.
    _STATE["L"] += np.histogram(kept[:, 0], bins=L_EDGES)[0]
    _STATE["ab"] += np.histogram2d(kept[:, 1], kept[:, 2], bins=(AB_EDGES, AB_EDGES))[0].astype(np.int64)
    _STATE["n_images"] += 1
    _STATE["n_pixels"] += int(inside.sum())
    _STATE["out_of_range"] += int((~inside).sum())


def _flush() -> None:
    out = os.environ.get(OUT_ENV)
    if not out:
        return
    parts = Path(out) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    tmp = parts / f"part_{os.getpid()}.tmp.npz"
    np.savez(tmp, L=_STATE["L"], ab=_STATE["ab"], n_images=_STATE["n_images"], n_pixels=_STATE["n_pixels"], out_of_range=_STATE["out_of_range"])
    os.replace(tmp, parts / f"part_{os.getpid()}.npz")


def install_patches() -> None:
    """Wrap e5_roi so the estimator's own valid-ring pixels are accumulated; estimates are returned unchanged."""
    from dermssl import e5_roi  # noqa: PLC0415

    original_ring, original_border, original_read = e5_roi.ring_estimate, e5_roi.border_estimate, e5_roi._read_skin  # noqa: SLF001

    def ring_estimate(*args, **kwargs):
        _STATE["context"] = "ring"
        try:
            return original_ring(*args, **kwargs)
        finally:
            _STATE["context"] = None

    def border_estimate(*args, **kwargs):
        _STATE["context"] = "border"
        try:
            return original_border(*args, **kwargs)
        finally:
            _STATE["context"] = None

    def read_skin(lab, valid, region_px):
        estimate = original_read(lab, valid, region_px)
        if _STATE["context"] == "ring" and estimate.status == e5_roi.OK:
            accumulate(lab[valid])
            _flush()
        return estimate

    e5_roi.ring_estimate, e5_roi.border_estimate, e5_roi._read_skin = ring_estimate, border_estimate, read_skin  # noqa: SLF001


def run(args, passthrough: list[str]) -> int:
    out = Path(args.out).expanduser()
    (out / "parts").mkdir(parents=True, exist_ok=True)
    if any((out / "parts").iterdir()):
        raise SystemExit(f"{out}/parts is not empty; use a fresh --out so stale parts cannot be summed")
    os.environ[OUT_ENV] = str(out)
    install_patches()
    import e5_fitzpatrick  # noqa: PLC0415

    return e5_fitzpatrick.main(passthrough)


def aggregate(args) -> int:
    out = Path(args.out).expanduser()
    files = sorted((out / "parts").glob("part_*.npz"))
    if not files:
        raise SystemExit("no part files found")
    L = np.zeros(len(L_EDGES) - 1, dtype=np.int64)
    ab = np.zeros((len(AB_EDGES) - 1, len(AB_EDGES) - 1), dtype=np.int64)
    n_images = n_pixels = oor = 0
    for path in files:
        with np.load(path) as part:
            L += part["L"]
            ab += part["ab"]
            n_images += int(part["n_images"])
            n_pixels += int(part["n_pixels"])
            oor += int(part["out_of_range"])
    e5 = json.loads(Path(args.e5_json).read_text(encoding="utf-8"))
    table = e5["per_image"]
    index = {name: position for position, name in enumerate(table["columns"])}
    ring_ok = sum(1 for row in table["rows"]
                  if row[index["roi_status"]] == "ok" and row[index["roi_estimator"]] == "perilesional_roi")
    if int(L.sum()) != n_pixels or int(ab.sum()) != n_pixels:
        raise SystemExit(f"invariant broken: sum(L)={L.sum()} sum(ab)={ab.sum()} n_pixels={n_pixels}")
    doc = {"dataset": args.dataset, "n_images": n_images, "n_pixels": n_pixels, "out_of_range_pixels": oor,
           "L_edges": L_EDGES.tolist(), "L_counts": L.tolist(), "a_edges": AB_EDGES.tolist(), "b_edges": AB_EDGES.tolist(),
           "ab_counts": ab.tolist(), "code_revision": args.code_revision, "masks_sha256": args.masks_sha256, "ids_sha256": args.ids_sha256,
           "canonical_input": CANONICAL, "ring_definition": RING_DEFINITION, "n_parts": len(files),
           "e5_ring_ok_images_in_rerun": ring_ok, "e5_rerun_json_sha256": hashlib.sha256(Path(args.e5_json).read_bytes()).hexdigest()}
    target = Path(args.output) if args.output else out / f"cielab_ring_hist_{args.dataset}.json"
    target.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    print(f"wrote {target}: images {n_images} (E5 rerun ring-ok {ring_ok}), pixels {n_pixels}, out of range {oor}")
    return 0 if n_images == ring_ok else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--out", required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--out", required=True)
    agg.add_argument("--dataset", required=True)
    agg.add_argument("--e5-json", required=True)
    agg.add_argument("--code-revision", required=True)
    agg.add_argument("--masks-sha256", required=True)
    agg.add_argument("--ids-sha256", required=True)
    agg.add_argument("--output", default=None)
    args, passthrough = parser.parse_known_args(argv)
    passthrough = [item for item in passthrough if item != "--"]
    return run(args, passthrough) if args.command == "run" else aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
