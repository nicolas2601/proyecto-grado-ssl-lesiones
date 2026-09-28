r"""S4-UNet A1 (docs/27): S-diag saliency of DINOv2 over the student_r3 pool.

Registered on 2026-09-28: DINOv2 replaces MoCo->CCAM as the saliency source.
Per cached image (224 px, ImageNet-normalised, fp32, the frozen public
ViT-B/14), the last-layer patch tokens go through the S-diag primary rule of
``dermssl.s_diag`` (L2 per token, per-image PCA, sign so the central block
is positive), NOT binarised. The 16x16 map is repeated to 224x224 (nearest
neighbour, as S-diag resizes) and written as ``saliency.npy`` (N, 224, 224,
float16), the layout ``ccam/saliency.npy`` had. Then the SAME uncertainty
module that made r0 turns it into pseudo-masks, with r0's exact flags, via
``make_pseudo_masks.py --source dino`` (min-max per image, no second sign
flip). Nothing is trained or fitted across images; no mask is read.

CPU, one process per shard (``--workers``, OMP/MKL threads = 1 each). Pod::

    P=$HOME/work/proyectonp; source $P/.venv/bin/activate
    export TORCH_HOME=$HOME/work/.torch; cd $P/repo_rel
    F=$P/salidas/final; S4=$P/salidas/s4_unet
    # 1. measured CPU cost first (no output written)
    python scripts/s4_dino_saliency.py --cache-dir $F/cache \
      --output-dir $S4/a1_saliency --workers 32 --timing-projection 640
    # 2. the saliency (setsid: survives the Jupyter terminal)
    setsid nohup python scripts/s4_dino_saliency.py --cache-dir $F/cache \
      --output-dir $S4/a1_saliency --workers 32 < /dev/null > $S4/a1_saliency.log 2>&1 &
    # 3. pseudo-masks with r0's flags (cluster-final.json "pseudo" section)
    python plan-segmentacion/camino-b-cnn/pseudo-mascaras/make_pseudo_masks.py \
      --saliency $S4/a1_saliency/saliency.npy --source dino \
      --output-dir $S4/pseudo_a1 --field $F/cache/field.npy --images $F/cache/images.npy \
      --low 0.35 --high 0.65 --max-center-distance 0.5 --max-area-fraction 0.6 \
      --min-area-fraction 0.003
    # 4. Pau's metric: pseudo_a1 vs r0 stats (coverage, no_region, by area)
    diff <(jq -S . $F/pseudo_r0/pseudo_masks_stats.json) \
         <(jq -S . $S4/pseudo_a1/pseudo_masks_stats.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts", REPO / "plan-segmentacion" / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl import run_provenance as rp  # noqa: E402
from dermssl import s_diag as sd  # noqa: E402
from s4_provenance import (  # noqa: E402
    parse_backbone, pool_fingerprint, sha256_file, weights_digest,
)

PATCH_SIZE = 14
SALIENCY_FILE = "saliency.npy"
SALIENCY_DTYPE = np.float16  # same dtype as ccam/saliency.npy
DEFAULT_BACKBONE = "dinov2_vitb14:public"
RULE = ("S-diag primary (docs/27): last-layer x_norm_patchtokens, L2 per token, per-image PCA "
        "first component, sign so the central grid//4 block is positive; not binarised; "
        "nearest-neighbour repeat to the cache resolution")
_BACKBONES: dict = {}

__all__ = ["saliency_map", "shards", "process_shard", "main", "sha256_file"]


def saliency_map(tokens: np.ndarray, grid: int, resolution: int) -> np.ndarray:
    """S-diag PCA map of one image, repeated to ``resolution`` (nearest)."""
    if resolution % grid:
        raise ValueError(f"resolution {resolution} is not a whole number of {grid} patches")
    factor = resolution // grid
    projection = sd.pca_first_component_map(tokens, grid)
    return np.repeat(np.repeat(projection, factor, axis=0), factor, axis=1).astype(np.float32)


def shards(total: int, workers: int) -> list[tuple[int, int]]:
    """Contiguous, disjoint ``[start, stop)`` ranges, at most ``workers`` of them."""
    count = max(1, min(workers, total))
    edges = np.linspace(0, total, count + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _load_backbone(arch: str, source: str):
    from dermssl.s4_unet import hub_backbone

    model = hub_backbone(arch, source)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.eval()


def _backbone(arch: str, source: str):
    """One model per process, loaded on the first shard it serves."""
    key = (arch, source)
    if key not in _BACKBONES:
        _BACKBONES[key] = _load_backbone(arch, source)
    return _BACKBONES[key]


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def process_shard(job: tuple) -> dict:
    """Fill rows ``[start, stop)`` of the output memmap; returns compute seconds."""
    import torch

    from dermssl.torch_c1_extraction import to_normalized_tensor

    images_path, out_path, start, stop, arch, source, batch_size = job
    model = _backbone(arch, source)
    images = np.load(images_path, mmap_mode="r")
    out = np.load(out_path, mmap_mode="r+")
    resolution = int(images.shape[1])
    grid = resolution // PATCH_SIZE
    began = time.perf_counter()
    for first in range(start, stop, batch_size):
        rows = range(first, min(first + batch_size, stop))
        batch = torch.stack([to_normalized_tensor(images[row]) for row in rows])
        with torch.no_grad():
            tokens = model.forward_features(batch)["x_norm_patchtokens"].float().numpy()
        for row, image_tokens in zip(rows, tokens):
            out[row] = saliency_map(image_tokens, grid, resolution).astype(SALIENCY_DTYPE)
    out.flush()
    return {"start": start, "stop": stop, "seconds": time.perf_counter() - began}


def _run_shards(jobs: list[tuple], workers: int) -> list[dict]:
    if workers <= 1:
        return [process_shard(job) for job in jobs]
    import multiprocessing

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"  # inherited by the spawned children before torch loads
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker) as pool:
        return pool.map(process_shard, jobs, chunksize=1)


def build(cache_dir: Path, out_path: Path, count: int, args) -> list[dict]:
    """Compute the saliency of the first ``count`` cached images into ``out_path``."""
    from cache_io import create_cache

    images_path = cache_dir / "images.npy"
    resolution = int(np.load(images_path, mmap_mode="r").shape[1])
    create_cache(out_path, (count, resolution, resolution), SALIENCY_DTYPE).flush()
    arch, source = args.backbone
    jobs = [(str(images_path), str(out_path), start, stop, arch, source, args.batch_size)
            for start, stop in shards(count, args.workers * args.shards_per_worker)]
    return _run_shards(jobs, args.workers)


def timing_projection(args, pool_images: int) -> dict:
    """Time ``--timing-projection`` images; project the whole pool at these workers."""
    count = min(args.timing_projection, pool_images)
    with tempfile.TemporaryDirectory() as scratch:
        began = time.perf_counter()
        results = build(args.cache_dir, Path(scratch) / SALIENCY_FILE, count, args)
        wall = time.perf_counter() - began
    per_image = sum(r["seconds"] for r in results) / count
    projected = per_image * pool_images / max(1, args.workers)
    return {"images_timed": count, "pool_images": pool_images, "workers": args.workers,
            "wall_seconds": wall, "seconds_per_image_per_worker": per_image,
            "projected_hours": projected / 3600.0}


def run_record(args, fingerprint: dict, results: list[dict], revision: str) -> dict:
    arch, source = args.backbone
    return {
        "experiment": "S4-UNet A1 saliency (docs/27, registered 2026-09-28)",
        "rule": RULE, "patch_size": PATCH_SIZE, "dtype": np.dtype(SALIENCY_DTYPE).name,
        "backbone": {"arch": arch, "source": source, "weights_sha256": weights_digest(source)},
        "pool": fingerprint, "code_revision": revision, "workers": args.workers,
        "compute_seconds": sum(r["seconds"] for r in results),
        "next_step": "make_pseudo_masks.py --source dino with the r0 flags",
        "reads_masks_or_labels": False, "complete": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, type=parse_backbone,
                        metavar="ARCH:SOURCE", help="SOURCE is 'public' or a state_dict file")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shards-per-worker", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timing-projection", type=int, default=0, metavar="N",
                        help="time N images, print the projected CPU hours, write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fingerprint = pool_fingerprint(args.cache_dir)
    if args.timing_projection > 0:
        print(json.dumps(timing_projection(args, fingerprint["n_images"])))
        return 0
    out = args.output_dir.expanduser()
    if out.exists():
        raise SystemExit(f"ABORT: {out} exists; A1 saliency is never overwritten")
    try:
        revision = rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    out.mkdir(parents=True)
    results = build(args.cache_dir, out / SALIENCY_FILE, fingerprint["n_images"], args)
    record = run_record(args, fingerprint, results, revision)
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": fingerprint["n_images"], "seconds": record["compute_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
