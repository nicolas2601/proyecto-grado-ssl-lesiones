r"""E5 (docs/33): descriptive Fitzpatrick analysis of HAM10000, ISIC 2019 and ISIC 2020. NOT validated.

Per dataset it writes ``<out-dir>/e5_fitzpatrick_<dataset>.json`` (+ ``.environment.json``): per image
the ITA bucket I-VI of two global estimators (Otsu lesion removal, whole image) with abstention, the
L* tercile of THIS dataset, and the perilesional-ring estimator (outside the automatic student_r3 mask,
without hair / specular / saturated / border pixels, with abstention); plus class x bucket, bucket x split and
tercile counts. The ITA estimator is the one that scored 22.9 % against dermatologists in PAD-UFES-20
(docs/30, PMC13459198) and the terciles are luminance, not skin tone: the JSON says so. Only HAM10000
has native masks (450 x 600); ISIC 2019 / 2020 use the automatic 224 x 224 pool masks, and an image whose mask
is quarantined or absent falls back to the image border as skin (``no_mask_border``), counted apart.

Images are decoded to a canonical resolution before any math (PIL draft + antialiased resize, shorter
side 450, never upsampled), on CPU, in ``--workers`` processes that return small dicts (memory-safe).
The result does not depend on the number of workers. Tree must be clean (``--allow-dirty`` only with
``--limit``). A finished result is skipped; ``--overwrite`` redoes it. ISIC 2020's split needs the
isolated scikit-learn 1.7.2 (docs/33 section 2).

Pod (P=$HOME/work/proyectonp; R=$P/repo_bench4; D=$HOME/work/datos)::

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$HOME/work/skl_try/1.7.2
    cd $R; O=resultados/banco_completo/e5; M=$P/salidas/c1/masks
    python scripts/e5_fitzpatrick.py --dataset ham10000 --data-root $D/ham10000 \
        --masks $M/c1_pseudo_masks.npy --masks-manifest $M/c1_pseudo_masks_manifest.json \
        --out-dir $O --workers 40
    T=$P/salidas/tone/inputs
    python scripts/e5_fitzpatrick.py --dataset isic2019 --data-root $D/isic2019_fullres \
        --image-dir $D/isic2019/train-image --pool-root $T --out-dir $O --workers 40
    python scripts/e5_fitzpatrick.py --dataset isic2020 --data-root $D/isic2020 --pool-root $T \
        --out-dir $O --workers 40

Add ``--dry-run`` first (split / mask check, no images read) or ``--limit 50`` for a smoke run.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker: 48 processes, not 48 x 48 threads

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import e5_data  # noqa: E402

from dermssl import e5_analyze as an  # noqa: E402
from dermssl import e5_document as doc  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402

DEFAULT_WORKERS = 8


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=e5_data.DATASETS, required=True)
    p.add_argument("--data-root", type=Path, required=True, help="folder holding the metadata / ground-truth CSV")
    p.add_argument("--image-dir", type=Path, default=None, help="folder searched for the images (default: data-root)")
    p.add_argument("--pool-root", type=Path, default=None,
                   help="isic only: salidas/tone/inputs (originals/, masks/, mask_manifest.json; 224 x 224)")
    p.add_argument("--masks", type=Path, default=None, help="ham10000 only: c1_pseudo_masks.npy (student_r3)")
    p.add_argument("--masks-manifest", type=Path, default=None, help="ham10000 only: c1_pseudo_masks_manifest.json")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--chunk-size", type=int, default=an.DEFAULT_CHUNK)
    p.add_argument("--limit", type=int, default=None, help="smoke: N images per split (own output name)")
    p.add_argument("--allow-dirty", action="store_true", help="smoke only")
    p.add_argument("--overwrite", action="store_true", help="redo a finished result")
    p.add_argument("--dry-run", action="store_true", help="verify the split and the masks, print the plan, stop")
    args = p.parse_args(argv)
    if not args.dry_run and args.out_dir is None:
        p.error("--out-dir is required unless --dry-run")
    if args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def revision(args) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty and args.limit:
            return "dirty"
        raise SystemExit(f"ABORT: {exc}") from exc


def out_path(args, name: str) -> Path:
    suffix = f"_smoke_{args.limit}" if args.limit else ""
    return args.out_dir.expanduser() / f"e5_fitzpatrick_{name}{suffix}.json"


def write_atomic(path: Path, text: str) -> None:
    """A crash leaves the old file or none, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _progress(started: float):
    def report(done: int, total: int) -> None:
        print(f"[e5] {done}/{total} images, {time.time() - started:.0f}s", flush=True)

    return report


def analyze_spec(args, spec) -> list[dict]:
    masked = spec.masks is not None
    return an.run_tasks(spec.tasks, workers=args.workers, chunk_size=args.chunk_size,
                        masks_path=args.masks.expanduser() if args.masks else None, use_hair=masked,
                        progress=_progress(time.time()))


def coverage_text(spec) -> str:
    """Mask state per split, printed before any image is read (the dry run shows it)."""
    lines = []
    for split in spec.splits:
        mine = [t.mask_state or "none" for t in spec.tasks if t.split == split]
        lines.append(f"  {split}: " + ", ".join(f"{s} {mine.count(s)}" for s in dict.fromkeys(mine)))
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = e5_data.load_spec(args)
    print(f"{spec.name}: {len(spec.tasks)} images, splits {list(spec.splits)}, masks {'yes' if spec.masks else 'no'}")
    print(coverage_text(spec))
    gaps = e5_data.require_intact_pool(spec) if spec.masks and spec.tasks[0].mask_pos is None else {}
    if gaps:
        print(f"  fallback sin máscara: {gaps['fallback_from_pool']} con la imagen del pool, "
              f"{gaps['fallback_from_source']} leídas de la fuente a 224 x 224")
    if args.dry_run:
        return 0
    out = out_path(args, spec.name)
    if out.is_file() and not args.overwrite:
        print(f"[skip] {out} exists (use --overwrite)")
        return 0
    code_revision = revision(args)
    hair = an.hair_parameters() if spec.masks else None
    records = analyze_spec(args, spec)
    document = doc.build_document(spec, records, code_revision=code_revision, smoke=args.limit, hair_detector=hair)
    write_atomic(out, doc.dump_document(document))
    run_environment.write_sidecar(out, args)
    failed = document["report"]["n_unreadable"]
    print(f"wrote {out} ({failed} unreadable)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
