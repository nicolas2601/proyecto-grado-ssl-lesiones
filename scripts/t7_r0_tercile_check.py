r"""T7 teacher check (docs/27): r0 pseudo-masks against HAM10000 reference masks by luminance tercile.

CPU only, run before any T7 GPU job. The bins are ``tone_tercile`` of the
published manifest (terciles of the skin median L*, not skin tone). For every
HAM10000 image evaluated by A2 that has a tercile, the r0 pseudo-mask A2 was
trained on is scored against the reference mask at the cache resolution
(224 px, reference read by the evaluator: nearest neighbour, > 127; Dice of
the evaluator). Images r0 excluded (every pixel 255) are counted per tercile
and left out of the Dice, as they are left out of training. Two forms:
(a) binary, 255 as background; (b) only over the confident pixels.

The registered conditional rule: if |G(r0)| >= |G(A2)| on the same images
(point estimates, form (a)), a dated addendum predicting "T7 no cumple por
techo del maestro" is committed before launching T7. ``addendum_required``
in the JSON says which case holds.

Refuses unless the T7 block of docs/27 is an ancestor of HEAD, unchanged and
older than this script, and the manifest, the r0 pseudo-masks, the cache
manifest and A2's per-image CSV have the registered sha256.

Pod command (CPU; P=~/work/proyectonp; clean clone in $P/repo_rel)::

    P=$HOME/work/proyectonp; source $P/.venv/bin/activate; cd $P/repo_rel
    F=$P/salidas/final; T7=$P/salidas/t7; mkdir -p $T7
    python scripts/t7_r0_tercile_check.py --cache-dir $F/cache \
      --pseudo-masks $F/pseudo_r0/pseudo_masks.npy \
      --manifest $F/strata_v2/fst_pseudo_manifest.csv \
      --ham-mask-dir $P/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl \
      --out $T7/r0_check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts", REPO / "plan-segmentacion" / "comun" / "evaluacion",
               REPO / "plan-segmentacion" / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import evaluate_segmentation as es  # noqa: E402
from cache_io import IGNORE, read_manifest  # noqa: E402
from dermssl import luminance_dice as ld  # noqa: E402
from dermssl import luminance_threshold as lt  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from eval_inputs import load_binary_mask  # noqa: E402

RESULTS = REPO / "plan-segmentacion" / "camino-b-cnn" / "resultados"
REGISTRATION_COMMIT = "d24a0f252e6c06a6bce5c85cf01de52139a64ae5"
REGISTRATION_DOC = "docs/27-plan-v7-mejoras-ssl.md"
BLOCK_START = "<!-- registro-t7-reentreno-luminancia:inicio -->"
BLOCK_END = "<!-- registro-t7-reentreno-luminancia:fin -->"
SCRIPT = "scripts/t7_r0_tercile_check.py"
MANIFEST_SHA256 = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
PSEUDO_R0_SHA256 = "8c9e47978070baa7bfa9f0174a80eb4e4236d45737801abf690804ae1eb0fc87"
CACHE_MANIFEST_SHA256 = "a83f927ab798b6f7a0614c3923fda77d082f887d10fdbe73416e34e9574ba71a"
A2_RUN = "2026-09-28-s4-a2"
REGISTERED_BOOTSTRAP = {"n_resamples": 1000, "seed": 20260923}
#: HAM10000 reference masks are ``<image_id>_segmentation.png``, as in A2's evaluation; not configurable.
MASK_SUFFIX, MASK_EXTENSION = "_segmentation", ".png"
JSON_FILE, TABLE_FILE, ROWS_FILE = "t7_r0_tercile_check.json", "README.md", "r0_per_image.csv"
ROW_COLUMNS = ("image_id", "tone_tercile", "excluded", "dice_binary", "dice_confident")
TERCILE_NAMES = {"dark": "oscuro", "medium": "medio", "light": "claro"}


def check_registration(repo: Path) -> str:
    return pg.check_registration(repo, commit=REGISTRATION_COMMIT, doc=REGISTRATION_DOC, script=SCRIPT,
                                 start=BLOCK_START, end=BLOCK_END)


def require_sha256(path: Path, expected: str, what: str) -> None:
    digest = rp.file_sha256(path)
    if digest != expected:
        raise ValueError(f"{what} {path} tiene sha256 {digest[:12]}, no el registrado {expected[:12]}")


def read_a2_reference(path: Path) -> dict[str, float]:
    """A2's per-image Dice, refused unless it matches the run's published ``SHA256SUMS``."""
    sums = {}
    for line in (path.parents[1] / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.removeprefix("./")] = digest
    require_sha256(path, sums[f"{path.parent.name}/{path.name}"], "el per_image.csv de A2")
    with open(path, newline="") as handle:
        return {r["image_id"]: float(r["dice"]) for r in csv.DictReader(handle)}


def teacher_dice(mask: np.ndarray, truth: np.ndarray) -> dict:
    """Form (a): 255 as background; form (b): only the confident pixels. All-255 is an exclusion."""
    confident = mask != IGNORE
    if not confident.any():
        return {"excluded": True, "dice_binary": None, "dice_confident": None}
    lesion = mask == 1
    return {"excluded": False,
            "dice_binary": es.compute_binary_metrics(lesion, truth)["dice"],
            "dice_confident": es.compute_binary_metrics(lesion[confident], truth[confident])["dice"]}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _tercile_summary(rows: list[dict], a2: dict[str, float]) -> dict:
    kept = [r for r in rows if not r["excluded"]]
    return {"n": len(rows), "n_excluded": len(rows) - len(kept),
            "excluded_fraction": (len(rows) - len(kept)) / len(rows) if rows else None,
            "dice_binary": _mean([r["dice_binary"] for r in kept]),
            "dice_confident": _mean([r["dice_confident"] for r in kept]),
            "dice_a2_same_images": _mean([a2[r["image_id"]] for r in kept])}


def analyze(rows: list[dict], a2: dict[str, float], n_resamples: int, seed: int) -> dict:
    """Per-tercile teacher Dice, both gaps on the same kept images and the registered addendum flag."""
    by = {t: [r for r in rows if r["tone_tercile"] == t] for t in lt.TERCILES}
    dark = [r for r in by[lt.DARK] if not r["excluded"]]
    rest = [r for t in lt.REST for r in by[t] if not r["excluded"]]
    pick = lambda group, key: [r[key] for r in group]  # noqa: E731
    a2_of = lambda group: [a2[r["image_id"]] for r in group]  # noqa: E731
    teacher = ld.unpaired_contrast(pick(dark, "dice_binary"), pick(rest, "dice_binary"), n_resamples, seed)
    confident = ld.unpaired_contrast(pick(dark, "dice_confident"), pick(rest, "dice_confident"),
                                     n_resamples, seed)
    a2_gap = ld.unpaired_contrast(a2_of(dark), a2_of(rest), n_resamples, seed)
    paired = lt.gap_reduction(a2_of(dark), pick(dark, "dice_binary"), a2_of(rest), pick(rest, "dice_binary"),
                              n_resamples, seed)
    return {"terciles": {t: _tercile_summary(by[t], a2) for t in lt.TERCILES},
            "teacher_gap_binary": teacher, "teacher_gap_confident": confident,
            "a2_gap_same_images": a2_gap, "reduction_a2_minus_teacher": paired["reduction"],
            "addendum_required": bool(abs(teacher["delta"]) >= abs(a2_gap["delta"])),
            "bootstrap": {"n_resamples": n_resamples, "seed": seed}}


def _num(value, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "—"
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}").replace(".", ",")


def _ci(stat: dict) -> str:
    return (f"{_num(stat['delta'], signed=True)} [{_num(stat['ci95_low'], signed=True)}; "
            f"{_num(stat['ci95_high'], signed=True)}]")


from dermssl.display_nomenclature import segmentation_presentation  # noqa: E402


@segmentation_presentation
def render_markdown(doc: dict) -> str:
    a = doc["analysis"]
    lines = ["# T7 — Chequeo del maestro r0 por tercil de luminancia (HAM10000)", "",
             f"Generado por `{SCRIPT}` desde `{JSON_FILE}`; no editar a mano. Registro y regla condicional: "
             "docs/27, bloque T7. Los bins son `tone_tercile` del manifiesto publicado (luminancia, no es tono "
             "de piel). Oráculo diagnóstico: no entra al entrenamiento.", "",
             f"Cobertura: {doc['coverage']['evaluated']} imágenes con tercil; "
             f"{doc['coverage']['without_tercile']} sin tercil; {doc['coverage']['not_in_pool']} fuera del pool.",
             "", "| Tercil | n | excluidas por r0 | Dice r0 (a) binaria | Dice r0 (b) confiable | "
             "Dice A2 (mismas imágenes) |", "|---|---:|---:|---:|---:|---:|"]
    for tercile, name in TERCILE_NAMES.items():
        row = a["terciles"][tercile]
        lines.append(f"| {name} | {row['n']} | {row['n_excluded']} ({_num(row['excluded_fraction'], 3)}) | "
                     f"{_num(row['dice_binary'])} | {_num(row['dice_confident'])} | "
                     f"{_num(row['dice_a2_same_images'])} |")
    rule = "**sí**: se agrega la adenda" if a["addendum_required"] else "**no**"
    lines += ["", "| Contraste oscuro − resto | Δ [IC 95 %] |", "|---|---|",
              f"| G(r0), forma (a) | {_ci(a['teacher_gap_binary'])} |",
              f"| G(r0), forma (b) | {_ci(a['teacher_gap_confident'])} |",
              f"| G(A2), mismas imágenes | {_ci(a['a2_gap_same_images'])} |",
              f"| \\|G(A2)\\| − \\|G(r0)\\| (pareado, descriptivo) | {_ci(a['reduction_a2_minus_teacher'])} |",
              "", f"Regla condicional (\\|G(r0)\\| ≥ \\|G(A2)\\|, estimaciones puntuales, forma (a)): {rule}.", ""]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("cache-dir", "pseudo-masks", "manifest", "ham-mask-dir", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--n-resamples", type=int, default=REGISTERED_BOOTSTRAP["n_resamples"])
    parser.add_argument("--seed", type=int, default=REGISTERED_BOOTSTRAP["seed"])
    args = parser.parse_args(argv)
    asked = {"n_resamples": args.n_resamples, "seed": args.seed}
    if asked != REGISTERED_BOOTSTRAP:
        raise SystemExit(f"ABORT: bootstrap {asked} no es el registrado {REGISTERED_BOOTSTRAP}")
    return args


def preflight() -> tuple[str, str]:
    try:
        return check_registration(REPO), rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def _inputs(args) -> tuple[dict, dict, dict, np.ndarray]:
    require_sha256(args.manifest, MANIFEST_SHA256, "el manifiesto")
    require_sha256(args.cache_dir / "manifest.csv", CACHE_MANIFEST_SHA256, "el manifiesto de la caché")
    require_sha256(args.pseudo_masks, PSEUDO_R0_SHA256, "las pseudo-máscaras r0")
    with open(args.manifest, newline="") as handle:
        terciles = {r["image_id"]: r["tone_tercile"] for r in csv.DictReader(handle)}
    index = {r["image_id"]: r["index"] for r in read_manifest(args.cache_dir / "manifest.csv")}
    a2 = read_a2_reference(RESULTS / A2_RUN / "evaluacion_ham10000" / "per_image.csv")
    return terciles, index, a2, np.load(args.pseudo_masks, mmap_mode="r")


def _mask_path(args, image_id: str) -> Path:
    return args.ham_mask_dir / f"{image_id}{MASK_SUFFIX}{MASK_EXTENSION}"


def reference_masks_digest(paths: list[tuple[str, Path]]) -> str:
    """sha256 of the ``<id>  <sha256 of the mask file>`` listing, in scoring order: pins the reference corpus."""
    listing = "".join(f"{image_id}  {rp.file_sha256(path)}\n" for image_id, path in paths)
    return hashlib.sha256(listing.encode()).hexdigest()


def _score(args, ids: list[str], terciles: dict, index: dict, masks: np.ndarray) -> list[dict]:
    resolution = int(masks.shape[1])
    rows = []
    for image_id in ids:
        truth = load_binary_mask(_mask_path(args, image_id), resolution)
        rows.append({"image_id": image_id, "tone_tercile": terciles[image_id],
                     **teacher_dice(np.asarray(masks[index[image_id]]), truth)})
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight()
    terciles, index, a2, masks = _inputs(args)
    in_pool = [i for i in sorted(a2) if i in index]
    ids = [i for i in in_pool if terciles.get(i) in lt.TERCILES]
    rows = _score(args, ids, terciles, index, masks)
    args.out.mkdir(parents=True, exist_ok=False)
    _write_rows(args.out / ROWS_FILE, rows)
    doc = {"experiment": "T7 teacher check (docs/27)",
           "coverage": {"evaluated": len(ids), "without_tercile": len(in_pool) - len(ids),
                        "not_in_pool": len(a2) - len(in_pool)},
           "analysis": analyze(rows, a2, args.n_resamples, args.seed),
           "provenance": {"registration_commit": registration, "code_revision": revision,
                          "manifest_sha256": MANIFEST_SHA256, "pseudo_r0_sha256": PSEUDO_R0_SHA256,
                          "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
                          "rows_sha256": rp.file_sha256(args.out / ROWS_FILE),
                          "reference_masks_sha256": reference_masks_digest([(i, _mask_path(args, i)) for i in ids])}}
    (args.out / JSON_FILE).write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf-8")
    run_environment.write_sidecar(args.out / JSON_FILE, args)
    (args.out / TABLE_FILE).write_text(render_markdown(doc), encoding="utf-8")
    print(f"addendum_required = {doc['analysis']['addendum_required']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
