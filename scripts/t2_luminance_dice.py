"""T2 (docs/27): Dice and HD95 of A2 and student_r3 by luminance tercile, exploratory.

Reads the ``per_image.csv`` files already written by ``evaluate_segmentation.py``
and the ``tone_tercile`` column of the published manifest (terciles of the skin
median L*, not skin tone). Refuses to run unless the registration in docs/27 is
an ancestor of HEAD, unchanged, and older than this script, and unless the
manifest is the registered one (sha256). Writes the tercile extract of the
evaluated ids, the JSON and a markdown table generated from it::

    python scripts/t2_luminance_dice.py \\
      --manifest ~/cc-uploads/final/strata_v2/fst_pseudo_manifest.csv \\
      --out plan-segmentacion/camino-b-cnn/resultados/2026-09-28-t2-luminance-dice
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dermssl import luminance_dice as ld
from dermssl import pad_ufes_guard as pg
from dermssl import run_provenance as rp

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "plan-segmentacion" / "camino-b-cnn" / "resultados"
REGISTRATION_COMMIT = "a899048803fec724dc50aebf6d0cf8efcb00dfa1"
REGISTRATION_DOC = "docs/27-plan-v7-mejoras-ssl.md"
BLOCK_START = "<!-- registro-t2-luminancia:inicio -->"
BLOCK_END = "<!-- registro-t2-luminancia:fin -->"
SCRIPT = "scripts/t2_luminance_dice.py"
MANIFEST_SHA256 = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
RUNS = {"A2": "2026-09-28-s4-a2", "student_r3": "2026-09-21-final"}
DATASETS = ("isic2018", "ham10000")
DATASET_NAMES = {"isic2018": "ISIC 2018", "ham10000": "HAM10000"}
TERCILE_NAMES = {"dark": "oscuro", "medium": "medio", "light": "claro"}
TERCILES_FILE = "terciles_eval_ids.csv"
JSON_FILE = "t2_luminance_dice.json"
TABLE_FILE = "README.md"


def check_registration(repo: Path) -> str:
    return pg.check_registration(repo, commit=REGISTRATION_COMMIT, doc=REGISTRATION_DOC, script=SCRIPT,
                                 start=BLOCK_START, end=BLOCK_END)


def extract_terciles(manifest: Path, ids: set[str]) -> dict[str, str]:
    """``tone_tercile`` of the evaluated ids, only from the registered manifest."""
    digest = rp.file_sha256(manifest)
    if digest != MANIFEST_SHA256:
        raise ValueError(f"el manifiesto {manifest} tiene sha256 {digest[:12]}, no el registrado")
    with open(manifest, newline="") as handle:
        return {r["image_id"]: r["tone_tercile"] for r in csv.DictReader(handle) if r["image_id"] in ids}


def write_terciles(path: Path, terciles: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "tone_tercile"])
        writer.writerows(sorted(terciles.items()))


def read_terciles(path: Path) -> dict[str, str]:
    with open(path, newline="") as handle:
        return {r["image_id"]: r["tone_tercile"] for r in csv.DictReader(handle)}


def per_image_path(results: Path, segmenter: str, dataset: str) -> Path:
    return results / RUNS[segmenter] / f"evaluacion_{dataset}" / "per_image.csv"


def read_per_image(path: Path) -> dict[str, dict[str, float]]:
    with open(path, newline="") as handle:
        return {r["image_id"]: {m: float(r[m]) for m in ld.METRICS} for r in csv.DictReader(handle)}


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def _dataset_cell(results: Path, dataset: str, terciles: dict, n_resamples: int, seed: int) -> dict:
    rows = {s: read_per_image(per_image_path(results, s, dataset)) for s in RUNS}
    first, *others = rows.values()
    if any(set(other) != set(first) for other in others):
        raise ValueError(f"{dataset}: los segmentadores no puntuan las mismas imagenes")
    return {"coverage": ld.coverage(first, terciles),
            "segmenters": {s: ld.analyze(r, terciles, n_resamples, seed) for s, r in rows.items()}}


def build_document(results: Path, terciles_csv: Path, n_resamples: int, seed: int) -> dict:
    """Everything but the run provenance, so it can be rebuilt from committed files."""
    terciles = read_terciles(terciles_csv)
    inputs = {"manifest_sha256": MANIFEST_SHA256, "terciles": _relative(terciles_csv),
              "terciles_sha256": rp.file_sha256(terciles_csv),
              "per_image": {s: {d: {"path": _relative(per_image_path(results, s, d)),
                                    "sha256": rp.file_sha256(per_image_path(results, s, d))}
                                for d in DATASETS} for s in RUNS}}
    return {
        "analysis": "T2, exploratorio: Dice por tercil de luminancia (docs/27)",
        "bins": "tone_tercile = tercil de la mediana de L* de la piel sobre el pool; no es tono de piel",
        "params": {"n_resamples": n_resamples, "seed": seed, "clear_margin": ld.CLEAR_MARGIN,
                   "min_n": ld.MIN_N},
        "inputs": inputs,
        "datasets": {d: _dataset_cell(results, d, terciles, n_resamples, seed) for d in DATASETS},
    }


def _num(value: float | None, digits: int) -> str:
    return "n/d" if value is None else f"{value:+.{digits}f}".lstrip("+").replace(".", ",")


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def _ci(stat: dict, digits: int, signed: bool = False) -> str:
    fmt = (lambda v: _signed(v, digits)) if signed else (lambda v: _num(v, digits))
    return f"{fmt(stat.get('mean', stat.get('delta')))} [{fmt(stat['ci95_low'])}; {fmt(stat['ci95_high'])}]"


def _bin_rows(doc: dict) -> list[str]:
    lines = []
    for d, cell in doc["datasets"].items():
        for s, res in cell["segmenters"].items():
            for t, name in TERCILE_NAMES.items():
                b = res["bins"][t]
                lines.append(f"| {DATASET_NAMES[d]} | {s} | {name} | {b['n']} | {_ci(b['dice'], 4)} | "
                             f"{_ci(b['hd95'], 2)} | {b['hd95']['n_dropped_non_finite']} |")
    return lines


def _contrast_rows(doc: dict) -> list[str]:
    lines = []
    for d, cell in doc["datasets"].items():
        for s, res in cell["segmenters"].items():
            rest, glob = res["dark_minus_rest"], res["dark_minus_global"]
            lines.append(f"| {DATASET_NAMES[d]} | {s} | {res['bins']['dark']['n']} | {res['rest_n']} | "
                         f"{_ci(rest['dice'], 4, True)} | **{rest['dice']['verdict']}** | "
                         f"{_ci(glob['dice'], 4, True)} | {_ci(rest['hd95'], 2, True)} |")
    return lines


def _coverage_rows(doc: dict) -> list[str]:
    return [f"| {DATASET_NAMES[d]} | {c['dark']} | {c['medium']} | {c['light']} | {c['unavailable']} | "
            f"{c['not_in_manifest']} | {c['total']} |"
            for d, c in ((d, cell["coverage"]) for d, cell in doc["datasets"].items())]


def render_markdown(doc: dict) -> str:
    p = doc["params"]
    return "\n".join([
        "# T2 — Dice por tercil de luminancia (exploratorio)", "",
        "Generado por `scripts/t2_luminance_dice.py` desde `t2_luminance_dice.json`; no editar a mano. "
        "Registro, predicción y salvedades: docs/27, bloque T2.", "",
        "Los bins son terciles de la mediana de L\\* de la piel sobre el pool (`tone_tercile` del manifiesto "
        f"publicado, sha256 `{doc['inputs']['manifest_sha256'][:12]}…`): luminancia, no es tono de piel. "
        "La luminancia está confundida con el diagnóstico, el tamaño de la lesión, el dermatoscopio y la "
        "iluminación, así que ningún contraste se atribuye a una causa.", "",
        f"Bootstrap: {p['n_resamples']} remuestreos, semilla {p['seed']}, IC percentil 95 %. Contraste "
        "oscuro − resto no pareado y estratificado por grupo. Veredicto registrado: **cumplida** si el IC "
        f"queda entero bajo {_signed(p['clear_margin'], 2)}; sin veredicto si un grupo tiene n < {p['min_n']}.",
        "", "## Cobertura del cruce por `image_id`", "",
        "| Conjunto | oscuro | medio | claro | cuarentena | fuera del manifiesto | total |",
        "|---|---:|---:|---:|---:|---:|---:|", *_coverage_rows(doc), "",
        "## Dice y HD95 por tercil", "",
        "| Conjunto | Segmentador | Tercil | n | Dice [IC 95 %] | HD95 px [IC 95 %] | HD95 no finito |",
        "|---|---|---|---:|---|---|---:|", *_bin_rows(doc), "",
        "## Contraste oscuro − resto (predicción de Pau)", "",
        "| Conjunto | Segmentador | n oscuro | n resto | ΔDice oscuro − resto [IC 95 %] | Veredicto | "
        "ΔDice oscuro − global [IC 95 %] (descriptivo) | ΔHD95 oscuro − resto px [IC 95 %] (descriptivo) |",
        "|---|---|---:|---:|---|---|---|---|", *_contrast_rows(doc), "",
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args(argv)


def preflight() -> tuple[str, str]:
    try:
        return check_registration(REPO), rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight()
    ids = {i for s in RUNS for d in DATASETS for i in read_per_image(per_image_path(RESULTS, s, d))}
    terciles_csv = args.out / TERCILES_FILE
    write_terciles(terciles_csv, extract_terciles(args.manifest.expanduser(), ids))
    doc = build_document(RESULTS, terciles_csv, args.n_resamples, args.seed)
    doc["provenance"] = {"registration_commit": registration, "code_revision": revision,
                         "manifest": str(args.manifest)}
    (args.out / JSON_FILE).write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf-8")
    (args.out / TABLE_FILE).write_text(render_markdown(doc), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
