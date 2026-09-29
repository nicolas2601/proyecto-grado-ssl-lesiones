"""T3 (docs/28): C1-v7 by luminance tercile, every mask condition against 'none', exploratory.

Reads the val probabilities C1-v7 already saved per cell (both grids,
student_r3 and A2 masks), the val labels of the bank cache and the split
index, all copied from the pod, plus the ``tone_tercile`` column of the
published manifest (terciles of the skin median L*, not skin tone). Nothing is
retrained. Refuses to run unless the registration in docs/28 is an ancestor of
HEAD, unchanged and older than this script, unless the manifest is the
registered one, and unless every cell reproduces its published c1_cells.json::

    python scripts/t3_c1_luminance.py --inputs ~/.cache/c1out/t3 \\
      --manifest ~/cc-uploads/final/strata_v2/fst_pseudo_manifest.csv

Input layout (``--inputs``): ``c1_split_index.json``; ``labels/`` with the
bank cache ``*__train__*.labels.npy`` and ``*__val__*.labels.npy``;
``student_r3/`` and ``a2/`` with ``cell__<backbone>__<condition>.proba.npy``
and ``.manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dermssl import c1_luminance as cl  # noqa: E402
from dermssl import luminance_dice as ld  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import paired_bootstrap as pb  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.c1_split_identity import ids_sha256  # noqa: E402
from dermssl.feature_cache import _digest, fingerprint_ids  # noqa: E402
from scripts import t2_luminance_dice as t2  # noqa: E402

REGISTRATION_COMMIT = "d05f563c18b2fc055327a4d2e018bee6e64497f0"
REGISTRATION_DOC = "docs/28-rediseno-brazo-propuesto-v7.md"
BLOCK_START = "<!-- registro-t3-luminancia:inicio -->"
BLOCK_END = "<!-- registro-t3-luminancia:fin -->"
SCRIPT = "scripts/t3_c1_luminance.py"
VAL_IDS_SHA256 = "6ff4c17f04a9eae2f0d058aaa41084ea65b9aff91e4d5e5b05c924b4465a2fc6"
BACKBONES = ("dinov2_public_vitb14", "v7_astar_s42", "v7_astar_s43", "v7_bstar_s42", "v7_bstar_s43")
ALL_CONDITIONS = (cl.CONTROL, *cl.CONDITIONS)
C1_V7 = REPO / "resultados" / "clasificacion" / "c1_v7"
PUBLISHED = {s: C1_V7 / s / "c1_cells.json" for s in cl.SEGMENTERS}
OUT = C1_V7 / "t3_luminance"
TERCILES_FILE, JSON_FILE, TABLE_FILE = "terciles_val_ids.csv", "t3_c1_luminance.json", "README.md"
MEL, N_CLASSES = 4, 7  # classes = sorted(dx): akiec bcc bkl df mel nv vasc
REPRODUCTION_TOLERANCE = 1e-12
ALPHA = 0.05
write_terciles, read_terciles = t2.write_terciles, t2.read_terciles


def _labels(root: Path, split: str, ids: list[str]) -> np.ndarray:
    hits = sorted((root / "labels").glob(f"*__{split}__*.labels.npy"))
    if len(hits) != 1:
        raise ValueError(f"esperaba 1 archivo de etiquetas de {split}, encontre {len(hits)}")
    if hits[0].name[: -len(".labels.npy")].rsplit("__", 1)[-1] != fingerprint_ids(ids):
        raise ValueError(f"las etiquetas de {split} son de otro orden de ids (fingerprint)")
    return np.load(hits[0])


def load_split(root: Path, val_ids_sha256: str) -> dict:
    """Val ids and labels, with the digest every cell manifest must carry."""
    index = json.loads((root / "c1_split_index.json").read_text(encoding="utf-8"))
    val_ids, train_ids = list(index["val_ids"]), list(index["train_ids"])
    if ids_sha256(val_ids) != val_ids_sha256:
        raise ValueError("el orden de val no es el registrado (c1_val_ids_sha256)")
    y_val = _labels(root, "val", val_ids)
    labels_sha = _digest(np.concatenate([_labels(root, "train", train_ids), y_val]))
    return {"val_ids": val_ids, "y": y_val, "labels_sha256": labels_sha, "val_sha256": ids_sha256(val_ids)}


def load_cell(grid: Path, backbone: str, condition: str, split: dict) -> np.ndarray:
    stem = grid / f"cell__{backbone}__{condition}"
    manifest = json.loads(Path(f"{stem}.manifest.json").read_text(encoding="utf-8"))
    proba = np.load(f"{stem}.proba.npy")
    for field, have in (("proba_sha256", _digest(proba)), ("c1_val_ids_sha256", split["val_sha256"]),
                        ("labels_sha256", split["labels_sha256"])):
        if manifest.get(field) != have:
            raise ValueError(f"{stem.name}: {field} no coincide con el manifiesto de la celda")
    return proba


def check_reproduces(segmenter: str, cell: str, y: np.ndarray, proba: np.ndarray, published: dict) -> None:
    want = published[cell]["metrics"]
    for metric in ("macro_f1", "balanced_accuracy"):
        have = pb.METRIC_FUNCTIONS[metric](y, proba)
        if abs(have - want[metric]) > REPRODUCTION_TOLERANCE:
            raise ValueError(f"{segmenter}/{cell}: {metric} {have!r} no reproduce el publicado {want[metric]!r}")


def _segmenter(root: Path, segmenter: str, split: dict, published: Path, terciles, n_resamples, seed) -> dict:
    cells = json.loads(Path(published).read_text(encoding="utf-8"))
    out = {}
    for backbone in BACKBONES:
        probas = {c: load_cell(root / segmenter, backbone, c, split) for c in ALL_CONDITIONS}
        for condition, proba in probas.items():
            check_reproduces(segmenter, f"{backbone}__{condition}", split["y"], proba, cells)
        out[backbone] = cl.tercile_contrasts(split["y"], terciles, probas, n_resamples=n_resamples,
                                             seed=seed, mel=MEL, n_classes=N_CLASSES)
    return out


def _key_cells(segmenters: dict) -> dict:
    return {(s, b, c): segmenters[s][b][cl.KEY_TERCILE]["contrasts"][c][cl.PRIMARY_METRIC]["verdict"]
            for s in cl.SEGMENTERS for b in cl.KEY_BACKBONES for c in cl.KEY_CONDITIONS}


def _multiplicity(n_key: int) -> dict:
    total = cl.n_contrasts(n_backbones=len(BACKBONES))
    return {"n_contrasts": total, "expected_ci_without_zero_by_chance": round(total * ALPHA, 6),
            "n_key_contrasts": n_key, "expected_key_improves_by_chance": round(n_key * ALPHA / 2, 6)}


def build_document(root: Path, terciles_csv: Path, *, n_resamples: int, seed: int,
                   published: dict | None = None, val_ids_sha256: str = VAL_IDS_SHA256) -> dict:
    """Everything but the run provenance."""
    published = published or PUBLISHED
    split = load_split(Path(root), val_ids_sha256)
    by_id = read_terciles(terciles_csv)
    coverage = ld.coverage(dict.fromkeys(split["val_ids"]), by_id)
    terciles = np.array([by_id.get(i, "not_in_manifest") for i in split["val_ids"]])
    segmenters = {s: _segmenter(Path(root), s, split, published[s], terciles, n_resamples, seed)
                  for s in cl.SEGMENTERS}
    key = _key_cells(segmenters)
    return {
        "analysis": "T3, exploratorio: C1-v7 por tercil de luminancia, cada condicion contra 'none' (docs/28)",
        "bins": "tone_tercile = tercil de la mediana de L* de la piel sobre el pool; no es tono de piel",
        "params": {"n_resamples": n_resamples, "seed": seed, "min_n": ld.MIN_N, "resampling": "imagen, pareado"},
        "inputs": {"manifest_sha256": t2.MANIFEST_SHA256, "terciles_sha256": rp.file_sha256(terciles_csv),
                   "val_ids_sha256": split["val_sha256"], "labels_sha256": split["labels_sha256"]},
        "coverage": coverage,
        "segmenters": segmenters,
        "key_cells": [{"segmenter": s, "backbone": b, "condition": c, "verdict": v} for (s, b, c), v in key.items()],
        "prediction": cl.prediction(key),
        "multiplicity": _multiplicity(len(key)),
    }


TERCILE_NAMES = {"dark": "oscuro", "medium": "medio", "light": "claro"}
METRIC_NAMES = {"macro_f1": "macro-F1", "balanced_accuracy": "bal-acc", "mel_recall": "sens. mel"}


def _fmt(v: float) -> str:
    return f"{v:+.4f}".replace(".", ",")


def _delta(entry: dict) -> str:
    if "point_delta" not in entry:
        return entry["verdict"]
    if "ci95_low" not in entry:
        return f"{_fmt(entry['point_delta'])} (sin potencia)"
    return f"{_fmt(entry['point_delta'])} [{_fmt(entry['ci95_low'])}; {_fmt(entry['ci95_high'])}] {entry['verdict']}"


def _rows(doc: dict, only_key: bool) -> list[str]:
    lines = []
    for s, backbones in doc["segmenters"].items():
        for b, bins in backbones.items():
            for t, name in TERCILE_NAMES.items():
                for c, metrics in bins[t]["contrasts"].items():
                    is_key = t == cl.KEY_TERCILE and b in cl.KEY_BACKBONES and c in cl.KEY_CONDITIONS
                    if is_key == only_key:
                        cols = " | ".join(_delta(metrics[m]) for m in cl.METRICS)
                        lines.append(f"| {s} | {b} | {name} | {c} | {bins[t]['n']} | {bins[t]['n_mel']} | {cols} |")
    return lines


def render_markdown(doc: dict) -> str:
    p, pred, mult, cov = doc["params"], doc["prediction"], doc["multiplicity"], doc["coverage"]
    header = ["| Segmentador | Backbone | Tercil | Condición | n | n mel | "
              + " | ".join(f"Δ {METRIC_NAMES[m]} [IC 95 %]" for m in cl.METRICS) + " |",
              "|---|---|---|---|---:|---:|---|---|---|"]
    return "\n".join([
        "# T3 — C1-v7 por tercil de luminancia (exploratorio)", "",
        "Generado por `scripts/t3_c1_luminance.py` desde `t3_c1_luminance.json`; no editar a mano. "
        "Registro, predicción y salvedades: docs/28, bloque T3.", "",
        "Los bins son terciles de la mediana de L\\* de la piel sobre el pool (`tone_tercile`, manifiesto sha256 "
        f"`{doc['inputs']['manifest_sha256'][:12]}…`): luminancia, no es tono de piel. Está confundida con el "
        "diagnóstico, el dermatoscopio y la iluminación; ningún Δ se atribuye al tono.", "",
        f"Δ = métrica(condición) − métrica(`none`), bootstrap pareado por imagen dentro del tercil, "
        f"{p['n_resamples']} remuestreos, semilla {p['seed']}, IC percentil 95 %. Con n < {p['min_n']}: sin potencia.", "",
        f"Cobertura de val: oscuro {cov['dark']}, medio {cov['medium']}, claro {cov['light']}, "
        f"sin tercil {cov['unavailable']}, fuera del manifiesto {cov['not_in_manifest']}, total {cov['total']}.", "",
        f"## Predicción de Paula: **{pred['verdict']}**", "",
        f"Celdas clave: {pred['n_key_cells']}. Mejoran: {', '.join(pred.get('improves', [])) or 'ninguna'}. "
        f"Robustas: {', '.join(pred.get('robust', [])) or 'ninguna'}. Empeoran: {pred.get('n_worsens', 'n/d')}.", "",
        f"Multiplicidad: {mult['n_contrasts']} contrastes en total, unos {mult['expected_ci_without_zero_by_chance']} "
        f"IC sin cero esperables por azar; en las {mult['n_key_contrasts']} celdas clave, unas "
        f"{mult['expected_key_improves_by_chance']} «mejora» por azar.", "",
        "## Celdas clave (tercil oscuro)", "", *header, *_rows(doc, True), "",
        "## Descriptivos (sin veredicto de predicción)", "", *header, *_rows(doc, False), "",
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args(argv)


def preflight() -> tuple[str, str]:
    try:
        registration = pg.check_registration(REPO, commit=REGISTRATION_COMMIT, doc=REGISTRATION_DOC,
                                             script=SCRIPT, start=BLOCK_START, end=BLOCK_END)
        return registration, rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight()
    inputs = args.inputs.expanduser()
    val_ids = json.loads((inputs / "c1_split_index.json").read_text(encoding="utf-8"))["val_ids"]
    terciles_csv = args.out / TERCILES_FILE
    write_terciles(terciles_csv, t2.extract_terciles(args.manifest.expanduser(), set(val_ids)))
    doc = build_document(inputs, terciles_csv, n_resamples=args.n_resamples, seed=args.seed)
    doc["provenance"] = {"registration_commit": registration, "code_revision": revision}
    (args.out / JSON_FILE).write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf-8")
    run_environment.write_sidecar(args.out / JSON_FILE, args)
    (args.out / TABLE_FILE).write_text(render_markdown(doc), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
