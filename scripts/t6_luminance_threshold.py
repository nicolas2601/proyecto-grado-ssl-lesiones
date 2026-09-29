r"""T6 (docs/27): one mask threshold per luminance tercile on A2's probabilities, exploratory.

Cheap analogue of Fitzpatrick Thresholding (arXiv 2510.06655). The bins are
``tone_tercile`` of the published manifest (terciles of the skin median L*,
not skin tone). The thresholds are chosen on HAM10000 fold-0 train and scored
on fold-0 val (the C1-v7 test, pinned by its sha256) and, descriptively, on
the ISIC 2018 ids with a tercile.

Refuses unless the T6 block of docs/27 is an ancestor of HEAD, unchanged and
older than this script, the manifest is the registered one, the rebuilt split
is C1-v7's, and the Dice at 0.5 recomputed from the saved probabilities is
identical to A2's published ``per_image.csv``. Writes the per-image Dice
curves, the JSON and a table generated from it.

Pod commands (P=~/work/proyectonp; clean clone in $P/repo_rel at the published
commit). Step 1 repeats A2's evaluation with the SAME flags and batch size as
the published one (s4_unet_train.py step 5), only adding the probabilities,
so the 0.5 Dice can be reproduced exactly::

    P=$HOME/work/proyectonp; source $P/.venv/bin/activate
    export TORCH_HOME=$HOME/work/.torch; cd $P/repo_rel
    F=$P/salidas/final; S4=$P/salidas/s4_unet; T6=$P/salidas/t6; mkdir -p $T6
    for SET in isic2018 ham10000; do
      case $SET in
        isic2018) IMG=$P/datos_isic2018/ISIC2018_Task1-2_Training_Input
                  GT=$P/masks_eval/isic2018/ground_truth ;;
        ham10000) IMG=$P/eval_ham10000_images
                  GT=$P/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl ;;
      esac
      python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \
        --checkpoint $S4/a2/best.pt --image-dir $IMG --mask-dir $GT \
        --output-dir $T6/evaluacion_$SET --eval-name ${SET}_s4_a2_t6 --batch-size 32 \
        --save-probabilities $T6/probs_$SET
    done
    # 2. CPU; the manifest must be the registered one (sha256 57df1e65...)
    python scripts/t6_luminance_threshold.py --manifest $F/strata_v2/fst_pseudo_manifest.csv \
      --ham-root $HOME/work/datos/ham10000 \
      --probs-ham $T6/probs_ham10000 --probs-isic $T6/probs_isic2018 \
      --ham-mask-dir $P/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl \
      --isic-mask-dir $P/masks_eval/isic2018/ground_truth --out $T6/result
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts", REPO / "plan-segmentacion" / "comun" / "evaluacion",
               REPO / "plan-segmentacion" / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import luminance_threshold as lt  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402

RESULTS = REPO / "plan-segmentacion" / "camino-b-cnn" / "resultados"
REGISTRATION_COMMIT = "4971fc1063ea8a80a8c96233ebc33a6a5311390c"
REGISTRATION_DOC = "docs/27-plan-v7-mejoras-ssl.md"
BLOCK_START = "<!-- registro-t6-umbral-luminancia:inicio -->"
BLOCK_END = "<!-- registro-t6-umbral-luminancia:fin -->"
SCRIPT = "scripts/t6_luminance_threshold.py"
MANIFEST_SHA256 = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
#: ``eval_ids_sha256`` of resultados/clasificacion/v7/final_eval.json: the fold-0 val ids, in order.
VAL_IDS_SHA256 = "6ff4c17f04a9eae2f0d058aaa41084ea65b9aff91e4d5e5b05c924b4465a2fc6"
A2_RUN = "2026-09-28-s4-a2"
DATASETS = ("ham10000", "isic2018")
DATASET_NAMES = {"isic2018": "ISIC 2018", "ham10000": "HAM10000"}
TERCILE_NAMES = {"dark": "oscuro", "medium": "medio", "light": "claro"}
LABELS = (*lt.TERCILES, "unavailable")
PROBABILITIES_FILE, IDS_FILE = "probabilities.npy", "ids.txt"
CURVES_FILE, JSON_FILE, TABLE_FILE = "dice_curves.csv", "t6_luminance_threshold.json", "README.md"
CURVE_COLUMNS = [f"dice_{t:.2f}" for t in lt.GRID]


def check_registration(repo: Path) -> str:
    return pg.check_registration(repo, commit=REGISTRATION_COMMIT, doc=REGISTRATION_DOC, script=SCRIPT,
                                 start=BLOCK_START, end=BLOCK_END)


def extract_terciles(manifest: Path, ids: set[str]) -> dict[str, str]:
    digest = rp.file_sha256(manifest)
    if digest != MANIFEST_SHA256:
        raise ValueError(f"el manifiesto {manifest} tiene sha256 {digest[:12]}, no el registrado")
    with open(manifest, newline="") as handle:
        return {r["image_id"]: r["tone_tercile"] for r in csv.DictReader(handle) if r["image_id"] in ids}


def _rebuild_split(ham_root: Path) -> dict:
    from c1_build_val_index import rebuild_split  # noqa: PLC0415  (pandas and sklearn, pod only)

    return rebuild_split(ham_root)[0]


def split_ids(ham_root: Path) -> tuple[list[str], list[str]]:
    """Fold-0 train and val of the bank, refused unless val is C1-v7's test set."""
    split = _rebuild_split(ham_root)
    train, val = list(split["train_ids"]), list(split["val_ids"])
    digest = si.ids_sha256(val)
    if digest != VAL_IDS_SHA256:
        raise ValueError(f"los val_ids reconstruidos tienen sha256 {digest[:12]}, no {VAL_IDS_SHA256[:12]}")
    return train, val


def read_per_image_dice(path: Path) -> dict[str, float]:
    with open(path, newline="") as handle:
        return {r["image_id"]: float(r["dice"]) for r in csv.DictReader(handle)}


def compute_curves(prob_dir: Path, mask_dir: Path, wanted: set[str], reference: dict[str, float],
                   suffix: str, extension: str) -> dict[str, np.ndarray]:
    """Dice at every grid threshold for the wanted ids; aborts unless the 0.5 column is the published one."""
    from eval_inputs import load_binary_mask  # noqa: PLC0415

    ids = (prob_dir / IDS_FILE).read_text(encoding="utf-8").split()
    maps = np.load(prob_dir / PROBABILITIES_FILE, mmap_mode="r")
    if missing := wanted - set(ids):
        raise ValueError(f"{len(missing)} ids pedidos no tienen probabilidad (p. ej. {sorted(missing)[:3]})")
    fixed, curves, mismatched = lt.GRID.index(lt.FIXED), {}, []
    for k, image_id in enumerate(ids):
        if image_id not in wanted:
            continue
        truth = load_binary_mask(mask_dir / f"{image_id}{suffix}{extension}", maps.shape[1])
        curves[image_id] = lt.dice_curve(np.asarray(maps[k]), truth)
        if curves[image_id][fixed] != reference[image_id]:
            mismatched.append(image_id)
    if mismatched:
        raise ValueError(f"{len(mismatched)} imagenes no reproducen el Dice a 0,5 publicado "
                         f"(p. ej. {mismatched[:3]}): las probabilidades no son las de A2")
    return curves


def write_curves(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "dataset", "part", "tone_tercile", *CURVE_COLUMNS])
        for r in sorted(rows, key=lambda r: r["image_id"]):
            writer.writerow([r["image_id"], r["dataset"], r["part"], r["tone_tercile"],
                             *(repr(float(v)) for v in r["curve"])])


def read_curves(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return [{**{k: r[k] for k in ("image_id", "dataset", "part", "tone_tercile")},
                 "curve": np.array([float(r[c]) for c in CURVE_COLUMNS])} for r in csv.DictReader(handle)]


def _coverage(rows: list[dict], part: str) -> dict:
    return {label: sum(r["part"] == part and r["tone_tercile"] == label for r in rows) for label in LABELS}


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def _dataset_view(rows: list[dict], dataset: str) -> tuple[dict, dict, dict, list[dict]]:
    subset = [r for r in rows if r["dataset"] == dataset]
    curves = {r["image_id"]: r["curve"] for r in subset}
    terciles = {r["image_id"]: r["tone_tercile"] for r in subset}
    ids = {p: [r["image_id"] for r in subset if r["part"] == p] for p in ("fit", "eval")}
    return curves, terciles, ids, subset


def _with_coverage(cell: dict, subset: list[dict], ids: dict) -> dict:
    return {**cell, "coverage": {p: _coverage(subset, p) for p in ("fit", "eval") if ids[p]}}


def build_document(curves_csv: Path, n_resamples: int, seed: int) -> dict:
    """Everything but the run provenance, rebuilt from the committed curves.

    The thresholds come from HAM10000 fit ids only; ISIC 2018 reuses them, descriptively.
    """
    rows = read_curves(curves_csv)
    curves, terciles, ids, subset = _dataset_view(rows, "ham10000")
    ham = lt.analyze(curves, terciles, ids["fit"], ids["eval"], n_resamples, seed)
    cells = {"ham10000": _with_coverage(ham, subset, ids)}
    curves, terciles, ids, subset = _dataset_view(rows, "isic2018")
    isic = {"evaluation": lt.evaluate(curves, terciles, ham["thresholds"], ids["eval"], n_resamples, seed,
                                      with_verdict=False)}
    cells["isic2018"] = _with_coverage(isic, subset, ids)
    return {
        "analysis": "T6, exploratorio: umbral de mascara por tercil de luminancia sobre A2 (docs/27)",
        "source": "Stothers et al., Fitzpatrick Thresholding for Skin Image Segmentation, arXiv 2510.06655",
        "bins": "tone_tercile = tercil de la mediana de L* de la piel sobre el pool; no es tono de piel",
        "params": {"grid": list(lt.GRID), "fixed": lt.FIXED, "n_resamples": n_resamples, "seed": seed,
                   "gap_margin": lt.GAP_MARGIN, "global_tolerance": lt.GLOBAL_TOLERANCE, "min_n": lt.MIN_N},
        "inputs": {"curves": _relative(curves_csv), "curves_sha256": rp.file_sha256(curves_csv),
                   "manifest_sha256": MANIFEST_SHA256, "val_ids_sha256": VAL_IDS_SHA256},
        "datasets": cells,
    }


def _num(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "n/d"
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}").replace(".", ",")


def _ci(stat: dict) -> str:
    return f"{_num(stat['delta'], signed=True)} [{_num(stat['ci95_low'], signed=True)}; " \
           f"{_num(stat['ci95_high'], signed=True)}]"


def _threshold_rows(doc: dict) -> list[str]:
    chosen = doc["datasets"]["ham10000"]["thresholds"]
    return [f"| {TERCILE_NAMES[t]} | {_num(c['threshold'], 2)} | {c['fit_n']} | {_num(c['fit_dice_at_fixed'])} | "
            f"{_num(c['fit_dice'])} |" for t, c in chosen.items()]


def _tercile_rows(doc: dict) -> list[str]:
    return [f"| {DATASET_NAMES[d]} | {TERCILE_NAMES[t]} | {b['n']} | {_num(b['dice_fixed'])} | "
            f"{_num(b['dice_adapted'])} |"
            for d, cell in doc["datasets"].items() for t, b in cell["evaluation"]["terciles"].items()]


def _contrast_rows(doc: dict) -> list[str]:
    lines = []
    for d, cell in doc["datasets"].items():
        ev = cell["evaluation"]
        verdict = ev.get("verdict", {})
        lines.append(f"| {DATASET_NAMES[d]} | {ev['gap']['n_dark']} | {ev['gap']['n_rest']} | "
                     f"{_ci(ev['gap']['gap_fixed'])} | {_ci(ev['gap']['gap_adapted'])} | "
                     f"{_ci(ev['gap']['reduction'])} | {verdict.get('gap', 'descriptivo')} | "
                     f"{_ci(ev['global'])} | {verdict.get('global', 'descriptivo')} | "
                     f"**{verdict.get('prediction', 'sin veredicto')}** |")
    return lines


def render_markdown(doc: dict) -> str:
    p = doc["params"]
    return "\n".join([
        "# T6 — Umbral de máscara por tercil de luminancia sobre A2 (exploratorio)", "",
        "Generado por `scripts/t6_luminance_threshold.py` desde `t6_luminance_threshold.json`; no editar a "
        "mano. Registro, predicción y salvedades: docs/27, bloque T6. Análogo barato de *Fitzpatrick "
        "Thresholding for Skin Image Segmentation* (arXiv 2510.06655).", "",
        "Los bins son terciles de la mediana de L\\* de la piel (`tone_tercile` del manifiesto publicado): "
        "no son tono de piel ni fototipo. Los umbrales se eligen con máscaras de referencia de HAM10000 "
        "fold-0 train (oráculo diagnóstico declarado) y se evalúan en fold-0 val (el test de C1-v7); "
        "ISIC 2018 es solo descriptivo.", "",
        f"Grilla {_num(p['grid'][0], 2)}-{_num(p['grid'][-1], 2)}, paso 0,05. Bootstrap pareado por imagen: "
        f"{p['n_resamples']} remuestreos, semilla {p['seed']}, IC percentil 95 %.", "",
        "## Umbrales elegidos (HAM10000 fold-0 train, dentro de la muestra)", "",
        "| Tercil | Umbral | n ajuste | Dice ajuste a 0,5 | Dice ajuste con el umbral |",
        "|---|---:|---:|---:|---:|", *_threshold_rows(doc), "",
        "## Dice por tercil en evaluación", "",
        "| Conjunto | Tercil | n | Dice a 0,5 | Dice adaptado |", "|---|---|---:|---:|---:|",
        *_tercile_rows(doc), "",
        "## Brecha oscuro − resto y Dice global (predicción de Pau)", "",
        "| Conjunto | n oscuro | n resto | Brecha a 0,5 [IC 95 %] | Brecha adaptada [IC 95 %] | "
        "Reducción R [IC 95 %] | Parte 1 | ΔDice global [IC 95 %] | Parte 2 | Predicción |",
        "|---|---:|---:|---|---|---|---|---|---|---|", *_contrast_rows(doc), "",
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("manifest", "ham-root", "probs-ham", "probs-isic", "ham-mask-dir", "isic-mask-dir", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--mask-suffix", default="_segmentation")
    parser.add_argument("--mask-extension", default=".png")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args(argv)


def preflight() -> tuple[str, str]:
    try:
        return check_registration(REPO), rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def _dataset_rows(args, dataset: str, terciles: dict, parts: dict[str, str]) -> list[dict]:
    probs = {"ham10000": args.probs_ham, "isic2018": args.probs_isic}[dataset].expanduser()
    masks = {"ham10000": args.ham_mask_dir, "isic2018": args.isic_mask_dir}[dataset].expanduser()
    reference = read_per_image_dice(RESULTS / A2_RUN / f"evaluacion_{dataset}" / "per_image.csv")
    ids = set((probs / IDS_FILE).read_text(encoding="utf-8").split()) & set(terciles)
    curves = compute_curves(probs, masks, ids, reference, args.mask_suffix, args.mask_extension)
    part = (lambda i: parts[i]) if dataset == "ham10000" else (lambda i: "eval")
    return [{"image_id": i, "dataset": dataset, "part": part(i), "tone_tercile": terciles[i], "curve": c}
            for i, c in curves.items()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight()
    train, val = split_ids(args.ham_root.expanduser())
    parts = {**dict.fromkeys(train, "fit"), **dict.fromkeys(val, "eval")}
    ids = {i for d in (args.probs_ham, args.probs_isic)
           for i in (d.expanduser() / IDS_FILE).read_text(encoding="utf-8").split()}
    terciles = extract_terciles(args.manifest.expanduser(), ids)
    rows = [r for d in DATASETS for r in _dataset_rows(args, d, terciles, parts)]
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write_curves(out / CURVES_FILE, rows)
    doc = build_document(out / CURVES_FILE, args.n_resamples, args.seed)
    doc["provenance"] = {"registration_commit": registration, "code_revision": revision,
                         "manifest": str(args.manifest),
                         "probabilities": {d: str(p) for d, p in zip(DATASETS, (args.probs_ham, args.probs_isic))}}
    doc["environment"] = run_environment.snapshot(args)
    (out / JSON_FILE).write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
    (out / TABLE_FILE).write_text(render_markdown(doc), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
