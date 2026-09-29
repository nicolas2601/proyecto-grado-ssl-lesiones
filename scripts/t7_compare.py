r"""T7 (docs/27): compare the three retrained arms against A2 by luminance tercile, exploratory.

The bins are ``tone_tercile`` of the published manifest (terciles of the skin
median L*, not skin tone). Arms: T7 (dark oversampling k = 2 and darkening),
(a) A2 with seed 43 (seed noise) and (b) T7 with augmentation only. Every
arm's ``run.json`` must equal A2's in everything but its registered changes.
HAM10000 (all ids with a tercile) carries the verdict; HAM10000 fold-0 val
and the ISIC 2018 replica are descriptive. Writes a JSON and a table
generated from it.

Pod commands, after the three trainings (P, F, S4 and T7 as in
``s4_unet_train.py``)::

    for RUN in t7 a2_s43 t7_aug; do for SET in isic2018 ham10000; do
      case $SET in
        isic2018) IMG=$P/datos_isic2018/ISIC2018_Task1-2_Training_Input
                  GT=$P/masks_eval/isic2018/ground_truth ;;
        ham10000) IMG=$P/eval_ham10000_images
                  GT=$P/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl ;;
      esac
      E=$T7/$RUN/evaluacion_$SET
      python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \
        --checkpoint $T7/$RUN/best.pt --image-dir $IMG --mask-dir $GT \
        --output-dir $E --eval-name ${SET}_t7_$RUN --overlay-dir $E/overlays \
        --hausdorff --batch-size 32 --fst-manifest $F/fst/fst_pseudo_manifest.csv \
        --strata-column tone_tercile --wandb-project dermssl-segmentacion
    done; done
    python scripts/t7_compare.py --t7 $T7/t7 --s43 $T7/a2_s43 --aug $T7/t7_aug \
      --manifest $F/strata_v2/fst_pseudo_manifest.csv --ham-root $HOME/work/datos/ham10000 \
      --out $T7/result
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts", REPO / "plan-segmentacion" / "comun" / "evaluacion",
               REPO / "plan-segmentacion" / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl import luminance_retrain as lr  # noqa: E402
from dermssl import luminance_sampling as ls  # noqa: E402
from dermssl import luminance_threshold as lt  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402

RESULTS = REPO / "plan-segmentacion" / "camino-b-cnn" / "resultados"
REGISTRATION_COMMIT = ls.REGISTRATION_COMMIT
SCRIPT = "scripts/t7_compare.py"
MANIFEST_SHA256 = ls.MANIFEST_SHA256
A2_RUN = "2026-09-28-s4-a2"
ARMS = ("T7", "A2-s43", "T7-aug")
REFERENCE = "A2"
SEEDS = {"T7": 42, "A2-s43": 43, "T7-aug": 42}
OVERSAMPLE = {"T7": ls.REGISTERED_K, "T7-aug": 1}
REGISTERED_DARKEN = {"probability": 0.5, "gamma_range": [1.0, 2.2], "brightness_range": [0.85, 1.0],
                     "contrast_range": [0.85, 1.0], "terciles": ["medium", "light"]}
#: Top-level run.json keys an arm may change: its registered changes and bookkeeping. Everything else is A2's.
ARM_KEYS = ("variant", "seed", "t7", "code_revision", "result", "complete", "environment")
#: Paths that depend on where the pod keeps the files; their content is pinned by the sha256 next to them.
FREE_PATHS = {"pool": ("cache_dir",), "pseudo_masks": ("path",)}
REGISTERED_BOOTSTRAP = {"n_resamples": 1000, "seed": 20260923}
#: evaluate_segmentation meta that must equal A2's: same evaluator path, binarisation, resolution and image set.
EVAL_KEYS = ("predictions", "resolution", "threshold", "tta", "n_images")
JSON_FILE, TABLE_FILE = "t7_compare.json", "README.md"
DATASET_NAMES = {"ham10000": "HAM10000 (veredicto)", "ham10000_fold0_val": "HAM10000 fold-0 val (descriptivo)",
                 "isic2018": "ISIC 2018, réplica (descriptivo)"}


def check_registration(repo: Path) -> str:
    return pg.check_registration(repo, commit=REGISTRATION_COMMIT, doc=ls.REGISTRATION_DOC, script=SCRIPT,
                                 start=ls.BLOCK_START, end=ls.BLOCK_END)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _comparable(record: dict) -> dict:
    out = {k: v for k, v in record.items() if k not in ARM_KEYS}
    for key, free in FREE_PATHS.items():
        if isinstance(out.get(key), dict):
            out[key] = {k: v for k, v in out[key].items() if k not in free}
    return out


def _differences(record: dict, a2: dict) -> list[str]:
    ours, theirs = _comparable(record), _comparable(a2)
    return sorted(k for k in ours.keys() | theirs.keys() if ours.get(k) != theirs.get(k))


def check_arm(arm: str, record: dict, a2: dict) -> None:
    """The arm is A2's recipe plus only its registered changes; anything else moved is refused."""
    _require(record.get("complete") is True, f"{arm}: la corrida no está completa")
    moved = _differences(record, a2)
    _require(not moved, f"{arm}: difiere de A2 fuera de sus cambios registrados en {moved}")
    _require(record["seed"] == SEEDS[arm], f"{arm}: semilla {record['seed']}, no {SEEDS[arm]}")
    if arm == "A2-s43":
        _require(record["variant"] == "A2" and "t7" not in record, f"{arm}: no es la receta de A2 (t7)")
        return
    t7 = record.get("t7", {})
    _require(record["variant"] == "A2-T7", f"{arm}: variante {record['variant']}")
    _require(t7.get("registration_commit") == REGISTRATION_COMMIT, f"{arm}: otro commit de registro")
    _require(t7.get("tercile_manifest_sha256") == ls.MANIFEST_SHA256, f"{arm}: otro manifiesto")
    _require(t7.get("oversample_k") == OVERSAMPLE[arm], f"{arm}: k = {t7.get('oversample_k')}")
    _require(t7.get("darken") == REGISTERED_DARKEN, f"{arm}: darken distinto del registrado")


def _read_per_image(path: Path) -> tuple[dict, dict]:
    """One row per image with a finite Dice; HD95 may be non-finite (empty mask) and is dropped later."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["image_id"] for r in rows]
    _require(len(ids) == len(set(ids)), f"{path}: image_id duplicados")
    dice = {r["image_id"]: float(r["dice"]) for r in rows}
    bad = [i for i, v in dice.items() if not math.isfinite(v)]
    _require(not bad, f"{path}: Dice no finito en {bad[:3]}")
    outside = [i for i, v in dice.items() if not 0.0 <= v <= 1.0]
    _require(not outside, f"{path}: Dice fuera de [0, 1] en {outside[:3]}")
    return dice, {r["image_id"]: float(r["hd95"]) for r in rows}


def read_a2(dataset: str) -> tuple[dict, dict]:
    """A2's per-image Dice and HD95, refused unless they match the run's published ``SHA256SUMS``."""
    path = RESULTS / A2_RUN / f"evaluacion_{dataset}" / "per_image.csv"
    sums = {}
    for line in (RESULTS / A2_RUN / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name] = digest
    expected = sums[f"./evaluacion_{dataset}/per_image.csv"]
    _require(rp.file_sha256(path) == expected, f"el per_image.csv de A2 ({dataset}) no es el publicado")
    return _read_per_image(path)


def read_a2_meta(dataset: str) -> dict:
    """The meta of A2's published evaluation (its per_image.csv is pinned by ``read_a2``)."""
    path = RESULTS / A2_RUN / f"evaluacion_{dataset}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))["meta"]


def check_evaluation(arm: str, dataset: str, arm_dir: Path, meta: dict, a2_meta: dict) -> None:
    """The arm was scored like A2 (same evaluator settings and image count) from its own best checkpoint."""
    for key in EVAL_KEYS:
        _require(meta.get(key) == a2_meta.get(key), f"{arm} ({dataset}): {key} = {meta.get(key)!r}, A2 usó "
                                                    f"{a2_meta.get(key)!r}")
    checkpoint = Path(str(meta.get("checkpoint", "")))
    _require(checkpoint.name == "best.pt" and checkpoint.parent.name == Path(arm_dir).name,
             f"{arm} ({dataset}): checkpoint {checkpoint} no es el best.pt de {arm_dir}")


def fold0_val_ids(ham_root: Path) -> list[str]:
    from t6_luminance_threshold import split_ids

    return split_ids(ham_root)[1]


def _finite_mean(values) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _per_tercile(dice: dict, hd95: dict, ids_by: dict) -> dict:
    return {name: {t: {"n": len(ids), "dice": _finite_mean([dice[name][i] for i in ids]),
                       "hd95": _finite_mean([hd95[name][i] for i in ids])} for t, ids in ids_by.items()}
            for name in dice}


def analyze_dataset(dice: dict, hd95: dict, terciles: dict, ids, n_resamples: int, seed: int,
                    with_verdict: bool) -> dict:
    """Per-tercile means, gaps on shared resamples, R and the global change of each arm against A2."""
    ids = list(ids)
    binned = sorted(i for i in ids if terciles.get(i) in lt.TERCILES)
    ids_by = {t: [i for i in binned if terciles[i] == t] for t in lt.TERCILES}
    dark, rest = ids_by[lt.DARK], [i for t in lt.REST for i in ids_by[t]]
    groups = {name: ([d[i] for i in dark], [d[i] for i in rest]) for name, d in dice.items()}
    gaps = lr.shared_gaps(groups, n_resamples, seed)
    out = {"n_dark": len(dark), "n_rest": len(rest), "n_without_tercile": len(ids) - len(binned),
           "terciles": _per_tercile(dice, hd95, ids_by), "gap": {name: lr.gap(gaps, name) for name in dice},
           "reduction": {arm: lr.reduction(gaps, REFERENCE, arm) for arm in ARMS},
           "global": {arm: lt.global_change([dice[REFERENCE][i] for i in binned], [dice[arm][i] for i in binned],
                                            n_resamples, seed) for arm in ARMS},
           "seed_contrast": lr.reduction_difference(gaps, "T7", "A2-s43"),
           "oversampling_contrast": lr.reduction_difference(gaps, "T7", "T7-aug")}
    if with_verdict:
        parts = {"gap": lt.gap_verdict(out["reduction"]["T7"]["ci95_low"]),
                 "global": lt.global_verdict(out["global"]["T7"]["ci95_low"])}
        prediction = lt.prediction_verdict(parts["gap"], parts["global"], len(dark), len(rest))
        seed_part = lr.seed_reading(out["seed_contrast"]["ci95_low"])
        out["verdict"] = {**parts, "prediction": prediction, "seed": seed_part,
                          "conclusion": lr.conclusion(prediction, seed_part)}
    return out


def _num(value, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "—"
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}").replace(".", ",")


def _ci(stat: dict) -> str:
    return (f"{_num(stat['delta'], signed=True)} [{_num(stat['ci95_low'], signed=True)}; "
            f"{_num(stat['ci95_high'], signed=True)}]")


def _dataset_lines(key: str, d: dict) -> list[str]:
    lines = [f"## {DATASET_NAMES[key]}", "", f"n oscuro = {d['n_dark']}, n resto = {d['n_rest']}, "
             f"sin tercil = {d['n_without_tercile']}.", "",
             "| Segmentador | Dice oscuro | Dice medio | Dice claro | HD95 oscuro | G oscuro − resto [IC 95 %] |",
             "|---|---:|---:|---:|---:|---|"]
    for name, by in d["terciles"].items():
        lines.append(f"| {name} | {_num(by['dark']['dice'])} | {_num(by['medium']['dice'])} | "
                     f"{_num(by['light']['dice'])} | {_num(by['dark']['hd95'], 2)} | {_ci(d['gap'][name])} |")
    lines += ["", "| Brazo | R = \\|G(A2)\\| − \\|G(brazo)\\| [IC 95 %] | ΔD global vs A2 [IC 95 %] |", "|---|---|---|"]
    lines += [f"| {arm} | {_ci(d['reduction'][arm])} | {_ci(d['global'][arm])} |" for arm in ARMS]
    lines += ["", f"S = R(T7) − R(A2-s43), pareado: {_ci(d['seed_contrast'])}. "
              f"R(T7) − R(T7-aug), descriptivo: {_ci(d['oversampling_contrast'])}.", ""]
    if "verdict" in d:
        v = d["verdict"]
        lines += [f"Parte 1 (brecha): **{v['gap']}**. Parte 2 (global): **{v['global']}**. "
                  f"Predicción de Pau: **{v['prediction']}**. Ruido de semilla: **{v['seed']}**.", "",
                  f"Lectura registrada: **{v['conclusion']}**.", ""]
    return lines


def render_markdown(doc: dict) -> str:
    lines = ["# T7 — A2 reentrenado con oscurecimiento y sobremuestreo del tercil oscuro (exploratorio)", "",
             f"Generado por `{SCRIPT}` desde `{JSON_FILE}`; no editar a mano. Registro, predicción y salvedades: "
             "docs/27, bloque T7. Los bins son `tone_tercile` del manifiesto publicado: luminancia, no es tono de "
             "piel. La luminancia está confundida con el diagnóstico, el dermatoscopio y la iluminación, y HAM10000 "
             "es transductivo (limitación del diseño).", "",
             f"Bootstrap: {doc['bootstrap']['n_resamples']} remuestreos, semilla {doc['bootstrap']['seed']}, "
             "IC percentil 95 %. G, R y S (la brecha y sus contrastes: estratificado por oscuro y resto, con los "
             "mismos índices para todos los segmentadores). ΔD global: pareado por imagen, sin estratificar, como "
             "en T6 y en el registro.", ""]
    for key in DATASET_NAMES:
        lines += _dataset_lines(key, doc["datasets"][key])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("t7", "s43", "aug", "manifest", "ham-root", "out"):
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


def _terciles(manifest: Path) -> dict[str, str]:
    _require(rp.file_sha256(manifest) == MANIFEST_SHA256, f"el manifiesto {manifest} no tiene el sha256 registrado")
    with open(manifest, newline="") as handle:
        return {r["image_id"]: r["tone_tercile"] for r in csv.DictReader(handle)}


def _load(args, dirs: dict) -> tuple[dict, dict]:
    a2_record = json.loads((RESULTS / A2_RUN / "run.json").read_text(encoding="utf-8"))
    for arm, path in dirs.items():
        check_arm(arm, json.loads((path / "run.json").read_text(encoding="utf-8")), a2_record)
    dice, hd95 = {}, {}
    for dataset in ("ham10000", "isic2018"):
        dice[dataset], hd95[dataset] = {}, {}
        dice[dataset][REFERENCE], hd95[dataset][REFERENCE] = read_a2(dataset)
        for arm, path in dirs.items():
            evaluation = path / f"evaluacion_{dataset}"
            meta = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))["meta"]
            check_evaluation(arm, dataset, path, meta, read_a2_meta(dataset))
            dice[dataset][arm], hd95[dataset][arm] = _read_per_image(evaluation / "per_image.csv")
            _require(dice[dataset][arm].keys() == dice[dataset][REFERENCE].keys(),
                     f"{arm} no se evaluó en las mismas imágenes que A2 ({dataset})")
    return dice, hd95


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight()
    dirs = {"T7": args.t7, "A2-s43": args.s43, "T7-aug": args.aug}
    try:
        terciles = _terciles(args.manifest)
        dice, hd95 = _load(args, dirs)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    views = {"ham10000": ("ham10000", list(dice["ham10000"][REFERENCE]), True),
             "ham10000_fold0_val": ("ham10000", fold0_val_ids(args.ham_root), False),
             "isic2018": ("isic2018", list(dice["isic2018"][REFERENCE]), False)}
    datasets = {key: analyze_dataset(dice[src], hd95[src], terciles, ids, args.n_resamples, args.seed, verdict)
                for key, (src, ids, verdict) in views.items()}
    doc = {"experiment": "T7 (docs/27)", "datasets": datasets,
           "bootstrap": {"n_resamples": args.n_resamples, "seed": args.seed},
           "provenance": {"registration_commit": registration, "code_revision": revision,
                          "manifest_sha256": MANIFEST_SHA256, "arms": {a: str(p) for a, p in dirs.items()}}}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / JSON_FILE).write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf-8")
    run_environment.write_sidecar(args.out / JSON_FILE, args)
    (args.out / TABLE_FILE).write_text(render_markdown(doc), encoding="utf-8")
    print(f"{datasets['ham10000']['verdict']['conclusion']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
