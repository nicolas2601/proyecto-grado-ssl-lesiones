"""Validate the project's FST estimator against PAD-UFES-20's ``fitspatrick`` (docs/30 5).

Exploratory. ITA with Groh's fixed thresholds, whole image and Otsu
background, per image, through ``validar_fitzpatrick17k.evaluate_image`` and
``accuracy_metrics`` (the docs/20 code, not a second implementation). Adds
the linear Cohen's kappa and the majority baseline on the same rows.

Refuses to run unless the docs/30 5 block was committed before this script
and is unchanged, the tree is clean, and every image matches its manifest.

    python scripts/pad_ufes_fst_validation.py --manifest $P/manifest.json \
        --pad-root $P --out resultados/fst/pad_ufes_fst_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import validar_fitzpatrick17k as vf  # noqa: E402
from pad_ufes_eval import load_pad, write_fst_table  # noqa: E402

from dermssl import final_eval as fe  # noqa: E402
from dermssl import pad_ufes as pu  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import pad_ufes_readings as pr  # noqa: E402
from dermssl import paired_bootstrap as pb  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402

#: The commit that registered docs/30 5, pushed on its own before this script.
FST_REGISTRATION_COMMIT = "d351f36df4a525b27661eced45eefaca38c888b2"
FST_BLOCK = ("<!-- registro-pad-fst:inicio -->", "<!-- registro-pad-fst:fin -->")
THIS_SCRIPT = "scripts/pad_ufes_fst_validation.py"
CONDITIONS = ("whole_image", "masked")
FST_LABELS = tuple(range(1, 7))
SLIGHT_KAPPA = 0.20  # docs/30 5 secondary prediction: below "slight" agreement


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pad-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def majority_baseline(actual: list[int]) -> dict:
    """Always predict the most frequent label of these rows (ties: the lighter type)."""
    if not actual:
        return {"label": None, "accuracy": None, "off_by_one_accuracy": None}
    counts = Counter(actual)
    label = min(counts, key=lambda k: (-counts[k], k))
    return {"label": label, "accuracy": counts[label] / len(actual),
            "off_by_one_accuracy": sum(abs(a - label) <= 1 for a in actual) / len(actual)}


def linear_kappa(actual, predicted) -> float | None:
    from sklearn.metrics import cohen_kappa_score  # noqa: PLC0415

    if len(actual) == 0:
        return None
    return float(cohen_kappa_score(actual, predicted, labels=list(FST_LABELS), weights="linear"))


def _accuracy(actual, predicted) -> float:
    return float(np.mean(np.asarray(actual) == np.asarray(predicted)))


def grouped_interval(actual, predicted, groups, stat) -> list[float] | None:
    """Patient-grouped percentile interval of ``stat(actual, predicted)``; descriptive."""
    if len(actual) == 0:
        return None
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    draw = pb.index_sampler(len(actual), groups)
    rng = np.random.RandomState(pr.BOOTSTRAP["rng_seed"])
    values = [stat(actual[p], predicted[p]) for p in (draw(rng) for _ in range(pr.BOOTSTRAP["n_resamples"]))]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def condition_report(rows: list[dict], condition: str) -> dict:
    pairs = [(r["actual_fst"], r[condition]["predicted_bucket"]) for r in rows]
    report = vf.accuracy_metrics(pairs)
    kept = [r for r, (_, predicted) in zip(rows, pairs) if predicted is not None]
    actual = [r["actual_fst"] for r in kept]
    predicted = [r[condition]["predicted_bucket"] for r in kept]
    groups = [r["patient_id"] for r in kept]
    report["majority_baseline"] = majority_baseline(actual)
    report["linear_kappa"] = linear_kappa(actual, predicted)
    report["ci95"] = {"accuracy": grouped_interval(actual, predicted, groups, _accuracy),
                      "linear_kappa": grouped_interval(actual, predicted, groups, linear_kappa)}
    base = report["majority_baseline"]["accuracy"]
    report["beats_majority"] = None if base is None else bool(report["accuracy"] > base)
    return report


def verdicts(reports: dict) -> dict:
    """FST-P1: no condition beats the majority; FST-P2: kappa < 0.20 in both."""
    beats = [reports[c]["beats_majority"] for c in CONDITIONS]
    kappas = [reports[c]["linear_kappa"] for c in CONDITIONS]
    p1 = None not in beats and not any(beats)
    p2 = None not in kappas and all(k < SLIGHT_KAPPA for k in kappas)
    return {"FST-P1": fe.MET if p1 else fe.NOT_MET, "FST-P2": fe.MET if p2 else fe.NOT_MET}


def load_rgb(path: Path) -> np.ndarray:
    """Same decoding as validar_fitzpatrick17k.run: sRGB float in [0, 1], no color constancy."""
    from PIL import Image  # noqa: PLC0415

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def evaluate_rows(pad: dict, loader=load_rgb) -> tuple[list[dict], int]:
    """Per image with a valid phototype; returns the rows and how many had none."""
    rows, skipped = [], 0
    for img_id, fst, patient in zip(pad["ids"], pad["fst"], pad["groups"]):
        if fst is None:
            skipped += 1
            continue
        row = vf.evaluate_image(loader(pad["paths"][img_id]), int(fst))
        rows.append({**row, "img_id": img_id, "patient_id": str(patient)})
    return rows, skipped


def build_document(rows: list[dict], pad: dict, skipped: int, provenance: dict) -> dict:
    reports = {c: condition_report(rows, c) for c in CONDITIONS}
    return {**provenance, "exploratory": True, "manifest_sha256": pad["manifest_sha256"],
            "color_constancy": "identity", "n_images": len(pad["ids"]), "n_without_fst": skipped,
            "fst_note": pu.fst_note(pu.fst_composition(pad["fst"])), "metrics": reports,
            "verdicts": verdicts(reports), "records": rows}


def preflight(args: argparse.Namespace) -> tuple[str, str]:
    if args.out.expanduser().exists():
        raise SystemExit(f"ABORT: {args.out} ya existe; la validacion no se repite")
    try:
        registration = pg.check_registration(REPO, commit=FST_REGISTRATION_COMMIT, doc=pg.REGISTRATION_DOC,
                                             script=THIS_SCRIPT, start=FST_BLOCK[0], end=FST_BLOCK[1])
        return registration, rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration, revision = preflight(args)
    pad = load_pad(args.manifest.expanduser(), args.pad_root.expanduser())
    rows, skipped = evaluate_rows(pad)
    provenance = {"code_revision": revision, "registration_commit": registration}
    document = build_document(rows, pad, skipped, provenance)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    run_environment.write_sidecar(out, args)
    print(write_fst_table(out, pad["fst"]))
    print(document["verdicts"], f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
