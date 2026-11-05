"""Zero-shot DDI: does the tone treatment narrow the phototype gap?

This is the measurement the tone line actually needed and never had. The paired
contrast of 2026-11-05 measured aggregate macro-F1 and came out negative
(PAD -0.0129, HAM -0.0112, see ``docs/65``), but the hypothesis of the line was
never about aggregate accuracy - it was about equity across skin tone. A method
can cost a little overall accuracy and still close the gap, and that would be a
legitimate positive result.

DDI is the only dataset here whose phototype is assigned by consensus of two
board-certified dermatologists rather than estimated, so it is the only way to
evaluate that claim without the ITA instrument this project discarded.

Protocol, fixed before looking at any number
--------------------------------------------
* Train on HAM, predict on DDI. Zero-shot: no DDI image is ever trained on.
* **Binary malignant/benign over all 656 images.** NOT the 368 that map to HAM's
  seven classes: that mapping keeps 61% of FST I-II and only 35% of FST V-VI,
  and drops the darkest bin's malignant rate from 23.2% to 13.7%. Filtering on
  it would reproduce the exact bias ``docs/64`` documents for ITA
  measurability. ``dermssl.ddi`` already declares the binary reading as primary.
* Reader mirrors the published one: row-normalized features, balanced class
  weights, logistic regression.
* Primary quantity: **sensitivity** (recall on malignant) per phototype bin.
  For a screening aid, missing a malignant lesion is the costly error, and the
  published fairness concern in dermatology AI is precisely that sensitivity
  degrades on darker skin.
* The gap is ``FST I-II minus FST V-VI``. A treatment that works makes it
  smaller. Bootstrap over images, 2000 resamples; DDI has no duplicate groups,
  each image is its own identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # repo layout
    from dermssl.ddi import HAM_MALIGNANT
    from dermssl.pad_ufes import PAD_TO_HAM
except ModuleNotFoundError:  # cluster: ddi.py sits flat beside this script
    # The frozen experiment tree ships its own ``dermssl`` package (needed for
    # ``dermssl.metrics``, which the reader imports) and it is hash-sealed, so
    # ``ddi.py`` cannot be dropped into it. Importing it flat avoids shadowing
    # that package with a stub. ``ddi.py`` has stdlib imports only, so this is
    # the same module either way.
    from ddi import HAM_MALIGNANT
    from pad_ufes import PAD_TO_HAM

BASE = Path("/home/jovyan/work/ddi-20261105")
TRAIN_FEATURES = {
    # Dermoscopy. Kept because its FAILURE is the evidence: a dermoscopy-trained
    # reader does not transfer to clinical photography at all (sensitivity 0.06
    # to 0.12 with specificity near 1.0 is a near-constant predictor), so any
    # gap computed from it compares two broken classifiers.
    "ham_dev": {
        "control": Path("/home/jovyan/work/validation_v8/clf-ab-repro-20261103-001/ham_dev.npz"),
        "tono_w1": Path("/home/jovyan/work/source-fit-revision-20261104/readout_tono_s42/ham_dev.npz"),
    },
    # Clinical smartphone photography, the same modality as DDI. This is the
    # valid comparison.
    "pad_adapt": {
        "control": Path("/home/jovyan/work/validation_v8/clf-ab-repro-20261103-001/pad_adapt.npz"),
        "tono_w1": Path("/home/jovyan/work/source-fit-revision-20261104/readout_tono_s42/pad_adapt.npz"),
    },
}
DDI_FEATURES = {
    "control": BASE / "ddi.control.npz",
    "tono_w1": BASE / "ddi.tono_w1.npz",
}
OVERLAY = Path("/home/jovyan/work/clf1_clinical_reference_20261024/results/content-exclusion-overlay-v1")
HAM_METADATA = (Path("/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv"),
                "06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5")
PAD_METADATA = (Path("/home/jovyan/work/pad_ufes_20/metadata.csv"),
                "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527")
BINS = ("12", "34", "56")
BOOT = 2000


def normalize(path: Path, order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        features, ids = data["features"], [str(s) for s in data["image_ids"]]
    if order is not None:
        index = {key: position for position, key in enumerate(ids)}
        features = features[[index[key] for key in order]]
        ids = list(order)
    return features / np.linalg.norm(features, axis=1, keepdims=True), ids


def rates(truth: np.ndarray, predicted: np.ndarray) -> dict:
    positives, negatives = truth == 1, truth == 0
    sensitivity = float(predicted[positives].mean()) if positives.any() else float("nan")
    specificity = float(1.0 - predicted[negatives].mean()) if negatives.any() else float("nan")
    return {
        "n": int(truth.size),
        "n_malignant": int(positives.sum()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
    }


def main() -> None:
    import sys
    sys.path.insert(0, "/home/jovyan/work/clf1_clinical_reference_20261024/scripts")
    from evaluate_overlay_readout import load_labels
    from sklearn.linear_model import LogisticRegression

    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    ddi_ids = [r["ddi_id"] for r in manifest["images"]]
    ddi_tone = np.asarray([r["skin_tone"] for r in manifest["images"]])
    ddi_truth = np.asarray([1 if r["malignant"] else 0 for r in manifest["images"]])

    # The PAD malignant set is DERIVED, not invented: compose the declared
    # PAD_TO_HAM mapping with HAM_MALIGNANT. BCC->bcc, MEL->mel, ACK->akiec and
    # SCC->akiec all land inside it; NEV->nv and SEK->bkl do not. Two mappings
    # that already existed, so no new clinical judgement is introduced here.
    pad_malignant = frozenset(k for k, v in PAD_TO_HAM.items() if v in HAM_MALIGNANT)

    sources = {}
    for role, kind, metadata, grouping in (
        ("ham_dev", "HAM", HAM_METADATA, HAM_MALIGNANT),
        ("pad_adapt", "PAD", PAD_METADATA, pad_malignant),
    ):
        rows = [json.loads(line) for line in (OVERLAY / f"{role}.jsonl").read_text().splitlines()
                if line.strip()]
        classes = np.asarray(load_labels(kind, rows, *metadata)).astype(str)
        sources[role] = {
            "ids": [row["image_id"] for row in rows],
            "truth": np.asarray([1 if c in grouping else 0 for c in classes]),
            "malignant_classes": sorted(grouping),
        }

    def interval(values: list[float]) -> dict:
        array = np.asarray(values)
        return {"n_boot": int(array.size), "mean": float(array.mean()),
                "ci95_low": float(np.percentile(array, 2.5)),
                "ci95_high": float(np.percentile(array, 97.5))}

    out = {
        "protocol": {
            "evaluate": "DDI, all 656, binary malignant/benign, zero-shot",
            "why_not_the_368_mapped": (
                "mapping to HAM's seven classes keeps 61% of FST I-II and 35% of FST V-VI "
                "and drops the darkest bin's malignant rate from 23.2% to 13.7%"
            ),
            "primary_quantity": "sensitivity per phototype bin; gap = FST I-II minus FST V-VI",
            "two_training_sources": (
                "pad_adapt is the valid one, same clinical-photography modality as DDI. "
                "ham_dev is dermoscopy and is reported only because its failure is the evidence "
                "of the domain shift."
            ),
        },
        "ddi_composition": {
            b: {"n": int((ddi_tone == b).sum()),
                "malignant": int(ddi_truth[ddi_tone == b].sum())} for b in BINS
        },
        "trained_on": {},
    }

    rng = np.random.default_rng(42)
    for role, data in sources.items():
        predictions = {}
        arms = {}
        for arm in ("control", "tono_w1"):
            x_train, _ = normalize(TRAIN_FEATURES[role][arm], data["ids"])
            x_ddi, _ = normalize(DDI_FEATURES[arm], ddi_ids)
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            model.fit(x_train, data["truth"])
            predicted = model.predict(x_ddi)
            predictions[arm] = predicted
            per_bin = {b: rates(ddi_truth[ddi_tone == b], predicted[ddi_tone == b]) for b in BINS}
            arms[arm] = {
                "overall": rates(ddi_truth, predicted),
                "by_phototype": per_bin,
                "gap_sensitivity_light_minus_dark":
                    per_bin["12"]["sensitivity"] - per_bin["56"]["sensitivity"],
                "gap_balanced_accuracy_light_minus_dark":
                    per_bin["12"]["balanced_accuracy"] - per_bin["56"]["balanced_accuracy"],
            }

        # Paired bootstrap over images: both arms predict the same frames, so
        # the difference of gaps is paired and the resample must be shared.
        diffs, gaps = [], {"control": [], "tono_w1": []}
        for _ in range(BOOT):
            pick = rng.integers(0, ddi_truth.size, ddi_truth.size)
            tone, truth = ddi_tone[pick], ddi_truth[pick]
            current = {}
            for arm in ("control", "tono_w1"):
                predicted = predictions[arm][pick]
                light = rates(truth[tone == "12"], predicted[tone == "12"])["sensitivity"]
                dark = rates(truth[tone == "56"], predicted[tone == "56"])["sensitivity"]
                if not (np.isfinite(light) and np.isfinite(dark)):
                    current = None
                    break
                current[arm] = light - dark
            if current is None:
                continue
            for arm in current:
                gaps[arm].append(current[arm])
            diffs.append(current["tono_w1"] - current["control"])

        usable = arms["control"]["overall"]["balanced_accuracy"] > 0.60
        out["trained_on"][role] = {
            "n_train": int(data["truth"].size),
            "n_train_malignant": int(data["truth"].sum()),
            "malignant_classes": data["malignant_classes"],
            "arms": arms,
            "bootstrap": {
                "gap_control": interval(gaps["control"]),
                "gap_tono_w1": interval(gaps["tono_w1"]),
                "gap_change_treated_minus_control": {
                    **interval(diffs),
                    "fraction_narrowing_the_gap": float((np.asarray(diffs) < 0).mean()),
                    "reading": "negative means the treatment NARROWED the phototype gap",
                },
            },
            "transfer_usable": bool(usable),
            "transfer_note": None if usable else (
                "Balanced accuracy at or below 0.60 on the control arm: the reader barely "
                "beats chance on DDI, so a gap computed here compares two near-constant "
                "predictors and must NOT be reported as a fairness result."
            ),
        }

    destination = BASE / "ddi_tone_gap.json"
    destination.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
