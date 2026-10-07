"""Figure 3: segmentation masks. (a) Dice distribution per segmenter and dataset, (b) a qualitative panel.

The sources are the committed ``per_image.csv`` of the two segmenters: ``student_r3`` (U-Net, 2026-09-21-final)
and ``A2`` (frozen DINOv2 + DPT decoder, 2026-09-28-s4-a2), on HAM10000 and ISIC 2018. The qualitative cases are
picked by Dice percentile of the MEAN Dice of both segmenters (p90 good, p50 median, p10 bad), never by eye; the
images and masks of those cases come from ``fig3_extract_cases.py`` (run once on the pod).

    PYTHONPATH=$HOME/work/tmp/mpl python scripts/figures/fig3_masks.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = "plan-segmentacion/camino-b-cnn/resultados"
SOURCES = {
    ("student_r3", "ham10000"): f"{RESULTS}/2026-09-21-final/evaluacion_ham10000/per_image.csv",
    ("student_r3", "isic2018"): f"{RESULTS}/2026-09-21-final/evaluacion_isic2018/per_image.csv",
    ("A2", "ham10000"): f"{RESULTS}/2026-09-28-s4-a2/evaluacion_ham10000/per_image.csv",
    ("A2", "isic2018"): f"{RESULTS}/2026-09-28-s4-a2/evaluacion_isic2018/per_image.csv",
}
SEGMENTERS = ("student_r3", "A2")
DATASETS = ("ham10000", "isic2018")
DATASET_LABEL = {"ham10000": "HAM10000", "isic2018": "ISIC 2018"}
CASE_ROLES = (("bueno", 90), ("mediano", 50), ("malo", 10))
CASES_JSON = "resultados/figuras/fig3_cases.json"
CASES_NPZ = "resultados/figuras/fuentes/fig3_casos.npz"
STEM = "fig3_masks"


def load_dice(repo: Path = REPO) -> dict[tuple[str, str], dict[str, float]]:
    """Dice per image id for every (segmenter, dataset), read from the committed CSV files."""
    out = {}
    for key, rel in SOURCES.items():
        with open(repo / rel, newline="", encoding="utf-8") as handle:
            out[key] = {row["image_id"]: float(row["dice"]) for row in csv.DictReader(handle)}
    return out


def select_cases(repo: Path = REPO) -> list[dict]:
    """The image nearest to p90 / p50 / p10 of the mean Dice of both segmenters, per dataset (ties: smallest id)."""
    dice = load_dice(repo)
    cases = []
    for dataset in DATASETS:
        first, second = (dice[(name, dataset)] for name in SEGMENTERS)
        ids = sorted(set(first) & set(second))
        mean = np.array([(first[i] + second[i]) / 2 for i in ids])
        for role, percentile in CASE_ROLES:
            image_id = ids[int(np.argmin(np.abs(mean - np.percentile(mean, percentile))))]
            cases.append({"dataset": dataset, "role": role, "percentile": percentile, "image_id": image_id,
                          "dice": {name: dice[(name, dataset)][image_id] for name in SEGMENTERS}})
    return cases


def figure_data(repo: Path = REPO) -> dict:
    """Numbers drawn by panel (a), plus the qualitative cases of panel (b)."""
    dice = load_dice(repo)
    summary: dict = {dataset: {} for dataset in DATASETS}
    for (segmenter, dataset), by_id in dice.items():
        values = np.array(list(by_id.values()))
        summary[dataset][segmenter] = {
            "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        }
    return {"dice": summary, "cases": select_cases(repo)}


def write_cases(repo: Path = REPO) -> Path:
    path = repo / CASES_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(select_cases(repo), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_cases())
