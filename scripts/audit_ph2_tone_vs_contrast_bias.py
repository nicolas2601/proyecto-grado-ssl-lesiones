"""C1: is the teacher 026 Dice bias on PH2 driven by skin tone or by
lesion-skin contrast? Same metric definitions as maestro's ISIC2018 measurement
(engram #2062/#2064, ~/work/ita-ladder-20261104/contrast.py) so the two
cohorts are directly comparable:

* ``ita``: ITA from the EXPERT mask (``fitzpatrick.estimate_fst(img, mask=gt)``),
  independent of the model's own segmentation error - the "arm4_gt_ita" of
  the ISIC ladder.
* ``contrast_L``: median L* outside the GT mask (skin) minus median L* inside
  (lesion) - the lightness axis the lesion border lives on.
* ``contrast_lab``: Euclidean norm of (median outside Lab - median inside Lab),
  all three channels.
* ``dice``: Dice of the pinned teacher 026 recipe (shaped=False, production)
  against the GT mask, reused from
  ``resultados/segmentacion/v8/hair_removal_shaped_dice_ph2/per_image.jsonl``
  (``dice_unshaped``) rather than rerun.

Spearman of each driver against Dice, the tone-vs-contrast confound check,
the rank-residualized partial correlation, and the tercile gap - mirroring
contrast.py's own print blocks so the two cohorts can be read side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

from dermssl.fitzpatrick import estimate_fst, srgb_to_lab

REPO = Path(__file__).resolve().parents[1]
PH2 = REPO / "data" / "ph2"
DICE_ROWS = REPO / "resultados" / "segmentacion" / "v8" / "hair_removal_shaped_dice_ph2" / "per_image.jsonl"
OUT_DIR = REPO / "resultados" / "segmentacion" / "v8" / "ph2_tone_vs_contrast_bias"


def partial(y: np.ndarray, x: np.ndarray, controls: list[np.ndarray]) -> tuple[float, float]:
    """Spearman of residuals after regressing out ``controls``, on ranks."""
    rank = stats.rankdata
    design = np.column_stack([np.ones(len(x))] + [rank(c) for c in controls])
    res_y = rank(y) - design @ np.linalg.lstsq(design, rank(y), rcond=None)[0]
    res_x = rank(x) - design @ np.linalg.lstsq(design, rank(x), rcond=None)[0]
    return stats.spearmanr(res_x, res_y)


def tercile_gap(driver: np.ndarray, dice: np.ndarray) -> dict:
    q = np.quantile(driver, [1 / 3, 2 / 3])
    groups = [dice[driver < q[0]], dice[(driver >= q[0]) & (driver < q[1])], dice[driver >= q[1]]]
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    return {
        "n_per_tercile": [int(g.size) for g in groups],
        "mean_per_tercile": [float(g.mean()) for g in groups],
        "gap_t3_minus_t1": float(groups[2].mean() - groups[0].mean()),
        "anova_F": float(f_stat), "anova_p": float(f_p),
        "kruskal_H": float(h_stat), "kruskal_p": float(h_p),
    }


def main() -> None:
    manifest = json.loads((PH2 / "manifest.json").read_text(encoding="utf-8"))
    dice_by_id = {
        json.loads(line)["image_id"]: json.loads(line)["dice_unshaped"]
        for line in DICE_ROWS.read_text(encoding="utf-8").splitlines()
    }

    rows, skipped = [], []
    for record in manifest["records"]:
        image_id = record["image_id"]
        img = np.asarray(Image.open(PH2 / record["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(PH2 / record["mask_path"]).convert("L")) > 127

        if gt.sum() < 50 or (~gt).sum() < 50:
            skipped.append((image_id, "too_few_pixels_in_a_partition"))
            continue

        est = estimate_fst(img, mask=gt)
        if est.bucket is None and est.status != "ok":
            skipped.append((image_id, f"estimate_fst_status={est.status}"))
            continue

        lab = srgb_to_lab(img)
        inside, outside = lab[gt], lab[~gt]
        dice = dice_by_id.get(image_id)
        if dice is None:
            skipped.append((image_id, "no_dice_row"))
            continue

        rows.append({
            "image_id": image_id,
            "ita": est.ita,
            "dice": dice,
            "area": float(gt.mean()),
            "contrast_L": float(np.median(outside[:, 0]) - np.median(inside[:, 0])),
            "contrast_lab": float(np.linalg.norm(
                np.median(outside, axis=0) - np.median(inside, axis=0)
            )),
        })

    ita = np.array([r["ita"] for r in rows])
    dice = np.array([r["dice"] for r in rows])
    area = np.array([r["area"] for r in rows])
    c_l = np.array([r["contrast_L"] for r in rows])
    c_lab = np.array([r["contrast_lab"] for r in rows])

    simple_corr = {}
    for name, v in [("ita", ita), ("contrast_L", c_l), ("contrast_lab", c_lab), ("area", area)]:
        r, p = stats.spearmanr(v, dice)
        simple_corr[name] = {"rho": float(r), "p": float(p)}

    confound = {}
    for name, v in [("contrast_L", c_l), ("contrast_lab", c_lab), ("area", area)]:
        r, p = stats.spearmanr(ita, v)
        confound[f"ita_vs_{name}"] = {"rho": float(r), "p": float(p)}

    r_p1, p_p1 = partial(dice, ita, [c_l, area])
    r_p2, p_p2 = partial(dice, c_l, [ita, area])
    partials = {
        "ita_given_contrast_L_and_area": {"rho": float(r_p1), "p": float(p_p1)},
        "contrast_L_given_ita_and_area": {"rho": float(r_p2), "p": float(p_p2)},
    }

    terciles = {"ita": tercile_gap(ita, dice), "contrast_L": tercile_gap(c_l, dice)}

    summary = {
        "dataset": "ph2",
        "n_used": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "spearman_vs_dice": simple_corr,
        "tone_vs_contrast_confound": confound,
        "partial_correlations": partials,
        "tercile_gap": terciles,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_image.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
