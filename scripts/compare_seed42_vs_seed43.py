"""Extract 021 and 022 stratified metrics and compare Seed 42 (019 vs 020) vs Seed 43 (021 vs 022)."""
import json
from pathlib import Path
import numpy as np


def compute_stratified(cases_jsonl_path: Path) -> dict:
    strata_defs = [
        ("<1%", 0.0, 0.01),
        ("1-2%", 0.01, 0.02),
        ("2-5%", 0.02, 0.05),
        ("5-10%", 0.05, 0.10),
        (">=10%", 0.10, 1.01),
    ]
    strata_records = {name: [] for name, _, _ in strata_defs}
    all_records = []

    for line in cases_jsonl_path.open(encoding="utf-8"):
        rec = json.loads(line)
        gt_a = float(rec["reference_foreground_fraction"])
        sens = float(rec["sensitivity"])
        spec = float(rec["specificity"])
        dice = float(rec["dice"])
        pred_a = sens * gt_a + (1.0 - spec) * (1.0 - gt_a)

        # pred_a is reconstructed, not read from the file, and precision, area ratio
        # and all five CIELAB metrics hang off it. Assert dice == 2*TP/(gt+pred) so a
        # corrupt or over-rounded sensitivity/specificity cannot silently contaminate
        # the published table.
        denom = gt_a + pred_a
        if denom > 1e-12:
            implied_dice = 2.0 * (sens * gt_a) / denom
            if abs(implied_dice - dice) > 1e-4:
                raise AssertionError(
                    f"Dice invariant violated for {rec['image_id']}: "
                    f"implied {implied_dice!r} vs recorded {dice!r}"
                )

        tp_f = sens * gt_a
        fn_f = (1.0 - sens) * gt_a
        prec = (tp_f / pred_a) if pred_a > 1e-12 else (1.0 if gt_a == 0 else 0.0)
        area_ratio = (pred_a / gt_a) if gt_a > 1e-12 else float("nan")
        zero_ov = (dice == 0.0 and gt_a > 0)
        empty_p = (pred_a <= 1e-12)
        misloc_z = (zero_ov and not empty_p)

        avail_skin_frac = 1.0 - pred_a
        unprot_gt_frac = 1.0 - sens
        skin_pool_contam = fn_f / max(avail_skin_frac, 1e-6)
        circle_degen = (pred_a < 0.005) or (pred_a > 0.95)

        item = {
            "image_id": rec["image_id"],
            "gt_area": gt_a,
            "pred_area": pred_a,
            "dice": dice,
            "precision": prec,
            "sensitivity": sens,
            "area_ratio": area_ratio,
            "zero_overlap": zero_ov,
            "empty_pred": empty_p,
            "mislocalized_zero": misloc_z,
            "avail_skin_frac": avail_skin_frac,
            "unprot_gt_frac": unprot_gt_frac,
            "skin_pool_contam": skin_pool_contam,
            "circle_degen": circle_degen,
        }
        all_records.append(item)
        for sname, lo, hi in strata_defs:
            if lo <= gt_a < hi:
                strata_records[sname].append(item)
                break

    def summarize(items):
        return {
            "count": len(items),
            "dice_mean": float(np.mean([x["dice"] for x in items])),
            "precision_mean": float(np.mean([x["precision"] for x in items])),
            "sensitivity_mean": float(np.mean([x["sensitivity"] for x in items])),
            # nanmedian: area_ratio is NaN when gt_area == 0, and the "<1%" stratum
            # is defined as 0.0 <= gt < 0.01, so it admits empty ground-truth masks.
            # A single one would otherwise turn the whole cell into NaN without warning.
            "median_area_ratio_pred_gt": float(np.nanmedian([x["area_ratio"] for x in items])),
            "zero_overlap_count": int(sum(1 for x in items if x["zero_overlap"])),
            "empty_pred_count": int(sum(1 for x in items if x["empty_pred"])),
            "mislocalized_zero_count": int(sum(1 for x in items if x["mislocalized_zero"])),
            "cielab_median_avail_skin_frac": float(np.median([x["avail_skin_frac"] for x in items])),
            "cielab_p05_avail_skin_frac": float(np.percentile([x["avail_skin_frac"] for x in items], 5)),
            "cielab_mean_unprotected_lesion_frac_of_gt": float(np.mean([x["unprot_gt_frac"] for x in items])),
            "cielab_median_unprotected_lesion_frac_of_gt": float(np.median([x["unprot_gt_frac"] for x in items])),
            "cielab_p95_skin_pool_lesion_contamination": float(np.percentile([x["skin_pool_contam"] for x in items], 95)),
            "cielab_circle_degenerate_mask_count": int(sum(1 for x in items if x["circle_degen"])),
        }

    return {
        "overall_2594": summarize(all_records),
        "by_size_stratum": {sname: summarize(strata_records[sname]) for sname, _, _ in strata_defs},
    }


def main():
    root = Path("/home/jovyan/work/dermlip_slic_v8")
    p021 = root / "evaluations/paired-raw-morph-20261027/021/c0167b4735bb35a1/cruda/cases.jsonl"
    p022 = root / "evaluations/paired-raw-morph-20261027/022/02859d9e0f46fab5/cruda/cases.jsonl"

    d021 = compute_stratified(p021)
    d022 = compute_stratified(p022)

    # Save outputs
    bench_dir = root / "evaluations/ph2_val_3groups_benchmark_20261027"
    (bench_dir / "eval_021_isic2018_2594_stratified.json").write_text(json.dumps(d021, indent=2), encoding="utf-8")
    (bench_dir / "eval_022_isic2018_2594_stratified.json").write_text(json.dumps(d022, indent=2), encoding="utf-8")

    # Load seed 42 results
    d019 = json.loads((bench_dir / "eval_019_reference_isic2018_2594_stratified.json").read_text(encoding="utf-8"))
    d020 = json.loads((bench_dir / "eval_020_abl_isic2018_2594_stratified.json").read_text(encoding="utf-8"))

    strata_order = ["<1%", "1-2%", "2-5%", "5-10%", ">=10%"]

    print("\n" + "=" * 110)
    print("TABLA 1: REPRODUCCIÓN PAREADA SEMILLA 43 (021 Línea Base vs 022 Copy-Paste en Fotos Reales ISIC 2018)")
    print("=" * 110)
    print(f"{'Estrato':<8} | {'N':<4} | {'Dice 021':<9} | {'Dice 022':<9} | {'ΔDice (s43)':<11} | {'Prec 021':<9} | {'Prec 022':<9} | {'Med AR 021':<11} | {'Med AR 022':<11} | {'Ceros (021 vs 022)'}")
    print("-" * 110)

    for s in strata_order:
        r21 = d021["by_size_stratum"][s]
        r22 = d022["by_size_stratum"][s]
        delta_d = r22["dice_mean"] - r21["dice_mean"]
        z21 = f"{r21['zero_overlap_count']} ({r21['empty_pred_count']}v/{r21['mislocalized_zero_count']}m)"
        z22 = f"{r22['zero_overlap_count']} ({r22['empty_pred_count']}v/{r22['mislocalized_zero_count']}m)"
        print(f"{s:<8} | {r21['count']:<4} | {r21['dice_mean']*100:6.2f}%  | {r22['dice_mean']*100:6.2f}%  | {delta_d*100:+8.2f}%   | {r21['precision_mean']*100:6.2f}%  | {r22['precision_mean']*100:6.2f}%  | {r21['median_area_ratio_pred_gt']:6.2f}×     | {r22['median_area_ratio_pred_gt']:6.2f}×     | {z21} vs {z22}")

    o21 = d021["overall_2594"]
    o22 = d022["overall_2594"]
    d_all = o22["dice_mean"] - o21["dice_mean"]
    print("-" * 110)
    print(f"{'Global':<8} | {o21['count']:<4} | {o21['dice_mean']*100:6.2f}%  | {o22['dice_mean']*100:6.2f}%  | {d_all*100:+8.2f}%   | {o21['precision_mean']*100:6.2f}%  | {o22['precision_mean']*100:6.2f}%  | {o21['median_area_ratio_pred_gt']:6.2f}×     | {o22['median_area_ratio_pred_gt']:6.2f}×     | {o21['zero_overlap_count']} vs {o22['zero_overlap_count']}")

    print("\n" + "=" * 110)
    print("TABLA 2: INVARIANZA ESTOCÁSTICA DEL COPY-PASTE (Semilla 42 vs Semilla 43)")
    print("=" * 110)
    print(f"{'Estrato':<8} | {'ΔDice Semilla 42 (020 - 019)':<30} | {'ΔDice Semilla 43 (022 - 021)':<30} | {'Med AR 019 -> 020':<20} | {'Med AR 021 -> 022':<20}")
    print("-" * 110)
    for s in strata_order:
        d_s42 = d020["by_size_stratum"][s]["dice_mean"] - d019["by_size_stratum"][s]["dice_mean"]
        d_s43 = d022["by_size_stratum"][s]["dice_mean"] - d021["by_size_stratum"][s]["dice_mean"]
        ar_s42 = f"{d019['by_size_stratum'][s]['median_area_ratio_pred_gt']:.2f}× -> {d020['by_size_stratum'][s]['median_area_ratio_pred_gt']:.2f}×"
        ar_s43 = f"{d021['by_size_stratum'][s]['median_area_ratio_pred_gt']:.2f}× -> {d022['by_size_stratum'][s]['median_area_ratio_pred_gt']:.2f}×"
        print(f"{s:<8} | {d_s42*100:+8.2f}%                       | {d_s43*100:+8.2f}%                       | {ar_s42:<20} | {ar_s43:<20}")

    d_all_42 = d020["overall_2594"]["dice_mean"] - d019["overall_2594"]["dice_mean"]
    d_all_43 = d022["overall_2594"]["dice_mean"] - d021["overall_2594"]["dice_mean"]
    ar_all_42 = f"{d019['overall_2594']['median_area_ratio_pred_gt']:.2f}× -> {d020['overall_2594']['median_area_ratio_pred_gt']:.2f}×"
    ar_all_43 = f"{d021['overall_2594']['median_area_ratio_pred_gt']:.2f}× -> {d022['overall_2594']['median_area_ratio_pred_gt']:.2f}×"
    print("-" * 110)
    print(f"{'Global':<8} | {d_all_42*100:+8.2f}%                       | {d_all_43*100:+8.2f}%                       | {ar_all_42:<20} | {ar_all_43:<20}")


if __name__ == "__main__":
    main()
