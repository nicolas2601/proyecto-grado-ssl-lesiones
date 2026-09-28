"""Records with an artifact or confidence < 1 are reported apart (docs/26 §6.4)."""

import numpy as np
from PIL import Image

import seg_paths  # noqa: F401
import fst_pseudo_manifest as fpm
from dermssl import fitzpatrick

ORIGINAL_SUMMARY_KEYS = {"n", "states", "ITA", "FST_distribution", "tone_distribution"}


def _write_pool(tmp_path, image_ids):
    lines = ["index,dataset,image_id,path,split"]
    for index, image_id in enumerate(image_ids):
        path = tmp_path / "images" / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (180, 130, 100)).save(path)
        lines.append(f"{index},ham10000,{image_id},{path},train")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    masks = np.zeros((len(image_ids), 8, 8), dtype=np.uint8)
    masks[:, 2:5, 2:5] = 1
    masks_path = tmp_path / "pseudo_masks.npy"
    np.save(masks_path, masks)
    return manifest_path, masks_path


def _estimate(*, artifact, confidence, L_median, status="ok"):
    return fitzpatrick.FSTEstimate(
        bucket=2 if status == "ok" else None,
        ita=45.0 if status == "ok" else float("nan"),
        skin_fraction=0.8,
        status=status,
        fallback_used=False,
        confidence=confidence,
        artifact_detected=artifact,
        L_median=L_median,
        b_median=12.0 if status == "ok" else 1.0,
        n_skin_px=50,
    )


# img-art: hair-like artifact; img-clean: clean; img-indet: no bucket (confidence 0).
ESTIMATES = {
    "img-art": _estimate(artifact=True, confidence=0.7, L_median=60.0),
    "img-clean": _estimate(artifact=False, confidence=1.0, L_median=70.0),
    "img-indet": _estimate(
        artifact=False, confidence=0.0, L_median=80.0, status=fitzpatrick.ITA_INDETERMINATE
    ),
}


def _run_patched(tmp_path, monkeypatch):
    manifest_path, masks_path = _write_pool(tmp_path, list(ESTIMATES))

    def fake(image, *, mask, mask_record, border_mask=None):
        return ESTIMATES[mask_record["image_id"]]

    monkeypatch.setattr(fpm.fitzpatrick, "estimate_fst_from_mask_record", fake)
    return fpm.run(
        fpm.parse_args(
            ["--manifest", str(manifest_path), "--pseudo-masks", str(masks_path),
             "--output-json", str(tmp_path / "fst.json")]
        )
    )


def test_records_carry_the_artifact_flag(tmp_path, monkeypatch):
    output = _run_patched(tmp_path, monkeypatch)
    flags = {row["image_id"]: row["artifact_detected"] for row in output["records"]}
    assert flags == {"img-art": True, "img-clean": False, "img-indet": False}


def test_a_quarantined_mask_has_no_artifact_flag_raised(tmp_path):
    manifest_path, masks_path = _write_pool(tmp_path, ["img-a"])
    np.save(masks_path, np.zeros((1, 8, 8), dtype=np.uint8))
    output = fpm.run(
        fpm.parse_args(
            ["--manifest", str(manifest_path), "--pseudo-masks", str(masks_path),
             "--output-json", str(tmp_path / "fst.json")]
        )
    )
    assert output["records"][0]["artifact_detected"] is False


def test_flagged_records_are_counted_with_their_own_distributions(tmp_path, monkeypatch):
    summary = _run_patched(tmp_path, monkeypatch)["summary"]["by_dataset"]["ham10000"]
    flagged = summary["flagged"]
    assert flagged["rule"] == "artifact_detected or confidence < 1"
    assert flagged["n"] == 2
    assert flagged["n_artifact_detected"] == 1
    assert flagged["n_confidence_below_1"] == 2
    assert flagged["states"] == {"ita_indeterminate": 1, "ok": 1}
    assert flagged["FST_distribution"] == {"2": 1}
    assert flagged["tone_distribution"] == {"dark": 1, "light": 1}


def test_artifact_records_get_their_own_block(tmp_path, monkeypatch):
    summary = _run_patched(tmp_path, monkeypatch)["summary"]["by_dataset"]["ham10000"]
    artifact = summary["flagged"]["artifact_detected"]
    assert artifact == {"n": 1, "FST_distribution": {"2": 1}, "tone_distribution": {"dark": 1}}


def test_the_overall_distributions_still_count_every_record(tmp_path, monkeypatch):
    summary = _run_patched(tmp_path, monkeypatch)["summary"]["by_dataset"]["ham10000"]
    assert ORIGINAL_SUMMARY_KEYS <= set(summary)
    assert summary["n"] == 3
    assert summary["FST_distribution"] == {"2": 2}
    assert summary["tone_distribution"] == {"dark": 1, "medium": 1, "light": 1}
