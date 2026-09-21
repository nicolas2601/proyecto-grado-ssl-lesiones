"""Tests for scripts/cluster_frozen_bank.py.

CPU-only, no network, no real GPU. The backbone builder seam
(``construct_backbone``) is monkeypatched in the end-to-end test so no
``torch.hub`` download ever happens.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import cluster_frozen_bank as bank  # noqa: E402
from dermssl import metrics as m  # noqa: E402

# --- stratified_by_label (src/dermssl/metrics.py) --------------------------
#
# Mirrors stratified_summary's OK/SKIPPED/empty-input contract, but keyed by
# whatever labels are actually observed in a manifest column instead of the
# fixed FST bucket set.


def _multiclass_rows(n, n_classes, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, n)
    proba = rng.dirichlet(np.ones(n_classes), size=n)
    return y, proba


def test_stratified_by_label_returns_empty_dict_for_empty_input():
    out = m.stratified_by_label(np.array([]), np.zeros((0, 2)), np.array([]), n_classes=2)
    assert out == {}


def test_stratified_by_label_keys_by_observed_labels_plus_unavailable():
    y, proba = _multiclass_rows(40, 2, seed=0)
    labels = np.array(["dark"] * 20 + ["light"] * 20)
    out = m.stratified_by_label(y, proba, labels, n_classes=2, min_n=10)
    assert set(out) == {"dark", "light", "unavailable"}
    assert out["unavailable"]["status"] == "SKIPPED"
    assert out["unavailable"]["count"] == 0


@pytest.mark.parametrize(
    "count, expected_status",
    [(40, "OK"), (10, "SKIPPED")],
    ids=["populated", "sparse_skipped"],
)
def test_stratified_by_label_status_by_population(count, expected_status):
    n_classes, total = 2, 100
    y, proba = _multiclass_rows(total, n_classes, seed=1)
    labels = np.array(["dark"] * count + ["light"] * (total - count), dtype=object)

    out = m.stratified_by_label(y, proba, labels, n_classes=n_classes, min_n=30)

    assert out["dark"]["status"] == expected_status
    assert out["dark"]["count"] == count
    if expected_status == "SKIPPED":
        assert "reason" in out["dark"]
        assert "macro_f1" not in out["dark"]
    else:
        assert out["dark"]["macro_f1"] == pytest.approx(
            m.summarize(y[:count], proba[:count], n_classes)["macro_f1"]
        )


def test_stratified_by_label_omits_never_observed_labels():
    """Unlike stratified_summary's fixed FST buckets, a label absent from the
    data has no known vocabulary slot to report SKIPPED in - it is just not a key."""
    y, proba = _multiclass_rows(40, 2, seed=2)
    labels = np.array(["dark"] * 40)
    out = m.stratified_by_label(y, proba, labels, n_classes=2, min_n=30)
    assert "light" not in out


def test_stratified_by_label_empty_and_none_values_join_unavailable():
    n_classes = 2
    y, proba = _multiclass_rows(50, n_classes, seed=3)
    labels = np.array([None] * 20 + [""] * 15 + ["dark"] * 15, dtype=object)

    out = m.stratified_by_label(y, proba, labels, n_classes=n_classes, min_n=30)

    assert out["unavailable"]["status"] == "OK"
    assert out["unavailable"]["count"] == 35
    assert out["dark"]["status"] == "SKIPPED"
    assert out["dark"]["count"] == 15


def test_stratified_by_label_ok_entry_has_same_metric_keys_as_summarize():
    n_classes = 3
    y, proba = _multiclass_rows(40, n_classes, seed=5)
    labels = np.array(["medium"] * 40)

    out = m.stratified_by_label(y, proba, labels, n_classes=n_classes, min_n=30)

    expected_keys = set(m.summarize(y, proba, n_classes)) | {"status", "count"}
    assert set(out["medium"]) == expected_keys


# --- gap / worst_label semantics on top of stratified_by_label -------------


def _strata_from_metric_values(values: dict[str, float | None]) -> dict:
    """Build a stratified_by_label-shaped dict with a given macro_f1/auc per label."""
    strata = {}
    for label, value in values.items():
        if value is None:
            strata[label] = {"status": "SKIPPED", "count": 3, "reason": "n=3 < min_n=30"}
        else:
            strata[label] = {"status": "OK", "count": 40, "macro_f1": value, "auc": value + 0.1}
    return strata


def test_compute_gap_needs_at_least_two_ok_labels():
    strata = _strata_from_metric_values({"dark": 0.5, "unavailable": None})
    gap, worst = bank.compute_gap(strata, "macro_f1")
    assert gap is None
    assert worst is None


def test_compute_gap_excludes_unavailable_label():
    strata = _strata_from_metric_values({"dark": 0.9, "light": 0.5, "unavailable": 0.1})
    gap, worst = bank.compute_gap(strata, "macro_f1")
    assert gap == pytest.approx(0.4)
    assert worst == "light"


def test_compute_gap_skips_skipped_labels():
    strata = _strata_from_metric_values({"dark": 0.9, "light": 0.5, "medium": None})
    gap, worst = bank.compute_gap(strata, "macro_f1")
    assert gap == pytest.approx(0.4)
    assert worst == "light"


# --- CLI: --backbone NAME=ARCH:SOURCE parsing -------------------------------


def test_parse_backbone_spec_valid():
    assert bank.parse_backbone_spec("ssl=dinov2_vits14:public") == (
        "ssl", "dinov2_vits14", "public",
    )


def test_parse_backbone_spec_accepts_path_source():
    name, arch, source = bank.parse_backbone_spec("ssl=dinov2_vits14:/tmp/backbone.pt")
    assert (name, arch, source) == ("ssl", "dinov2_vits14", "/tmp/backbone.pt")


@pytest.mark.parametrize(
    "raw",
    ["dinov2_vits14:public", "ssl=dinov2_vits14", "=dinov2_vits14:public",
     "ssl=:public", "ssl=dinov2_vits14:", "ssl"],
)
def test_parse_backbone_spec_rejects_malformed_input(raw):
    with pytest.raises(Exception):
        bank.parse_backbone_spec(raw)


def test_build_arg_parser_parses_repeated_backbone_flags():
    parser = bank.build_arg_parser()
    args = parser.parse_args([
        "--ham-root", "/data/ham",
        "--backbone", "ssl=dinov2_vits14:public",
        "--backbone", "sup=dinov2_vitb14:/ckpt/backbone.pt",
        "--output-dir", "/tmp/out",
    ])
    assert args.backbones == [
        ("ssl", "dinov2_vits14", "public"),
        ("sup", "dinov2_vitb14", "/ckpt/backbone.pt"),
    ]
    assert args.probes == list(bank.DEFAULT_PROBES)
    assert args.fractions == list(bank.DEFAULT_FRACTIONS)
    assert args.seeds == list(bank.DEFAULT_SEEDS)
    assert args.strata_column == bank.DEFAULT_STRATA_COLUMN
    assert args.min_n == bank.DEFAULT_MIN_N
    assert args.resolution == bank.DEFAULT_RESOLUTION


def test_help_does_not_import_torch(monkeypatch):
    """--help must work in a torch-less shell; nothing in argparse setup imports it."""
    monkeypatch.setitem(sys.modules, "torch", None)
    parser = bank.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


# --- split protocol: StratifiedGroupKFold(5, shuffle=True, random_state=42) -


def _synthetic_metadata(n_per_class=8, n_classes=3, lesions_share=2):
    """A HAM10000-shaped DataFrame: image_id, lesion_id, dx.

    Every ``lesions_share`` consecutive images within a class share one
    lesion_id, so the split protocol has real grouping to respect.
    """
    import pandas as pd

    classes = [f"cls{i}" for i in range(n_classes)]
    rows = []
    counter = 0
    for cls in classes:
        for i in range(n_per_class):
            lesion = f"{cls}_lesion{i // lesions_share}"
            rows.append({"image_id": f"img{counter:04d}", "lesion_id": lesion, "dx": cls})
            counter += 1
    return pd.DataFrame(rows)


def test_build_split_reproduces_the_notebook_fold():
    from sklearn.model_selection import StratifiedGroupKFold

    meta = _synthetic_metadata()
    result = bank.build_split(meta)

    all_ids = meta.image_id.tolist()
    classes = sorted(meta.dx.unique())
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls2idx[d] for d in meta.dx])
    groups = meta.lesion_id.values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(sgkf.split(all_ids, y_all, groups))
    expected_train = [all_ids[i] for i in tr_idx]
    expected_val = [all_ids[i] for i in va_idx]

    assert result["train_ids"] == expected_train
    assert result["val_ids"] == expected_val
    assert result["classes"] == classes


def test_build_split_has_zero_lesion_overlap():
    meta = _synthetic_metadata()
    result = bank.build_split(meta)
    lookup = meta.set_index("image_id")
    train_lesions = set(lookup.loc[result["train_ids"], "lesion_id"])
    val_lesions = set(lookup.loc[result["val_ids"], "lesion_id"])
    assert train_lesions & val_lesions == set()
    assert result["overlap_lesions"] == 0


def test_filter_metadata_to_available_drops_images_missing_on_disk():
    meta = _synthetic_metadata(n_per_class=2, n_classes=2)
    image_index = {row.image_id: Path(f"/fake/{row.image_id}.jpg")
                   for row in meta.itertuples() if row.image_id != "img0000"}
    filtered = bank.filter_metadata_to_available(meta, image_index)
    assert "img0000" not in filtered.image_id.tolist()
    assert len(filtered) == len(meta) - 1


# --- tone manifest join: missing image_id / empty value -> "unavailable" ---


def test_load_tone_manifest_rows_reads_the_contract_columns(tmp_path):
    manifest_path = tmp_path / "fst_pseudo_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=bank.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "image_id": "img0001", "fst_bucket": "2", "tone_tercile": "dark",
            "status": "ok", "L_median": "40.0", "b_median": "12.0",
            "ita": "10.0", "n_skin_px": "500",
        })
    rows = bank.load_tone_manifest_rows(manifest_path)
    assert rows["img0001"]["tone_tercile"] == "dark"


def test_strata_labels_for_maps_missing_and_empty_to_unavailable():
    manifest_rows = {
        "img0001": {"tone_tercile": "dark"},
        "img0002": {"tone_tercile": ""},
        "img0003": {"tone_tercile": "light"},
    }
    val_ids = ["img0001", "img0002", "img0003", "img0004"]  # img0004 missing entirely
    labels = bank.strata_labels_for(val_ids, manifest_rows, "tone_tercile")
    assert list(labels) == ["dark", "unavailable", "light", "unavailable"]


def test_strata_labels_for_without_manifest_marks_everything_unavailable():
    labels = bank.strata_labels_for(["img0001", "img0002"], None, "tone_tercile")
    assert list(labels) == ["unavailable", "unavailable"]


# --- end-to-end smoke: tiny synthetic dataset, monkeypatched backbone ------


def _write_synthetic_ham10000(root: Path, n_per_class=8, n_classes=3, resolution=32):
    from PIL import Image

    meta = _synthetic_metadata(n_per_class=n_per_class, n_classes=n_classes)
    meta.to_csv(root / "HAM10000_metadata.csv", index=False)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for image_id in meta.image_id:
        pixels = rng.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images_dir / f"{image_id}.jpg")
    return meta


def _write_tone_manifest(root: Path, image_ids: list[str]) -> Path:
    path = root / "fst_pseudo_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=bank.MANIFEST_COLUMNS)
        writer.writeheader()
        for i, image_id in enumerate(image_ids):
            # Every third id is left out entirely to exercise "unavailable".
            if i % 3 == 0:
                continue
            writer.writerow({
                "image_id": image_id,
                "fst_bucket": "2", "tone_tercile": "dark" if i % 2 == 0 else "light",
                "status": "ok", "L_median": "40.0", "b_median": "12.0",
                "ita": "10.0", "n_skin_px": "500",
            })
    return path


def _fake_construct_backbone(arch, source, resolution, device):
    import torch.nn as nn

    return nn.Sequential(nn.Flatten(), nn.Linear(3 * resolution * resolution, 8))


def test_end_to_end_smoke_writes_consistent_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(bank, "construct_backbone", _fake_construct_backbone)

    ham_root = tmp_path / "ham"
    ham_root.mkdir()
    meta = _write_synthetic_ham10000(ham_root, n_per_class=8, n_classes=3, resolution=32)
    manifest_path = _write_tone_manifest(ham_root, meta.image_id.tolist())
    output_dir = tmp_path / "out"

    result = bank.main([
        "--ham-root", str(ham_root),
        "--backbone", "tiny=tiny_test_arch:public",
        "--tone-manifest", str(manifest_path),
        "--probes", "logreg_balanced", "knn",
        "--fractions", "1.0",
        "--seeds", "0",
        "--resolution", "32",
        "--device", "cpu",
        "--output-dir", str(output_dir),
        "--smoke",
    ])

    assert result["mode"] == "smoke"
    assert (output_dir / "bank.json").is_file()
    assert (output_dir / "bank_rows.csv").is_file()
    assert (output_dir / "strata.json").is_file()
    assert (output_dir / "bank.md").is_file()

    on_disk = json.loads((output_dir / "bank.json").read_text())
    assert on_disk["backbones"]["tiny"]["frozen_trainable_params"] == 0
    assert on_disk["backbones"]["tiny"]["dim"] == 8
    assert on_disk["split"]["overlap_lesions"] == 0
    assert len(on_disk["rows"]) > 0
    assert "tiny" in on_disk["strata"]
    assert set(on_disk["strata"]["tiny"]["labels"]) >= {"unavailable"}

    # No NaN survives to JSON.
    raw_text = (output_dir / "bank.json").read_text()
    assert "NaN" not in raw_text

    with (output_dir / "bank_rows.csv").open(newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(on_disk["rows"])


def test_end_to_end_smoke_without_tone_manifest_produces_no_strata(tmp_path, monkeypatch):
    monkeypatch.setattr(bank, "construct_backbone", _fake_construct_backbone)

    ham_root = tmp_path / "ham"
    ham_root.mkdir()
    _write_synthetic_ham10000(ham_root, n_per_class=8, n_classes=3, resolution=32)
    output_dir = tmp_path / "out"

    result = bank.main([
        "--ham-root", str(ham_root),
        "--backbone", "tiny=tiny_test_arch:public",
        "--probes", "knn",
        "--fractions", "1.0",
        "--resolution", "32",
        "--device", "cpu",
        "--output-dir", str(output_dir),
        "--smoke",
    ])

    assert result["strata"] == {}


def test_end_to_end_smoke_one_failing_probe_does_not_abort_the_grid(tmp_path, monkeypatch):
    """logreg_balanced needs scikit-learn but its class_weight kwarg is fine;
    a genuinely broken probe name still leaves the run with completed rows and
    a recorded error string instead of raising."""
    monkeypatch.setattr(bank, "construct_backbone", _fake_construct_backbone)

    ham_root = tmp_path / "ham"
    ham_root.mkdir()
    _write_synthetic_ham10000(ham_root, n_per_class=8, n_classes=3, resolution=32)
    output_dir = tmp_path / "out"

    from dermssl import probes as probes_mod

    def _broken_probe(*args, **kwargs):
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setitem(probes_mod.PROBES, "knn", _broken_probe)

    result = bank.main([
        "--ham-root", str(ham_root),
        "--backbone", "tiny=tiny_test_arch:public",
        "--probes", "knn",
        "--fractions", "1.0",
        "--resolution", "32",
        "--device", "cpu",
        "--output-dir", str(output_dir),
        "--smoke",
    ])

    assert len(result["rows"]) > 0
    assert all(row["error"] == "RuntimeError: synthetic probe failure" for row in result["rows"])
