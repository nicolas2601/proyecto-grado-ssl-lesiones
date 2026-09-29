"""Corrida A/B (sin GPU): comparar estimate_fst_whole_image contra
estimate_fst sobre imagenes reales de disco - a diferencia de
kaggle_fst_manifest.py (torch, solo static tests), este script es pura
numpy y se puede ejecutar de verdad en el test suite."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from env_sidecar import sidecar_env  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import diagnostico_fst_comparativa as dfc  # noqa: E402


def _write_disk_image(path: Path, size=64, radius=15,
                       skin_rgb=(0.85, 0.65, 0.60), lesion_rgb=(0.30, 0.20, 0.10)):
    """Mismo generador sintetico que tests/test_fitzpatrick.py::disk_srgb_image,
    pero escrito a disco como jpg - lo que este script realmente lee."""
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.empty((size, size, 3))
    img[:] = skin_rgb
    img[dist <= radius] = lesion_rgb
    arr = (img * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_find_images_only_matches_known_extensions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    found = dfc.find_images(tmp_path)

    assert [p.name for p in found] == ["a.jpg", "b.png"]


def test_find_images_dedupes_by_image_id(tmp_path):
    """El HAM10000 real en Kaggle trae las mismas imagenes dos veces, bajo
    carpetas con distinto casing (HAM10000_images_part_1 vs
    ham10000_images_part_1 - indistinguibles en un filesystem case-
    insensitive como el de Windows, por eso este test usa nombres de
    carpeta distintos para simular el mismo stem repetido sin depender de
    casing). find_images debe dedupar por image_id (el stem), quedandose
    con la primera ruta en orden alfabetico."""
    (tmp_path / "carpeta_a").mkdir()
    (tmp_path / "carpeta_b").mkdir()
    (tmp_path / "carpeta_a" / "ISIC_0000001.jpg").write_bytes(b"")
    (tmp_path / "carpeta_b" / "ISIC_0000001.jpg").write_bytes(b"")
    (tmp_path / "carpeta_a" / "ISIC_0000002.jpg").write_bytes(b"")

    found = dfc.find_images(tmp_path)

    assert [dfc.image_id(p) for p in found] == ["ISIC_0000001", "ISIC_0000002"]
    # se queda con la ruta de "carpeta_a" (gana en orden alfabetico de la ruta completa)
    assert found[0].parent.name == "carpeta_a"


def test_find_images_is_recursive_like_ham10000s_part_1_part_2_layout(tmp_path):
    """HAM10000 en Kaggle reparte las imagenes en subcarpetas
    (HAM10000_images_part_1/, part_2/) - find_images tiene que bajar ahi,
    no solo mirar el nivel superior de --image-root."""
    (tmp_path / "part_1").mkdir()
    (tmp_path / "part_2").mkdir()
    (tmp_path / "part_1" / "img_a.jpg").write_bytes(b"")
    (tmp_path / "part_2" / "img_b.jpg").write_bytes(b"")

    found = dfc.find_images(tmp_path)

    assert [p.name for p in found] == ["img_a.jpg", "img_b.jpg"]


def test_run_writes_csv_with_both_corridas_for_each_image(tmp_path):
    _write_disk_image(tmp_path / "img_ok.jpg")
    out_csv = tmp_path / "out.csv"
    args = dfc.parse_args([
        "--image-root", str(tmp_path),
        "--dataset-id", "smoke_dataset",
        "--output-csv", str(out_csv),
    ])

    rows = dfc.run(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset_id"] == "smoke_dataset"
    assert row["image_id"] == "img_ok"
    assert row["status_sin_mascara"] == "ok"
    assert row["status_con_mascara"] == "ok"
    assert row["bucket_sin_mascara"] != ""
    assert row["bucket_con_mascara"] != ""
    assert isinstance(row["bucket_cambio"], bool)

    with out_csv.open(encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 1
    assert set(written[0]) == set(dfc.MANIFEST_COLUMNS)


def test_run_diverges_between_corridas_when_lesion_dominates(tmp_path):
    """Misma imagen de lesion grande que
    test_estimate_fst_whole_image_diverges_from_masked_when_lesion_dominates
    en test_fitzpatrick.py - acá via el CLI de punta a punta."""
    _write_disk_image(tmp_path / "lesion_grande.jpg", radius=28)
    args = dfc.parse_args([
        "--image-root", str(tmp_path),
        "--dataset-id", "smoke_dataset",
        "--output-csv", str(tmp_path / "out.csv"),
    ])

    rows = dfc.run(args)

    assert rows[0]["bucket_cambio"] is True


def test_smoke_flag_caps_the_number_of_images(tmp_path):
    for i in range(dfc.SMOKE_MAX_IMAGES + 5):
        _write_disk_image(tmp_path / f"img_{i:03d}.jpg")
    args = dfc.parse_args([
        "--image-root", str(tmp_path),
        "--dataset-id", "smoke_dataset",
        "--output-csv", str(tmp_path / "out.csv"),
        "--smoke",
    ])

    rows = dfc.run(args)

    assert len(rows) == dfc.SMOKE_MAX_IMAGES


def test_run_raises_when_no_images_found(tmp_path):
    args = dfc.parse_args([
        "--image-root", str(tmp_path),
        "--dataset-id", "vacio",
        "--output-csv", str(tmp_path / "out.csv"),
    ])

    with pytest.raises(SystemExit):
        dfc.run(args)


def test_summary_dict_matches_the_projects_results_json_shape():
    """Mismo espiritu que results_{modelo}.json en build_baseline_tests.py:
    metadata (dataset/n/smoke) + metricas, serializable directo con
    json.dumps."""
    rows = [
        dfc.build_row("d", Path("a.jpg"),
                       whole=_ok(bucket=1), masked=_ok(bucket=1)),
        dfc.build_row("d", Path("b.jpg"),
                       whole=_ok(bucket=2), masked=_ok(bucket=3)),
    ]

    summary = dfc.summary_dict("d", rows, smoke=True)

    assert summary["dataset"] == "d"
    assert summary["n"] == 2
    assert summary["smoke"] is True
    assert summary["status_sin_mascara"] == {"ok": 2}
    assert summary["buckets_sin_mascara"] == {1: 1, 2: 1}
    assert summary["comparables"] == 2
    assert summary["bucket_cambio"] == 1
    assert summary["bucket_cambio_pct"] == 50.0
    json.dumps(summary)  # no debe fallar por tipos no serializables


def test_summary_dict_includes_the_transition_matrix():
    rows = [
        dfc.build_row("d", Path("a.jpg"), whole=_ok(bucket=1), masked=_ok(bucket=1)),
        dfc.build_row("d", Path("b.jpg"), whole=_ok(bucket=1), masked=_ok(bucket=2)),
        dfc.build_row("d", Path("c.jpg"), whole=_ok(bucket=1), masked=_ok(bucket=2)),
    ]

    summary = dfc.summary_dict("d", rows)

    assert summary["transicion_buckets"] == {"1": {"1": 1, "2": 2}}


def test_summary_dict_bootstrap_ci_is_reproducible_and_brackets_the_point_estimate():
    rows = [dfc.build_row("d", Path(f"{i}.jpg"),
                           whole=_ok(bucket=1), masked=_ok(bucket=2 if i < 3 else 1))
            for i in range(10)]  # 3/10 = 30% cambia

    summary_a = dfc.summary_dict("d", rows)
    summary_b = dfc.summary_dict("d", rows)

    ci = summary_a["bucket_cambio_ic95_bootstrap"]
    assert summary_a["bucket_cambio_ic95_bootstrap"] == summary_b["bucket_cambio_ic95_bootstrap"]
    assert ci[0] <= 30.0 <= ci[1]


def test_summary_dict_artifact_breakdown_isolates_artifact_rows():
    from dermssl.fitzpatrick import FSTEstimate

    def estimate(bucket, artifact):
        return FSTEstimate(bucket=bucket, ita=20.0, skin_fraction=1.0, status="ok",
                            fallback_used=False, confidence=1.0, artifact_detected=artifact)

    rows = [
        dfc.build_row("d", Path("con_artefacto_cambia.jpg"),
                       whole=estimate(1, True), masked=estimate(2, False)),
        dfc.build_row("d", Path("sin_artefacto_no_cambia.jpg"),
                       whole=estimate(1, False), masked=estimate(1, False)),
    ]

    breakdown = dfc.summary_dict("d", rows)["bucket_cambio_por_artefacto"]

    assert breakdown["con_artefacto"] == {"total": 1, "cambia": 1, "pct": 100.0}
    assert breakdown["sin_artefacto"] == {"total": 1, "cambia": 0, "pct": 0.0}


def test_write_summary_json_writes_valid_indented_json(tmp_path):
    summary = {"dataset": "d", "n": 1}
    out = tmp_path / "resumen.json"

    dfc.write_summary_json(summary, out)

    assert json.loads(out.read_text(encoding="utf-8")) == summary
    assert out.read_text(encoding="utf-8").startswith("{\n")  # indent=2, no un solo renglon


def test_run_also_writes_a_summary_json_next_to_the_csv(tmp_path):
    _write_disk_image(tmp_path / "img.jpg")
    out_csv = tmp_path / "out.csv"
    args = dfc.parse_args([
        "--image-root", str(tmp_path),
        "--dataset-id", "d",
        "--output-csv", str(out_csv),
    ])

    dfc.run(args)

    out_json = out_csv.with_suffix(".json")
    assert out_json.is_file()
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["dataset"] == "d"
    assert summary["n"] == 1
    assert sidecar_env(out_json)["schema"] == 1


def test_run_defaults_to_json_primary_without_creating_a_csv(tmp_path, monkeypatch):
    _write_disk_image(tmp_path / "img.jpg")
    monkeypatch.chdir(tmp_path)
    out_json = tmp_path / "diagnostic.json"
    args = dfc.parse_args([
        "--image-root", str(tmp_path),
        "--dataset-id", "d",
        "--output-json", str(out_json),
    ])

    dfc.run(args)

    assert out_json.is_file()
    assert list(tmp_path.glob("*.csv")) == []


def _ok(bucket):
    from dermssl.fitzpatrick import FSTEstimate
    return FSTEstimate(bucket=bucket, ita=20.0, skin_fraction=1.0, status="ok",
                        fallback_used=False, confidence=1.0, artifact_detected=False)


def test_build_row_leaves_bucket_cambio_blank_when_a_corrida_fails():
    """Si cualquiera de las dos corridas no llega a "ok", bucket_cambio no
    debe comparar None contra un numero - eso mezclaria "cambio real de
    bucket" con "una corrida fallo", dos cosas distintas para el resumen."""
    from dermssl.fitzpatrick import FSTEstimate

    ok = FSTEstimate(bucket=3, ita=20.0, skin_fraction=1.0, status="ok",
                      fallback_used=False, confidence=1.0, artifact_detected=False)
    failed = FSTEstimate(bucket=None, ita=float("nan"), skin_fraction=0.0,
                          status="no_skin", fallback_used=True, confidence=0.0,
                          artifact_detected=False)

    row = dfc.build_row("d", Path("x.jpg"), whole=ok, masked=failed)

    assert row["bucket_cambio"] == ""


def test_build_diagnostic_manifest_keeps_full_and_masked_unavailable_explicit():
    """The JSON-primary report must never turn an unavailable FST result
    into an empty CSV-only value; both estimator paths remain auditable."""
    from dermssl.fitzpatrick import FSTEstimate

    whole = FSTEstimate(
        bucket=2,
        ita=48.0,
        skin_fraction=1.0,
        status="ok",
        fallback_used=False,
        confidence=1.0,
        artifact_detected=False,
    )
    masked = FSTEstimate(
        bucket=None,
        ita=float("nan"),
        skin_fraction=0.0,
        status="mask_ineligible",
        fallback_used=False,
        confidence=0.0,
        artifact_detected=False,
        mask_producer="unet",
        mask_version="checkpoint-a",
        mask_sha256="a" * 64,
    )
    row = dfc.build_row("ham10000", Path("ISIC_1.jpg"), whole, masked)

    manifest = dfc.build_diagnostic_manifest("ham10000", [row])

    assert manifest["schema_version"] == "1.0"
    assert manifest["records"][0]["whole_image"]["bucket"] == 2
    assert manifest["records"][0]["masked"]["bucket"] == "unavailable"
    assert manifest["records"][0]["masked"]["status"] == "mask_ineligible"
    assert manifest["records"][0]["masked"]["mask_provenance"] == {
        "producer": "unet",
        "version": "checkpoint-a",
        "sha256": "a" * 64,
    }


def test_build_diagnostic_manifest_keeps_a_measured_masked_bucket_numeric():
    row = dfc.build_row("ham10000", Path("ISIC_2.jpg"), _ok(2), _ok(3))

    manifest = dfc.build_diagnostic_manifest("ham10000", [row])

    assert manifest["records"][0]["whole_image"]["bucket"] == 2
    assert manifest["records"][0]["masked"]["bucket"] == 3
    assert manifest["records"][0]["masked"]["status"] == "ok"


def test_diagnostic_manifest_serializes_unavailable_ita_as_valid_json_null():
    """json.dumps serializa float('nan') como el token literal `NaN`, que no
    es JSON valido (RFC 8259) - rompe cualquier parser estricto (el visor de
    JupyterLab, JSON.parse en el navegador, jq, etc.), aunque Python lo
    pueda releer sin quejarse. Un campo `ita` no disponible debe viajar como
    `null`, igual que ya se hace con `bucket` -> "unavailable"."""
    from dermssl.fitzpatrick import FSTEstimate

    masked = FSTEstimate(bucket=None, ita=float("nan"), skin_fraction=0.0,
                          status="degenerate_mask", fallback_used=True,
                          confidence=0.0, artifact_detected=False)
    row = dfc.build_row("d", Path("x.jpg"), whole=_ok(2), masked=masked)

    manifest = dfc.build_diagnostic_manifest("d", [row])
    dumped = json.dumps(manifest)

    assert manifest["records"][0]["masked"]["ita"] is None
    assert "NaN" not in dumped
    assert json.loads(dumped)["records"][0]["masked"]["ita"] is None
