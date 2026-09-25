from __future__ import annotations

from pathlib import Path

import pytest

from demo.backend.report import build_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_report_reads_the_latest_sealed_classification_bank() -> None:
    report = build_report(REPO_ROOT)

    proposed = report["classification"]["ssl_proposed_vitb14"]
    plain = report["classification"]["ssl_plain_vitb14"]
    public = report["classification"]["dinov2_public_vitb14"]

    assert proposed["macro_f1"] == pytest.approx(0.5237, abs=1e-4)
    assert plain["macro_f1"] == pytest.approx(0.5856, abs=1e-4)
    assert public["macro_f1"] == pytest.approx(0.6158, abs=1e-4)
    assert report["classification_verdict"]["global"] == "loss"
    assert report["source"]["bank"] == "resultados/clasificacion/bank.json"


def test_report_does_not_promote_preliminary_tone_result() -> None:
    report = build_report(REPO_ROOT)
    text = " ".join(report["claims"]["wins"] + report["claims"]["losses"])

    assert "5/6" not in text
    assert "intervalos" in text.lower()
    assert report["claims"]["losses"]


def test_report_exposes_improvement_plan_from_sealed_result() -> None:
    report = build_report(REPO_ROOT)
    steps = report["next_steps"]

    assert steps[0]["title"] == "Separar las señales del objetivo"
    assert any("tres vistas" in step["detail"].lower() for step in steps)
    assert any("validaci" in step["detail"].lower() for step in steps)


def test_report_includes_all_site_data_surfaces() -> None:
    report = build_report(REPO_ROOT)

    assert len(report["training"]["arms"]) == 4
    assert len(report["contrasts"]) == 24
    assert report["classification_meta"]["headline_probe"] == "logreg"
    assert report["downloads"]["release_tag"] == "proposed-cluster-v1"
    assert report["site_meta"]["branch"] == "proposed-cluster"
