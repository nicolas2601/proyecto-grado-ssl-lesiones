"""Contratos puros del ledger y la ayuda compartida del flujo Kaggle."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.cli import build_spanish_parser  # noqa: E402
from dermssl.experiment_contract import (  # noqa: E402
    new_ledger,
    record_stage,
    validate_ledger,
)


def test_spanish_parser_replaces_argparse_default_help(capsys):
    parser = build_spanish_parser(description="Flujo de prueba")

    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "opciones" in help_text
    assert "muestra esta ayuda y termina" in help_text


def test_recorded_stage_has_the_required_schema_and_artifact_hashes(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text("entrada", encoding="utf-8")
    output_path.write_text("salida", encoding="utf-8")
    ledger = new_ledger("run-1", "a" * 64)

    stage = record_stage(
        ledger,
        name="stage1",
        status="PASS",
        artifacts={"input": input_path, "output": output_path},
    )

    assert stage == ledger["stages"][0]
    assert stage["schema_version"] == "1.0"
    assert stage["run_id"] == "run-1"
    assert stage["input_sha256"] == hashlib.sha256(b"entrada").hexdigest()
    assert stage["output_sha256"] == hashlib.sha256(b"salida").hexdigest()
    assert stage["status"] == "PASS"
    assert stage["valid_for_thesis"] is True
    validate_ledger(ledger)


def test_record_stage_rejects_unknown_status():
    ledger = new_ledger("run-1", "a" * 64)

    with pytest.raises(ValueError, match="estado"):
        record_stage(ledger, name="stage1", status="UNKNOWN", artifacts={})


def test_ledger_rejects_stage2_without_failed_stage1():
    ledger = new_ledger("run-1", "a" * 64)
    record_stage(ledger, name="stage1", status="PASS", artifacts={})
    record_stage(ledger, name="stage2", status="COMPLETE", artifacts={})

    with pytest.raises(ValueError, match="Stage 2"):
        validate_ledger(ledger)


def test_ledger_rejects_stage2_recorded_before_later_stage1_failure():
    ledger = new_ledger("run-1", "a" * 64)
    record_stage(ledger, name="stage2", status="COMPLETE", artifacts={})
    record_stage(ledger, name="stage1", status="FAIL", artifacts={})

    with pytest.raises(ValueError, match="Stage 2"):
        validate_ledger(ledger)


def test_ledger_accepts_completed_and_skipped_stage2_statuses_after_stage1_failure():
    for status, canonical_status, valid_for_thesis in (
        ("completed", "COMPLETE", True),
        ("skipped", "SKIPPED", False),
    ):
        ledger = new_ledger("run-1", "a" * 64)
        record_stage(ledger, name="stage1", status="FAIL", artifacts={})

        stage = record_stage(ledger, name="stage2", status=status, artifacts={})

        assert stage["status"] == canonical_status
        assert stage["valid_for_thesis"] is valid_for_thesis
        validate_ledger(ledger)


def test_record_stage_rejects_noncanonical_stage_name_that_bypasses_stage2_gate():
    ledger = new_ledger("run-1", "a" * 64)

    with pytest.raises(ValueError, match="nombre"):
        record_stage(ledger, name="stage_2", status="completed", artifacts={})


def test_smoke_stage_is_not_valid_for_thesis():
    ledger = new_ledger("run-1", "a" * 64)

    stage = record_stage(ledger, name="stage1", status="SMOKE", artifacts={})

    assert stage["valid_for_thesis"] is False
    validate_ledger(ledger)


def test_spanish_parser_localizes_usage_and_required_argument_errors(capsys):
    parser = build_spanish_parser(description="Flujo de prueba")
    parser.add_argument("--entrada", required=True)

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([])

    error_text = capsys.readouterr().err
    assert error_text.startswith("uso:")
    assert "se requieren los siguientes argumentos: --entrada" in error_text
