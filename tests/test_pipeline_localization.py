"""Contratos de localización de las interfaces visibles del pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import diagnostico_fst_comparativa as diagnostico  # noqa: E402
import generate_tone_transforms  # noqa: E402
import kaggle_circle_stage2  # noqa: E402
import kaggle_fst_manifest  # noqa: E402
import kaggle_push  # noqa: E402
import kaggle_run_bank  # noqa: E402
import pretrain  # noqa: E402
from dermssl import circle  # noqa: E402
from dermssl.pretrain import config  # noqa: E402


def test_pretrain_help_uses_spanish_for_visible_guidance(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pretrain.py", "--help"])

    with pytest.raises(SystemExit, match="0"):
        pretrain.parse_args()

    help_text = capsys.readouterr().out
    assert "Ejecuta un preentrenamiento SSL" in help_text
    assert "presupuesto reducido" in help_text.lower()
    assert "\nUna GPU:\n" in help_text


def test_diagnostic_help_uses_spanish_for_visible_guidance(capsys):
    with pytest.raises(SystemExit, match="0"):
        diagnostico.parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "Diagnóstico A/B" in help_text
    assert "Carpeta con las imágenes" in help_text


def test_diagnostic_console_messages_are_in_spanish(tmp_path, capsys):
    output_json = tmp_path / "diagnostico.json"
    image = np.full((64, 64, 3), (220, 170, 150), dtype=np.uint8)
    image[20:44, 20:44] = (70, 45, 30)
    Image.fromarray(image).save(tmp_path / "muestra.jpg")

    diagnostico.run(
        diagnostico.parse_args(
            [
                "--image-root",
                str(tmp_path),
                "--dataset-id",
                "muestra",
                "--output-json",
                str(output_json),
            ]
        )
    )

    assert "Se escribieron 1 registros JSON de comparación FST" in capsys.readouterr().out


def test_validation_errors_are_translated_without_changing_contracts():
    with pytest.raises(ValueError, match="método desconocido"):
        config.validate(
            config.PretrainConfig(
                method="invalid",
                image_root="images",
                output_dir="out",
                split_hash="a" * 64,
            )
        )

    with pytest.raises(ValueError, match="máscara explícita"):
        circle.shift_skin_tone(np.zeros((2, 2, 3)), None, 1)


@pytest.mark.parametrize(
    ("parser", "guidance", "uses_shared_parser"),
    [
        (kaggle_push.main, "Usuario de Kaggle", False),
        (kaggle_run_bank.main, "Muestra el estado", True),
        (kaggle_fst_manifest.parse_args, "Imágenes de entrenamiento", True),
        (generate_tone_transforms.parse_args, "Carpeta de imágenes originales", True),
        (kaggle_circle_stage2.parse_args, "Paquete de características", True),
        (pretrain.parse_args, "Método SSL que se ejecutará", True),
    ],
)
def test_kaggle_utilities_help_uses_spanish_for_visible_guidance(
    parser, guidance, uses_shared_parser, capsys
):
    with pytest.raises(SystemExit, match="0"):
        parser(["--help"])

    help_text = capsys.readouterr().out
    assert guidance in help_text
    if uses_shared_parser:
        assert "opciones" in help_text
        assert "muestra esta ayuda y termina" in help_text


@pytest.mark.parametrize(
    "parser",
    [
        kaggle_run_bank.main,
        kaggle_fst_manifest.parse_args,
        generate_tone_transforms.parse_args,
        kaggle_circle_stage2.parse_args,
        pretrain.parse_args,
    ],
)
def test_shared_cli_parsers_localize_usage_and_required_errors(parser, capsys):
    with pytest.raises(SystemExit, match="2"):
        parser([])

    error_text = capsys.readouterr().err
    assert error_text.startswith("uso:")
    assert "se requieren los siguientes argumentos" in error_text


def test_stage2_skipped_result_uses_a_spanish_reason(tmp_path):
    stage1_gate = tmp_path / "stage1.json"
    feature_bundle = tmp_path / "features.npz"
    output = tmp_path / "stage2.json"
    stage1_gate.write_text('{"status": "PASS"}', encoding="utf-8")
    feature_bundle.write_bytes(b"")

    kaggle_circle_stage2.main(
        [
            "--feature-bundle",
            str(feature_bundle),
            "--stage1-gate",
            str(stage1_gate),
            "--output-json",
            str(output),
        ]
    )

    assert "sesgo medible" in output.read_text(encoding="utf-8")


def test_kaggle_bank_report_uses_spanish_state_labels(tmp_path, monkeypatch, capsys):
    state_file = tmp_path / "state.json"
    state_file.write_text('{"alexnet": "pending"}', encoding="utf-8")
    monkeypatch.setattr(kaggle_run_bank, "status", lambda _slug: "COMPLETE")

    kaggle_run_bank.report("baselines", ["alexnet"], "usuario", state_file)

    report = capsys.readouterr().out
    assert "estado=pendiente" in report
    assert "kaggle=completado" in report


def test_fst_manifest_legacy_invocation_keeps_its_default_csv_output():
    args = kaggle_fst_manifest.parse_args(
        [
            "--unet-train-images",
            "train-images",
            "--unet-train-masks",
            "train-masks",
            "--target-image-root",
            "target-images",
        ]
    )

    assert args.output_csv == "fst_manifest.csv"
    assert args.output_json == "fst_manifest.json"


def test_pretrain_legacy_invocation_works_without_split_hash():
    args = pretrain.parse_args(
        [
            "--method",
            "simsiam",
            "--image-root",
            "images",
            "--output-dir",
            "output",
        ]
    )

    cfg = config.validate(pretrain.build_config(args))

    assert cfg.split_hash == pretrain.LEGACY_SPLIT_HASH


def test_pretrain_transformed_invocation_requires_split_hash():
    args = pretrain.parse_args(
        [
            "--method",
            "simsiam",
            "--image-root",
            "images",
            "--output-dir",
            "output",
            "--transformed-image-root",
            "transformed-images",
        ]
    )

    with pytest.raises(ValueError, match="split_hash"):
        pretrain.build_config(args)
