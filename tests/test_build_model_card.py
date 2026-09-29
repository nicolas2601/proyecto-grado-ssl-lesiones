"""The release model card is generated from the committed evaluation files."""

import numpy as np
import pytest

from scripts import build_model_card as bmc


@pytest.fixture(scope="module")
def text():
    return bmc.render()  # ~10 s: 1000-resample bootstraps for five rows


def test_committed_model_card_is_fresh(text):
    assert bmc.TARGET.read_text(encoding="utf-8") == text, "run: python scripts/build_model_card.py"


def test_every_evaluated_model_has_a_row_with_intervals(text):
    for name in ("`A_s42`", "`A_s43`", "`A_s44`", "`B_s42`", "`B_s43`", "`B_s44`"):
        row = next(line for line in text.splitlines() if line.startswith(f"| {name}"))
        assert row.count("[") == len(bmc.COLUMNS)


def test_rows_read_the_json_value_not_a_copy():
    docs, probas = bmc.load()
    row = bmc.model_row("x", docs[43], probas[43], "proposed", bmc.pcm.HAM10000_CLASSES)
    assert f"{docs[43]['results']['proposed']['macro_f1']:.4f}" in row


def test_a_probability_file_that_disagrees_with_the_json_is_refused():
    docs, probas = bmc.load()
    shuffled = {**probas[42], "plain": np.roll(probas[42]["plain"], 1, axis=1)}
    try:
        bmc.model_row("x", docs[42], shuffled, "plain", bmc.pcm.HAM10000_CLASSES)
    except ValueError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("a mismatched probability file was accepted")


def test_card_states_research_use_license_and_registered_rules(text):
    for phrase in ("not a medical device", "CC BY-NC 4.0", "non-commercial",
                   "99.5%", "0.002", "PAD-UFES-20"):
        assert phrase in text, phrase


def _parity(value: float, fp16: bool):
    row = {"value": value, "evaluated": 0.6, "delta": value - 0.6, "evaluated_source": "x"}
    names = ("macro_f1", "balanced_accuracy", "accuracy", "mel_sensitivity", "mel_auc", "ece")
    block = {"n_images": 2004, "n_predictions_changed": 3, "metrics": {n: row for n in names}}
    gate = {"publish": fp16, "argmax_agreement": 0.999, "delta_macro_f1": 0.0005,
            "rule": {"min_argmax_agreement": 0.995, "max_abs_delta_macro_f1": 0.002}}
    return {"pass": True, "deployment_metrics": {"onnx_fp32": block,
                                                 "onnx_fp16": {**block, "gate": gate}}}


def test_release_card_shows_deployment_metrics_from_parity(tmp_path):
    import json

    for name in ("A_s42", "B_s43"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "parity.json").write_text(json.dumps(_parity(0.61, name == "A_s42")))
    text = bmc.render(tmp_path)
    assert "métricas de despliegue" in text
    row = next(line for line in text.splitlines() if line.startswith("| `A_s42` | ONNX fp32"))
    assert "0.6100 (+0.0100)" in row and row.rstrip().endswith("| 3 |")
    assert "| `A_s42` | ONNX fp16 (published) |" in text
    assert "| `B_s43` | ONNX fp16 (not published) |" in text


def test_committed_card_explains_where_deployment_metrics_appear(text):
    assert "métricas de despliegue" in text and "parity.json" in text


def test_licencias_section_names_both_license_files_and_no_open_flags(text):
    assert "## Licencias" in text
    for phrase in ("LICENSE-WEIGHTS", "Apache-2.0", "ISIC 2019", "ISIC 2020",
                   "challenge.isic-archive.com/data", "doi:10.7910/DVN/DBW86T"):
        assert phrase in text, phrase
    assert "verify before publishing" not in text


def _reading_doc(p3ic: str, p2: str, a_f1: float, a_delta: float) -> dict:
    return {"results": {"plain": {"macro_f1": a_f1}},
            "predictions": {"P3-IC": {"verdict": p3ic}, "P2": {"verdict": p2},
                            "P1": {"evidence": {"macro_f1": {"point_delta": a_delta}}}}}


def test_the_reading_counts_the_seeds_it_is_given():
    docs = {42: _reading_doc("cumplida", "no cumplida", 0.60, 0.02),
            43: _reading_doc("cumplida", "no cumplida", 0.64, 0.06),
            44: _reading_doc("no cumplida", "no cumplida", 0.62, 0.04)}
    text = bmc.reading(docs)
    assert "B\\* beats public DINOv2 in macro-F1 with 2 of 3 seeds" in text
    assert "adds nothing measurable over A\\* with any seed" in text
    assert "spans 0.0400 across seeds (max - min)" in text and "+0.0400" in text
    docs[44] = _reading_doc("cumplida", "cumplida", 0.62, 0.04)
    text = bmc.reading(docs)
    assert "with all three seeds" in text and "P2 met with 1 of 3 seeds" in text


def test_committed_card_carries_the_generated_reading(text):
    docs, _ = bmc.load()
    assert bmc.reading(docs) in text and "both seeds" not in text


def test_card_lists_the_released_models_and_marks_robustness_seeds(text):
    what = next(line for line in text.splitlines() if "research models, one per arm" in line)
    assert what.startswith("Four research models")
    assert "`A_s42`, `A_s43` (A\\*" in what and "`B_s42`, `B_s43` (B\\*" in what
    assert "s44" not in what
    for name in ("`A_s44`", "`B_s44`"):
        row = next(line for line in text.splitlines() if line.startswith(f"| {name}"))
        assert "not released" in row
    for name in ("`A_s42`", "`B_s43`"):
        row = next(line for line in text.splitlines() if line.startswith(f"| {name}"))
        assert "not released" not in row
