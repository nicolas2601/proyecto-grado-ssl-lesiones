"""Reader-facing terminology must not rename machine identifiers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_proposed_versions_have_descriptive_names():
    from dermssl.display_nomenclature import display_text

    assert display_text("B71 + F4 frente a B71") == "Proposed con segmentación frente a Proposed sin segmentación"


def test_segmentation_rounds_and_control_are_not_extra_proposals():
    from dermssl.display_nomenclature import display_text

    assert display_text("student_r3 frente a A71") == (
        "experimento de segmentación U-Net, ronda final 3 frente a control SSL sin consistencia de tono"
    )


def test_paths_keys_commands_and_metrics_are_preserved():
    from dermssl.display_nomenclature import display_text

    text = "`B71__clean` `student_r3/best.pt` [source](B71/README.md) F1 macro 0,7141"
    assert display_text(text) == text


def test_experiment_table_labels_are_readable_without_changing_f1_metric():
    from dermssl.display_nomenclature import display_text

    assert display_text("| F4 | F1 macro | D0 |") == (
        "| Proposed con segmentación (experimental) | F1 macro | referencia pública sin adaptación |"
    )


def test_sentence_punctuation_and_paired_names_are_translated():
    from dermssl.display_nomenclature import display_text

    assert display_text("B71.") == "Proposed sin segmentación."
    assert display_text("B71/A71") == "Proposed sin segmentación / control SSL sin consistencia de tono"


def test_segmentation_context_disambiguates_a2_without_renaming_classification_pilot():
    from dermssl.display_nomenclature import segmentation_text

    assert segmentation_text("A2 frente a A2-s43") == (
        "experimento de segmentación DINOv2 con pseudo-máscaras CCAM frente a "
        "experimento de segmentación DINOv2 con pseudo-máscaras CCAM, semilla 43"
    )
    assert segmentation_text("`a2/best.pt` `A2`") == "`a2/best.pt` `A2`"


def test_variant_codes_are_descriptive_in_table_cells_but_f1_metric_stays_f1():
    from dermssl.display_nomenclature import display_text

    assert display_text("| F1 | F1 macro | F2 |") == (
        "| experimento con imagen completa y recorte | F1 macro | experimento con recorte de resolución original |"
    )


def test_display_translation_is_idempotent():
    from dermssl.display_nomenclature import display_text

    text = "B71-limpio y student_r3; B71+F4"
    assert display_text(display_text(text)) == display_text(text)
