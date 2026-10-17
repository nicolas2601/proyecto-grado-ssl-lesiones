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
        "Experimento de segmentación 3 frente a Modelo de control"
    )


def test_paths_keys_commands_and_metrics_are_preserved():
    from dermssl.display_nomenclature import display_text

    text = "`B71__clean` `student_r3/best.pt` [source](B71/README.md) F1 macro 0,7141"
    assert display_text(text) == text


def test_experiment_table_labels_are_readable_without_changing_f1_metric():
    from dermssl.display_nomenclature import display_text

    assert display_text("| F4 | F1 macro | D0 |") == (
        "| Proposed con segmentación (experimental) | F1 macro | Modelo de referencia |"
    )


def test_sentence_punctuation_and_paired_names_are_translated():
    from dermssl.display_nomenclature import display_text

    assert display_text("B71.") == "Proposed sin segmentación."
    assert display_text("B71/A71") == "Proposed sin segmentación / Modelo de control"


def test_segmentation_context_disambiguates_a2_without_renaming_classification_pilot():
    from dermssl.display_nomenclature import segmentation_text

    assert segmentation_text("A2 frente a A2-s43") == (
        "Experimento de segmentación 5 frente a "
        "Experimento de segmentación 5, repetición 43"
    )
    assert segmentation_text("`a2/best.pt` `A2`") == "`a2/best.pt` `A2`"


def test_variant_codes_are_descriptive_in_table_cells_but_f1_metric_stays_f1():
    from dermssl.display_nomenclature import display_text

    assert display_text("| F1 | F1 macro | F2 |") == (
        "| Imagen y recorte | F1 macro | Recorte original |"
    )


def test_display_translation_is_idempotent():
    from dermssl.display_nomenclature import display_text

    text = "B71-limpio y student_r3; B71+F4"
    assert display_text(display_text(text)) == display_text(text)


def test_markdown_escaped_star_names_are_translated():
    from dermssl.display_nomenclature import display_text

    assert display_text(r"A\* frente a B\*") == (
        "Modelo de control frente a "
        "Proposed sin segmentación"
    )
    assert display_text(r"`A_*` y `B_*`") == r"`A_*` y `B_*`"


def test_less_than_comparison_does_not_hide_the_rest_of_a_table_row():
    from dermssl.display_nomenclature import display_text

    assert display_text(r"Dice < student_r3 | A\* | <br>") == (
        "Dice < Experimento de segmentación 3 | Modelo de control | <br>"
    )


def test_generated_labels_in_code_spans_are_translated_but_paths_are_not():
    from dermssl.display_nomenclature import display_text

    row = "| `A71` − `public_A71` | Macro-F1 | `resultados/x/A71__clean.json` |"
    assert display_text(row, code_labels=True) == (
        "| Modelo de control − DINOv2 público con el protocolo del modelo de control | Macro-F1 | "
        "`resultados/x/A71__clean.json` |"
    )
    assert display_text(row) == row  # prose and glossary keep their literal codes


def test_experiment_blocks_and_seeds_have_plain_names():
    from dermssl.display_nomenclature import display_text

    assert display_text("E3b frente a E1; s42 y s44") == (
        "Prueba con segmentación en los modelos de comparación frente a Prueba de pesos limpios; "
        "repetición 42 y repetición 44"
    )
    assert display_text("T7-aug y T7") == "Prueba de oscurecimiento (solo aumento de datos) y Prueba de oscurecimiento"


def test_overloaded_s_codes_depend_on_context():
    from dermssl.display_nomenclature import display_text

    assert display_text("S1", context="segmentation") == "Mejora de máscaras: unir regiones"
    assert display_text("cómputo S1", context="bench") == "cómputo lote 64"
    assert display_text("S1") == "S1"  # without context the code is ambiguous, so it is never guessed


def test_prediction_codes_keep_their_meaning_inside_v7_reports_only():
    from dermssl.display_nomenclature import display_text

    assert display_text("P3-IC", context="v7") == "Predicción de objetivo (intervalo de confianza)"
    assert display_text("P3-IC") == "P3-IC"


def test_slash_separated_repetitions_are_plain_too():
    from dermssl.display_nomenclature import display_text

    assert display_text("media s42/s43/s44") == "media repeticiones 42, 43 y 44"
    assert display_text("media s42/s43") == "media repeticiones 42 y 43"
