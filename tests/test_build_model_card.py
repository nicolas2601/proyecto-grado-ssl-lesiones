"""The release model card is generated from the committed evaluation files."""

import numpy as np
import pytest

from scripts import build_model_card as bmc


@pytest.fixture(scope="module")
def text():
    return bmc.render()  # ~10 s: 1000-resample bootstraps for five rows


def test_committed_model_card_is_fresh(text):
    assert bmc.TARGET.read_text(encoding="utf-8") == text, "run: python scripts/build_model_card.py"


def test_every_released_model_has_a_row_with_intervals(text):
    for name in ("`A_s42`", "`A_s43`", "`B_s42`", "`B_s43`"):
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
                   "99.5%", "0.002", "Apache License 2.0", "PAD-UFES-20"):
        assert phrase in text, phrase
