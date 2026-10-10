"""The qualitative extractor must preserve the evaluator's batch boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig3_extract_cases as extractor  # noqa: E402


def test_predict_case_reuses_original_evaluation_batch(monkeypatch):
    pairs = [(str(index), Path(f"{index}.jpg"), Path(f"{index}.png")) for index in range(70)]
    observed = {}

    def load_image(path, resolution):
        assert resolution == 2
        return np.full((2, 2, 3), int(path.stem), dtype=np.uint8)

    def predict_masks(model, images, device, batch_size):
        observed["values"] = images[:, 0, 0, 0].tolist()
        observed["batch_size"] = batch_size
        predictions = np.zeros((batch_size, 2, 2), dtype=np.float32)
        predictions[7] = 0.9
        return predictions

    monkeypatch.setattr(extractor.eval_inputs, "load_image", load_image)
    monkeypatch.setattr(extractor.eval_student, "predict_masks", predict_masks)
    image, prediction = extractor.predict_case(object(), pairs, 39, 2, object())

    assert observed == {"values": list(range(32, 64)), "batch_size": 32}
    assert image[0, 0, 0] == 39
    assert prediction.shape == (2, 2) and prediction.all()


def test_last_batch_uses_only_remaining_pairs(monkeypatch):
    pairs = [(str(index), Path(f"{index}.jpg"), Path(f"{index}.png")) for index in range(70)]
    calls = []
    monkeypatch.setattr(extractor.eval_inputs, "load_image",
                        lambda path, resolution: np.zeros((2, 2, 3), dtype=np.uint8))

    def predict_masks(model, images, device, batch_size):
        calls.append((len(images), batch_size))
        return np.zeros((batch_size, 2, 2), dtype=np.float32)

    monkeypatch.setattr(extractor.eval_student, "predict_masks", predict_masks)
    _, prediction = extractor.predict_case(object(), pairs, 69, 2, object())
    assert calls == [(6, 6)] and not prediction.any()
