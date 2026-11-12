from __future__ import annotations

import hashlib
import json

import pytest

from demo.backend.screening import POLICY_NAME, ScreeningPolicy

MODEL_SHA = "d" * 64
HEAD_SHA = "e" * 64
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEVELOPMENT = {
    "at_zero": {"target_recall": 0.56, "target_to_nevus": 0.2, "nevus_recall": 0.83, "macro_f1": 0.61},
    "nested": {"target_recall": 0.75, "target_to_nevus": 0.13, "nevus_recall": 0.76, "macro_f1": 0.60},
}


def _policy(**overrides):
    head = {"task": "HAM", "target": "mel", "log_offset": 1.0, "head_sha256": HEAD_SHA, "development": DEVELOPMENT}
    head.update(overrides.pop("head", {}))
    policy = {
        "schema": "dermssl.screening-policy.v1",
        "rule_id": "melanoma-priority-v1",
        "release": "v10.1.0",
        "model_sha256": MODEL_SHA,
        "heads": {"MOCO_HAM7": head},
    }
    policy.update(overrides)
    return policy


def _write(tmp_path, policy) -> str:
    content = json.dumps(policy).encode()
    (tmp_path / POLICY_NAME).write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _prediction(probs, model_key="MOCO_HAM7", head_sha=HEAD_SHA, model_sha=MODEL_SHA):
    scores = dict(zip(HAM_CLASSES, probs))
    pred = max(scores, key=scores.get)
    return {
        "pred": pred,
        "classes": list(HAM_CLASSES),
        "probs": scores,
        "model_key": model_key,
        "identity": {"model_sha256": model_sha, "head_sha256": head_sha},
    }


NEVUS_CLOSE_CALL = [0.01, 0.01, 0.02, 0.01, 0.30, 0.60, 0.05]  # 0.30 * e > 0.60
NEVUS_CLEAR = [0.01, 0.01, 0.02, 0.01, 0.10, 0.80, 0.05]  # 0.10 * e < 0.80


def test_close_call_raises_a_melanoma_alert_without_changing_pred(tmp_path):
    policy = ScreeningPolicy(tmp_path, _write(tmp_path, _policy()))
    prediction = _prediction(NEVUS_CLOSE_CALL)

    block = policy.annotate(prediction)

    assert prediction["pred"] == "nv"
    assert block["status"] == "ok" and block["reason"] is None
    assert block["pred"] == "mel" and block["alert"] is True and block["differs_from_pred"] is True
    assert block["target"] == "mel" and block["log_offset"] == 1.0
    assert block["rule_id"] == "melanoma-priority-v1"
    assert block["development"] == DEVELOPMENT


def test_clear_nevus_keeps_the_screening_pred(tmp_path):
    block = ScreeningPolicy(tmp_path, _write(tmp_path, _policy())).annotate(_prediction(NEVUS_CLEAR))

    assert block["pred"] == "nv" and block["alert"] is False and block["differs_from_pred"] is False


def test_melanoma_argmax_is_an_alert_that_does_not_differ(tmp_path):
    block = ScreeningPolicy(tmp_path, _write(tmp_path, _policy())).annotate(
        _prediction([0.01, 0.01, 0.02, 0.01, 0.70, 0.20, 0.05])
    )

    assert block["alert"] is True and block["differs_from_pred"] is False


def test_missing_pin_is_unavailable_and_never_raises(tmp_path):
    _write(tmp_path, _policy())
    policy = ScreeningPolicy(tmp_path, None)

    assert policy.annotate(_prediction(NEVUS_CLOSE_CALL)) == {"status": "unavailable", "reason": "policy_not_configured"}
    assert policy.status()["ready"] is False


def test_pin_mismatch_is_unavailable(tmp_path):
    _write(tmp_path, _policy())
    block = ScreeningPolicy(tmp_path, "a" * 64).annotate(_prediction(NEVUS_CLOSE_CALL))

    assert block == {"status": "unavailable", "reason": "policy_invalid"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "other"},
        {"head": {"log_offset": -0.5}},
        {"head": {"log_offset": float("nan")}},
        {"head": {"log_offset": 9.0}},
        {"head": {"target": "melanoma"}},
        {"model_sha256": "short"},
    ],
)
def test_invalid_policy_is_unavailable(tmp_path, overrides):
    content = json.dumps(_policy(**overrides)).encode()
    (tmp_path / POLICY_NAME).write_bytes(content)
    policy = ScreeningPolicy(tmp_path, hashlib.sha256(content).hexdigest())

    assert policy.annotate(_prediction(NEVUS_CLOSE_CALL)) == {"status": "unavailable", "reason": "policy_invalid"}


def test_policy_bound_to_another_head_is_unavailable(tmp_path):
    policy = ScreeningPolicy(tmp_path, _write(tmp_path, _policy()))

    assert policy.annotate(_prediction(NEVUS_CLOSE_CALL, head_sha="f" * 64))["reason"] == "classifier_mismatch"
    assert policy.annotate(_prediction(NEVUS_CLOSE_CALL, model_sha="f" * 64))["reason"] == "classifier_mismatch"


def test_model_without_a_policy_head_is_unavailable(tmp_path):
    policy = ScreeningPolicy(tmp_path, _write(tmp_path, _policy()))

    assert policy.annotate(_prediction(NEVUS_CLOSE_CALL, model_key="MOCO_PAD6"))["reason"] == "model_not_covered"


def test_status_reports_the_pinned_policy(tmp_path):
    pin = _write(tmp_path, _policy())

    status = ScreeningPolicy(tmp_path, pin).status()

    assert status == {"ready": True, "reason": "ready", "rule_id": "melanoma-priority-v1", "sha256": pin}


class FakeMoCo:
    def __init__(self, probs):
        self.probs = probs

    def status(self):
        return {"backbone_ready": True, "classification_ready": True, "reason": "ready", "release": "v10.1.0"}

    def predict(self, image, model_key="MOCO_HAM7", mode="base"):
        return _prediction(self.probs, model_key=model_key)


def _client(tmp_path, pin):
    from fastapi.testclient import TestClient

    from demo.backend.v10_app import create_app

    screening = ScreeningPolicy(tmp_path, pin)
    app = create_app(moco_service=FakeMoCo(NEVUS_CLOSE_CALL), report_path=None, screening_service=screening)
    return TestClient(app)


def _png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 150, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_carries_the_screening_block_next_to_the_unchanged_pred(tmp_path):
    client = _client(tmp_path, _write(tmp_path, _policy()))

    body = client.post("/api/predict?model=MOCO_HAM7", files={"file": ("a.png", _png(), "image/png")}).json()

    assert body["pred"] == "nv"
    assert body["screening"]["status"] == "ok" and body["screening"]["alert"] is True


def test_predict_without_a_policy_still_classifies(tmp_path):
    client = _client(tmp_path, None)

    response = client.post("/api/predict?model=MOCO_HAM7", files={"file": ("a.png", _png(), "image/png")})

    assert response.status_code == 200
    assert response.json()["screening"] == {"status": "unavailable", "reason": "policy_not_configured"}


def test_analyze_classification_carries_the_screening_block(tmp_path):
    client = _client(tmp_path, _write(tmp_path, _policy()))

    body = client.post("/api/analyze?model=MOCO_HAM7", files={"file": ("a.png", _png(), "image/png")}).json()

    assert body["classification"]["screening"]["pred"] == "mel"


def test_status_reports_the_screening_policy(tmp_path):
    pin = _write(tmp_path, _policy())
    client = _client(tmp_path, pin)

    body = client.get("/api/status?model=MOCO_HAM7").json()

    assert body["screening"] == {"ready": True, "reason": "ready", "rule_id": "melanoma-priority-v1", "sha256": pin}
