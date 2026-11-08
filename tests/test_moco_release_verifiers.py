import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts/release_v10"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_http_verifier_handles_plain_text_cors_preflight(monkeypatch):
    http = module("verify_http")
    def fake_run(command, **kwargs):
        Path(command[command.index("--dump-header") + 1]).write_text("Access-Control-Allow-Origin: test\n")
        return SimpleNamespace(stdout="OK\n200")
    monkeypatch.setattr(http.subprocess, "run", fake_run)
    status, payload, headers = http.request("http://127.0.0.1/", method="OPTIONS")
    assert status == 200 and payload == {} and headers["access-control-allow-origin"] == "test"


def test_http_probability_verifier_rejects_nan_and_wrong_classes():
    http = module("verify_http")
    assert http.compare({"classes": ["a", "b"], "probs": {"a": .8, "b": .2}, "pred": "a"}, [.8, .2], ["a", "b"])["passed"]
    for actual in ({"classes": ["b", "a"], "probs": {}},
                   {"classes": ["a", "b"], "probs": {"a": float("nan"), "b": .2}}):
        with pytest.raises(ValueError):
            http.compare(actual, [.8, .2], ["a", "b"])


def test_promotion_pins_the_verified_oci_index_not_its_config_digest():
    promotion = module("promote_api")
    assert promotion.NEW_IMAGE == "sha256:aa55c4e8bfeb6276dce0fe811e306e9531111dba85acc9965d9d89d67b399008"
    assert promotion.OLD_IMAGE == "sha256:2a67b4d785da9c23c2db72caa619a0a1341774c61b9bf1195b2ef62002838748"
    assert promotion.NEW_IMAGE != promotion.OLD_IMAGE
