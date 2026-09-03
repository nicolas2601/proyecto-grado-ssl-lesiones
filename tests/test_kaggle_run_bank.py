"""The bank runner has to survive Kaggle's two-concurrent-GPU-session limit,
a push CLI that exits 0 on failure, and stale COMPLETE statuses left behind by
an earlier smoke version of the same kernel."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import kaggle_run_bank as krb  # noqa: E402


PUSH_OK = ("Kernel version 3 successfully pushed.  Please check progress at "
           "https://www.kaggle.com/code/nicorghost/baselines-isic2020-alexnet\n")
PUSH_BUSY = "Kernel push error: Maximum batch GPU session count of 2 reached.\n"


def test_push_success_is_detected():
    ok, detail = krb.parse_push(PUSH_OK, "")
    assert ok and "alexnet" in detail


def test_push_error_is_not_swallowed_even_though_the_cli_exits_zero():
    ok, detail = krb.parse_push(PUSH_BUSY, "")
    assert not ok
    assert "Maximum batch GPU session count" in detail


def test_push_error_is_recognised_as_a_busy_slot_not_a_real_failure():
    assert krb.is_slot_busy(PUSH_BUSY)
    assert not krb.is_slot_busy("Kernel push error: notebook not found")


def test_unparseable_push_output_counts_as_failure():
    ok, _ = krb.parse_push("", "")
    assert not ok


@pytest.mark.parametrize("raw,expected", [
    ('nicorghost/x has status "KernelWorkerStatus.RUNNING"', "RUNNING"),
    ('nicorghost/x has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
    ('nicorghost/x has status "KernelWorkerStatus.ERROR"', "ERROR"),
    ('nicorghost/x has status "KernelWorkerStatus.QUEUED"', "QUEUED"),
])
def test_parse_status(raw, expected):
    assert krb.parse_status(raw) == expected


def test_parse_status_of_a_kernel_that_never_ran():
    assert krb.parse_status("404 Client Error: Not Found for url: ...") == "UNKNOWN"


def test_result_of_a_real_run_must_not_carry_the_smoke_flag(tmp_path):
    path = tmp_path / "results_alexnet_isic2020.json"
    path.write_text(json.dumps({"modelo": "alexnet", "smoke": True}))
    with pytest.raises(krb.StaleResult):
        krb.verify_result(path, real=True)


def test_result_of_a_real_run_is_accepted_when_smoke_is_false(tmp_path):
    path = tmp_path / "results_alexnet_isic2020.json"
    path.write_text(json.dumps({"modelo": "alexnet", "smoke": False}))
    assert krb.verify_result(path, real=True) == {"modelo": "alexnet", "smoke": False}


def test_missing_result_file_is_reported(tmp_path):
    with pytest.raises(krb.StaleResult):
        krb.verify_result(tmp_path / "nope.json", real=True)


def test_expected_result_name_follows_the_bank_naming(tmp_path):
    assert krb.result_name("baselines", "alexnet") == "results_alexnet.json"
    assert (krb.result_name("baselines_isic2020", "vit_b16")
            == "results_vit_b16_isic2020.json")
    assert krb.result_name("finetune", "resnet50") == "results_resnet50.json"


def test_state_roundtrip_lets_the_runner_resume(tmp_path):
    state_file = tmp_path / "state.json"
    krb.save_state(state_file, {"alexnet": "done", "vgg19": "running"})
    assert krb.load_state(state_file) == {"alexnet": "done", "vgg19": "running"}


def test_state_of_a_fresh_bank_is_empty(tmp_path):
    assert krb.load_state(tmp_path / "absent.json") == {}


def test_pending_models_skip_the_ones_already_done():
    state = {"alexnet": "done", "vgg19": "failed", "resnet50": "running"}
    models = ["alexnet", "vgg19", "resnet50", "vit_b16"]
    assert krb.pending(models, state) == ["vit_b16"]


def test_free_slots_account_for_the_running_kernels():
    assert krb.free_slots({"a": "running", "b": "running"}, limit=2) == 0
    assert krb.free_slots({"a": "running", "b": "done"}, limit=2) == 1
    assert krb.free_slots({}, limit=2) == 2
