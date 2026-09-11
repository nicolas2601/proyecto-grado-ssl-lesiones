"""The bank runner has to survive Kaggle's two-concurrent-GPU-session limit,
a push CLI that exits 0 on failure, and stale COMPLETE statuses left behind by
an earlier smoke version of the same kernel."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_finetune_results_land_where_the_cluster_writes_them(tmp_path):
    """Kaggle and the cluster run the same protocol, so their outputs have to
    be comparable without a path translation. `finetune_baseline.py` writes
    `<results-dir>/<dataset>/`, so the Kaggle driver does too."""
    assert krb.output_dir("finetune", tmp_path) == tmp_path / "finetune" / "ham10000"
    assert (krb.output_dir("finetune_isic2019", tmp_path)
            == tmp_path / "finetune" / "isic2019")


def test_frozen_banks_keep_their_flat_layout(tmp_path):
    """Only the fine-tuning banks have a cluster counterpart to line up with."""
    assert krb.output_dir("baselines", tmp_path) == tmp_path / "baselines"
    assert (krb.output_dir("baselines_isic2019", tmp_path)
            == tmp_path / "baselines_isic2019")


def test_expected_result_name_follows_the_bank_naming(tmp_path):
    """The fine-tuning case lived here asserting `results_resnet50.json`, which
    is the buggy name. Written before any fine-tuning bank had ever run, it
    never failed - it certified the defect. Covered now by
    test_finetune_results_carry_the_ft_suffix_the_notebooks_write, against a
    file that exists on disk."""
    assert krb.result_name("baselines", "alexnet") == "results_alexnet.json"
    assert (krb.result_name("baselines_isic2020", "vit_b16")
            == "results_vit_b16_isic2020.json")


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


def test_a_download_that_fails_is_retried_before_the_model_is_written_off():
    """A COMPLETE kernel's output is not readable the instant it completes.

    dinov2_vitb14 on the ISIC 2019 bank was marked failed with "missing result
    file" while its JSON was sitting on Kaggle and downloaded fine seconds
    later: `download` ignored the CLI exit code, so one transient miss burned a
    GPU run permanently.
    """
    calls = []

    def flaky(cmd):
        calls.append(cmd)
        ok = len(calls) >= 3          # first two attempts fail
        return SimpleNamespace(returncode=0 if ok else 1, stdout="", stderr="boom")

    assert krb.download_with_retry("u/s", Path("/tmp"), runner=flaky,
                                   attempts=3, delay=0)
    assert len(calls) == 3


def test_a_download_that_never_succeeds_reports_failure_instead_of_lying():
    def always_fails(cmd):
        return SimpleNamespace(returncode=1, stdout="", stderr="nope")

    assert not krb.download_with_retry("u/s", Path("/tmp"), runner=always_fails,
                                       attempts=2, delay=0)


def test_a_stale_result_is_re_fetched_before_the_model_is_written_off(tmp_path):
    """Right after a new version completes, Kaggle can still serve the previous
    version's output. resnet50's real fine-tuning run was marked failed because
    the first download returned the smoke JSON; the real one was there moments
    later. One verification miss must not retire a finished GPU run.
    """
    path = tmp_path / "results_resnet50_ft.json"
    path.write_text(json.dumps({"modelo": "resnet50", "smoke": True}))
    attempts = []

    def fetch_then_settle(slug, dest, **kw):
        attempts.append(slug)
        if len(attempts) >= 2:      # second fetch brings the real output
            path.write_text(json.dumps({"modelo": "resnet50", "smoke": False}))
        return True

    assert krb.verify_with_refetch(path, real=True, slug="u/s", dest=tmp_path,
                                   downloader=fetch_then_settle, attempts=3,
                                   delay=0)
    assert len(attempts) == 2


def test_a_result_that_stays_stale_is_still_reported_as_failed(tmp_path):
    path = tmp_path / "results_resnet50_ft.json"
    path.write_text(json.dumps({"modelo": "resnet50", "smoke": True}))

    assert not krb.verify_with_refetch(path, real=True, slug="u/s", dest=tmp_path,
                                       downloader=lambda *a, **k: True,
                                       attempts=2, delay=0)


def test_the_driver_reports_that_it_is_alive_between_polls():
    """Silence must not be the normal state of a healthy run.

    Between a push and a completion the loop printed nothing, so a bank whose
    models take an hour each looked identical to a hung process for that whole
    hour. Third instance today of the same shape: the buffered log, the guard
    that checked vocabulary, and the mute wait for a GPU slot. Absence of
    signal reads exactly like the silence of a failure.

    Falsified: asserting `"heartbeat" in source` passed with the print deleted
    and only the explanatory comment left - category 3 of docs/20, committed
    three hours after cataloguing it. The assertion has to name the mechanism.
    """
    import ast
    source = (REPO / "scripts" / "kaggle_run_bank.py").read_text()
    tree = ast.parse(source)
    drive = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "drive")
    # The print has to live in the SAME block as the sleep. Counting prints
    # anywhere in drive() passed with the heartbeat deleted, because the push
    # and failure messages alone met the threshold.
    bloques_con_sleep = [
        node for node in ast.walk(drive)
        if isinstance(node, ast.If)
        and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "sleep"
                for c in ast.walk(node))
    ]
    assert bloques_con_sleep, "drive() ya no duerme entre polls"
    innermost = bloques_con_sleep[-1]
    assert any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "print"
               for stmt in innermost.body for c in ast.walk(stmt)), (
        "el bloque que duerme no imprime nada: un banco sano es "
        "indistinguible de uno colgado durante horas")


def test_the_driver_does_not_buffer_its_diagnostics_away():
    """A long run piped to a file must report as it goes.

    Python block-buffers stdout when it is not a tty, so a driver that runs for
    hours writes zero bytes until it exits. Both bank logs were empty while the
    run was in flight, which is exactly when the output is needed: every `failed`
    diagnosed today had to be reconstructed from files on disk because the
    reason never reached the log.
    """
    source = (REPO / "scripts" / "kaggle_run_bank.py").read_text()
    assert "reconfigure(line_buffering=True)" in source or "flush=True" in source, \
        "el driver no fuerza el vaciado de stdout: su log queda vacio en vuelo"


def test_finetune_results_carry_the_ft_suffix_the_notebooks_write():
    """The driver has to look for the file that actually gets written.

    Deriving the name from `protocol.DATASETS[...].json_suffix` gives
    `results_<model>.json` for ham10000, but the fine-tuning notebooks write
    `results_<model>_ft.json`. The driver spent the whole bank looking for a
    file that never existed and retiring finished runs as "missing result
    file" - four of them before the log became readable and said so.
    """
    assert krb.result_name("finetune", "resnet50") == "results_resnet50_ft.json"
    assert (krb.result_name("finetune_isic2019", "resnet50")
            == "results_resnet50_ft_isic2019.json")


def test_frozen_banks_keep_their_own_names():
    assert krb.result_name("baselines", "resnet50") == "results_resnet50.json"
    assert (krb.result_name("baselines_isic2019", "resnet50")
            == "results_resnet50_isic2019.json")


def test_an_unreadable_status_is_reported_instead_of_polled_forever():
    """UNKNOWN is not RUNNING, and treating it as such hangs the bank silently.

    `kaggle kernels status` returns nothing parseable when the network is down,
    which happened during this session. UNKNOWN is not terminal, so the driver
    would keep polling a kernel it can no longer see, forever, printing nothing.
    """
    seen = []
    krb._unknown_streak.clear()
    for _ in range(krb.UNKNOWN_TOLERANCE - 1):
        assert krb.status_or_report("u/s", reader=lambda s: "UNKNOWN",
                                    report=seen.append) == "UNKNOWN"
    assert krb.status_or_report("u/s", reader=lambda s: "UNKNOWN",
                                report=seen.append) == "UNREACHABLE"
    assert seen and "u/s" in seen[-1]


def test_a_readable_status_clears_the_streak():
    """A blip must not accumulate across a healthy poll."""
    krb._unknown_streak.clear()
    krb.status_or_report("u/s", reader=lambda s: "UNKNOWN", report=lambda m: None)
    assert krb.status_or_report("u/s", reader=lambda s: "RUNNING",
                                report=lambda m: None) == "RUNNING"
    assert "u/s" not in krb._unknown_streak


def test_free_slots_account_for_the_running_kernels():
    assert krb.free_slots({"a": "running", "b": "running"}, limit=2) == 0
    assert krb.free_slots({"a": "running", "b": "done"}, limit=2) == 1
    assert krb.free_slots({}, limit=2) == 2
