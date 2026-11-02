"""Concurrency tests for the per-lot exclusive lock.

Covers the failure that actually happened on 2026-11-01: a second queue
controller started while the first was alive, and a duplicate launcher truncated
the winner's console log with ``>``.

Legacy PID recovery reads /proc and is Linux-only. Kernel exclusion, publication
races, crash recovery and the log regression run on Windows and Linux.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from lot_lock import LotLock, LotLockError  # noqa: E402

LOT = "test-lot-026-031"
needs_proc = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="owner identity is read from /proc"
)


def test_acquire_writes_owner_identity(tmp_path: Path) -> None:
    lock_path = tmp_path / "lot.lock"
    with LotLock(lock_path, LOT) as lock:
        assert lock_path.is_file()
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["lot_id"] == LOT
        assert payload["owner"]["pid"] == os.getpid()
        assert lock.took_over_from is None
    # Released on exit, so the next controller can start.
    assert not lock_path.exists()


def test_second_holder_in_same_process_is_refused(tmp_path: Path) -> None:
    """The live-owner branch: our own PID is unambiguously alive."""
    lock_path = tmp_path / "lot.lock"
    with LotLock(lock_path, LOT):
        with pytest.raises(LotLockError) as excinfo:
            LotLock(lock_path, LOT).acquire()
    assert "already held" in str(excinfo.value)


def test_two_competing_controllers_only_one_launches(tmp_path: Path) -> None:
    """Two processes race for the same lot; exactly one proceeds to launch.

    This is the scenario that produced four live controllers. The loser must
    fail before launching anything, not after.
    """
    lock_path = tmp_path / "lot.lock"
    launched = tmp_path / "launched"
    script = tmp_path / "controller.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / "scripts")!r})
            from lot_lock import LotLock, LotLockError
            try:
                lock = LotLock({str(lock_path)!r}, {LOT!r}).acquire()
            except LotLockError:
                print("REFUSED")
                sys.exit(3)
            # Hold it long enough for the sibling to collide with a live owner.
            with open({str(launched)!r}, "a", encoding="utf-8") as handle:
                handle.write("launch\\n")
            time.sleep(2.5)
            lock.release()
            print("LAUNCHED")
            """
        ).strip(),
        encoding="utf-8",
    )

    first = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    # Give the first process time to take the lock before the second tries.
    deadline = 10.0
    waited = 0.0
    while not launched.exists() and waited < deadline:
        import time as _time

        _time.sleep(0.1)
        waited += 0.1
    assert launched.exists(), "first controller never acquired the lock"

    second = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=30
    )
    first_out, _ = first.communicate(timeout=30)

    assert second.returncode == 3, f"second controller should refuse, got {second.returncode}"
    assert "REFUSED" in second.stdout
    assert "LAUNCHED" in first_out
    # The decisive assertion: exactly one launch happened.
    assert launched.read_text(encoding="utf-8").count("launch") == 1


@pytest.mark.parametrize("stale", [False, True], ids=["new-lock", "stale-recovery"])
def test_contender_is_refused_before_owner_metadata_is_published(tmp_path: Path, stale: bool) -> None:
    """Pause the first holder before publication, then force a competing acquire.

    The old implementation had no exclusion during metadata initialization and
    recovered stale owners by unlinking whatever inode happened to be at path.
    """
    lock_path = tmp_path / "lot.lock"
    if stale:
        import socket

        lock_path.write_text(json.dumps({
            "lot_id": LOT,
            "locking_protocol": "kernel-guard-v1",
            "owner": {"pid": -1, "hostname": socket.gethostname()},
        }), encoding="utf-8")
    imports = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / 'scripts')!r}); "
        "from lot_lock import LotLock, LotLockError\n"
    )
    holder_code = imports + textwrap.dedent(f"""
        original = LotLock._payload
        def pause(self):
            print('READY', flush=True)
            sys.stdin.readline()
            return original(self)
        LotLock._payload = pause
        with LotLock({str(lock_path)!r}, {LOT!r}):
            print('HELD', flush=True)
    """)
    contender_code = imports + textwrap.dedent(f"""
        try:
            with LotLock({str(lock_path)!r}, {LOT!r}):
                print('WRONG: duplicate holder')
        except LotLockError:
            sys.exit(3)
    """)
    first = subprocess.Popen(
        [sys.executable, "-c", holder_code], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert first.stdout.readline().strip() == "READY"
        contender = subprocess.run(
            [sys.executable, "-c", contender_code], capture_output=True, text=True, timeout=15,
        )
        assert contender.returncode == 3, contender.stdout + contender.stderr
    finally:
        stdout, stderr = first.communicate("continue\n", timeout=15)
    assert first.returncode == 0, stderr
    assert "HELD" in stdout


def test_process_exit_releases_guard_without_deleting_its_inode(tmp_path: Path) -> None:
    """An abrupt exit leaves metadata, but the OS releases the actual lock."""
    lock_path = tmp_path / "lot.lock"
    code = (
        f"import sys, os; sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / 'scripts')!r}); "
        "from lot_lock import LotLock; "
        f"lock = LotLock({str(lock_path)!r}, {LOT!r}).acquire(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
    guard = lock_path.with_name(lock_path.name + ".guard")
    inode = guard.stat().st_ino
    with LotLock(lock_path, LOT) as lock:
        assert lock.took_over_from is not None
        assert guard.stat().st_ino == inode
    assert guard.exists()
    assert guard.stat().st_ino == inode


def test_unverifiable_legacy_owner_is_not_stolen(tmp_path: Path, monkeypatch) -> None:
    import lot_lock
    import socket

    lock_path = tmp_path / "lot.lock"
    lock_path.write_text(json.dumps({
        "lot_id": LOT, "owner": {"pid": 123, "hostname": socket.gethostname()},
    }), encoding="utf-8")
    before = lock_path.read_bytes()
    monkeypatch.setattr(lot_lock, "_owner_is_live", lambda owner: None)
    with pytest.raises(LotLockError):
        LotLock(lock_path, LOT).acquire()
    assert lock_path.read_bytes() == before


@needs_proc
def test_stale_lock_from_dead_process_is_taken_over(tmp_path: Path) -> None:
    """A controller that died must not wedge the lot forever."""
    lock_path = tmp_path / "lot.lock"
    # Popen, not run: we need the pid of a process we know has exited.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = dead.pid
    assert dead.wait(timeout=30) == 0

    lock_path.write_text(
        json.dumps(
            {
                "lot_id": LOT,
                "role": "queue-controller",
                "acquired_at_utc": "2026-11-01T18:24:00Z",
                "owner": {
                    "pid": dead_pid,
                    "start_ticks": 1,  # wrong on purpose: PID reuse must not read as live
                    "hostname": __import__("socket").gethostname(),
                    "boot_id": None,
                },
            }
        ),
        encoding="utf-8",
    )

    with LotLock(lock_path, LOT) as lock:
        assert lock.took_over_from is not None
        assert lock.took_over_from["owner"]["pid"] == dead_pid
        assert json.loads(lock_path.read_text(encoding="utf-8"))["owner"]["pid"] == os.getpid()


@needs_proc
def test_recycled_pid_is_not_mistaken_for_a_live_owner(tmp_path: Path) -> None:
    """Our own PID with the wrong start time must read as dead, not alive.

    Without the start-time check, any stale lock whose PID was later reused
    would block the lot permanently.
    """
    lock_path = tmp_path / "lot.lock"
    lock_path.write_text(
        json.dumps(
            {
                "lot_id": LOT,
                "role": "queue-controller",
                "acquired_at_utc": "2026-11-01T18:24:00Z",
                "owner": {
                    "pid": os.getpid(),
                    "start_ticks": 1,
                    "hostname": __import__("socket").gethostname(),
                    "boot_id": None,
                },
            }
        ),
        encoding="utf-8",
    )
    with LotLock(lock_path, LOT) as lock:
        assert lock.took_over_from is not None


def test_corrupt_lock_file_fails_closed(tmp_path: Path) -> None:
    lock_path = tmp_path / "lot.lock"
    lock_path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(LotLockError):
        LotLock(lock_path, LOT).acquire()
    assert lock_path.read_text(encoding="utf-8") == "{ this is not json"


def test_empty_owner_file_is_not_stolen(tmp_path: Path) -> None:
    """An empty file may belong to a process still publishing its identity."""
    lock_path = tmp_path / "lot.lock"
    lock_path.touch()
    with pytest.raises(LotLockError):
        LotLock(lock_path, LOT).acquire()
    assert lock_path.read_bytes() == b""


def test_foreign_host_owner_requires_explicit_review(tmp_path: Path) -> None:
    lock_path = tmp_path / "lot.lock"
    payload = {
        "lot_id": LOT,
        "owner": {"pid": 123, "hostname": "another-cluster-host", "start_ticks": 1},
    }
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    before = lock_path.read_bytes()
    with pytest.raises(LotLockError):
        LotLock(lock_path, LOT).acquire()
    assert lock_path.read_bytes() == before


def test_release_does_not_delete_a_lock_owned_by_someone_else(tmp_path: Path) -> None:
    lock_path = tmp_path / "lot.lock"
    lock = LotLock(lock_path, LOT).acquire()
    # Simulate a takeover by another process while we still think we hold it.
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["owner"]["pid"] = os.getpid() + 100000
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    lock.release()
    assert lock_path.is_file(), "release must not remove another owner's lock"


def test_generated_launcher_appends_and_refuses_duplicates() -> None:
    """Regression for the log truncation that destroyed four headers.

    The generator must emit ``>>`` and a pgrep guard. With ``>`` a losing
    launcher zeroed the winner's console log, taking the
    ``Loaded Labeled datasets`` line with it.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/prepare_dermlip_ima_ablation_026_031.py"
    ).read_text(encoding="utf-8")

    assert '>> "$LOG" 2>&1 &' in source, "launcher must append to the console log"
    assert '> "$LOG" 2>&1 &' not in source.replace('>> "$LOG" 2>&1 &', ""), (
        "launcher must not truncate the console log"
    )
    assert "pgrep -f" in source, "launcher must refuse to start a duplicate trainer"
    assert "REFUSING" in source
