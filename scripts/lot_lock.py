"""Exclusive per-lot lock for cooperating queue controllers.

Direct launchers do not acquire this guard. Their process check is advisory;
the execution claim remains the final barrier against duplicate training.

Why this exists. On 2026-11-01 the queue controller for lot 026-031 was
relaunched five times in eleven minutes while its pool pins were being debugged,
without stopping the previous controller. Four controllers ended up alive and
monitoring the same outputs. Counted from runs/ima_ablation_026_031_queue.log,
runs 026 and 027 received ten launches each and 028-031 four each: 36 launches
for six runs.

The damage was contained by the execution claim, which serialises at the output:
each run ended with exactly one claim file and one run_id, so training itself
launched once. But two things did break, and neither is caught by the claim:

1. Nothing stopped a second controller from existing. The claim rejects the
   second *training*, late, after a process already started and failed.
2. The launchers wrote the console log with ``>``, which truncates. A losing
   process truncated the winner's log in four of six runs: 028-031 start with
   about 1,700 null bytes, their ``Loaded Labeled datasets`` header is gone, and
   with it the only record of the effective ``--ima-batch``. That line survives
   only for 026 and 027.

So the lock is not a substitute for the claim. The claim protects the outputs;
the lock protects everything around them, logs included, and it fails early
instead of after a launch.

Exclusion is enforced by a kernel lock on a persistent companion .guard file.
Never unlink that file: contenders must all lock the same inode. Owner metadata
is published atomically while holding the guard. Legacy metadata is recovered
only when its local owner is provably dead; unknown or foreign owners fail closed.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


class LotLockError(RuntimeError):
    """Raised when the lot is already owned by a live process."""


def _proc_start_ticks(pid: int) -> int | None:
    """Process start time in clock ticks since boot, or None if pid is gone.

    Field 22 of /proc/<pid>/stat. Read together with the PID it distinguishes a
    live owner from a recycled PID. The comm field can contain spaces and
    parentheses, so the line is split after the last ``)``.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    try:
        return int(raw[raw.rindex(")") + 1:].split()[19])
    except (ValueError, IndexError):
        return None


def _identity(pid: int) -> dict[str, Any]:
    return {
        "pid": pid,
        "start_ticks": _proc_start_ticks(pid),
        "hostname": socket.gethostname(),
        "boot_id": _boot_id(),
    }


def _boot_id() -> str | None:
    """Kernel boot id, so a lock from before a reboot is never read as live."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _owner_is_live(owner: dict[str, Any]) -> bool | None:
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return None
    if owner.get("hostname") != socket.gethostname():
        return None
    boot = _boot_id()
    if owner.get("boot_id") is not None and boot is not None and owner["boot_id"] != boot:
        return False
    current = _proc_start_ticks(pid)
    if current is None:
        if Path("/proc/self/stat").exists() and not Path(f"/proc/{pid}").exists():
            return False
        return None
    recorded = owner.get("start_ticks")
    if recorded is None:
        # Written without a start time: fall back to bare existence and say so.
        return True
    return current == recorded


class LotLock:
    """Context manager holding an exclusive lock for one lot.

    The guard remains on disk after release. Removing it while any contender
    has it open would split ownership across two inodes. JSON is diagnostic
    metadata, never the mechanism that decides mutual exclusion.
    """

    def __init__(self, path: Path, lot_id: str, role: str = "controller") -> None:
        self.path = Path(path)
        self.lot_id = lot_id
        self.role = role
        self.took_over_from: dict[str, Any] | None = None
        self._held = False
        self._guard_fd: int | None = None
        self._lease_id = uuid.uuid4().hex

    def _payload(self) -> dict[str, Any]:
        return {
            "locking_protocol": "kernel-guard-v1",
            "lease_id": self._lease_id,
            "lot_id": self.lot_id,
            "role": self.role,
            "acquired_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "owner": _identity(os.getpid()),
        }

    def _read_owner(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            raise LotLockError(f"Cannot verify owner metadata at {self.path}; review it explicitly") from error
        if not isinstance(data, dict):
            raise LotLockError(f"Invalid owner metadata at {self.path}")
        owner = data.get("owner")
        if not isinstance(owner, dict):
            raise LotLockError(f"Missing owner identity at {self.path}")
        return data

    def _lock_guard(self) -> None:
        guard = self.path.with_name(self.path.name + ".guard")
        fd = os.open(guard, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as error:
            os.close(fd)
            raise LotLockError(f"Lot {self.lot_id} is already held or its guard cannot be locked: {guard}") from error
        self._guard_fd = fd

    def _unlock_guard(self) -> None:
        fd, self._guard_fd = self._guard_fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _write_owner(self) -> None:
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._payload(), handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def acquire(self) -> "LotLock":
        if self._held:
            raise LotLockError(f"Lot {self.lot_id} is already held by this lock instance")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_guard()
        try:
            existing = self._read_owner()
            if existing is not None:
                owner = existing["owner"]
                if existing.get("lot_id") != self.lot_id or owner.get("hostname") != socket.gethostname():
                    raise LotLockError(f"Foreign or mismatched lot owner at {self.path}; explicit review required")
                # For this protocol, holding the kernel guard proves the previous
                # lease was released or its process exited. No PID-based takeover.
                if existing.get("locking_protocol") != "kernel-guard-v1" and _owner_is_live(owner) is not False:
                    raise LotLockError(
                        f"Legacy lot owner is live or cannot be verified at {self.path}; explicit review required"
                    )
                self.took_over_from = existing
            self._write_owner()
            self._held = True
            return self
        except BaseException:
            self._unlock_guard()
            raise

    def release(self) -> None:
        if not self._held:
            return
        try:
            existing = self._read_owner()
            if existing is not None and existing.get("lease_id") == self._lease_id and existing["owner"].get("pid") == os.getpid():
                self.path.unlink()
        finally:
            self._held = False
            self._unlock_guard()

    def __enter__(self) -> "LotLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
