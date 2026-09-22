"""The deadline has to be falsified against the failure it claims to stop.

A 120 s deadline was configured and a step still ran thirty-five minutes.
`timeout_method = thread` cannot interrupt a blocking read inside a C
extension, so the guard existed, read as reasonable, and did nothing. A
defence that answers "yes, I am configured" while being unable to act is
worse than none: nobody looks at it again.

Two things follow, and this file tests both.

A deadline verified against `time.sleep()` passes under the thread method.
The real failure is a socket read that blocks in a syscall, so that is what
runs here.

And it runs under `-n 2 --dist loadfile`, the topology CI uses. The signal
method needs the main thread; each xdist worker has its own, which is very
probably fine and is exactly the kind of "very probably" that cost this
project three separate bugs today.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pytest_timeout", reason="the deadline plugin is what is under test")
pytest.importorskip("xdist", reason="the deadline must be verified under the CI topology")

BLOCKED_READ = '''
import socket


def test_blocks_in_a_syscall_forever():
    """Blocks in recv(), the way a stalled torch.hub fetch does."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname())
    # La referencia importa: si se descarta, el socket se cierra al recolectarse
    # y recv() devuelve EOF de inmediato en vez de bloquear. La primera version
    # de esta prueba hacia eso y "pasaba" en 0,01 s sin colgarse nunca.
    conn, _ = server.accept()
    assert conn is not None
    client.recv(1)          # el par nunca escribe: bloquea en el syscall
'''


def _run_inner(tmp_path: Path, extra: list[str]) -> subprocess.CompletedProcess:
    (tmp_path / "test_hangs.py").write_text(textwrap.dedent(BLOCKED_READ), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path / "test_hangs.py"),
         "-q", "-p", "no:cacheprovider", "--timeout=5",
         "--timeout-method=signal", *extra],
        capture_output=True, text=True, timeout=180,
    )


def test_the_deadline_kills_a_blocked_syscall(tmp_path):
    """Plain run: the mechanism works at all."""
    result = _run_inner(tmp_path, [])
    assert result.returncode != 0, "la prueba colgada termino en verde"
    assert "Timeout" in result.stdout + result.stderr, (
        "el proceso murio sin que el deadline lo nombrara")


def test_the_deadline_still_fires_under_the_ci_topology(tmp_path):
    """Under -n 2 --dist loadfile, which is what CI actually runs.

    Verifying the deadline in a flat run would verify a configuration CI
    does not use.
    """
    result = _run_inner(tmp_path, ["-n", "2", "--dist", "loadfile"])
    assert result.returncode != 0, (
        "bajo xdist la prueba colgada termino en verde: el deadline no "
        "llega a los workers")
    combined = result.stdout + result.stderr
    assert "Timeout" in combined or "timeout" in combined, (
        "el worker murio sin que el deadline lo nombrara")
