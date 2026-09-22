"""The guard that protects the other guards: count what gets collected.

Installing torch in CI made five test files collect for the first time. They
had never run, and nothing said so - pytest exits 0 when it collects less
than you think. Under-collection is the same failure this repo keeps
finding, applied to the instrument instead of the system: absence reads as
success.

A missing dependency, an unregistered marker under --strict-markers, a
renamed directory, a typo in a path insert - each quietly removes tests and
leaves the run green. This asserts a floor. It has to be raised by hand when
files are added, which is the point: the number is a claim someone made, not
a number that drifts.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Raise these when adding files. Never lower them to make a run pass: a drop
# means tests stopped being collected, which is the thing being guarded.
#
# The floor depends on the environment, and saying so is the honest version.
# With every optional dependency present - which is what CI_REQUIRE_FULL_DEPS
# asserts - every file on disk must collect. Without them some modules cannot
# import, so the floor is lower and the guard only catches gross drops.
MIN_TESTS_FULL = 1250
MIN_TESTS_PARTIAL = 1150
MIN_FILES_PARTIAL = 75


def _collect() -> tuple[int, int]:
    """Ask pytest what it can see, without running anything.

    ``--continue-on-collection-errors`` on purpose: a module that fails to
    import must still let the rest be counted, or one broken file hides how
    much else disappeared with it.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "--continue-on-collection-errors"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    files, tests = set(), 0
    for line in completed.stdout.splitlines():
        if "::" not in line:
            continue
        path = line.split("::", 1)[0].strip()
        if path.endswith(".py"):
            files.add(path)
            tests += 1
    return len(files), tests


def _files_on_disk() -> int:
    return len(list((REPO / "tests").glob("test_*.py")))


def test_the_suite_still_collects_what_it_claims():
    files, tests = _collect()
    full_env = os.environ.get("CI_REQUIRE_FULL_DEPS") == "1"

    if full_env:
        on_disk = _files_on_disk()
        assert files >= on_disk, (
            f"hay {on_disk} archivos de test en disco y solo {files} se "
            "recolectaron, con todas las dependencias presentes. Los que "
            "faltan no fallan: desaparecen, y la corrida sale verde.")
        floor = MIN_TESTS_FULL
    else:
        assert files >= MIN_FILES_PARTIAL, (
            f"se recolectaron {files} archivos, el piso sin dependencias "
            f"completas es {MIN_FILES_PARTIAL}.")
        floor = MIN_TESTS_PARTIAL

    assert tests >= floor, (
        f"se recolectaron {tests} tests, el piso es {floor}. "
        "Una corrida verde con menos tests no es una corrida verde.")
