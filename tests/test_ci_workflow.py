"""The CI config is code too, and it rots the same way.

Three things bit this repo before CI existed and must not be re-introducible:
a bare `pytest` that silently skips whole modules when scikit-learn is absent,
generated notebooks that drift from their builder, and hand-typed numbers in
documentation. Each one has an assertion here.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def ci() -> str:
    assert WORKFLOW.is_file(), f"no CI workflow at {WORKFLOW}"
    return WORKFLOW.read_text()


def test_ci_installs_the_full_test_dependencies(ci):
    """A bare `pytest` skips test_probes and test_notebook_metrics_binary."""
    assert "requirements-dev.txt" in ci


def test_ci_demands_that_nothing_skips_for_a_missing_dependency(ci):
    """The variable has to be SET, not merely mentioned.

    `"CI_REQUIRE_FULL_DEPS" in ci` also matches the comment that explains it,
    so removing the env block would have left this green.
    """
    assert 'CI_REQUIRE_FULL_DEPS: "1"' in ci


def test_ci_regenerates_both_notebook_builders(ci):
    """The finetune builder reuses the baseline backbone cell: fixing one and
    regenerating only the other leaves stale notebooks on disk."""
    assert "build_baseline_tests.py" in ci
    assert "build_finetune_tests.py" in ci


def test_ci_fails_when_generated_files_drift(ci):
    """The gate compares the working tree against the builders' output and
    exits non-zero on any difference."""
    assert "notebooks/" in ci and "exit 1" in ci


def test_the_drift_gate_also_sees_brand_new_generated_files(ci):
    """`git diff` is blind to untracked paths.

    Adding the isic2019 bank made both builders emit a whole new directory,
    `notebooks/finetune_isic2019/`, and the gate stayed green with eight
    uncommitted notebooks on disk - the exact drift it exists to catch.
    """
    assert "git status --porcelain" in ci or "--intent-to-add" in ci


def test_ci_runs_the_linter(ci):
    assert "ruff" in ci


def _on_block(ci: str) -> str:
    match = re.search(r"^on:\n((?:[ \t].*\n|\n)*)", ci, re.M)
    assert match, "the workflow has no `on:` block"
    return match.group(1)


def test_ci_runs_only_when_dispatched_by_hand(ci):
    """Decision of 2026-09-29: GitHub Actions billing blocked every run, so CI
    is manual-only. It neither runs nor spends minutes on its own, and it can
    be re-enabled by restoring the push trigger. The evidence for every
    reported HEAD is the full local suite on that exact commit."""
    triggers = _on_block(ci)
    assert "workflow_dispatch" in triggers
    assert "push" not in triggers and "pull_request" not in triggers, (
        "un disparador automatico vuelve a gastar minutos de Actions")


def test_ci_pins_action_versions_instead_of_tracking_a_moving_tag(ci):
    """Unpinned actions turn a green pipeline red without a commit."""
    for action in re.findall(r"uses:\s*(\S+)", ci):
        assert "@" in action, f"{action} has no version"
        assert not action.endswith("@master"), f"{action} tracks a moving tag"


def test_requirements_dev_pins_scikit_learn():
    text = (REPO / "requirements-dev.txt").read_text()
    assert "scikit-learn" in text


def test_requirements_dev_does_not_hardcode_a_test_count():
    """`266 passed` in a comment is stale the moment a test is added - the same
    rot that hit the result tables and the checkpoint table."""
    text = (REPO / "requirements-dev.txt").read_text()
    assert not re.search(r"\d+\s+passed", text), (
        "hard-coded test count in requirements-dev.txt: it goes stale silently")


def test_a_pytest_piped_into_tee_still_fails_the_step(ci):
    """GitHub's default shell is `bash -e`, without pipefail: `pytest ... | tee` takes
    tee's exit code. Run 36531248587 went green while this step printed '1 failed'
    on py3.11 and py3.12. Every step that pipes pytest must set pipefail first."""
    steps = re.split(r"\n\s*- name: ", ci)
    piped = [s for s in steps if re.search(r"pytest[^\n]*\|\s*tee", s)]
    assert piped, "the random-order step pipes pytest into tee; finding none means this check is stale"
    for step in piped:
        run = step.split("run: |", 1)[-1]
        assert "set -o pipefail" in run.split("pytest", 1)[0], step.splitlines()[0]
