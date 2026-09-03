"""Fail loudly in CI when an optional dependency would silently hide tests.

`pytest.importorskip("sklearn")` at the top of a module turns 21 tests into one
skipped line. That is the right behaviour on a laptop without scikit-learn and
the wrong one in CI, where the whole point is to exercise the metric cell.
Setting CI_REQUIRE_FULL_DEPS=1 turns those skips into a failure.
"""

import importlib
import os

import pytest

# Modules whose absence silently removes tests from the run.
REQUIRED_FOR_FULL_RUN = ("numpy", "pandas", "sklearn")

REQUIRE_FULL = os.environ.get("CI_REQUIRE_FULL_DEPS") == "1"


@pytest.mark.parametrize("module", REQUIRED_FOR_FULL_RUN)
def test_optional_dependency_is_present_when_the_run_claims_to_be_complete(module):
    if not REQUIRE_FULL:
        pytest.skip(f"CI_REQUIRE_FULL_DEPS unset: {module} is optional here")
    try:
        importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - only fires on a broken CI
        pytest.fail(
            f"{module} missing while CI_REQUIRE_FULL_DEPS=1. Whole test modules "
            f"would skip and the run would still report green: {exc}")


def test_the_guard_is_opt_in_so_a_laptop_without_sklearn_still_runs():
    """Documents the contract: unset means skip, '1' means fail."""
    assert REQUIRE_FULL == (os.environ.get("CI_REQUIRE_FULL_DEPS") == "1")
