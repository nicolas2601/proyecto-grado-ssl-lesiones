"""docs/USO_MODELOS.md: tagged blocks exist, install only public packages, and run.

The clean-venv run (scripts/check_release_guides.py) needs network; here the
same Python blocks run unchanged in the current interpreter against a tiny
release, which catches a broken guide in CI without downloading anything.
"""

import os

import pytest

from scripts import check_release_guides as crg

FORMS = ("a", "b", "c")


@pytest.fixture(scope="module")
def blocks():
    return crg.extract_blocks(crg.GUIDE.read_text(encoding="utf-8"))


def test_every_form_has_an_install_and_a_code_block(blocks):
    for form in FORMS:
        assert f"guide-{form}" in blocks and f"guide-{form}-install" in blocks, form


def test_install_blocks_use_only_published_packages(blocks):
    for form in FORMS:
        install = blocks[f"guide-{form}-install"]
        for forbidden in ("dermssl", "-e ", "git+", "requirements", "proyecto"):
            assert forbidden not in install, (form, forbidden)
        assert all(line.startswith("python -m pip install ")
                   for line in install.strip().splitlines()), form


def test_parse_and_check_output():
    reference = {"mel": 0.25, "nv": 0.75}
    crg.check_output("nv\t0.750000\nmel\t0.250000\n", reference)
    with pytest.raises(AssertionError, match="differ"):
        crg.check_output("nv\t0.700000\nmel\t0.300000\n", reference)
    with pytest.raises(AssertionError, match="classes"):
        crg.check_output("nv\t1.000000\n", reference)


@pytest.fixture(scope="module")
def tiny_release(tmp_path_factory):
    for module in ("torch", "transformers", "onnxruntime", "safetensors"):
        pytest.importorskip(module)
    import release_helpers as rh

    source = rh.require_dinov2_source()
    root = tmp_path_factory.mktemp("release")
    return root, crg.build_tiny_release(root, source)


@pytest.mark.parametrize("form", FORMS)
def test_guide_code_runs_unchanged_and_matches_the_repository(blocks, tiny_release, form):
    import sys

    root, reference = tiny_release
    env = dict(os.environ)
    stdout = crg.run_block(sys.executable, blocks[f"guide-{form}"], root, env)
    crg.check_output(stdout, reference)
